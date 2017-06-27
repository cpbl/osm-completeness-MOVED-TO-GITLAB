#!/usr/bin/python
# coding=utf-8
"""
The multilevel modeling code requires PostGIS rasters of:
- density (we use Landscan)
- country codes (GADM id_0)

Both rasters should have global coverage at 30 arc-second resolution, i.e. 21600 x 43200

This script will load the rasters into a PostGIS database. This will almost certainly require some tweaking to match your database setup.

To invoke:
python setup_postgis.py

You will need the following files in your input directory:
gadm28.shp (global administrative boundaries). Download the whole-world shapefile from http://www.gadm.org/version2
LandScan2013.zip (density raster in ESRI GRID format). Download after requesting permission from http://web.ornl.gov/sci/landscan/landscan_data_avail.shtml. 

You can use other data sources (e.g. OSM boundary data), but you'll need to match the format of gadm and Landscan

An alternative to loading the rasters in PostGIS is to create them as numpy arrays in Python, 
and save them as a numpy .npz file to your scratch directory
They should be saved as a 21600x43200x2 array, with the density and countrycode rasters stacked as in:
    np.dstack([density_array, countrycode_array])
The file gadmCodes.tsv gives the GADM id_0 country codes, should you wish to create the array yourself from another source
"""

import os, sys, math
import numpy as np
import pandas as pd
from history_config import *
import history_tools as ht
from cpblUtilities.parallel import runFunctionsInParallel

GADMFILE=paths['input']+'gadm28.shp'
GADM_ID_LEVELS = ['id_0','id_1','id_2','id_3','id_4','id_5']
TILESIZE = 30   # each tile is 30x30 pixels. Consistent across all rasters, so hardcode here

def import_raw_GADM_into_postGIS(forceUpdate=False):   
    """Imports the gadm shapefiles into PostGIS and cleans some things up
    """
    db=ht.dbConnection()
    if not(os.path.exists(GADMFILE)):
        raise Exception('%s not found. Download it from gadm.org' % GADMFILE)
    if gadmTableName in db.list_tables(schema=pgLogin['gadmschema']) and not forceUpdate:
        print(' Skipping import_raw_GADM_into_postGIS because '+gadmTableName+' table already exists...')
        return
    if forceUpdate:
        print('Deleting all gadm tables...')
        for table in db.list_tables(schema=pgLogin['gadmschema']):
            if gadmTableName in table:
                db.execute('DROP TABLE '+table)
    print('Loading GADM data into pgis...')
    os.system("""shp2pgsql -s 4326 -g the_geom -I -W "latin1" """+GADMFILE+' '+pgLogin['gadmschema']+'.'+gadmTableName+"""  | psql -q %s """%(db.psql_command_line_flags())) 

    # Drop Antarctica, because that intersects the South Pole and means reprojection not possible
    db.execute('''SELECT count(*) FROM '''+gadmTableName+''' WHERE ST_Intersects(the_geom, ST_GeomFromText('Linestring(-180 -90, 180 -90)', 4326))''')
    nDrop = db.fetchall()[0][0]
    print 'Dropping %d rather cold features that intersect the South Pole' % nDrop
    db.execute('''DELETE FROM '''+gadmTableName+''' WHERE ST_Intersects(the_geom, ST_GeomFromText('Linestring(-180 -90, 180 -90)', 4326))''')

    if 'gadm28.shp' in GADMFILE:
        # Drop country XCA, (id_0=44 in 2.8), which appears to be a single row = the Caspian Sea
        db.execute('''DELETE FROM '''+gadmTableName+''' WHERE ISO='XCA';''')
        # Drop erroneous geometries that were picked up in tests and manually inspected (e.g. bits of Manitoba in the middle of Greenland)
        gids = [97928, 31917] 
        for gid in gids:
            cmd = """WITH t1 AS (SELECT the_geom g1 FROM %s WHERE gid=%s),
                      t2 AS (SELECT the_geom g2 FROM %s, t1 WHERE gid!=%s AND ST_Intersects(the_geom, g1) AND NOT ST_Touches(the_geom, g1)),
                      t3 AS (SELECT ST_Union(ST_CollectionExtract(ST_Intersection(g1, g2),3)) AS geom_intersect FROM t1, t2)
                  UPDATE %s SET the_geom = ST_Multi(ST_SymDifference(g1, geom_intersect)) FROM t1, t3 
                  WHERE gid=%s""" % (gadmTableName, gid, gadmTableName, gid, gadmTableName, gid) 
            db.execute(cmd)

    print('Fixing invalid geoms')
    invalidGeoms = db.execfetch('''UPDATE %s SET the_geom = ST_MakeValid(the_geom) WHERE not(ST_IsValid(the_geom)) RETURNING gid, iso;''' % gadmTableName)
    print('Fixed %d invalid geoms (please check in QGIS that the fix works) with the following gids and isos:\n %s' % (len(invalidGeoms),invalidGeoms))
    invalidGeoms = db.execfetch('SELECT gid, iso FROM %s WHERE not(ST_IsValid(the_geom))' % gadmTableName)
    assert invalidGeoms is None or len(invalidGeoms)==0

    db.execute('DROP INDEX IF EXISTS %s_the_geom_idx ;' % gadmTableName)  # duplicate index created by shp2pgsql
    db.execute('CREATE INDEX %s_idx_iso ON %s.%s (ISO);' % (gadmTableName, pgLogin['gadmschema'], gadmTableName))
    db.execute('CREATE INDEX %s_idx_id1 ON %s.%s (id_1);' % (gadmTableName, pgLogin['gadmschema'], gadmTableName))
    db.execute('CREATE INDEX %s_spat_idx  ON %s.%s  USING gist  (the_geom);' % (gadmTableName, pgLogin['gadmschema'], gadmTableName))
    return(gadmTableName)

def createGADMCountryRaster(iso=None, hemisph=None, id_1s=None, forceUpdate=False):
    """
    Rasterize GADM ids for an individual country (in parallel), and align with the Landscan density raster
       Then merge them all together into a global raster
       If hemisph=='E' or 'W', only does one hemisphere (for countries that straddle 180 degrees
       If id_1s is not None, do a subset (needed for Russia, which is too big)
    """
    rasterSchema = pgLogin['landscanschema']
    db = ht.dbConnection(schema=rasterSchema)
    if 'gadmraster' in db.list_tables(schema=rasterSchema) and not forceUpdate:
        print 'GADM raster table already exists...skipping'
        return
        
    # Create a raster table for each individual country, in parallel
    if iso is None:
        isoDict = ht.get_all_GADM_countries()
        EWcountries = ['UMI','NZL','USA','KIR','FJI','RUS']  # cross 180 degrees longitude
        parallel = True
        funcs,names=[],[]

        # Split Russia and Canada up
        russiaId_1s = [ii[0] for ii in db.execfetch("SELECT DISTINCT id_1::integer FROM "+gadmTableName+" WHERE iso='RUS'")]
        canadaId_1s = [ii[0] for ii in db.execfetch("SELECT DISTINCT id_1::integer FROM "+gadmTableName+" WHERE iso='CAN'")]
        funcs+=[[createGADMCountryRaster,['RUS'],{'hemisph':'E', 'id_1s':russiaId_1s[:40],  'forceUpdate':forceUpdate}]]
        funcs+=[[createGADMCountryRaster,['RUS'],{'hemisph':'E', 'id_1s':russiaId_1s[40:60],'forceUpdate':forceUpdate}]]
        funcs+=[[createGADMCountryRaster,['RUS'],{'hemisph':'E', 'id_1s':russiaId_1s[60:],  'forceUpdate':forceUpdate}]]
        funcs+=[[createGADMCountryRaster,['CAN'],{'id_1s':canadaId_1s[:3], 'forceUpdate':forceUpdate}]] # Alberta, BC, Manitoba
        funcs+=[[createGADMCountryRaster,['CAN'],{'id_1s':canadaId_1s[3:], 'forceUpdate':forceUpdate}]]
        names+=['GADM raster RUS_E1','GADM raster RUS_E2','GADM raster RUS_E3','GADM raster CAN1','GADM raster CAN2']

        # Now add regular countries
        for iso in isoDict:
            if iso in EWcountries or iso=='CAN': continue
            funcs+=[[createGADMCountryRaster,[iso], {'forceUpdate':forceUpdate}]]
            names+=['GADM raster %s'%iso]
        for iso in EWcountries: # do the eastern hemisphere for countries that cross the dateline
            for hemisph in ['E','W']:
                if iso=='RUS' and hemisph=='E': continue
                funcs+=[[createGADMCountryRaster,[iso],{'hemisph':hemisph,'forceUpdate':forceUpdate}]]
                names+=['GADM raster %s_%s'%(iso,hemisph)]

        runFunctionsInParallel(funcs,names=names,maxAtOnce=None, parallel=parallel)
        
        # Aggregate all rids
        db.execute('DROP TABLE IF EXISTS gadmraster')
        tableNames = [tt for tt in db.list_tables() if tt.startswith('gadmraster_')]
        unionNames = ' UNION ALL '.join(['SELECT * FROM '+tname for tname in tableNames])
        cmd = """CREATE TABLE """+rasterSchema+""".gadmraster AS 
                    SELECT rid, ST_Union(rast) AS rast FROM  ("""+unionNames+""") t1
                    GROUP BY rid"""
        print 'Creating union of all individual country rasters...'
        db.execute(cmd)
        # add constraints and index
        print 'Adding constraints and index...'
        rc = ht.rasterCon('gadmraster')
        rc.addConstraints(setAllExceptExtent=True)
        rc.execute("""ALTER TABLE gadmraster ADD PRIMARY KEY (rid);""")
        rc.addSpatialIndex()        
        for tname in tableNames: db.execute("DROP TABLE "+tname)

        return
    
    countryRastName = 'gadmraster_'+iso.replace('-','_')  # the replace is for SP-
    if hemisph is not None: countryRastName+=hemisph
    if id_1s is not None: countryRastName+=str(id_1s[0])
    if countryRastName in db.list_tables() and not forceUpdate: return
        
    db.execute('DROP TABLE IF EXISTS '+countryRastName)
    # max id value is 41932. So we can fit this in a 16BUI, with a nodata value of 65535
    idLevels = ', '.join(GADM_ID_LEVELS).replace('iso', 'id_0').replace("'","")
    pixTypes   = "ARRAY[" + ", ".join(["'16BUI'"]*len(GADM_ID_LEVELS)) + "]"
    noDataVals = "ARRAY[" + ", ".join(["65535"]*len(GADM_ID_LEVELS)) + "]"
    addBandArgs = ", ".join(["ROW(Null, '16BUI', 65535, 65535)"]*len(GADM_ID_LEVELS))
    ls = ht.rasterCon('landscan')
    ycoord = 'YMax' if ls.get_scale()[0]<0 else 'YMin'  # take the max of the bbox if the yscale is negtive
        
    # What's the strategy here? These people suggest using ST_MapAlgebra to burn in tiles, but that's SLOW and runs out of memory
    # http://geospatialelucubrations.blogspot.com/2014/05/a-guide-to-rasterization-of-vector.html
    # So we create a union of (i) a blank tile in the bottom left corner, which controls the tile alignment
    # This is the raster that is created with ST_AddBand(ST_MakeEmptyRaster(rast)...
    # and (ii) the rasterized geometry. 
    
    # Clause to get GADM polygons is different if the country crosses 180 degrees.
    if hemisph is not None: 
        polyClause = """WITH gadmpolys AS (SELECT * FROM (SELECT """+idLevels+""", (ST_Dump(the_geom)).geom AS geom 
                        FROM """+gadmTableName+""" WHERE iso = '"""+iso+"""') gp1 WHERE ST_X(ST_Centroid(geom)) > 0)"""  
        if hemisph=='W': polyClause = polyClause.replace('>', '<')   
    else:
        polyClause = """WITH gadmpolys AS (SELECT """+idLevels+""", the_geom AS geom 
                        FROM """+gadmTableName+""" WHERE iso = '"""+iso+"""')"""
    if id_1s is not None: polyClause=polyClause[:-1] + """ AND id_1 IN """+str(tuple(id_1s))+""")"""

    cmd = """CREATE TABLE """+pgLogin['landscanschema']+"."+countryRastName+""" AS """ + polyClause+"""                 
                SELECT ST_Tile(ST_Union(rast), ARRAY"""+str(range(1,len(GADM_ID_LEVELS)+1))+""", 30, 30, True) rast FROM (
                    SELECT ST_AddBand(ST_MakeEmptyRaster(rast), ARRAY["""+addBandArgs+"""]::addbandarg[]) rast
	                    FROM """+rasterSchema+""".landscan,
	                        (SELECT ST_SetSRID(ST_Point(ST_XMin(bbox), ST_"""+ycoord+"""(bbox)), 4326) AS pt FROM 
                            (SELECT ST_Envelope(ST_Collect(geom)) AS bbox FROM gadmpolys) t1) t2
	                    WHERE ST_Intersects(pt, rast)
            UNION
                SELECT ST_AsRaster(geom, 1/120::double precision, -1/120::double precision, 0, 0, """+pixTypes+""", 
                                   value:=ARRAY["""+idLevels+"""]::integer[], nodataval:="""+noDataVals+""")  AS rast
                FROM gadmpolys) t3;"""
    db.execute(cmd)
    db.execute("""ALTER TABLE """+countryRastName+""" ADD COLUMN rid integer""")
    cmd = """UPDATE """+countryRastName+""" r1 SET rid = r2.rid FROM landscan r2 
                WHERE ST_Intersects(r1.rast, r2.rast) 
                    AND round(ST_UpperLeftX(r1.rast)::numeric, 10)=round(ST_UpperLeftX(r2.rast)::numeric, 10)
                    AND round(ST_UpperLeftY(r1.rast)::numeric, 10)=round(ST_UpperLeftY(r2.rast)::numeric, 10)"""
    db.execute(cmd)
    if hemisph is not None: 
        db.execute("""DELETE FROM """+countryRastName+""" WHERE rid is Null AND ST_UpperLeftX(rast)::integer=180""")
    db.execute("""ALTER TABLE """+countryRastName+""" ADD PRIMARY KEY (rid);""")

def loadLandScan(forceUpdate=False):
    """ 
    Uploads the LandScan population and density data to postgres
    Note: units are people per sq km
    
    The challenges are (i) that population counts and areas are in different rasters, and 
      (ii) the area raster values are actually a lookup (values are rownumbers to the lookup table)
      (ArcGIS does the lookup automatically, but GDAL doesn't seem to handle this yet)
      
    The landscan raster will have a single 3 band raster:
        Band 1: population (number of people)
        Band 2: area (sq km)
        Band 3: density (persons km-2)
    """
    lsPath = paths['scratch']+'landscan/'
    if not(os.path.exists(lsPath)): os.mkdir(lsPath)
    rasterSchema = pgLogin['landscanschema']
    
    ls = ht.rasterCon('landscan') 
    if 'landscan' in ls.list_tables(schema=rasterSchema) and not forceUpdate: return
    
    # unzip files
    os.system('unzip %sLandScan2013.zip -d %s' % (paths['input'], lsPath))
    
    # upload rasters. Note that we don't have an extent constraint (which is very slow, so use -x). 
    # Use 30x30 tile size (i.e, tile raster into 1 degree squares)
    print('Uploading population raster')
    cmd = 'raster2pgsql -I -b 1 -t "%sx%s" -s 4326 %s -d -C -x %s.landscan | psql -q %s' % (TILESIZE, TILESIZE, lsPath + 'ArcGIS/Population/lspop2013', rasterSchema, ls.psql_command_line_flags())
    print(cmd)
    os.system(cmd)
    os.system('psql -c "ALTER TABLE %s.landscan OWNER to osmusers" %s' % (rasterSchema, ls.psql_command_line_flags()))

    # The only reason we load this in is to confirm that the alignment is OK. We don't use this raster, the values are actually lookups, not areas
    # To do this, change if 1: to if 0:
    if 1:
        print('Uploading area raster. This is for testing only. If you are on a slow connection, you may want to skip this step')
        cmd = 'raster2pgsql -I -b 1 -t "%sx%s" -s 4326 %s -d -C -x %s.landscanarea | psql -q %s' % (TILESIZE, TILESIZE, lsPath + 'ArcGIS/AreaGrid/areagrd', rasterSchema, ls.psql_command_line_flags())
        print(cmd)
        os.system(cmd)
        cmd = 'psql -c "ALTER TABLE %s.landscanarea OWNER to osmusers" %s' % (rasterSchema, ls.psql_command_line_flags())
        print(cmd)
        os.system(cmd)

        # The tricky bit is how to figure out how to match the row number of the population raster with the row number of the area raster
        # If we calculate it, it avoids problematic alignment issues due to floating point precision
        tilesPerRow = 360 * 60 * 2 / TILESIZE  # 30 arc-second tiles, 30 pixels per tile
        for rid in [1, 1+tilesPerRow]: # check that our tile calculation is right
            ls.execute('SELECT ST_UpperLeftx(p.rast) AS popy, ST_UpperLeftx(a.rast) AS areay FROM landscan as p, landscanarea a WHERE p.rid=%d AND a.rid=%d' % (rid, rid))
            assert ls.fetchall()[0] == [-180, -180]

        # how many extra rows (pixels, not tiles) do we have on the population raster?
        popUpperLeftY, areaUpperLeftY = ls.execfetch('SELECT ST_UpperLeftY(p.rast) AS popy, ST_UpperLeftY(a.rast) AS areay FROM landscan as p, landscanarea a WHERE p.rid=1 AND a.rid=1')[0]
        extraHeaderPixels = int(np.round((popUpperLeftY-areaUpperLeftY) * 60 * 2, 5))   # round is to deal with floating point precision issues
        assert extraHeaderPixels==np.round((popUpperLeftY-areaUpperLeftY) * 60 * 2, 5)
 
        # how many landscanarea rows do we have?
        nAreaRows = ls.execfetch('SELECT MAX(rid) FROM landscanarea')[0][0]*1.0/tilesPerRow
 
        # another check on our math
        result = ls.execfetch('''SELECT pmax, amax, ST_UpperLeftY(p.rast) AS popy, ST_UpperLeftY(a.rast) AS areay FROM landscan AS p, landscanarea AS a,
                        (SELECT max(rid) AS pmax FROM landscan) t1, (SELECT max(rid) AS amax FROM landscanarea) t2
                        WHERE p.rid=pmax AND a.rid=amax''')[0]
        assert result[0] - result[1] == extraHeaderPixels / TILESIZE * tilesPerRow
        assert np.round(result[2], 10) == np.round(result[3], 10)  # bottom row should be the same
    
    # Load the lookup table. Manually exported from ArcGIS. 
    # Note that row1 is at the BOTTOM, so this confuses things
    areaLookup = pd.read_csv(paths['bin']+'areaLookup.csv', usecols=['VALUE', 'AREA'], thousands=',')
    areaLookup.rename(columns={'VALUE':'areaRow'}, inplace=True)
    areaLookup['areaRid'] = areaLookup.areaRow.apply(lambda x: (math.floor((x-1)/TILESIZE)*-1+nAreaRows-1)*tilesPerRow+1)
    areaLookup['popRid']  =  areaLookup.areaRid.apply(lambda x: x+extraHeaderPixels/30*tilesPerRow)
    areaLookup['pixelY']  = areaLookup.areaRow.apply(lambda x: TILESIZE-(x-1)%TILESIZE) # within each raster image, what's the pixel?
    areaLookup.set_index('pixelY', inplace=True)

    # Now we have to use a loop, because otherwise the length of the lookup slows things down crazily
    # The loop allows us to use a smaller lookup
    # For use of reclass, see http://gis.stackexchange.com/questions/127300/postgis-replace-pixel-value-by-lookup-table
    maxRid = ls.execfetch('SELECT MAX(rid) FROM landscan')[0][0]
    minRid = extraHeaderPixels*tilesPerRow/TILESIZE+1  # this is the first rid that has area info
    
    # add two empty bands for tiles where we don't have an area estimate
    ndValue = ls.get_noDataValues()
    ls.dropConstraints(['nodata_values','num_bands','pixel_types','out_db'])  # otherwise, we can't add a new band
    ls.execute('''UPDATE landscan
                    SET rast = ST_AddBand(rast, ARRAY[ROW(2, '64BF', %(ndv)s, %(ndv)s), ROW(NULL, '64BF', %(ndv)s, %(ndv)s)]::addbandarg[]) 
                    WHERE rid<%(minRid)s''' %dict(ndv=ndValue,minRid=minRid))
    
    for ii, rid in enumerate(range(minRid, maxRid+1, tilesPerRow)):
        if ii%50==0: print 'Now doing rid %d of %d' % (rid-minRid, maxRid-minRid+1)

        areaLookup_partial = str(areaLookup[areaLookup.popRid==rid].AREA.to_dict())[1:-1]
        assert areaLookup_partial!='' and (areaLookup.popRid==rid).sum()==30

        # Add in area as band 2, and density as band 3
        # Note: ST_MapAlgebra(pop.rast, 1, '16BSI', '[rast1.y]', ndv) returns the pixel row number (1-based)
        # ST_Reclass(ST_MapAlgebra(pop.rast, 1, '16BSI', '[rast1.y]', ndv), 1, '%s', '32BF', ndv) uses the pixel row number to look up the area
        ls.execute('''UPDATE landscan
                        SET rast = ST_AddBand(rast, array[ST_Reclass(ST_MapAlgebra(rast, 1, '16BSI', '[rast1.y]', %(ndv)s), 1, '%(alp)s', '64BF', %(ndv)s),
                                     ST_MapAlgebra(rast, ST_Reclass(ST_MapAlgebra(rast, 1, '16BSI', '[rast1.y]', %(ndv)s), 1, '%(alp)s', '64BF', %(ndv)s), 
                                            '[rast1]/[rast2]', '64BF', 'INTERSECTION', 'Null', 'Null', Null)]) 
                        WHERE rid>=%(minRid)s AND rid<%(maxRid)s''' %dict(alp=areaLookup_partial,ndv=ndValue,minRid=rid,maxRid=rid+tilesPerRow))
    
    print('Adding back raster constraints')
    ls.addConstraints(['nodata_values','num_bands','pixel_types','out_db']) # Add back constraints
    
    # Floating point precision errors mean that the grid is slightly off. 
    print('Snapping raster to grid')
    ls.execute('UPDATE landscan SET rast=ST_SnapToGrid(rast, 0, 0, 1/120::double precision, -1/120::double precision, maxerr:=0.0001)')
    ls.addSpatialIndex()

    # clean up
    ls.execute('DROP TABLE IF EXISTS landscanarea')
    cmd = 'rm -r %s' % lsPath
    print(cmd)
    os.system(cmd)
    print('Completed loadLandScan\n')

    return

if __name__ == '__main__':
    runmode=None if len(sys.argv)<2 else sys.argv[1].lower()
    forceUpdate = any(['forceupdate' in arg.lower() for arg in sys.argv])
    print 'forceUpdates is %s' % forceUpdate   
    if runmode in [None,'gadm']:
        import_raw_GADM_into_postGIS(forceUpdate=forceUpdate)
    if runmode in [None,'landscan']:
        loadLandScan(forceUpdate=forceUpdate)
    if runmode in [None,'gadmraster']:
        createGADMCountryRaster(forceUpdate=forceUpdate)