#!/usr/bin/python
"""
Various tools for use with osm history analysis
"""
from history_config import *
import numpy as np

class dbConnection():
    def __init__(self, schema=None, curType='DictCursor'):
        """
        This returns a connection to the database.
        Adjust the pgLogin dictionary in history_config.py to connect to your own database

        curType right now is either DictCursor or the default (tuples)
        note that default is default for psycopg2, not the default for this function
        Note: this is a lightweight adaptation of the version in postgres_tools.py
        """
        import psycopg2, psycopg2.extras
        assert curType in ['DictCursor', 'default']
        self.curType=curType
        # Connect to database
        coninfo = ' '.join([{'db':'dbname','pw':'password'}.get(key,key)+' = '+val   for key,val in pgLogin.items() if 'schema' not in key])
        con = psycopg2.connect(coninfo)
        con.set_isolation_level(0)   # autocommit - see http://stackoverflow.com/questions/1219326/how-do-i-do-database-transactions-with-psycopg2-python-db-api
        self.cursor=None
        self.default_schema=pgLogin['gadmschema'] if schema is None else schema
        self.connection=con
        # See documentation for schemas. Search path order determines  schema use. Putting the default first ensures new tables will be created there if not specified explicitly.
        self.search_path=[pgLogin['gadmschema'],pgLogin['landscanschema'],'public']
        self.execute('SET search_path = '+','.join(self.search_path))   
        assert not self.cursor.closed
        self.refreshCursor=True

    def cur(self):
        """ Create a new cursor. This can be done frequently.
        The cursor is a lightweight object, and can be deleted after each use. That might help with postgres memory use.

        The latest cursor is always available as self.cursor but the only intended use is outside calls of the form:
          thisobject.cur().execute('pg command')
        """
        import psycopg2, psycopg2.extras
        if self.curType=='DictCursor':
            cur = self.connection.cursor(cursor_factory=psycopg2.extras.DictCursor)  # So evertying coming from the database will be in POython dict format.
        elif self.curType=='default': # will return data as tuples - easier to convert to pandas
            cur = self.connection.cursor()
        del self.cursor
        self.cursor=cur
        return cur

    def execute(self,cmd):
        if self.cursor is None:
            self.cur()
        return (self.cursor.execute(cmd))

    def fetchall(self):
        fout=self.cursor.fetchall()
        if self.refreshCursor:
                del self.cursor
                self.cursor=None
        return fout

    def execfetch(self,cmd):
        """Execute an SQL command and fetch (fetchall) the result"""
        self.execute(cmd)
        return self.fetchall()
        
    def list_tables(self,schema=None):
        """ List all tables in the schema. Views are not tables, so don't list them.
        TODO: Could add an "all_schemas" option?
        """
        if not schema:
            schema=self.default_schema
        cmd="SELECT table_name FROM information_schema.tables WHERE table_type != 'VIEW' and table_schema = '"+schema+"';"
        return [tt[0] for tt in self.execfetch(cmd)]

    def psql_command_line_flags(self):
        """ Return a string to be used in a psql command line to identify host, user, and database.

        Not yet tested with passwords
        """
        outs = '' if 'host' not in pgLogin else '-h %s ' % pgLogin['host']
        outs += ' -d %s -U %s ' % (pgLogin['db'], pgLogin['user'])
        return outs
  
def get_all_GADM_countries():
    """Returns a dict of countryISO, gadm id_0 and name"""
    import pandas as pd
    GADMcountries = pd.read_csv(paths['bin']+'gadmCodes.tsv',sep='\t').set_index('isoalpha3')
    return GADMcountries.to_dict('index')
    
def country2ISOLookup():
    """
    Returns dict of dicts
    cname2ISO is country name to ISO alpha 3 lookup
    ISO2cname is the reverse
    ISO2shortName is a consistent dict of short names (they might not be official)
        which also strip out non-ASCII128 characters (sorry, Cote d'Ivoire...)
    """
    import pandas as pd
    lookup = pd.read_table(paths['bin']+'countrycodes.tsv', dtype={'ISO3digit':'str'})
    cname2ISO = lookup.set_index('countryname').ISOalpha3.to_dict()
 
    ISO2cname = lookup.set_index('ISOalpha3').countryname.to_dict()
    ISO2cname['ALL'] = 'World'
    ISOalpha2ISOdigit = lookup.set_index('ISOalpha3').ISO3digit.to_dict()
    ISOdigit2ISOalpha = lookup.set_index('ISO3digit').ISOalpha3.to_dict()

    lookup = pd.read_table(paths['bin']+'shortNames.tsv', dtype={'ISO3digit':'str'})
    ISOalpha2shortName = lookup.set_index('ISOalpha3').shortName.to_dict()
    ISOalpha2shortName['ALL'] = 'World'
    
    # Add in some alternate names manually (different variants of country name)
    # these are only used in going TO iso from the country name
    cname2ISO.update({'Russia': 'RUS', 'The Bahamas': 'BHS', 'United Republic of Tanzania': 'TZA', 'Ivory Coast': 'CIV', 
                      'Republic of Serbia': 'SRB', 'Guinea Bissau': 'GNB', 'Iran': 'IRN', 'Democratic Republic of the Congo': 'COD',
                      'Republic of Congo': 'COG', 'Syria': 'SYR', 'Venezuela': 'VEN', 'Bolivia': 'BOL', 'South Korea': 'KOR', 'Laos': 'LAO',
                      'Brunei': 'BRN', 'East Timor': 'TLS', 'Vietnam': 'VNM',  'North Korea': 'PRK', 'Moldova': 'MDA', 'Vatican City': 'VAT',
                      'Macedonia': 'MKD', 'United Kingdom': 'GBR', 'Tanzania':'TZA', 'Cape Verde':'CPV', 'Reunion':'REU', 'Falkland Islands':'FLK', 
                      'Micronesia':'FSM', 'United States':'USA'})
    
    return {'cname2ISO':cname2ISO, 'ISO2cname': ISO2cname, 'ISOalpha2ISOdigit': ISOalpha2ISOdigit, 'ISOdigit2ISOalpha': ISOdigit2ISOalpha, 'ISOalpha2shortName':ISOalpha2shortName }

def country2WBregionLookup():
    """
    Returns lookup of World Bank region to country (by ISOalpha3)
    Indexed by WB region code ('GroupCode')
    """
    import pandas as pd
    WB2ISO = loadOtherCountryData().reset_index()[['WBcode','ISOalpha3']]
    WBregions = pd.read_csv(paths['bin']+'WBregions.csv')
    WBregions = WBregions.merge(WB2ISO, left_on='CountryCode', right_on='WBcode').set_index('GroupCode')
    WBregions.drop('CountryCode', axis=1, inplace=True) # duplicate to WBcode
    topWBregions = ['World','Africa',
                          'European Union',
       'East Asia & Pacific (all income levels)',
                          'South Asia', 'North America' ,
       'Latin America & Caribbean (all income levels)',
         'Central Europe and the Baltics',
                          'High income',                          'Middle income',       'Low income',]
    # Put the above list first
    notIntop = list(set(WBregions.GroupName.unique())-set(topWBregions))
    lwb = len(WBregions)
    WBregions = WBregions.reset_index().set_index('GroupName').loc[topWBregions+notIntop].reset_index().set_index('GroupCode')
    assert len(WBregions) == lwb
    return WBregions
        
def loadOtherCountryData():
    """Loads pandas dataframe with World Bank and other data
    See README-data-release.txt for details, and country_datadictionary.tsv for the data dictionary"""
    import pandas as pd
    df = pd.read_pickle(paths['input']+'countries_compiled.pandas')
    colsToUse = [cc for cc in df.columns if not(cc.startswith('frcComplete_')) and not(cc.startswith('length_')) and not(cc in ['OSMlength','method','max_length','fitfunction','MSE'])]  # only the World Bank, etc. data - not the columns that are generated as part of the analysis
    assert df.index.name=='ISOalpha3'
    return df[colsToUse]

def compileAllPlots(figs,figFn,figSize=None):
    """Take a list of figures and clean up and generate a single PDF"""
    from cpblUtilities import mergePDFs
    from cpblUtilities.mathgraph import remove_underscores_from_figure
    if figSize is None:
        try:
            figSize = figSizePage # if in globals
        except:
            figSize = (7, 8.75)  # for full-page figures

    outList = []
    for fignum, fig in enumerate(figs):
        fig.set_size_inches(figSize)
        remove_underscores_from_figure()
        fig.tight_layout()
        outFn = figFn+str(fignum)+'.pdf'
        fig.savefig(outFn)
        outList.append(outFn)
    mergePDFs(outList, figFn+'.pdf')  
    
class rasterCon(dbConnection):
    """  Inherit postgres stuff from dbConnection and add raster-specific methods."""
    def __init__(self, name=None, schema=pgLogin['landscanschema'], host=pgLogin['host'], curType='DictCursor'):
        dbConnection.__init__(self,curType=curType)
        self.name = name.lower()
        self.schema = schema.lower()
        self.alignments = {}

    def get_noDataValues(self):
        """Returns no data value, or a list if there is more than one band"""
        if not(hasattr(self, 'noDataValues')):
            bands = range(1, self.get_nBands()+1)
            cmd = 'SELECT '+', '.join(['ST_BandNoDataValue(rast, '+str(bb)+')' for bb in bands])+' FROM '+self.name+' LIMIT 1;'
            result = self.execfetch(cmd)[0]
            self.noDataValues = result[0] if len(result)==1 else dict([(ii,result[ii-1]) for ii in bands])
        return self.noDataValues
        
    def get_bbox(self):
        """Returns  ((MINX, MINY), (MINX, MAXY), (MAXX, MAXY), (MAXX, MINY), (MINX, MINY))"""
        if not(hasattr(self, 'bbox')):
            self.get_tileSizeDegrees()
            assert self.tileSizeDegrees[1]>0 # not set up to deal with tiles that go from right to left
            cmd = 'SELECT MIN(ST_UpperLeftX(rast)), MAX(ST_UpperLeftX(rast)), MIN(ST_UpperLeftY(rast)), MAX(ST_UpperLeftY(rast))  FROM '+self.name+';'
            xmin,xmax,ymin,ymax = self.execfetch(cmd)[0]
            self.xmin = xmin
            self.xmax = xmax+self.tileSizeDegrees[1]
            self.ymin = ymin if self.tileSizeDegrees[0] > 0 else ymin+self.tileSizeDegrees[0]
            self.ymax = ymax if self.tileSizeDegrees[0] < 0 else ymax+self.tileSizeDegrees[0]
            self.bbox = ((self.xmin, self.ymin), (self.xmin, self.ymax), (self.xmax, self.ymax), (self.xmax, self.ymin), (self.xmin, self.ymin))
        return self.bbox
    
    def get_scale(self):
        """Returns tuple of scaleY and scaleX"""
        if not(hasattr(self, 'scale')): 
            self.scale = tuple(self.execfetch('SELECT ST_ScaleY(rast), ST_ScaleX(rast) FROM '+self.name+' LIMIT 1')[0])
        assert self.scale[0]<0 and self.scale[1]>0   # They don't have to be, but we need to check these functions work with inverted scales
        return self.scale
    
    def get_tileSizeDegrees(self):
        """Returns height first, width second as tuple"""
        if self.getSRID()!=4326: raise Exception('rasterCon().get_tileSizeDegrees() only deals with SRID 4326 for now')
        if not(hasattr(self, 'tileSizeDegrees')):
            # avoid floating point precision issues by doing this in PostGIS
            cmd = 'SELECT ST_Height(rast)*ST_ScaleY(rast), ST_Width(rast)*ST_ScaleX(rast) FROM '+self.name+' LIMIT 1;'
            self.tileSizeDegrees = tuple(self.execfetch(cmd)[0])  
        return self.tileSizeDegrees
            
    def get_tileSizePixels(self):
        if not(hasattr(self, 'tileSizePixels')):
            cmd = 'SELECT ST_Height(rast), ST_Width(rast) FROM '+self.name+' LIMIT 1;'
            self.tileSizePixels = tuple(self.execfetch(cmd)[0])
        return self.tileSizePixels

    def get_nTiles(self):
        if not(hasattr(self, 'nTiles')):
            self.nTiles = self.execfetch('SELECT COUNT(*) FROM '+self.name+';')[0][0]
        return self.nTiles
        
    def get_nTilesYX(self):
        """Returns tuple of rows and columns"""
        th, tw = self.get_tileSizeDegrees()
        bbox = self.get_bbox()
        nRows = abs(int((bbox[1][1]-bbox[0][1])/th))
        nCols = abs(int((bbox[2][0]-bbox[0][0])/tw))
        self.nTilesXY = (nRows, nCols)
        return self.nTilesXY

    def get_nBands(self):
        if not(hasattr(self, 'nBands')): self.nBands = self.execfetch('SELECT ST_NumBands(rast)  FROM '+self.name+' LIMIT 1')[0][0]
        return self.nBands    

    def get_lonLat(self, lonlat=None, band=1):
        if self.getSRID()!=4326: raise Exception('rasterCon().get_lonLat() only deals with SRID 4326 for now')
        if isinstance(lonlat, list): lonlat = tuple(lonlat)
        assert isinstance(lonlat, tuple)   
        cmd = """WITH p AS (SELECT ST_SetSRID(ST_MakePoint%(lonlat)s, 4326) pt)                  
                 SELECT ST_Value(rast,%(band)s,pt) FROM %(name)s, p 
                 WHERE ST_Intersects(rast,pt);""" % dict(name=self.name, band=band, lonlat=str(lonlat))
        result = [rr[0] for rr in self.execfetch(cmd) if rr[0] is not None]
        return np.nan if result==[] else result[0] if len(result)==1 else result

    def sameAlignnment(self, raster2):
        if not(raster2 in self.alignments):
            cmd ='SELECT DISTINCT ST_SameAlignment(r1.rast,r2.rast) FROM '+self.name+' r1, '+raster2+' r2 WHERE r1.rid=r2.rid;'
            result = self.execfetch(cmd)[0]
            self.alignments.update({raster2: (result[0] is True and len(result)==1)})
        return self.alignments[raster2]
 
    def getSRID(self):
        if not(hasattr(self, 'srid')):
            srid = self.execfetch("SELECT srid from raster_columns where r_table_name='"+self.name+"';")
            if srid:
                self.srid = srid[0][0]
            else: # table not yet defined
                self.srid = None
        return self.srid
    
    def addSpatialIndex(self):
        self.execute('DROP INDEX IF EXISTS '+self.name+'_spat_idx;')
        self.execute('CREATE INDEX '+self.name+'_spat_idx ON '+self.name+' USING gist (ST_ConvexHull(rast));')
        return
    
    def addConstraints(self, constraints=None, setall=None, setAllExceptExtent=None):
        # Raster constraints make sure all tiles have the same srid, scale, etc. See http://postgis.net/docs/manual-2.2/RT_AddRasterConstraints.html
        assert (constraints is not None) ^ (setall is not None) ^ (setAllExceptExtent is not None)
        if isinstance(constraints, str): constraints = [constraints]
        allConstraints = ['srid','scale_x','scale_y','blocksize_x','blocksize_y','same_alignment','regular_blocking','num_bands','pixel_types','nodata_values','out_db','extent']
        if setall:
            constraints = allConstraints
        if setAllExceptExtent:
            constraints = allConstraints[:-1]
        cmd = "SELECT AddRasterConstraints('%(schema)s'::name, '%(name)s'::name, 'rast'::name, "%dict(schema=self.schema, name=self.name)
        cmd+=", ".join([cc+":=True" if cc in constraints else cc+":=False" for cc in allConstraints ])+");"
        self.execute(cmd)
        return

    def dropConstraints(self, constraints=None, dropall=None):
        assert (dropall is None) ^ (constraints is None)
        if dropall:
            cmd = "SELECT DropRasterConstraints('%(schema)s'::name, '%(name)s'::name, 'rast'::name)"%dict(schema=self.schema, name=self.name)
        else:
            if isinstance(constraints, str): constraints = [constraints]
            allConstraints = ['srid','scale_x','scale_y','blocksize_x','blocksize_y','same_alignment','regular_blocking','num_bands','pixel_types','nodata_values','out_db','extent']
            cmd = "SELECT DropRasterConstraints('%(schema)s'::name, '%(name)s'::name, 'rast'::name, "%dict(schema=self.schema, name=self.name)
            cmd+=", ".join([cc+":=True" if cc in constraints else cc+":=False" for cc in allConstraints ])+");"
        self.execute(cmd)
        return        

    def get_gadmArray(self,GADM_ids,bands=1):
        """Returns an array of the given raster, only for the area of the given gadm id
        GADM_ids is an array of [iso, id_1, id_2, id_3, id_4, id_5] up to the desired depth
        The first id can also be an integer, in which case it is interpreted as id_0"""
        if self.getSRID()!=4326: raise Exception('rasterCon().get_gadmArray() only deals with SRID 4326 for now')
        assert self.sameAlignnment('gadmraster')
        if isinstance(GADM_ids,str) or isinstance(GADM_ids,int):
            GADM_ids=[GADM_ids]
        else:
            GADM_ids=list(GADM_ids)

        from gadm import GADM_TABLE
        if isinstance(GADM_ids[0], str):  # replace iso with id_0
            GADM_ids[0] = self.execfetch("SELECT DISTINCT id_0::int FROM "+GADM_TABLE+" WHERE iso='"+GADM_ids[0]+"';")[0][0]
        whereClause = ' AND '.join(['id_'+str(ii)+'='+str(gadmId) for ii, gadmId in enumerate(GADM_ids)])
   
        # Get the tiles that we need
        cmd = """SELECT DISTINCT rid FROM %(rasterName)s, 
                   (SELECT the_geom AS geom FROM %(gadmName)s WHERE %(whereClause)s) t1
                    WHERE ST_Intersects(rast, geom)""" %dict(rasterName=self.name, gadmName=GADM_TABLE, whereClause=whereClause)
        rids = [rr[0] for rr in self.execfetch(cmd)]
        if rids is None or rids==[]: return np.array()
        array = self.asarray(bands=bands, rids=rids)  # get the array that we are interested in 
        rcGadm = rasterCon('gadmraster') # get the gadm array for the relevant tiles, so we can mask 
        gadmBands = range(1, len(GADM_ids)+1)
        arrayGadm = rcGadm.asarray(bands=gadmBands, rids=rids)
        assert arrayGadm.ndim==3 # asarray should have a 3rd dimension, even if there is only 1 gadmLevel (i.e. band) passed
        assert array.shape[:2]==arrayGadm.shape[:2]   
        for ii, gadmId in enumerate(GADM_ids):
            if array.ndim==2:
                np.putmask(array,arrayGadm[:,:,ii]!=gadmId,np.nan)  # set array to np.nan outside the specified gadmId
            else:  # need to tile the mask over the third dimension
                assert array.ndim==3 
                np.putmask(array,np.tile(np.expand_dims(arrayGadm[:,:,ii]!=gadmId,2),(1,1,len(bands))),np.nan)  # set array to np.nan outside the specified gadmId
            
        return array   

    def get_gadmAgg(self,GADM_ids,bands=1, areaWgted=True):
        """Returns mean and sum of values in a given gadm
        Area weighted if areaWgted=True"""
        if self.getSRID()!=4326: raise Exception('rasterCon().get_gadmAgg() only deals with SRID 4326 for now')
        if isinstance(bands, int): bands = [bands]
        array = self.get_gadmArray(GADM_ids,bands)
        if areaWgted:
            areaRc = rasterCon('landscan')
            areaArray = areaRc.get_gadmArray(GADM_ids,bands=2)
        else:
            areaArray = np.ones(array.shape)[:,:,0]
        arraySum = [np.nansum(array[:,:,bb-1]) for bb in bands]
        arrayMean = [np.nansum(array[:,:,bb-1]*areaArray)/np.nansum(areaArray) for bb in bands]
    
        if len(bands)==1:
            arraySum=arraySum[0]
            arrayMean = arrayMean[0]
        
        return {'sum': arraySum, 'mean': arrayMean}

    def saveImage(self,filename,format='GTiff', forceUpdate=True):
        assert format in ['JPEG','GTiff'] # others not tested
        if os.path.exists(filename) and not forceUpdate: 
            print( '%s already exists.' % filename)
        else:  
            cmd = '''gdal_translate -of %s PG:"%s table='%s' mode='2'" "%s"''' % (format, self.gdal_command_line_flags(), self.name, filename)
            print( 'Executing '+cmd)
            os.system(cmd)
        return

    def asarray(self, bands=1, rids=None, padEmptyRows=False, noDataToNan=True):
        """Based on http://gis.stackexchange.com/questions/130139/downloading-raster-data-into-python-from-postgis-using-psycopg2
        padEmptyRows will expand the number of rows so the whole world is captured (i.e. 21600 x 432600 for a 30-arc-second raster
        empty columns are always padded"""
        if bands=='all': bands = range(1,self.get_nBands()+1)
        assert isinstance(bands,int) or isinstance(bands, list)
    
        # Get subset of rids if necessary
        if isinstance(rids, int): rids=[rids]
        assert isinstance(rids, list) or rids is None
        if rids is None:
            fromClause = 'FROM '+self.name        
        elif len(rids)>1: 
            fromClause = 'FROM (SELECT rid, rast FROM '+self.name+' WHERE rid IN ' +str(tuple(rids))+') t1 '
        else:
            assert len(rids)==1
            fromClause = 'FROM (SELECT rid, rast FROM '+self.name+' WHERE rid =' +str(rids[0])+') t1 '

        tileDict = {}
        useGdal = False  # would be better, but there seems to be a bug in ST_AsGDALRaster. http://gis.stackexchange.com/questions/198787/getting-st-asgdalraster-to-recognize-pixel-type
        if useGdal:
            self.execute("""SELECT rid, ST_AsGDALRaster(rast, 'JPEG') """+fromClause+""" 
                           ORDER BY round(ST_UpperLeftY(rast)::numeric, 9) DESC, round(ST_UpperLeftX(rast)::numeric, 9);""")
        else: 
            bandText = 'ARRAY'+str([bands]) if isinstance(bands,int) else 'ARRAY'+str(bands)
            self.execute("""SELECT rid, (ST_DumpValues(rast,"""+bandText+""")).* """+fromClause+""" 
                           ORDER BY round(ST_UpperLeftY(rast)::numeric, 9) DESC, round(ST_UpperLeftX(rast)::numeric, 9), nband;""")
        for ii, tile in enumerate(self.cursor): # iterate over tiles, pull each into a dict of numpy arrays
            if ii>0: assert tile[0]>=rid  # confirm that rids are in correct order, i.e. this rid is greater than last rid
            rid = tile[0]
            if useGdal: 
                vsipath = '/vsimem/from_postgis'
                gdal.FileFromMemBuffer(vsipath, bytes(tile[1]))
                ds = gdal.Open(vsipath)
                if isinstance(bands,int):
                    rbs = ds.GetRasterBand(bands)  # we have to do this in 2 steps...not sure why
                    tileDict[rid] = rbs.ReadAsArray()
                else:
                    rbs = [ds.GetRasterBand(bb) for bb in bands]
                    tileDict[rid] = np.dstack([rb.ReadAsArray() for rb in rbs])
                gdal.Unlink(vsipath)
       
            else:
                if rid in tileDict:  # we've already got at least one bann
                    tileDict[rid] = np.dstack([tileDict[rid], np.array(tile[2], dtype=float)]) # dype float  will convert None values to float as well 
                else: 
                    tileDict[rid] = np.array(tile[2], dtype=float)
                
        # negative y scale indicates top to bottom for gdal tiles, so for a positive we need to flip each tile
        yscale = self.get_scale()[0]
        if yscale>0:  
            for rid in tileDict: tileDict[rid] = np.flipud(tileDict[rid]) 
                
        # assemble into a single array
        # create empty array where rid does not exist
        emptyArray = np.full(self.get_tileSizePixels(), np.nan)
        if isinstance(bands, list) and len(bands)>1: emptyArray = np.dstack([emptyArray for jj in range(len(bands))]) # 3D array
        if rids is None:
            if padEmptyRows:
                minRid = 1  
                nRows, nCols = (int(abs(180/self.get_tileSizeDegrees()[0])), int(abs(360/self.get_tileSizeDegrees()[1])))
            else:
                minRid = self.execfetch('SELECT MIN(rid) FROM '+self.name)[0][0]
                minRid = 1+int(minRid/self.get_nTilesYX()[1])*self.get_nTilesYX()[1]  # make sure it starts at -180 degrees
                nRows, nCols = self.get_nTilesYX()
            assert self.get_nTiles() == (ii+1.)/len(bands) if isinstance(bands, list) else ii+1
            nMissingTiles = sum([row*nCols+col+minRid not in tileDict for col in range(nCols) for row in range(nRows)])
            if nMissingTiles>0: print 'Typically due to uninhabited regions at high latitude, etc: %d of %d tiles are missing. Padding with empty array.' % (nMissingTiles, len(tileDict.keys()))
            array = np.vstack([np.hstack([tileDict[row*nCols+col+minRid] if row*nCols+col+minRid in tileDict else emptyArray for col in range(nCols)]) for row in range(nRows)])
        else: # rids may not be contiguous or rectangular, so let's fill in
            nRows, nCols = abs(int(180./self.get_tileSizeDegrees()[0])), abs(int(360./self.get_tileSizeDegrees()[1]))
            ridArray = np.array(range(1, nRows*nCols+1)).reshape((nRows, nCols))
            ridMask = np.ma.MaskedArray(ridArray, np.logical_not(np.in1d(ridArray, rids)))
            # extract subarray of non-null values
            nonZeroRows = np.nonzero(np.ma.count(ridMask, 1))
            nonZeroCols = np.nonzero(np.ma.count(ridMask, 0))
            # now get subset of the rid array
            ridArray = ridArray[np.min(nonZeroRows):np.max(nonZeroRows)+1, np.min(nonZeroCols):np.max(nonZeroCols)+1]
            nRows, nCols = ridArray.shape
        
            array = np.vstack([np.hstack([tileDict[ridArray[row, col]] if ridArray[row,col] in rids else emptyArray for col in range(nCols)]) for row in range(nRows)])
    
        # convert nodata values to nan. If we do it without gdal, they are nan already
        array = array.astype(float) 
        if useGdal:         
            if isinstance(bands, list):
                for bb in bands: array[:,:,bb-1][array[:,:,bb-1]==self.get_noDataValues()[bb]] = np.nan 
            else:
                ndValue = self.get_noDataValues() if self.get_nBands()==1 else self.get_noDataValues()[bands]
                array[array==ndValue] = np.nan 
        if isinstance(bands,list) and len(bands)==1: # promote to 3D array
            array = np.expand_dims(array,2)
        return array

def arrayToImage(array,cmap,fname,legTitle=None,useLog=False,cbarPos=(0.1,0.5),vsize=0.3,ndColor=None,alphas=None):
    """Saves a numpy array to an image. 
    cbarPos is None gives no colorbar, otherwise cbarPos gives the top LH corner in (0,1) space
    useLog can be False, True or plus1 (where array will be log(x+1))
    vsize gives the vertical dimension of the colorbar as a fraction of the plot height. (doesn't do horizontal yet..)
    If alphas is an array of the same shape, it will be used to provide the alpha values (note: it should be prenormalized to (0,1)
    Tick labeling on colorbar is still a kludge
    """
    
    try:
        import Image
    except:
        print('PIL Image library not found. Skipping array creation')  # Pillow is the alternative, but I get memory errors.
        return
        
    from cpblUtilities.color import addColorbarNonImage
    
    import matplotlib as mpl
    mpl.use(mpl_backend)
    import matplotlib.pyplot as plt
    assert useLog in [False, True, 'plus1']
    
    if ndColor is None: ndColor = [255,255,255,0] # RGBA transparent, but may not have desired effect 
    
    # convert to 0:1 range
    if useLog=='plus1':
        array = np.log(array+1)
    elif useLog is True:
        array = np.log(array)
    minVal, maxVal = np.nanmin(array), np.nanmax(array)
    scale = float(maxVal-minVal)
    
    norm = mpl.colors.Normalize(vmin=minVal, vmax=maxVal)   
    #imageArray = np.uint8(cmap((array-minVal)/scale)*255)
    imageArray = np.zeros((array.shape[0],array.shape[1],4),dtype=np.uint8)
    mask = np.isnan(array)
    imageArray[mask] = np.array(ndColor).astype(np.uint8)
    imageArray[np.logical_not(mask)] = np.uint8(cmap(norm(array[np.logical_not(mask)]))*255)
    
    if alphas is not None:
        if np.nanmin(alphas)<0 or np.nanmax(alphas)>1:
            raise Exception("Alpha array must be normalized to 0:1!")
        assert alphas.shape==array.shape
        alphas[np.isnan(alphas)]=0
        imageArray[:,:,3] = (alphas*255).astype(np.uint8)
        im = Image.fromarray(imageArray, 'RGBA')
    else:
        try:
            im = Image.fromarray(imageArray)
        except:
            print('Failed at creating image from array. Saving as failedimagearray.npz')
            np.savez_compressed('failedimagearray.npz', array=imageArray)
            return
            
    if cbarPos is not None:       # create colorbar
        plt.close(6354)
        plt.figure(6354,figsize=(1+0.4*(legTitle is not None),3)) # Create a dummy axis to hang the colorbar on
        hax=plt.gca()
        cb = mpl.colorbar.ColorbarBase(hax, cmap=cmap, norm=norm,label=legTitle)
        hax.patch.set_visible(False)  # transparent
        if useLog=='plus1':
            tickLabels = mpl.ticker.LogLocator(numticks=7).tick_values(max(np.exp(minVal)-1,0.0000000001), np.exp(maxVal)-1)[3:-1]
            ticks = np.log(tickLabels+1)
            tickLabels = ['%.5g'%flt for flt in tickLabels]
        elif useLog is True:
            tickLabels = mpl.ticker.LogLocator(numticks=7).tick_values(np.exp(minVal), np.exp(maxVal))[1:-1]
            ticks = np.log(tickLabels)
            tickLabels = ['%.5g'%flt for flt in tickLabels]
        else:
            tickLabels = mpl.ticker.MaxNLocator(nbins=5).tick_values(minVal, maxVal)
            ticks = tickLabels
        cb.set_ticks(ticks)
        cb.set_ticklabels(tickLabels)
        plt.tight_layout()       
        plt.savefig(paths['scratch']+'cbartmp.png', transparent=True)
        cb_img=Image.open(paths['scratch']+'cbartmp.png')

        # add cbar to image
        bbox = im.getbbox()
        hpos = int((bbox[2]-bbox[0])*cbarPos[0]+bbox[0])
        vpos = int((bbox[3]-bbox[1])*cbarPos[1]+bbox[1])
        bbox = cb_img.getbbox() # should be 80x300 if figsize is 0.8x3
        scale = int(imageArray.shape[0]*float(vsize)/bbox[3])
        cb_img = cb_img.resize((bbox[2]*scale,bbox[3]*scale)) 
        im.paste(cb_img, (hpos,vpos))
        os.remove(paths['scratch']+'cbartmp.png')
    
    im.save(fname)
    print 'Saved array image to %s' %fname
    return
