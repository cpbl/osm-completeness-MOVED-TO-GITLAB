#!/usr/bin/python
# coding=utf-8

"""
Downloads and parses the entire OSM planet history file
Assigns a density, country and sub-national ids to every version of every way in OSM
This takes 3-4 weeks on a 50-core server

Note that if you don't want to run history_process_raw.py (or don't have the resources),
you can use the premade files: osmCompleteness_level*.hd5, provided at https://alum.mit.edu/www/cpbl/PLoS2017roads

You can call as:
python process_raw.py               

Or stage-by-stage:
python process_raw.py chunks        Divides the history file into chunks
python process_raw.py parse         Parses the chunks (identifies length and country)
python process_raw.py collapse      Collapses the ways by country and sub-national id and density

"""
"""
The code requires a PostgreSQL database with 
- a table of country and sub-national geometries (e.g. Global Administrative Areas - GADM)
- a density raster (e.g. Landscan)

Install these with 
python setup_postgis.py

The Postgres call calculates the length of each OSM way, and intersects the way with the GADM and Landscan table
If you do not have access to a PostgreSQL database, you will need to adapt the parseWays() function to use
 Shapely or another approach to calculate the length of the way and assign it to geographic units

Note that this code is far from optimized. 
It was built around disk and memory constraints (all nodes would not fit in memory, nor in an indexed postgres database).
However, if your system has different specifications, you can probably speed this up by a factor of 10 by keeping nodes in memory rather than pytables on disk.
You could consider also filtering the history to only include relevant ways (with libosmium), and/or a tiling approach.

"""

import os, sys, time
import numpy as np
import pandas as pd
from cpblUtilities.parallel import runFunctionsInParallel
from history_config import *
import history_tools as ht

def getFullHistory(update=False):
    """
    June2015: parses the OSM history file, and saves the nodes to a hd5 store
    This has several advantages over storing on postgres, most notably read/write directly to pandas
    These nodes are then used to construct the history of the ways
    
    Ideally, we'd keep the nodes in memory (but there are too many) or write to a postgres database (but index is too big to fit on disk)
    hd5 store is a middle ground that provides relatively fast access with less disk space and minimal memory requirements

    update=True parses only way versions with a timestamp that starts with 2015, or whatever is set in parseWays

    """
 
    nodesLookup = createNodesLookup()  # make sure this exists, create it if not
    if not(os.path.exists(paths['scratch']+'waychunks/')): os.mkdir(paths['scratch']+'waychunks/')
        
    # we now have a set of gzipfiles that are split at the way id xml tags
    # after trial and error, the most that we can do at once is about 35, with a list of jobs of 450
    gzipFiles=[1]
    maxJobs = 450
    while gzipFiles!=[]:
        gzipFiles = [ff for ff in os.listdir(paths['scratch']) if ff.startswith('tmpWayChunk') and ff.endswith('.gz') 
            and not(os.path.exists(paths['scratch']+'waychunks/'+ff[:-3]+'_out'+update*'_update'+'.h5')) and not(os.path.exists(paths['scratch']+'waychunks/'+ff[:-3]+'_out'+update*'_update'+'.h5.txt'))][:maxJobs]
        runFunctionsInParallel([[parseWays, [agz, update]] for agz in gzipFiles], names=gzipFiles, maxAtOnce=34, offsetsSeconds=0.5)        
    
    # collapse the ways and then zip up the .h5 files
    # better to do this manually if we are dealing with updates as well
    if 1:
        for fn in gzipFiles: os.remove(paths['scratch'] + fn)
        nodeStoreFns = [ff for ff in os.listdir(paths['scratch']) if ff.startswith('nodeHistory') and ff.endswith('.h5')]
        for fn in nodeStoreFns: os.remove(paths['scratch'] + fn)
        print "done!"
    else:
        print 'All done! Remember to delete gzip and .h5 files (tmpWayChunk*.gz and nodeHistory*.h5)'   

def createHistoryChunks():
    """
    This reads the big bzf file of the planet history
    Saves the nodes as .h5 files (one for each chunk)
    Saves the relevant bits of the way files as .gz for later parsing
    """
    import bz2, gzip
    
    # source file is http://planet.openstreetmap.org/planet/full-history/history-latest.osm.bz2
    # most recent version downloaded Nov 7, 2015
    # wget http://planet.openstreetmap.org/planet/full-history/history-latest.osm.bz2
    
    # bz2file is not available on apollo, so we need to use a single stream version. 
    # See http://wiki.openstreetmap.org/wiki/Planet.osm/full#Data_Format
    # Use this line: bzip2 -cd history-latest.osm.bz2 | bzip2 -c > history-latest-singlestream.osm.bz2
    waysTmpFile = paths['scratch']+'wayHistoryChunk'  # stem of file name
    
    if not(os.path.exists(osmHistoryFile)):
        print 'Downloading planet history file and converting it to single stream. This will take about a week.'
        os.chdir(paths['input'])
        os.system('wget http://planet.openstreetmap.org/planet/full-history/history-latest.osm.bz2')
        os.system('bzip2 -cd history-latest.osm.bz2 | bzip2 -c > history-latest-singlestream.osm.bz2')
        os.remove('history-latest.osm.bz2')
        
    bzf = bz2.BZ2File(osmHistoryFile, 'r')     
       
    for i in range(0,3): bzf.readline()     # pass over header rows
    leftover = ''
    counter, wayCounter = 0, 0
    chunkSize=2000000000   # 2GB chunks. This seems to be the largest that a Python string can hold
    numGzChunks = 10
    gzChunkSize = chunkSize/numGzChunks  # target size for each gz file with ways (to be multiprocessed)
    
    while bzf:
        starttime = time.time()
        chunk = bzf.read(chunkSize)  
        if chunk=='': break
        if counter%10==0: print 'Reading chunk %d' % (counter)
        counter+=1
        if not('<node' in chunk or '<way' in chunk): # read the next one
            lastCloseTagSt    = chunk.rfind('</')
            lastCloseTagEnd = chunk[lastCloseTagSt:].find('>\n') + 2
            leftover = chunk[lastCloseTagSt+lastCloseTagEnd:]
        else:  # could have nodes, ways or both
            if '<node' in chunk:
                oldleftover = leftover  # need to keep this in case a chunk includes both nodes and ways
                leftover, nodesDf = parseNodes(leftover+chunk)

                # write a separate .h5 file for each chunk, which is indexed
                nodeStore = pd.HDFStore(paths['scratch']+'nodeHistory'+str(counter)+'.h5', 'w')
                nodeStore.put('nodes', nodesDf, format='table') 
                nodeStore.close()
                print '\tWrote %d nodes in %.2f minutes' % (len(nodesDf), (time.time() - starttime)/60.)
                del nodesDf

            if '<way' in chunk:
                # here, the only goal is to save the .gz files in a format that can be directly parsed (xml)
                # i.e., we save smaller chunks, each starting and ending with a tag
                if '<node' in chunk: leftover = oldleftover  # for the first chunk
                
                for jj in range(0,numGzChunks):
                    wayCounter+=1
                    snippet = leftover+chunk[jj*gzChunkSize:(jj+1)*gzChunkSize] 
                    # find last newline and close tag
                    lastCloseTagSt    = snippet.rfind('</')
                    lastCloseTagEnd = snippet[lastCloseTagSt:].find('>\n') + 2
                    if lastCloseTagSt==-1 or lastCloseTagEnd==1:  # no tag found, but write a blank text file so we know that no chunks were missed
                        leftover=snippet
                        ff = open(paths['scratch']+ 'tmpWayChunk'+str(wayCounter)+'.txt', 'w')
                        ff.close()
                    else:
                        leftover = snippet[lastCloseTagSt+lastCloseTagEnd:] 
                        waysGz = gzip.open(paths['scratch'] + 'tmpWayChunk'+str(wayCounter)+'.gz', 'w')
                        waysGz.write(snippet[:lastCloseTagSt+lastCloseTagEnd])
                        waysGz.close()
                leftover+=chunk[(jj+1)*gzChunkSize:] # any remainder
                    
    # We've now saved all the nodes, plus the ways for the first continent.
    # Now do the ways for the remaining continents, using the gz files saved before
    bzf.close()   
    nodesLookup = createNodesLookup(forceUpdate=True)  # forceupdate=True because we want to overwrite any old version

def createNodesLookup(forceUpdate=False):
    """
    From a list of .h5 files, find out the range of index values that are in that file
    This will speed up subsequent lookups dramatically
    """
    nodesLookupFn = paths['scratch'] + 'nodesLookup.pandas'
    if os.path.exists(nodesLookupFn) and not forceUpdate:
        return pd.read_pickle(nodesLookupFn)
    print 'Creating lookup for node .h5 files'
    nodeStoreFns = [ff for ff in os.listdir(paths['scratch']) if ff.startswith('nodeHistory') and ff.endswith('.h5')]
    nodesLookup = pd.DataFrame(index=range(len(nodeStoreFns)), columns = ['fn', 'minNode', 'maxNode'])

    for ii, ff in enumerate(nodeStoreFns):
        ns = pd.HDFStore(paths['scratch']+ff,'r')
        df = ns.select('nodes', columns=['index']).reset_index()
        ns.close()
        nodesLookup.ix[ii] = [ff, df['id'].min(), df['id'].max()]
    
    nodesLookup.sort_values(by='minNode', inplace=True)
    # check that nodes go in sequence (this speeds up lookups later on)
    # minimum of file is weakly greater than maximum of previous file 
    assert (nodesLookup.minNode-nodesLookup.maxNode.shift(1)).min()>=0
    
    nodesLookup.to_pickle(nodesLookupFn)
    return nodesLookup
      
def parseNodes(inChunk):
    """
    For use with getNodesHistory(). Based on parseNodes
    Takes an xml chunk and parses it to extract the node_id, lat, long, version, timestamp
    Returns a dataframe with just that info
    
    Note that ET.fromstring() is very slow for large chunks, which is why the multiprocessing is needed here
    """    
    import xml.etree.cElementTree as ET
    from multiprocessing import Queue, Process
    import pandas as pd
    import numpy as np
    
    maxJobs = 40
    chunkLen = len(inChunk)/maxJobs
        
    def readETree(c, outQ):
        # This is the worker function - creates the xml tree and returns the relevant parts as a dataframe
        tree = ET.fromstring(c)
        dfInput = []
        for cc in tree:
            if cc.tag!='node': continue
            dfInput += [dict(cc.items())]
        
        if dfInput==[]:  # nothing in this chunk
            outQ.put(pd.DataFrame())
        else:
            treeDf = pd.DataFrame(dfInput)
            treeDf = treeDf[treeDf.visible.str.lower()=='true']  # drop visible==False
            # clean up and reduce in size
            cols =   ['id', 'lat', 'lon', 'date', 'version']
            dtypes = ['int64', 'float', 'float', 'int32', 'int16']
            treeDf['date'] = treeDf.timestamp.apply(lambda x: int(x[0:4]+x[5:7]+x[8:10]))
            treeDf = treeDf[cols]
            for col, dt in zip(cols, dtypes):
                treeDf[col] = treeDf[col].astype(dt)

            outQ.put(treeDf)

    # split chunk up to farm out to multiprocessing  
    jobs = []
    outQ = Queue()
    combinedDf = pd.DataFrame()
    leftover2 = ''
    
    for i in range(0,maxJobs):  
        snippet = leftover2+inChunk[i*chunkLen:(i+1)*chunkLen] 
        # find last newline and close tag
        lastCloseTagSt    = snippet.rfind('</')
        lastCloseTagEnd = snippet[lastCloseTagSt:].find('>\n') + 2
        if lastCloseTagSt==-1 or lastCloseTagEnd==1:  # no tag found
            leftover2=snippet
        else:
            leftover2 = snippet[lastCloseTagSt+lastCloseTagEnd:] 
            jobs.append(Process(target=readETree,args=('<rows>'+snippet[:lastCloseTagSt+lastCloseTagEnd]+'</rows>', outQ)))
            jobs[-1].start()
    leftover2+=inChunk[(i+1)*chunkLen:]   # don't forget to add any remainder (after dividing chunk into equal snippets of chunkLen)
    
    for j in jobs: 
        combinedDf = pd.concat([combinedDf, outQ.get()])
    for job in jobs: job.join()    # wait for all jobs to finish
    
    # eliminate duplicate versions from nodesDf. This can arise if a tag has changed but lat/long are the same
    # sort, then drop if the row is identical to the previous (http://stackoverflow.com/questions/19463985/pandas-drop-consecutive-duplicates)
    # note we can't use drop_duplicates, because this misses nodes that get deleted and then resurrected later on
    combinedDf.sort_values(by=['id','version'], inplace=True)
    combinedDf = combinedDf[np.any(combinedDf[['id','lat', 'lon']]!=(combinedDf[['id','lat', 'lon']].shift()), axis=1)] 
     
    return leftover2, combinedDf[['id', 'lat', 'lon', 'date', 'version']].set_index('id')

def parseWays(gzf, update=False):
    """
    call in parallel
    reads the gzf, parses it and saves the result

    gzf must be pre-split (i.e., start and end of the file must be an xml tag)
    
    update=True parses only way versions with a timestamp that starts with 2015, or whatever is set below)
        to changes the behavior of update, you'll need to tweak the lines below
    """
    import xml.etree.cElementTree as ET
    import gzip
    
    assert gzf.endswith('.gz')
    outFn = gzf[:-3]+'_out'+update*'_update'+'.h5'
    if os.path.exists(paths['scratch']+'waychunks/'+outFn): return 
    
    # open file and read into xml
    f = gzip.open(paths['scratch'] + gzf, 'r')
    c = f.read()
    f.close()     
    tree = ET.fromstring('<rows>'+c+'</rows>')
    del c  # save memory
    
    ways = []
    cur = ht.dbConnection()  #curType='default')

    # define types of roads that are considered separately (a subset of the highways)
    roadTypes = ['motorway','motorway_link','trunk','trunk_link','primary','primary_link',
                 'secondary','secondary_link','tertiary', 'tertiary_link','residential','road','unclassified','living_street']
                   
    # First pass through the tree - get list of nodes, so we can do a single call to the hdf store 
    bigNodeList = []
    for cc in tree:
        if cc.tag!='way' or cc.get('visible')=='false': continue
        if update and not(cc.get('timestamp').startswith('2015')): continue
        if not any([tt.get('k')=='highway' for tt in cc.findall('tag')]): continue
        bigNodeList += [int(nd.get('ref')) for nd in cc.findall('nd')]
    bigNodeList = np.sort(np.unique(bigNodeList)).tolist()
    
    if len(bigNodeList)==0: # no ways
        f = open(paths['scratch']+'waychunks/'+outFn+'.txt', 'w')
        f.close()
        return

    nodesLookup = createNodesLookup()
    bigNodeDf = pd.DataFrame()

    for ii, row in nodesLookup.iterrows():
        startValue, stopValue = int(row['minNode']), int(row['maxNode'])
        rowNodes = [nn for nn in bigNodeList if startValue<=nn and stopValue>=nn]
        if rowNodes==[]: continue
        nodeStore = pd.HDFStore(paths['scratch']+row['fn'],'r')
        # We need to get nodes in batches of <32: see http://stackoverflow.com/questions/22777284/improve-query-performance-from-a-large-hdfstore-table-with-pandas
        bigNodeDf = pd.concat([bigNodeDf]+[nodeStore.select('nodes', 'index='+str(rowNodes[jj:jj+31])) for jj in xrange(0, len(rowNodes), 31)])
        nodeStore.close()

    bigNodeDf.sort_index(inplace=True)   # needed for .loc to be able to slice
    
    for cc in reversed(tree):  # iterate in reversed order, so we can skip earlier versions in the same month
        # skip if not a way, or if the way is not a road (k='highway')
        if cc.tag!='way': continue
        if update and not(cc.get('timestamp').startswith('2015')): continue   # change to whatever value you want to update from
        
        # this means that we will have some deleted non-highways in the dataset, because we don't know their tags
        if cc.get('visible')=='true' and not any([tt.get('k')=='highway' for tt in cc.findall('tag')]): continue
        
        wayDate = int(cc.get('timestamp')[0:4] + cc.get('timestamp')[5:7] + cc.get('timestamp')[8:10])
        osm_id = cc.get('id')
        if osm_id=='22529607': continue  # bogus way, throws an error in PostGIS (geometry out of bounds?)
        wayVersion = cc.get('version')
        
        # figure out if this way is a 'real' road or not (as opposed to a ped way)
        roadFlag = any([tt.get('k')=='highway' and tt.get('v') in roadTypes for tt in cc.findall('tag')])
        serviceFlag = any([tt.get('k')=='highway' and tt.get('v')=='service' for tt in cc.findall('tag')])

        # see if there is a more recent version of that way in the same month
        # Jan2016: no, we want daily resolution
        #if any ([int(osm_id)==ww[0] and str(wayDate)[:-2]==str(ww[1])[:-2] and int(wayVersion)<int(ww[2]) for ww in ways]): 
        #    continue

        if cc.get('visible')=='true':
            # Get list of nodes in that way       
            nodes = [int(nd.get('ref')) for nd in cc.findall('nd')]  # remember to keep the order!
            if len(nodes)==0: 
                ways += [[int(osm_id), int(wayDate), int(wayVersion), roadFlag, serviceFlag, len(nodes), len(nodes), 1] + [np.nan]*8]
                continue

            try:
                # this was the main performance bottleneck. Jan2016: move to a non-unique index, which is now blisteringly fast in comparison
                #nodesDf = (bigNodeDf.loc[nodes]).reset_index(level=1)
                nodesDf = bigNodeDf.loc[nodes]                
            except:  # no nodes in bigNodeDf
                ways += [[int(osm_id), int(wayDate), int(wayVersion), roadFlag, serviceFlag, len(nodes), len(nodes), 1] + [np.nan]*8]
                continue
            
            # Restrict to the right time period - get latest version before date of way
            nodesDf = nodesDf[nodesDf.date<=wayDate]           
            nodesDf = nodesDf.sort_values(by='version').groupby(level=0).last()
            
            # missing nodes are usually those that were redacted. E..g. http://www.openstreetmap.org/redactions/1
            NmissingNodes = len([id for id in nodes if not(id in nodesDf.index)])

            # get countries that intersect the line string. Note WKT is long lat not lat long
            # 4326 is WGS lat/long. 3857 is the projection for GADM
            # note that we don't split up the linestring if it intersects multiple administrative units (instead, we double count)
            # that would take something like this: http://gis.stackexchange.com/questions/78013/postgis-calculate-the-length-of-just-the-portion-of-linestring-that-intersect-w

            pointList = [(nodesDf.lat[int(id)], nodesDf.lon[int(id)]) for id in nodes if id in nodesDf.index]
            if len(pointList)<2:
                 ways += [[int(osm_id), int(wayDate), int(wayVersion), roadFlag, serviceFlag, len(nodes), NmissingNodes, 1] + [np.nan]*8]
                 continue
            lineString = "ST_GeomFromText('LINESTRING(" + ", ".join([str(pt[1]) + " " + str(pt[0]) for pt in pointList])+")', 4326)"
            cmd = """SELECT DISTINCT CAST(ST_Length(%s::geography) AS int), id_0::int, id_1::int, id_2::int, id_3::int, id_4::int, id_5::int FROM %s AS g 
                     WHERE ST_INTERSECTS(g.the_geom, %s)""" % (lineString, gadmTableName, lineString)
            try:
                cur.execute(cmd)
            except:  # catch errors due to invalid latitude of 90 degrees
                assert 90 in [lat for lat, lon in pointList] or -90 in [lat for lat, lon in pointList]
                ways += [[int(osm_id), int(wayDate), int(wayVersion), roadFlag, serviceFlag, len(nodes), NmissingNodes, 1] + [np.nan]*8]
                continue
            result = cur.fetchall()
            
            if result is None or len(result)==0 or (len(result)==1 and len(result[0])==1): # does not intersect GADM, so get length only. 
                # Note: this condition was not caught until 1May2016, so ways that did not intersect were just ignored in the Jan 2016 version
                cmd = """SELECT DISTINCT CAST(ST_Length(%s::geography) AS int)""" % (lineString)
                cur.execute(cmd)
                result = [[cur.fetchall()[0][0]]+[np.nan]*6]
            
            # get density from LandScan raster
            # See http://movingspatial.blogspot.com/2012/07/postgis-20-intersect-raster-and-polygon.html
            # The ST_Union is needed in case a way goes across multiple raster tiles
            if result[0][0]<500000:
                cmd = """WITH w AS (SELECT %s AS geom_way)
                     SELECT (ST_SummaryStats(ST_Union(ST_Clip(rast, geom_way)), 3)).mean
                     FROM landscan, w WHERE ST_Intersects(geom_way, rast)"""  % (lineString)
                try:
                    cur.execute(cmd)
                    density = cur.fetchall()[0][0]   
                except:
                    # usually, all pixel values are Null. This happens occasionally when a road is next to the sea or a lake, and the pixel geometry is imprecise
                    density = np.nan
            else:   # way is too long - more than 0.5 M km, so getting density is problematic
                density = np.nan
            # add osm_id, timestamp,  visible and N nodes
            result = [[int(osm_id), int(wayDate), int(wayVersion), roadFlag, serviceFlag, len(nodes), NmissingNodes, 1, density] + list(resLine) for resLine in result]
            ways += result
        else:  # not visible. we'll have duplicates of these, because we don't know which continent they are in
            ways += [[int(osm_id), int(wayDate), int(wayVersion), roadFlag, serviceFlag, np.nan, np.nan, 0] + [np.nan]*8]
        assert ways == [] or len(ways[-1])==16 # make sure we wrote the right number of fields
        #tmp = os.listdir(paths['input'])  # was a workaround for amb - keeps the connection to OKAI alive. But no longer needed with mounting of OKAI

        #if (time.time()-starttime)>10: print 'osm id %s version %s took a long time (%.2f seconds)!' % (osm_id, wayVersion, (time.time()-starttime))
        
    # return a dataframe with the ways
    if ways==[]:  # nothing in this chunk
        f = open(paths['scratch']+'waychunks/'+outFn+'.txt', 'w')
        f.close()
    else:
        cols =   ['osm_id', 'date', 'version', 'roadFlag', 'serviceFlag', 'Nnodes', 'NmissingNodes', 'visible', 'density', 'length','id0', 'id1', 'id2','id3','id4','id5']
        treeDf = pd.DataFrame(ways, columns=cols)
        treeDf.to_hdf(paths['scratch']+'waychunks/tmp'+outFn, 'ways', mode='w')
        # rename once write has successfully completed, in case it crashed midway
        os.rename(paths['scratch']+'waychunks/tmp'+outFn, paths['scratch']+'waychunks/'+outFn)  
        print "Process ended: wrote %d ways for file %s" % (len(treeDf), gzf)
        return

def collapseHistory(gadmLevel=0, timeLevel='D', density=True, forceUpdate=False, id0s=None, test=False):
    """
    Once we've stored all the ways in an .h5 file, we can collapse them
    This function loads in the various continents, drops duplicates (because of overlapping bboxes)
    
    gadmLevel and timeLevel can be a list or a single value (more efficient to do them as a list)
    gadmLevel = -1 does the whole world 
    
    density=True also groups by density bin and decile (in addition to the gadmLevels)
    
    Saves a pandas dataframe to scratch

    2015 Dec: 
     - If we want more detail than id1, we should restrict to just one country.
     - We can also collapse by density instead of small GADM. In this case, also specify just one (or a few?) countries:
           pass gadmLevel=0, density=True, id0=mycountryid

    if gadmLevel is a list, it will need to be [-1,0,1] but not include 2 or higher.
    """    

    assert timeLevel in ['D','M'] or all([tt in ['D','M'] for tt in timeLevel])
    if isinstance(gadmLevel, int): gadmLevel = [gadmLevel]
    if isinstance(timeLevel, str): timeLevel = [timeLevel]
    if timeLevel==['M','D']: timeLevel = ['D','M'] # need to do day first, because we lose data when going to month
    if any([gg>1 for gg in gadmLevel]) and 'D' in timeLevel: raise(Warning, 'Not advised to use daily resolution with gadmLevel of 2 or greater. This will be slow!')
    
    print 'Loading ways dataframe before collapsing it' # This is relatively fast compared to everything else
    waysFns = [ff for ff in os.listdir(paths['scratch']+'waychunks/') if ff.startswith('tmpWayChunk') and ff.endswith('.h5')]
    if test:
        print('Restricting to 20 hd5 files for debugging..')
        waysFns = waysFns[:20]        

    print 'Found %d files with waychunks.  Loading/concatenating...' % (len(waysFns))
    wdf = pd.concat([pd.read_hdf(paths['scratch']+'waychunks/'+fn, 'ways') for fn in waysFns])
    print 'Done loading %d ways with columns: %s' % (len(wdf), str(wdf.columns.values))
    
    # Drop long roads. These were all manually checked and identified as errors, or there was a new version within a few days.
    # E.g. version 2 of osm_id 110554804 erroneously goes from Japan to Brazil (one node was misplaced)
    # Usually the lat/lon coordinate got flipped, or a coordinate was entered as the North Pole
    # Most would be caught anyway since we have daily resolution, and the error was fixed at the end of the day. But this speeds things up a little later on...
    oldLen = len(wdf)
    wdf = wdf[(pd.isnull(wdf.length)) | (wdf.length<1000000)]
    print('Dropped %d long ways out of %d records' % (oldLen-len(wdf), oldLen))
    
    print('Creating validTo date for each osm id')
    # In general, this will be the validFrom date of the subsequent row, or 2099 if the subsequent row has a different osm_id
    # The complications come from situations where there are duplicates (ways that cross more than one gadm id)
    wdf.rename(columns={'date':'validFrom'}, inplace=True)
     
    wdf.sort_values(by=['osm_id','version'], inplace=True)
    wdf['validTo'] = wdf.validFrom.shift(-1)
    # if next id is different, replace validTo with a random date that is a long way in the future
    wdf['nextId'] = wdf.osm_id.shift(-1)
    wdf.loc[wdf.nextId!=wdf.osm_id, 'validTo'] = 20991231

    # For rows where this approach doesn't work because of duplicate osm_ids (that cross a gadm polygon)
    #    we need to make the values np.nan, and then backfill using fillna
    wdf['nextV'] = wdf.version.shift(-1)
    wdf.loc[(wdf.nextId==wdf.osm_id) & (wdf.nextV==wdf.version), 'validTo'] = np.nan
    wdf.validTo.fillna(method='bfill', inplace=True)

    if 1: # check that all ways and versions have same validTo dates
        chck = wdf.groupby(['osm_id','version'])[['validFrom', 'validTo']].std()
        assert np.all(chck[pd.notnull(chck.validFrom)]==0)           

    oldLen = len(wdf)
    wdf = wdf[wdf.visible==True]  # we can do this now, as visible=False was only to get the validTo date.
    wdf.drop(['visible', 'nextId', 'nextV'], axis=1, inplace=True)
    print('Dropped %d non-visible ways out of %d records' % (oldLen-len(wdf), oldLen))

    # Restrict to particular countries, if desired. This needs to be done AFTER the creation of validTo dates (in case a new version of the road is no longer in a given country)
    if id0s is not None:
        if isinstance(id0s, int): id0s=[id0s] # Let's look for a country-level file. 
        wdf = wdf[wdf.id0.isin(id0s)]
    
    if density: # Put density into decile bins
        wdf['density'] = wdf.density.astype(np.float32) # avoids precision problems later on
        print('        Preparing density ...')
        if 0: # let's compute the deciles separately by country instead
            wdf['densDecileWorld'] = pd.qcut(wdf.density, 10, labels=range(10)).astype(float)
            wdf.loc[pd.isnull(wdf.density), 'densDecileWorld'] = -1
            wdf.densDecileWorld = wdf.densDecileWorld.astype(int)

        # calculate deciles by country. Can't use qcut because of non-unique bin edges for some small countries: http://stackoverflow.com/questions/18921570/binning-with-zero-values-in-pandas
        # in principle, we could do this by all gadm levels too...
        # double groupby is needed so we don't weight osm_ids more if they have more versions
        wdf['densDecile'] = np.nan
        for rf in [True, False]:
            print('        Calculating by-country deciles for roadFlag %s ...' % rf)
            allDeciles = wdf[(wdf.roadFlag==rf) & (pd.notnull(wdf.density))].groupby(['id0', 'osm_id']).density.mean().groupby(level=0).quantile(np.arange(0,1.1,0.1))  # pd.notnull is needed to avoid error with id216, where density is all null
            print('        Aggregating by-country deciles for roadFlag %s ...' % rf)
            for id0 in allDeciles.index.get_level_values(0).unique():
                if np.isnan(id0): continue
                # deal with non-unique labels
                deciles = allDeciles.ix[id0].astype(np.float32).drop_duplicates()   # cut can't deal with duplicate bins, we assign to the lowest decile
                decileLabels=(deciles.index.values*10).astype(int)[:-1]
                mask = (wdf.id0==id0) & (wdf.roadFlag==rf) & (pd.notnull(wdf.density))
                # reset the first and last deciles; because we average over osm_id above, some versions of some ways might not be caught
                # for floating point errors, we add or subtract 1
                deciles.iloc[0] = wdf.loc[mask, 'density'].min()-1
                deciles.iloc[-1] = wdf.loc[mask, 'density'].max()+1
                wdf.loc[mask, 'densDecile'] = pd.cut(wdf.loc[mask, 'density'], deciles, labels=decileLabels, include_lowest=True)        
        wdf.densDecile.fillna(-1, inplace=True)
        
        if 0: # do log-spaced bins, i.e. spaced at exp(0.1) intervals
            bins =  np.hstack([np.array([0]), np.exp(np.arange(-1.3, np.log(wdf.density.max())+0.1, 0.1))])
            wdf['densBinWorld'] = wdf.density.apply(np.digitize, bins=bins)
            wdf.loc[pd.isnull(wdf.density), 'densBinWorld']    = -1

    for gl in gadmLevel:
        loopstarttime = time.time()
        print 'Collapsing history for gadmLevel %s at %d...' % (gl, loopstarttime)
        # most of the collapse is the same regardless of the timelevel, so let's do most of the work inside this level of the loop
        geogLevels  = ['id0', 'id1', 'id2', 'id3', 'id4', 'id5'][0:gl+1] 
        groupLevels = geogLevels+ density*['densDecile']#, 'densBin']
        waysDf = wdf.copy()

        # drop duplicates - sort means that we keep the duplicates with the lowest number of missing nodes
        # this also drops duplicates for lower-level geographies that we aren't using
        oldLen = len(waysDf)
        waysDf.sort_values(by=['osm_id','version', 'NmissingNodes'], inplace=True)               
        waysDf.drop_duplicates(subset=['osm_id','version']+geogLevels, inplace=True)    
        print '%d duplicate ways dropped after %d' % (oldLen-len(waysDf), loopstarttime)
        
        waysDf.length.fillna(0, inplace=True)  # makes sure that they show up in aggregate count, even though length will not register
        
        """
        Create single-string gadm id
        N.B.: For 12000 ways, id1==0 while id2 is nonzero. So "0" is sometimes a real GADM index. 
        However, when id1 is NaN that is always a correct indication that lower levels are also NaN, ie: wdf[np.logical_and(pd.isnull(wdf.id1), pd.notnull(wdf.id2))][['id0','id1','id2','id3','id4'] 
        So lets only truncate when we hit a NaN. 
        """
        if 0: # Let's not do this...code below rewritten to avoid a join on a MultiIndex instead
            # This method might be somewhat convoluted (create string id column by column), but an apply function that does it in one step is memory intensive and far, far slower 
            for gg in groupLevels:
                waysDf['s'+gg] = ''
                waysDf.loc[pd.notnull(waysDf[gg]),'s'+gg] = waysDf.loc[pd.notz(waysDf[gg]),gg].astype(int).astype(str)
                waysDf.loc[pd.isnull(waysDf[gg]),'s'+gg] = ''
                if gg!=groupLevels[-1]: waysDf['s'+gg]+='-' # add separator for all level except the last one
            waysDf['sid'] = waysDf[['s'+gg for gg in groupLevels]].sum(axis=1)  # sum here is a string concat
            waysDf.drop(['s'+gg for gg in groupLevels], axis=1, inplace=True)
        
        for tl in timeLevel:
            print '\t..and timeLevel %s...' % tl
            outFn = paths['output']+'osmHistory_level'+str(gl)+tl+density*'_density'+test*'_test'+'.hd5'
            if os.path.exists(outFn) and not forceUpdate:
                print '%s already exists - skipping collapse' % outFn
                continue

            if tl=='M':  # remove last two digits from date. Note months refer to midnight on the last day of the month
                waysDf.validFrom=(waysDf.validFrom/100.).astype(int)
                waysDf.validTo=(waysDf.validTo/100.).astype(int)

            print 'Calculating cumulative sum for ways at %d...' %loopstarttime
            dfDict = {}
            for aggCol in ['validFrom', 'validTo']:
                # collapse using cumulative sum by gadm id, roadFlag and date. See http://stackoverflow.com/questions/22650833/pandas-groupby-cumulative-sum                
                # the first groupby aggregates by validFrom date and gadm ids, densities, etc. 
                # The second groupby doesn't aggregate, but does the cumulative sum. Note: cumsum is over one fewer group than sum
                #cumDf = waysDf.groupby(['roadFlag','sid']+[aggCol]).length.agg(['count', 'sum']).groupby(level=[0,1]).cumsum() 
                cumDf = waysDf.groupby(['roadFlag']+groupLevels+[aggCol]).length.agg(['count', 'sum']).groupby(level=range(len(groupLevels)+1)).cumsum() 
                print 'Done double groupby for %s after %d' % (aggCol,loopstarttime)

                # Convert date index column to a datetime index - this allows resampling
                cumDf.index.names=cumDf.index.names[:-1]+['date']  # index names for validTo and validFrom have to match to do the join
                indexNames = cumDf.index.names[:-1]  # index names, excluding date
                cumDf.reset_index(inplace=True)
                if tl=='M': # date is actually end of the month, so let's add the right number of days
                    numDays  = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
                    cumDf.date = cumDf.date.apply(lambda x: x*100+numDays[int(str(x)[-2:])])

                cumDf.date = pd.to_datetime(cumDf.date.astype(str),format='%Y%m%d')# if tl=='M' else '%Y%m%d' )                            
                cumDf.set_index('date', inplace=True)
                cumDf.drop('2099-12-31', inplace=True)  # this was only for ways that were never dropped
                
                dfDict[aggCol] = cumDf   
                
            # Before resampling, make sure that the indices span the same amount of time. Otherwise, we assume zero cumulative ways dropped in any last days that don't overlap with validTo
            # We also need to make sure that the start of the new dataframe has count and sum 0. Otherwise, values from the country before will be propagated forward
            
            # First. set the minimum date (by group)
            vfDates = pd.DataFrame(dfDict['validFrom'].reset_index().groupby(indexNames).date.min()).set_index('date', append=True)
            vtDates = pd.DataFrame(dfDict['validTo'].reset_index().groupby(indexNames).date.min()).set_index('date', append=True)
            maxDate =  max(dfDict['validFrom'].reset_index()['date'].max(), dfDict['validTo'].reset_index()['date'].max())

            for aggCol, ownDates, otherDates in [('validFrom', vfDates, vtDates), ('validTo', vtDates, vfDates)]:
                # Avoid overwriting dates that are already in the dataframe
                datesToAdd = otherDates.ix[otherDates.index.difference(ownDates.index)].reset_index().set_index('date')
                # no ways are retired on the first day. Otherwise, we will ffill later during the resample, so we leave them as np.nan
                datesToAdd['count'] = np.nan
                datesToAdd['sum']   = np.nan
                dfDict[aggCol] = pd.concat([dfDict[aggCol], datesToAdd])  
            
                # no ways are retired on the first day, so replace with zero
                minDates = pd.DataFrame(dfDict[aggCol].reset_index().groupby(indexNames).date.min())
                minDates = minDates.reset_index().set_index(['date']+indexNames).index
                dfDict[aggCol].set_index(indexNames, append=True, inplace=True)
                dfDict[aggCol].sort_index(inplace=True)
                dfDict[aggCol].loc[minDates,'count'] = dfDict[aggCol].loc[minDates,'count'].fillna(0)
                dfDict[aggCol].loc[minDates,'sum'] = dfDict[aggCol].loc[minDates,'sum'].fillna(0) 
                dfDict[aggCol].reset_index(level=range(1,len(dfDict[aggCol].index.names)), inplace=True)
                assert dfDict[aggCol].index.name=='date'
            
            # Now, set the maximum date (same for all groups, so we have a common ending date)
            for aggCol in ['validFrom', 'validTo']:
                ownDates = pd.DataFrame(dfDict[aggCol].reset_index().groupby(indexNames).date.max()).set_index('date', append=True)
                maxDates = ownDates.reset_index(level='date', drop=True)
                maxDates['date'] = maxDate
                maxDates.set_index('date', append=True, inplace=True)
                datesToAdd = maxDates.ix[maxDates.index.difference(ownDates.index)].reset_index().set_index('date')
                dfDict[aggCol] = pd.concat([dfDict[aggCol], datesToAdd])   

                # Resampling makes sure that all intermediate dates are in the index
                dfDict[aggCol].sort_index(inplace=True)
                dfDict[aggCol] = dfDict[aggCol].groupby(indexNames).resample(tl, fill_method='ffill')
                print 'Resampled %s after %d' % (aggCol,loopstarttime)                       
                    
            # cumDf = cumDf.combine_first(newRows)   # old way
            # See http://stackoverflow.com/questions/15799162/resampling-within-a-pandas-multiindex
            # cumDf = cumDf.reindex(dfDict['validFrom'].index, method='ffill') # resampling is orders of magnitude faster.
                      
            if 0: # This is horribly slow, because it's a join on multiindex. 
                results = dfDict['validFrom'].subtract(dfDict['validTo'], fill_value=0)
                results.rename(columns={'count':'freq', 'sum':'length'}, inplace=True)
                results.reset_index(inplace=True)
            # Instead, we can concat because the two dataframes have identical indexes
            assert len(dfDict['validTo'])==len(dfDict['validFrom']) # not a robust check, but we do a full check on the alignment below
            dfDict['validTo'].reset_index(inplace=True)
            dfDict['validTo'].rename(columns=dict([(cc, 'vt'+cc) for cc in dfDict['validTo'].columns.values]), inplace=True)
            results = pd.concat([dfDict['validFrom'].reset_index(), dfDict['validTo']], axis=1)
            for cc in dfDict['validFrom'].index.names: assert np.all(results[cc] == results['vt'+cc]) # check that the concat aligned properly
            results['freq'] = (results['count'].subtract(results.vtcount)).astype(int)
            results['length'] = (results['sum'].subtract(results.vtsum)).astype(int)
            results.drop([cc for cc in results.columns.values if cc.startswith('vt') or cc in ['count', 'sum']], axis=1, inplace=True)
            del dfDict
            
            assert results.length.min>=0 and results.freq.min>=0
            print 'Combined validFrom and validTo after %d' %loopstarttime
            
            # add back in id0s from concatenated list, but we don't do that any more
            if 0: results['id0'] = results.sid.apply(lambda x: x.split('-')[0])

            # convert ids to country names
            if gl==-1:
                results['id0'] = -1
                results['ISOalpha3'] = 'ALL'
            elif gl==0:  # this takes too long for gadmLevel 1 and 2. Do this when we need to later on.
                id02ISO = dict([(ii['id_0'], iso) for iso, ii in ht.get_all_GADM_countries().items()])
                results['ISOalpha3'] = results.id0.apply(lambda x: id02ISO[int(x)])

            # do some checks, by comparing final totals
            assert results.index.name is None
            lastdate1 = results.sort_values('date').groupby(['roadFlag']+groupLevels)[['freq','length']].last()
            lastdate2 = waysDf[waysDf.validTo==20991231].groupby(['roadFlag']+groupLevels).length.agg(['count','sum'])
            testDf = lastdate1.join(lastdate2, how='outer').fillna(0)
            assert all(testDf.freq==testDf['count']) and all(testDf.length==testDf['sum'])
            
            results.set_index('date').to_hdf(outFn, 'history', mode='w', format='f', complevel=1, complib='zlib')
            
            if density:  # save another version without the density information, collapsed to all density levels
                colsToGroup = ['date','roadFlag']+geogLevels+['ISOalpha3']*('ISOalpha3'in results.columns)
                results = (results.groupby(colsToGroup)['freq','length'].sum()).reset_index().set_index('date')
                outFn2 = outFn.replace('_density','')
                results.to_hdf(outFn2, 'history', mode='w', format='f', complevel=1, complib='zlib')
            print '...done with collapse. Saved %s' % outFn
    
    return

if __name__ == '__main__':
    runmode=None if len(sys.argv)<2 else sys.argv[1].lower()
    forceUpdate = any(['forceupdate' in arg.lower() for arg in sys.argv])
    print 'runmode is %s. forceUpdate is %s' % (runmode, forceUpdate)   

    if runmode in [None,'all'] or 'chunks' in runmode:
        createHistoryChunks() 
    if runmode in [None,'all'] or 'parse' in runmode:
        getFullHistory() 
    if runmode in [None,'all'] or 'collapse' in runmode:
        collapseHistory(gadmLevel=[-1,0,1], timeLevel=['D'], forceUpdate=forceUpdate)     
