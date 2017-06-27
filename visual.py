"""
Undertakes the visual assessment for the analysis of OSM history

This is a stand-alone set up functions that require a Postgres database with OSM
The code is provided here for completeness, but it is not configured for general use.
Instead, use the provided data file, visual_assessment.pandas or visual_assessment.tsv

There are two parts to this:
1. Prepare the sample (in regular Python), by running choosePoints()
2. Do the assessment (load this file in the QGIS Python console). This is klunkier, but you can do it as follows:
   addOpenLayer()  # default is Google, can also use Bing and OSM
   a = sampleArray(contname) # for each continent
   # Now for each point, call (use the up arrow in the Python console to do this quickly)
   a.nextPoint()
   a.missCount(nMissingSegs) # where nMissingSegs is counted from the screen
   a.saveArray()  # do this periodically
"""

import os, sys
# may be necessary if QGIS doesn't have the path to the git repository
sys.path.append('/Users/amb/Documents/Research/sprawl/bin/osm')
from history_config import *
import history_tools as ht

# specific sample size
nPointsPop   = 25
nPointsHiDen = 20

pgPassword = None  # replace with your password, as getpass won't work in QGIS console
pgPort = '9997' #5432        # useful to change if you are connecting via a tunnel

#List of continents
contList = ['africa', 'asia', 'southamerica', 'centralamerica', 'australiaoceania', 'europe', 'northamerica']

#Dictionary of countries by continent, and the reverse
countryDict = {
'africa': ['Angola', 'Rwanda', 'Sudan', 'South Sudan', 'Senegal', 'Sierra Leone', 'Burundi', 'Somalia', 'Benin', 'Burkina Faso', 'Kenya', 'Swaziland', 'Chad', 'Togo', 'Liberia', 'Botswana', 'Libya', 'Central African Republic', 'Tunisia', 'Lesotho', 'Tanzania', 'Uganda', 'Morocco', 'Cameroon', 'Democratic Republic of the Congo', 'Madagascar', 'Republic of Congo', 'Mali', 'South Africa', 'Mozambique', 'Zambia', 'Djibouti', 'Mauritania', 'Zimbabwe', 'Malawi', 'Namibia', 'Algeria', 'Niger', 'Egypt', 'Nigeria', 'Eritrea', 'Ethiopia', 'Gabon', 'Ghana', 'Guinea', 'Comoros', 'Mauritius', 'Mayotte', 'Gambia', 'Guinea-Bissau', 'Equatorial Guinea', 'Cape Verde', 'Reunion', 'Seychelles', 'Western Sahara'],
'asia': ['Afghanistan','Singapore', 'Turkey', 'United Arab Emirates', 'Saudi Arabia', 'Armenia', 'Indonesia','India', 'Iran', 'Iraq', 'Azerbaijan', 'Israel', 'Jordan', 'Japan', 'Bangladesh', 'Kazakhstan', 'Kyrgyzstan', 'Cambodia', 'Syria', 'South Korea', 'Kuwait', 'Thailand', 'Laos', 'Brunei', 'Tajikistan', 'Lebanon', 'Bhutan', 'Turkmenistan', 'East Timor', 'Sri Lanka', 'China', 'Uzbekistan', 'Vietnam', 'Myanmar', 'Yemen', 'Mongolia', 'Malaysia', 'Nepal', 'Oman', 'Pakistan', 'Philippines', 'Georgia', 'North Korea', 'Qatar', 'Bahrain', 'Maldives'],
'southamerica':['Venezuela','Trinidad and Tobago', 'Falkland Islands', 'French Guiana', 'Colombia', 'Argentina', 'Bolivia', 'Brazil', 'Peru', 'Chile', 'Ecuador', 'Paraguay', 'Suriname', 'Uruguay', 'Guyana'],
'centralamerica':['Honduras', 'Mexico', 'Costa Rica', 'Belize', 'Panama', 'El Salvador', 'Nicaragua', 'Guatemala', 'Puerto Rico', 'Dominican Republic', 'Haiti', 'Jamaica', 'Cuba', 'Bahamas', 'Dominica','Grenada'],
'australiaoceania':['New Zealand', 'New Caledonia', 'Fiji', 'Vanuatu', 'Solomon Islands', 'Papua New Guinea', 'Australia', 'Antigua and Barbuda', 'Marshall Islands', 'Nauru', 'Palau', 'Samoa', 'Tonga', 'Tuvalu', 'Micronesia', 'Kiribati'], 
'europe':['Romania', 'Kosovo', 'Cyprus', 'Russia', 'Albania', 'Croatia', 'Hungary', 'Ireland', 'Austria', 'Iceland', 'Italy', 'Belgium', 'Serbia', 'Slovakia', 'Slovenia', 'Bulgaria', 'Sweden', 'Bosnia and Herzegovina', 'Belarus', 'Switzerland', 'Lithuania', 'Luxembourg', 'Latvia', 'Ukraine', 'Moldova', 'Macedonia', 'Montenegro', 'Czech Republic', 'Germany', 'Denmark', 'Spain', 'Netherlands', 'Estonia', 'Norway', 'Finland', 'France', 'United Kingdom', 'Poland', 'Portugal', 'Greece', 'Andorra', 'Liechtenstein', 'Malta', 'Monaco', 'Vatican City', 'San Marino'],
'northamerica':['Canada', 'United States'],
}
contDict = {country:cont for cont in countryDict for country in countryDict[cont]}


"""
These functions are run in regular Python
"""
def choosePoints(forceUpdate=False):
    """Saves and returns a dataframe with a stratified random sample for each country"""
    import pandas as pd
    import osmTools as osmt    
    from scipy import stats

    sampleFn = paths['working']+'visualAssessmentSample.npy'
    if os.path.exists(sampleFn) and not forceUpdate:
        return np.load(sampleFn)

    #Load density and country array
    arrays = loadArrays()
    
    gadmCountries = ht.get_all_GADM_countries()
    df = ht.loadOtherCountryData()
    urbanpc = df.urbanPop2012.astype(float)/df.pop2012
    
    sample = {} # dict of country-level samples
    
    # For each country, get 25 points weighted by log population density
    for iso in gadmCountries:
        countryName = gadmCountries[iso]['name']
        if not(countryName in contDict):
            print 'Skipping %s' % countryName
            continue
        print 'Sampling %s' % countryName
        gadmId = gadmCountries[iso]['id_0']
        countryArray = arrays[arrays[:,2]==gadmId]
        countryArray = countryArray[countryArray[:,0]>0]  # exclude zero pop points
        if len(countryArray)==0: continue
              
        # log population density weighted sample 
        countrySample = getSample(countryArray, nPointsPop, weights=True)
        countrySample['source'] = 'popWeighted'

        # sample of high-density areas
        if iso in urbanpc.index.values and pd.notnull(urbanpc[iso]):
            totalPop = np.sum(countryArray[:,0])
            popCutoff = totalPop * urbanpc[iso]/1.5/100.
            
            # need to sort countryArray by density
            sortOrder = countryArray[:,1].argsort()[::-1]
            countryArray = countryArray[sortOrder]
            
            cumPop = np.cumsum(countryArray[:,0])
            probHiDens = (cumPop>=popCutoff).sum()*1./len(countryArray)*nPointsHiDen  # probability of one point being chosen
            
            countryArrayHiDen = countryArray[cumPop<=popCutoff]
            if len(countryArrayHiDen)<=1: countryArrayHiDen = countryArray[0:1]  # for Monaco and other places where all the pop is in one grid cell
            cutOffDensity = countryArrayHiDen[-1,1]
            densCutoffPctile = stats.percentileofscore(countryArray[:,1], cutOffDensity) /100. # note: excludes 0-population cells
            
            sampleHiDens = getSample(countryArrayHiDen, nPointsHiDen)
            sampleHiDens['source'] = 'highDensity'
            sampleHiDens['densCutoffPctile'] = densCutoffPctile
            sampleHiDens['densCutoff'] = np.exp(cutOffDensity)
            
            # Probability of selection (for grid cells over the density threshold) is 1/(1-q)/N, 
            # where q is the quantile of the density threshold. N (number of grid cells in the country) is normalized to 1
            sampleHiDens['Pselection'] = 1. / (1-densCutoffPctile)
            
            countrySample = pd.concat([countrySample, sampleHiDens]).reset_index(drop=True)
        
        countrySample['country']   = countryName
        countrySample['continent'] = contDict[countryName]
        
        sample[iso] = countrySample
            
    # now concat dict 
    allCountries = pd.concat(sample)
    allCountries.index.names=['ISOalpha3','id']
    allCountries.reset_index(level=1, inplace=True)
        
    # add empty fields to be filled in during sampling
    for newCol in ['NmissingSegs', 'NosmNodes', 'NosmSegs','latproj','longproj','totSegs','frcComplete']:
        allCountries[newCol] = np.nan
    allCountries['comments'] = ''  # text string
    
    #allCountries.to_pickle(sampleFn)
    numpyArray = allCountries.to_records()  # hard to import pandas into QGIS, so use numpy
    np.save(sampleFn, numpyArray)
    return numpyArray    

def getSample(countryArray, nPts, weights=False):
    """Returns a random sample from countryArray. Weighted by col 1 if weights are passed"""
    import pandas as pd

    ids = np.arange(countryArray.shape[0])
    if weights is True: 
        pWeights = countryArray[:,1]/countryArray[:,1].sum()  # log density weighted
    else:
        pWeights = None

    # randomly choose grid cells, and then a point within that grid cell
    sampleIds = np.random.choice(ids, size = nPts, p=pWeights)    
    latOffset = np.random.random(nPts)/-120.   # each grid cell is 1/120 degrees, negative for lats as they start at -90
    lonOffset = np.random.random(nPts)/120.   
    
    lats    = countryArray[:,3][sampleIds] + latOffset
    lons    = countryArray[:,4][sampleIds] + lonOffset
    density = np.exp(countryArray[:,1][sampleIds])
    # note: probabilities are normalized to be comparable with hi-density stratum (i.e, we multiply by N)
    probs   = pWeights[sampleIds]*len(countryArray) if weights is True else [np.nan]*nPts
    sample = pd.DataFrame([lats, lons, density, probs], index = ['lat','long','density', 'Pselection']).T
    
    return sample        

def loadArrays():
    """Loads numpy arrays with density and lat information
       in 3rd dimension, they are ordered by 
       log population, density, gadm id0, lat and long"""
    import rasterTools as rt

    rastFn = paths['scratch']+'densityrasters.npz'
    if not(os.path.exists(rastFn)):
        print 'Loading rasters from PostGIS'
        rc = rt.rasterCon('landscan')
        lsarray = rc.asarray(bands=[1,3], padEmptyRows=True) # band 3 is density
        lsarray[:,:,1] = np.log(lsarray[:,:,1])  # convert density to log density
        lsarray[:,1][np.isinf(lsarray[:,1])] = np.nan
        
        # add gadm country code
        rc = rt.rasterCon('gadmraster')
        gaarray = rc.asarray(bands=1, padEmptyRows=True) # band 1 is id_0
        
        inRasters = np.dstack([lsarray, gaarray])
        assert inRasters.shape==(21600, 43200, 3)

        # add lat and long information
        longs = np.broadcast_to(np.arange(-180,180,1/120.), (21600, 43200))
        lats  = np.swapaxes(np.broadcast_to(np.arange(90,-90,-1/120.), (43200, 21600)), 0, 1)
        inRasters = np.dstack([inRasters, lats, longs]).reshape(21600*43200,5)
    
        np.savez_compressed(rastFn, rast=inRasters)
    else:
        print 'Loading rasters from file'
        arrays = np.load(rastFn)
        inRasters = arrays['rast']

    assert inRasters.shape==(21600*43200, 5)
    return inRasters

"""
These functions are for use in QGIS Python console
"""

def addOpenLayer(layerName='satellite'):
    """Add Google, OpenStreetMap or Bing aerial to canvas"""
    layerName = layerName.lower()
    layeract= {'satellite':'Google Maps', 'streets':'Google Maps', 'bing':'Bing Maps', 'osm':'OpenStreetMap'}[layerName]
    layerNum= {'satellite':3, 'streets':1, 'bing':1, 'osm':0}[layerName]  # which menu item to choose
    
    webmenu = qgis.utils.iface.webMenu()
    olmenu = False
    for act in webmenu.actions():
        if 'OpenLayers plugin' in act.text():
            olmenu = act
    if olmenu:
        for act in olmenu.menu().actions():
            if layeract in act.text():
                act.menu().actions()[layerNum].trigger()
    else:
        print 'You need to install OpenLayers'

def removeOpenLayers():
    """Remove all OpenLayers layers from canvas"""
    layers = QgsMapLayerRegistry.instance().mapLayers()
    for layer in layers:
        if 'OpenLayers' in layer or 'OpenStreetMap' in layer: QgsMapLayerRegistry.instance().removeMapLayer(layer)   

class sampleArray():
    """Holds the array of sampled and to-be-sampled points in QGIS
    Allows point to be displayed, and missing segments to be entered"""
    def __init__(self,continent=None,path=None):
        if path is None: path = paths['working']
        self.arr=np.load(path+'visualAssessmentSample.npy')
        self.path=path
        if continent is not None: addContinent(continent)
        self.currentContinent=continent
        self.rowId=None

    def nextPoint(self, continent=None):
        """Brings up the next point in the continent for analysis
        After this, call addPoint(n) to write to the array, where n is the number of missing segments"""
        if continent is None:
            if self.currentContinent is None:
                raise Exception('Cannot get next point. No continent defined')
            else: 
                continent=self.currentContinent
        if continent!=self.currentContinent: # start new continent
            addContinent(continent)
            self.currentContinent=continent
    
        # get first row that has not yet been done
        rowIds = np.where((self.arr['continent']==continent) & (np.isnan(self.arr['NmissingSegs'])))[0]
        if len(rowIds)==0:
            print 'No points remaining in continent %s' % continent
            self.rowId=None
            return
        self.rowId = rowIds[0]
        row = self.arr[self.rowId]
    
        print 'Sample point %s in country %s' % (row['id'], row['country'])
        x, y = convertSpatialRef(row['long'], row['lat'])
        centerCanvas(x,y)
        
        #Select features within current extent
        canvas = qgis.utils.iface.mapCanvas()
        ext = canvas.extent()
        allLayers = canvas.layers()
        seglyr, junclyr = allLayers[0], allLayers[1]
        assert str(seglyr.name()) == defaults['osm']['seg_table'].replace('REGION',continent)
    
        seglyr.select(ext, True)
        junclyr.select(ext, True)
        #Count selected features
        scount = seglyr.selectedFeatureCount()
        jcount = junclyr.selectedFeatureCount()
        print "Edges found: %s. Nodes found: %s" % (scount, jcount) 
        
        # remove selection
        seglyr.removeSelection()
        junclyr.removeSelection()
    
        #Write coordinates, density, and count to the array
        self.arr[self.rowId]['longproj'] = x
        self.arr[self.rowId]['latproj'] = y
        self.arr[self.rowId]['NosmSegs']  = scount
        self.arr[self.rowId]['NosmNodes'] = jcount
        
        jpgFn='_'.join([row['country'], str(row['id']), str(row['lat'])[0:6], str(row['long'])[0:7]])
        printscreen(jpgFn)       
    
    def missCount(self, nMissingSegs, comments=None):
        """Add the count of missing segments and optional comment to the array
        This is clunky, but because QGIS doesn't take raw_input(), we do it as a function"""
        if self.rowId is None: 
            print('No point selected. Use nextPoint')
        elif np.isnan(self.arr[self.rowId]['longproj']):
            print('Point not yet displayed. Use nextPoint')
        elif not(np.isnan(self.arr[self.rowId]['NmissingSegs'])):
            print('Already done this point. Use undo() to do it again')
        else:
            self.arr[self.rowId]['NmissingSegs'] = nMissingSegs
            self.arr[self.rowId]['totSegs']      = nMissingSegs+self.arr[self.rowId]['NosmSegs'] 
            self.arr[self.rowId]['frcComplete']  = float(self.arr[self.rowId]['NosmSegs'])/self.arr[self.rowId]['totSegs'] 
            if comments is not None:
                assert isinstance(comments, str)
                self.arr[self.rowId]['comments'] = comments
            
    def undo(self):
        """Undoes missCount() for the last point (in case a mistake is made)"""
        for col in ['NmissingSegs','totSegs','frcComplete','comments']:
            self.arr[self.rowId][col] = np.nan
    
    def saveArray(self):
        np.save(self.path+'visualAssessmentSample.npy', self.arr)
        print 'Updated array visualAssessmentSample.npy saved to %s' % self.path
    
def addContinent(continent):
    """Adds a layer to the canvas with the OSM edges and nodes from the project database"""
    print 'Displaying OSM edges and nodes from %s...this may take a while' % continent
    juncs = defaults['osm']['junc_table'].replace('REGION',continent)
    segs  = defaults['osm']['seg_table'].replace('REGION',continent)

    #Set database connection
    uri = QgsDataSourceURI()
    pgLogin = {'db': defaults['server']['postgres_db'], 'user': defaults['server']['postgres_role'], 'host':'localhost'}
    uri.setConnection(pgLogin['host'], pgPort, pgLogin['db'], pgLogin['user'], pgPassword)
    
    #Add the road_juncs layer 
    face = qgis.utils.iface
    uri.setDataSource(defaults['osm']['osmSchema'], ('{0}'.format(juncs)), "geom")
    junclyr = face.addVectorLayer(uri.uri(), ('{0}'.format(juncs)), "postgres")
    
    #Add labels to the layer
    palyr = QgsPalLayerSettings()
    palyr.readFromLayer(junclyr)
    palyr.enabled = True
    palyr.fieldName = 'degree7'
    palyr.placement= QgsPalLayerSettings.QuadrantAboveRight
    palyr.setDataDefinedProperty(QgsPalLayerSettings.Size,True,True,'10','')
    palyr.writeToLayer(junclyr)
    
    # Add edges
    uri.setDataSource(defaults['osm']['osmSchema'], ('{0}'.format(segs)), "geom")
    seglyr = face.addVectorLayer(uri.uri(), ('{0}'.format(segs)), "postgres")
    # Increase line width
    registry = QgsSymbolLayerV2Registry.instance()
    lineMeta = registry.symbolLayerMetadata("SimpleLine")
    symbol = QgsSymbolV2.defaultSymbol(seglyr.geometryType())
    #Create new symbology
    lineLayer = lineMeta.createSymbolLayer({'width': '0.75', 'color': '255,3,112'})
    #Remove old symbology and append new
    symbol.deleteSymbolLayer(0)
    symbol.appendSymbolLayer(lineLayer)
    #Render it
    renderer = QgsSingleSymbolRendererV2(symbol)
    seglyr.setRendererV2(renderer)
 
def convertSpatialRef(x, y, inputEPSG=4326, outputEPSG=3857):
    """Converts x and y in inputEPSG to outputEPSG
    Default is to convert from WGS84 lat/long to Projected Mercator"""
    import ogr, osr
    print x, y
    ogrpoint = ogr.Geometry(ogr.wkbPoint)
    ogrpoint.AddPoint(x, y)
    
    # Create coordinate transformation
    inSpatialRef = osr.SpatialReference()
    inSpatialRef.ImportFromEPSG(inputEPSG)
    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(outputEPSG)
    coordTransform = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)
    
    # Transform point
    ogrpoint.Transform(coordTransform)
    
    return ogrpoint.GetX(), ogrpoint.GetY() 

def centerCanvas(x, y, zoom=5000):
    """Centers and zooms the QGIS canvas"""
    map_pos = QgsPoint(x, y)
    rect = QgsRectangle(map_pos, map_pos)
    canvas = qgis.utils.iface.mapCanvas()
    canvas.setExtent(rect)
    canvas.refresh()
    canvas.zoomScale(5000)

def printscreen(jpgFn):
    """Prints the canvas to a JPG, so we have a permanent record"""
    jpgPath = paths['working']+'visualAssessmentJPGS/'
    if not(os.path.exists(jpgPath)):
        print '%s does not exist. Creating it' % jpgPath
        os.mkdir(jpgPath)
    canvas = qgis.utils.iface.mapCanvas()
    allLayers = canvas.layers()
    seglyr = allLayers[1]
    junclyr = allLayers[0]
    canvas.saveAsImage(jpgPath+jpgFn+'jpg', None, "JPG")

if __name__ == '__main__':
    forceUpdate = any(['forceupdate' in arg.lower() for arg in sys.argv])
    print 'forceUpdates is %s' % forceUpdate   
    df = choosePoints(forceUpdate=forceUpdate)
    print 'Sample dataframe created. Now open history_visual.py in the QGIS Python console, and run as follows.'
    print 'Setup: addOpenLayer(), a = sampleArray(contname)'
    print 'For each point: a.nextPoint(), a.missCount(nMissingSegs)'
    print 'a.saveArray() periodically'
    
    