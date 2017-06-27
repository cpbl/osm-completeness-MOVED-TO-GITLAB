import os, datetime

# Use the command "git clone git@github.com:cpbl/osm-completeness.git" to create the following folder (osm_completeness) with the code in it.

rootpath = '../osm-completeness/'  # replace if you want to keep your files somewhere else
if not(os.path.exists(rootpath)):
    raise Exception('Root path %s does not exist. Please create it ("git clone git@github.com:cpbl/osm-completeness.git") or change rootpath in config.py' % rootpath)

# bin is where your .stan files should exist
paths = {'input':rootpath+'input/', 
         'output':rootpath+'output/',
         'working':rootpath+'working/',
         'scratch':rootpath+'scratch/',
         'completenessFits':rootpath+'scratch/completenessFits/',
         'bin':'./'}

for pathname, path in paths.iteritems():
    if not(os.path.exists(path)) and pathname not in ['bin','completenessFits']:
        os.mkdir(path)
        print('Created new %s directory: %s' % (pathname, path))
if not(os.path.exists(paths['completenessFits'])): os.mkdir(paths['completenessFits'])

osmHistoryFile = paths['input'] + 'history-latest-singlestream.osm.bz2'
pointsFn = paths['input'] + 'visual_assessment.pandas'

# connection parameters to your PostgreSQL database
pgLogin = {'db':'yourdbname', 'user':'yourusername', 'pw':'', schema:'yourschema', host:'localhost'}
# name of the PostGIS table where the Global Administrative Areas boundaries are loaded (through setup_postgis.py)
gadmTableName='gadm28'

# various fit parameters
xdateMAX=datetime.datetime(2016,1,4,0,0) # if you use the published data release
#xdateMAX=datetime.datetime(2017,4,3,0,0)  # if you use the updated data. Or change this if you use an even-more recent version
xdateMIN=datetime.datetime(2006,3,15,0,0)
COMPLETE=0.99
yscaleFactor = 10**9  # to make it easier for the optimization in history_fits

# colors for charts (not including black)
# based on http://colorbrewer2.org, 5 classes, print friendly
mpl_backend='Agg'
texAvailable = False  # set to True will give you nicer looking axis labels, etc.
c5s = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00'] 
figSize = (5.2, 3.35)  # aspect ratio same as below
figSizeBig = (7, 4.5) # for top-10 countries full column. 
figSizePage = (7, 8.75)  # for full-page figures
gtDate = '2015-02-11'  # OSM database used for ground truth
