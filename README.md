
# OSM Completeness
## v1.0 June 2017

This is the code release associated with Barrington-Leigh and Millard-Ball, "The world’s user-generated road map is more than 80%
complete", PLoS One, 2017.  Please use the appropriate academic citation to give credit for or to refer to this work.
The code is released under the GNU GPL v3 license.

Everything was developed and run only under GNU/Linux
operating systems (Ubuntu and RHEL). Our server had .8TB of RAM and 50
cores, but below we provide output data files in case you don't want
do all the computation.  If you do want to run the full processing
sequence, you might simply rent time on Amazon AWS or its ilk.

This code repository should reside permanently here: https://github.com/cpbl/osm-completeness (if not there, look here: https://alum.mit.edu/www/cpbl/PLoS2017roads/osm-completeness)

It replicates the PLOS One paper version (See the permanent link
https://alum.mit.edu/www/cpbl/PLoS2017roads). However, it is adapted to:
(i) Use the data files from the PLOS One SI
(ii) Simplify the paths and dependencies

------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------
Instructions:
------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------

There are two parts to the analysis: 
* Modeling of the visual assessment data
* Parametric fits that detect saturation in additions to the OSM database
    * Parsing the raw OSM history file
    * Finding the best-fit parametric model for each country

To replicate the analysis, follow each step below.

1. Clone this repository

```
git clone https://github.com/cpbl/osm-completeness.git
cd osm-completeness
```

You may want to edit some of the configuration settings in `history_config.py` to match your directory structure and system settings.

2. Download the data and install the dependencies listed below

3. Load the country boundaries and density rasters into your PostGIS database

`python setup_postgis.py`

If you only want to run the parametric fits in Step 5, you can skip this step.

4. Parse the raw osm history file and create country- and subnational-level files with the number of ways on each date.

`python process_raw.py`

Most users will **not** want to do this. It takes 3-4 weeks on a 50-core server, and requires you to set up a PostgreSQL database with the Global Administrative Areas boundary files. 

Instead, you can use the premade files: `osmCompleteness_level*.hd5`, 
provided at https://alum.mit.edu/www/cpbl/PLoS2017roads
Specifically,  you should unzip [this file](http://sprawl.research.mcgill.ca/PLoS2017/Barrington-Leigh-Millard-Ball-PLoSOne2017-data-release-all.zip) *inside* your copy of the osm-completeness code repository folder:

```
wget http://sprawl.research.mcgill.ca/PLoS2017/Barrington-Leigh-Millard-Ball-PLoSOne2017-data-release-all.zip
unzip Barrington-Leigh-Millard-Ball-PLoSOne2017-data-release-all.zip
```

This will put the premade files `osmCompleteness_level*.hd5` in your working folder (specified in `history_config.py`), and the remaining steps will run fine.

5. Run the parametric fits. This finds the best-fitting sigmoid curve for each country and subnational unit:

`python fits.py`

6. Run the multilevel regression and poststratification models. These models are estimated from the visual assessment data. 

`python multilevel.py`

7. Generate the final plots and country-level tables

`python analysis.py`

`visual.py` is the code that we used to perform the visual assessment. It needs to be run through the QGIS Python console, and will require some modifications to run on your own QGIS installation. It is included here for sake of completeness, but you don't need to use the code.

------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------
Dependencies: data
------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------

You'll need to download the following files to your `input` folder (specified in `history_config.py`).  For instance, using `wget` from your input folder:

```
wget https://alum.mit.edu/www/cpbl/PLoS2017roads/visual_assessment.pandas
wget https://alum.mit.edu/www/cpbl/PLoS2017roads/countries_compiled.pandas
```

or, alternatively, unzip the file
http://sprawl.research.mcgill.ca/PLoS2017/Barrington-Leigh-Millard-Ball-PLoSOne2017-data-release-small.zip
*inside* your copy of the osm-completeness code repository folder.

If you don't want to run `process_raw.py` (very likely), download `osmCompleteness_level*.hd5` to your `working` folder:

wget https://alum.mit.edu/www/cpbl/PLoS2017roads/osmHistory_level0D_density.hd5
wget https://alum.mit.edu/www/cpbl/PLoS2017roads/osmHistory_level0D.hd5
wget https://alum.mit.edu/www/cpbl/PLoS2017roads/osmHistory_level-1D_density.hd5
wget https://alum.mit.edu/www/cpbl/PLoS2017roads/osmHistory_level1D_density.hd5
wget https://alum.mit.edu/www/cpbl/PLoS2017roads/osmHistory_level-1D.hd5
wget https://alum.mit.edu/www/cpbl/PLoS2017roads/osmHistory_level1D.hd5

All of the above data dependencies can be met simply by unzipping the file (n.b. "-all" rather than "-small")
http://sprawl.research.mcgill.ca/PLoS2017/Barrington-Leigh-Millard-Ball-PLoSOne2017-data-release-all.zip
*inside* your copy of the osm-completeness code repository folder.

In order to run the parametric fits, you will also need to download the [Global Administrative Areas](http://gadm.org) boundary files and the [Landscan](http://web.ornl.gov/sci/landscan/landscan_data_avail.shtml) density raster to your `input` folder. The direct link to download the Global Administrative Areas data is [here](http://biogeo.ucdavis.edu/data/gadm2.8/gadm28.shp.zip). For Landscan, you will need to register in order to obtain access. Then, download `LandScan2013.zip` to your `input` folder.

------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------
Dependencies: ours and others' open-source code
------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------

The following Python packages are required. Note: later versions may also work, but have not been tested.

```
numpy 1.12.1
pandas 0.20.1
matplotlib 2.0.2
scipy 0.19.0
tables 3.4.2
psycopg2 2.7.1
pystan 2.15.0.1
svgutils 0.2.0
PyPDF2 1.26.0
```

Install them with:

`pip install numpy pandas matplotlib scipy tables psycopg2 pystan svgutils PyPDF2`

cpblUtilities. Use `git clone https://github.com/cpbl/cpblUtilities.git`

postgres/PostGIS installation (to run `process_raw_osm.py` and `python multilevel.py`). See [here](http://postgis.net/install/) for installation instructions.

[PIL](http://www.pythonware.com/products/pil/) is optional, but needed if you want to create the high-resolution gridded maps.

The following required files are included directly in this code repository:

```
shortNames.tsv
countrycodes.tsv
gadmCodes.tsv
WBregions.csv
areaLookup.csv
*.stan
```

