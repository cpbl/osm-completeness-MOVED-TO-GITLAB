#!/usr/bin/python
# coding=utf-8
"""
Plots and other analysis of OSM completeness data

Call as:
python analysis.py 

Optional argument can be plots or tables to only do part of the analysis

"""
from history_config import *
import os, sys, datetime, math
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use(mpl_backend)
import matplotlib.pyplot as plt

import history_tools as ht
from cpblUtilities.mathgraph import figureFontSetup, saveAllFiguresToPDF
from cpblUtilities.parallel import runFunctionsInParallel

if texAvailable: 
    figureFontSetup(uniform=11)
else:
    plt.rcParams.update({'font.size': 11, 'ytick.labelsize': 11, 'xtick.labelsize': 11,'legend.fontsize': 11, 'axes.labelsize': 11, 'figure.figsize': (4.6, 4)})

def aggregateVisualAssessmentPoints(forceUpdate=False):
    """
    Aggregates results from the visual assessment to the country level
    Bootstraps standard errors and CIs
    Returns and saves a dataframe with country-level variables
    """
    countryOutFn = paths['input'] + 'visual_assessment_countries.pandas'
    if os.path.exists(countryOutFn) and not forceUpdate:
        return pd.read_pickle(countryOutFn)
    
    pointsDf = pd.read_pickle(pointsFn)
    assert pointsDf.index.name=='ISOalpha3'

    colsToGroup = ['NpresentSegs', 'NmissingSegs']
    countryDf = pointsDf.groupby(level=0)[colsToGroup].var()
    countryDf.rename(columns = dict([(cname, cname+'_var') for cname in colsToGroup]),inplace=True)
    countryDf = countryDf.join(pointsDf.groupby(level=0)[colsToGroup].sum())
    
    countryDf['N'] = pointsDf.groupby(level=0).NpresentSegs.count()    
    countryDf['totSegs'] = countryDf.NpresentSegs + countryDf.NmissingSegs
    countryDf['totSegs_var'] = countryDf.NpresentSegs_var + countryDf.NmissingSegs_var
    countryDf['frcComplete_visual'] = pointsDf.groupby(level=0).apply(lambda adf: float(sum(adf.NpresentSegs*adf.weight)) / sum((adf.NpresentSegs+adf.NmissingSegs)*adf.weight))    
    countryDf['urbanFrcComplete_visual'] = pointsDf[pointsDf.sampleType=='highDensity'].groupby(level=0).apply(lambda adf: float(sum(adf.NpresentSegs)) / sum((adf.NpresentSegs+adf.NmissingSegs)))    
    
    countryDf.sort_values(by='frcComplete_visual', inplace=True)
    
    # Assume a joint distribution of (n_i, p_i). Thus the bootstrap samples from the observations sites using the same process as the original sample was drawn
    # choose the bootstrap samples using the original probabilities. http://www.sagepub.com/sites/default/files/upm-binaries/21122_Chapter_21.pdf, p. 602
    def _bootstrap_unequalObservations(adf, reps=1000):
        Nsites=len(adf)  # country-level
        bMask, tMask = adf.sampleType=='popWeighted', adf.sampleType=='highDensity'
        assert bMask.sum()+tMask.sum() == Nsites
        sFrcComplete=[]
        # np.random.choice needs standardized weights that sum to 1, within each stratum, and take the inverse
        adf['Pselection'] = 1./adf.weight
        bWeightsStd = (adf[bMask].Pselection / adf[bMask].Pselection.sum()).values
        tWeightsStd = (adf[tMask].Pselection / adf[tMask].Pselection.sum()).values

        for ii in range(reps):
            #ii=np.random.choice(range(Nsites), size=Nsites, replace=True) # choose element indices with replacement
            ii = np.array([], dtype=int)
            if tMask.sum()>0: ii=               np.random.choice(np.where(tMask)[0], size=tMask.sum(), replace=True, p=tWeightsStd)   
            if bMask.sum()>0: ii=np.hstack([ii, np.random.choice(np.where(bMask)[0], size=bMask.sum(), replace=True, p=bWeightsStd)]) 
            totSegs_=adf.totSegs.values[ii]
            obsEdges_=adf.NpresentSegs.values[ii]
            weights_= 1.0/adf.Pselection.values[ii]
        
            sFrcComplete+= [sum(obsEdges_ * weights_) / sum(totSegs_ * weights_)]

        #return sFrcComplete   # return all replications
        return pd.Series([np.mean(sFrcComplete), np.std(sFrcComplete), np.percentile(sFrcComplete, 5), np.percentile(sFrcComplete, 95)], index=[vv+'_boot' for vv in ['frcComplete_visual', 'frcComplete_visual_se', 'frcComplete_visual_5pct', 'frcComplete_visual_95pct']])  
        
    print('   Bootstrapping....')
    countryDf = countryDf.join(pointsDf.groupby(level=0).apply(_bootstrap_unequalObservations))
        
    countryDf.to_pickle(countryOutFn)
    
    return countryDf

def compileCountries(stanModel='multilevel_GT_sq',forceUpdate=False):
    """
    Returns a dataframe of countries aggregating the following:
    - ground truth estimates from bootstrapping
    - ground truth estimates from Stan (using the model specified in stanModel)
    - best fit from chooseBestFit (frcComplete_model is how far along relative to the asymptote, frcComplete_fit compares data to asymptote)
    - other data (GDP, etc.)
    """
    outFn = paths['scratch']+'countries_history_compiled.pickle'
    if os.path.exists(outFn) and not forceUpdate:
        return pd.read_pickle(outFn)
    print('Compiling estimates from bootstrap, Stan and fits')
    
    from fits import chooseBestFit, loadAggregatedHistory
        
    # Bootstrap estimates
    colsToUse = ['frcComplete_visual', 'frcComplete_visual_boot','frcComplete_visual_se_boot', 'frcComplete_visual_5pct_boot', 'frcComplete_visual_95pct_boot']
    countries = aggregateVisualAssessmentPoints(forceUpdate=forceUpdate)[colsToUse]
    assert countries.index.name == 'ISOalpha3'
    
    # Stan estimates
    colsToUse = ['frcComplete_data', 'frcComplete_MRP_insample', 'frcComplete_MRP_insample_5pct', 'frcComplete_MRP_insample_95pct',
                 'totSegs_MRP_outofsample', 'totSegs_MRP_outofsample_5pct', 'totSegs_MRP_outofsample_95pct', 
                 'osmSegs_MRP_outofsample', 'osmSegs_MRP_outofsample_5pct', 'osmSegs_MRP_outofsample_95pct', 
                 'frcComplete_MRP_outofsample', 'frcComplete_MRP_outofsample_5pct', 'frcComplete_MRP_outofsample_95pct']
    joinDf = pd.read_pickle(paths['working'] + 'countriesFit_'+stanModel+'.pickle')[colsToUse]
    assert [ii for ii in joinDf.index if ii not in countries.index] == ['ALL']  # Stan shouldn't have any extra countries, only 'ALL'
    countries = countries.join(joinDf, how='outer')  # outer because we need 'ALL' from joinDf
    assert all(pd.isnull(countries.frcComplete_data) | (countries.frcComplete_data.round(8) == countries.frcComplete_visual.round(8)))
    countries.drop('frcComplete_data', axis=1, inplace=True)
    
    # Best country-level fit
    joinDf = chooseBestFit(gadmLevel=0)
    assert not any([ii not in joinDf.index for ii in countries.index])
    countries = countries.join(joinDf, how='outer')  # outer because some countries have parametric fits but no GT data
    
    # Now we need to deal with the fact that the fits use Jan 2016 (or later if we update...) data
    # The groundtruth is based on an earlier version of the OSM dataset (Feb 5-11, 2015, depending on continent)
    # So we need to (i)   rename the visual assessment estimate to Feb2015, and 
    #               (ii)  scale visual assessment estimates to the latest data
    #               (iii) scale the parametric fits to Feb2015
    # g gives the growth in the length of OSM segments from Feb2015 to the latest data, 
    #   so length_Feb2015 * g = maxlength, and frcComplete_Feb2015 * g = frcComplete
    
    renameCols = [cc for cc in countries.columns if 'frcComplete_visual' in cc or 'frcComplete_MRP' in cc]
    renameCols = dict([(cc, cc.replace('frcComplete','frcComplete_Feb2015')) for cc in renameCols])
    countries.rename(columns=renameCols, inplace=True)

    countriesEarlier = loadAggregatedHistory(gadmLevel=0).ix[gtDate].set_index('ISOalpha3') # 
    countries['maxLength_Feb2015'] = countriesEarlier['length'].combine_first(countries.maxLength)  # uses max length if there are no recent additions
    countries['g'] = countries.maxLength.astype(float) / countries.maxLength_Feb2015
    countries.loc[(countries.g<1), 'g'] = 1.0  # don't allow for declining growth
    countries['frcComplete_Feb2015_fit'] = countries.frcComplete_fit / countries.g
    for col in renameCols:
        assert col not in countries.columns  # should have been renamed
        countries[col] = countries[renameCols[col]] * countries.g
        #countries.loc[countries[col]>1, col] = 1.0   # set max at 1.0. Aug2016: no, not here. Do only for 'best'
    countries['length_MRP_insample'] = countries.maxLength / countries.frcComplete_MRP_insample 
    countries['length_MRP_outofsample'] = countries.maxLength / countries.frcComplete_MRP_outofsample 
    assert countries.index.name == 'ISOalpha3'
    
    # Fit from adding up gadmLevel 1 estimates, including adding up gadmLevel1 estimates earlier
    level1Df = chooseBestFit(gadmLevel=1,excludeJustaline=True).set_index('ISOalpha3')
    #why do we need these estimates for 2015?
    #countriesEarlier = loadAggregatedHistory(gadmLevel=1).ix[gtDate].set_index(['id0','id1']) 
    #level1Df['maxLength_Feb2015'] = countriesEarlier['length'].combine_first(level1Df.maxLength) 
    joinDf = addUpEstimates(level1Df, '_lev1_')
 
    assert len(joinDf)+1==len(countries) and [ii for ii in countries.index if ii not in joinDf.index] == ['ALL'] # ALL should be in countries but not fits
    countries = countries.join(joinDf)
    assert countries.index.name == 'ISOalpha3'
    
    # for the world, we call level1Agg the country-level fits. But let's weight them by Stan road length
    # Otherwise, Cameroon has the longest road network in the world!
    for cc in ['fit','model']:
        mask = np.logical_not((countries.index=='ALL') | pd.isnull(countries.length_MRP_outofsample) | pd.isnull(countries['frcComplete_'+cc]))
        countries.loc['ALL','frcComplete_lev1_'+cc] =  (countries.loc[mask, 'length_MRP_outofsample'] * countries.loc[mask, 'frcComplete_'+cc]).sum() / countries.loc[mask, 'length_MRP_outofsample'].sum()

    # add in fit from adding up density quintile estimates
    quintileDf = chooseBestFit('quintiles', excludeJustaline=True).reset_index(level=1)
    #countriesEarlier = loadAggregatedHistory(gadmLevel=0, density=True).ix[gtDate]
    #countriesEarlier = densityDeciles2Quintiles(countriesEarlier).set_index(['ISOalpha3','densQuintile'])
    #quintileDf['maxLength_Feb2015'] = countriesEarlier['length'].combine_first(quintileDf.maxLength) 
    joinDf = addUpEstimates(quintileDf, '_quintiles_')
    assert all([cc for cc in joinDf.index if cc in countries.index])
    countries = countries.join(joinDf)
    assert countries.index.name == 'ISOalpha3'
    
    # add in other data
    joinDf = ht.loadOtherCountryData()
    countries = countries.join(joinDf)
    
    assert countries.index.name == 'ISOalpha3'
    assert 'KSV' not in countries.index.values and 'SP-' not in countries.index.values # we should have updated to 'XKO' and 'XSP'
    
    countries.to_pickle(outFn)
    return countries
 
def addUpEstimates(subDf, stubName):
    """
    Adds up sub-national estimates of frcComplete (by gadm level 1 or density quintile) to the country level
    Weights each level1 or quintile by length
    Returns a dataframe with the national totals (grouped by ISOalpha3)
    """
    assert subDf.index.name == 'ISOalpha3'
    aggDf = subDf.groupby(level=0)['length_fit','length_model'].sum()
    for cc in ['fit','model']:
        aggDf['tmpnumerator'] = subDf[pd.notnull(subDf['length_'+cc])].groupby(level=0)['maxLength'].sum().astype(float)
        aggDf['frcComplete'+stubName+cc] = aggDf.tmpnumerator / aggDf['length_'+cc]
        aggDf.loc[aggDf['frcComplete'+stubName+cc]>1,'frcComplete'+stubName+cc] = 1
    return aggDf[[cc for cc in aggDf.columns if stubName in cc]]
 
def chooseStanorFit(stanModel='multilevel_GT_sq'):
    """
    We have several measures of completeness from Stan and the parametric fits
    This chooses the best one and report it as frcComplete_best
    take parametric fit if it agrees with Stan (within 95% CI or 5% of point estimate), otherwise Stan
    completeAgree is true if the following show completeness: (i) Stan; (ii) frcComplete_fit OR frcComplete_model; (iii) frcComplete_lev1_fit OR frcComplete_density_fit
    The choice between frcComplete_fit OR frcComplete_model is because some countries (e.g. USA, Ethopia) have a drop in OSM streets at the end
    
    We also:
        - top code frcComplete_best at 1.0 (fits can indicate more than 100% completeness)
        - set length_best to nan if frcComplete_best < 0.05 (usually here, we get way too high estimates of length
    """
    tol = 0.05 # how close the measures of frcComplete can be to agree
    df = compileCountries(stanModel)
    df['completeAgree'] = (df.frcComplete_MRP_outofsample>=1-tol ) & ((df.frcComplete_fit>=1-tol ) | (df.frcComplete_model>=1-tol )) & ((df.frcComplete_lev1_fit>=1-tol ) | (df.frcComplete_quintiles_fit>=1-tol ))
    
    def _chooseStanOrFit_onecountry(adf, tol):
        if pd.isnull(adf.frcComplete_MRP_outofsample) and pd.notnull(adf.frcComplete_fit):
            measure='fit'
        elif pd.notnull(adf.frcComplete_MRP_outofsample) and pd.isnull(adf.frcComplete_fit):
            measure='stan'
        elif pd.isnull(adf.frcComplete_MRP_outofsample) and pd.isnull(adf.frcComplete_fit):
            measure = np.nan
        elif adf.frcComplete_fit>=min(adf.frcComplete_MRP_outofsample_5pct, adf.frcComplete_MRP_outofsample-tol) and min(1., adf.frcComplete_fit)<=max(adf.frcComplete_MRP_outofsample_95pct, adf.frcComplete_MRP_outofsample+tol):
            measure = 'fit'
        else:
            measure = 'stan'
        return measure
        
    df['measure_best'] = df.apply(lambda x: _chooseStanOrFit_onecountry(x, tol), axis=1)
    
    df.loc[df.measure_best=='fit','frcComplete_best'] = df.frcComplete_fit
    df.loc[df.measure_best=='stan','frcComplete_best'] = df.frcComplete_MRP_outofsample
    df.loc[df.frcComplete_best>1, 'frcComplete_best'] = 1.0   # set max at 1.0
    df['length_best'] = df.maxLength / df.frcComplete_best
    print 'Setting these countries lengths to nan (frcComplete_best < 0.05):'
    print df[df.frcComplete_best<0.05].countryname.values
    df.loc[df.frcComplete_best<0.05, 'length_best'] = np.nan # no good estimate of length here
    
    return df
    
def figMapCompleteness(fieldToMap=None):
    """
    Saves a figure of estimated completeness and road length per capita, using our best estimate of the completeness (GT or fitted)
    """
    if fieldToMap is None:
        # call this function recursively
        for fld in ['frcComplete_best', 'roadspc',]:#'logroadspc']:
            figMapCompleteness(fld)
        return
    
    legendDict = {'frcComplete_best': 'fraction of streets in OSM database',
                  'roadspc': 'Road length per capita (m)'}
    legendLabel = legendDict[fieldToMap] if fieldToMap in legendDict else ''
    
    countries = chooseStanorFit()
    countries.index.name='id'  # to be recognized by the svg
    #countries['logroadspc']= (countries.length_best / countries.pop2012).map(np.log)

    if fieldToMap == 'roadspc':
        countries['roadspc'] = countries.length_best / countries.pop2012
        countries.loc[countries.roadspc>30,'roadspc']=30 # top code to avoid outliers distorting scale
    # this is the place to add other variables that we might want to map
    
    assert fieldToMap in countries.columns
    
    countries = countries[fieldToMap]
    countries = countries[pd.notnull(countries)]
    
    if fieldToMap=='frcComplete_best': # constrain to max of 1
        countries = countries.apply(lambda x: min(1.0, x))
    
    svgfn=paths['bin']+'BlankMap-World6-noAntarctica_ISO3.svg'
    
    from cpblUtilities.mapping import colorize_svg
    from cpblUtilities.color import linearColormapLookup,assignColormapEvenly,getIndexedColormap
    from cpblUtilities.mathgraph import tightBoundingBoxInkscape
    # use a matplotlib color map
    # cpbl has something more fancy, but this misses out countries for me
    
    if fieldToMap == 'roadspc':
        data2color = assignColormapEvenly('Blues', countries.values,asDict=False,missing=[1,1,1]) 
    else:
        data2color = linearColormapLookup('Blues', countries.values)

    map = colorize_svg(countries, outfilename=paths['output']+'OSMmap-'+fieldToMap+'.svg', 
                       addcolorbar=True, cbylabel=legendLabel, data2color=data2color,#'Blues',
        customfeatures=dict(cbarpar=dict(
        expandx=1, movebartox=150, movebartoy=400,
            scalebar=2.0,
            fontsize= 24
        )),
        CSSselector='class', 
        blanksvgfile=svgfn)
        
    # The legend needs to indicate that 30 is topcoded (for roads pc). Kludge..
    if fieldToMap=='roadspc':
        print 'Warning: need to replace 30 with 30+ in legend (do this manually in Inkscape)'
    
    # Tighten up boundaries; also, make the SVG conform to standards:
    tightBoundingBoxInkscape(paths['output']+'OSMmap-'+fieldToMap+'.svg',use_xvfb=True)

    return

def plotCompletenessMeasures():
    """Plots all our various measures of completeness"""
    countries = compileCountries()
    assert countries.index.name == 'ISOalpha3'
    countries.sort_values(by='maxLength', ascending=False, inplace=True)
    shortNames = ht.country2ISOLookup()['ISOalpha2shortName']
 
    nrows = 40 
    rowNums = [(ii*nrows, min(len(countries),(ii+1)*nrows)) for ii in range(int(math.ceil(len(countries)/nrows)))]
    GTbarHeight, stanBarHeight = 0.8, 0.5
    barOffset = (GTbarHeight-stanBarHeight)/2.
      
    plt.close('all')    
    figs = []
    for ii, (srow, erow) in enumerate(rowNums):
        fig, ax = plt.subplots(figsize=figSizePage)
        plotDf = countries.iloc[srow:erow]
        plotDf = plotDf.reindex(index=plotDf.index[::-1])
        yVals = np.arange(0,len(plotDf))
        
        # plot bootstrapped GT (as two separate bars either side of point estimate)
        widths = plotDf.frcComplete_visual_boot-plotDf.frcComplete_visual_5pct_boot
        ax.barh(yVals, widths, left=plotDf.frcComplete_visual_5pct_boot, height=GTbarHeight, color=c5s[0], alpha=0.5, zorder=1, label='Bootstrap (95pc CI)')
        widths = plotDf.frcComplete_visual_95pct_boot-plotDf.frcComplete_visual_boot
        ax.barh(yVals, widths, left=plotDf.frcComplete_visual_boot, height=GTbarHeight, color=c5s[0], alpha=0.5, zorder=1)
        
        # plot GT data (note: these are the values scaled up from 2015 levels)
        ax.scatter(plotDf.frcComplete_visual, yVals+0.5*GTbarHeight, c=c5s[0], alpha=0.5, marker='o', zorder=3, label='Point estimate')

        # plot Stan range (note: these are the values scaled up from 2015 levels)
        widths = plotDf.frcComplete_MRP_outofsample-plotDf.frcComplete_MRP_outofsample_5pct
        ax.barh(yVals+barOffset, widths, left=plotDf.frcComplete_MRP_outofsample_5pct, height=stanBarHeight, color=c5s[1], alpha=0.9, zorder=2, label='Multilevel model (95pc CI)')
        widths = plotDf.frcComplete_MRP_outofsample_95pct-plotDf.frcComplete_MRP_outofsample
        ax.barh(yVals+barOffset, widths, left=plotDf.frcComplete_MRP_outofsample, height=stanBarHeight, color=c5s[1], alpha=0.9, zorder=2)
        
        # plot the parametric fits
        ax.scatter(plotDf.frcComplete_fit, yVals+0.5*GTbarHeight, marker='s', c=c5s[2], alpha=0.9, zorder=5, label='Country-level fit')
        ax.scatter(plotDf.frcComplete_lev1_fit, yVals+0.5*GTbarHeight, marker='v', c=c5s[2], alpha=0.9, zorder=4, label='Sub-country fit')
        ax.scatter(plotDf.frcComplete_quintiles_fit, yVals+0.5*GTbarHeight, marker='d', c=c5s[2], alpha=0.9, zorder=4, label='Density quintile fit')

        # plot country names to left of bars
        isos = plotDf.index.values
        xpos = plotDf[['frcComplete_MRP_outofsample_5pct','frcComplete_visual_5pct_boot','frcComplete_visual','frcComplete_fit','frcComplete_lev1_fit','frcComplete_quintiles_fit']].min(axis=1).fillna(0).values
        for iso, xx,yy in zip(isos, xpos, yVals):
            ax.text(xx-0.02,yy+0.5*GTbarHeight,shortNames[iso].replace('&','\&'),ha='right',va='center',size=9)
        
        #ax.set_yticklabels(plotDf.index.values)
        ax.set_yticklabels([])
        ax.set_xlim([-0.1,1.2])
        ax.set_ylim([-1,len(plotDf)+7])
        ax.set_xlabel('Fraction complete, '+xdateMAX.strftime("%B %Y"), size=16)
        
        # vertical lines for orientation
        ax.vlines(np.arange(0,1.1,0.2), ymin=-1, ymax=len(plotDf), alpha=0.2)
        
        # dummy plots to add to labels
        ax.plot(0,0,c='w',lw=0,label='Visual assessment')
        ax.plot(0,0,c='w',lw=0,label='Parametric model')
        legHandles, legLabels = ax.get_legend_handles_labels()
        legOrder = [0,2,6,7,1,3,4,5]   # not sure why scatter puts its labels before barh
        legHandles = [legHandles[ii] for ii in legOrder]
        legLabels  = [legLabels[ii]  for ii in legOrder]
        plt.legend(labels=legLabels,handles=legHandles,loc=9, ncol=2)
        figs.append(fig)

    outFigFn = paths['output'] + 'allCompletenessMeasures'    
    ht.compileAllPlots(figs,outFigFn,figSize=figSizePage)
       
def plotIRFvsOSM():
    """Plot for PLOS1. Our estimates of road length vs IRF"""
    df = chooseStanorFit()
    df['roads_OSM'] = df.length_best/1000.  # because IRF are in km
    fig, ax = plt.subplots(figsize=figSize)
    ax.scatter(df.roads_km.values, df.roads_OSM.values, s=10, c=c5s[0])
    ax.plot((0,10**7),(0,10**7), c=c5s[0],lw=0.5)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Roads reported by IRF, km') 
    ax.set_ylabel('Estimated roads from OSM data (km)')
    ax.axis('image')  # square-shaped plot
    ax.set_xlim([10,10**7])
    ax.set_ylim([10,10**7])
    fig.tight_layout()
    
    figFn = paths['output']+'fig-IRFvsOSM.pdf'
    fig.savefig(figFn)
    print('Wrote IRF vs OSM fig to %s' % figFn)

def country_summary_table(latex=None):
    """Exports summary table for use in the appendix"""
    df = chooseStanorFit().reset_index()
    df = df[pd.notnull(df.frcComplete_best)]
    df['length_pc'] = df.length_best *1.0 / df.pop2012
    df.length_best = df.length_best / 1000.
    
    colsToUse = ['countryname','pop2012','length_best', 'length_pc','frcComplete_best', 'frcComplete_fit', 'fitfunc_name', 
                'frcComplete_lev1_fit','frcComplete_quintiles_fit',
                'frcComplete_MRP_outofsample','frcComplete_MRP_outofsample_5pct', 'frcComplete_MRP_outofsample_95pct']
       
    ISO2cname = ht.country2ISOLookup()['ISOalpha2shortName'] 
    df['countryname']=df.ISOalpha3.map(ISO2cname)
    #df['countryname']=df.countryname.map(lambda ss:     "Ivory Coast"  if "d'Ivoire" in ss else {"United States":'USA','United Kingdom':'UK'}.get(ss,ss))
    df=df[colsToUse]
    df.sort_values(by='length_best',inplace=True,ascending=False)

    from cpblUtilities.mathgraph import round_to_n_sigfigs, format_to_n_sigfigs
    

    df.length_best=df.length_best/1e3 # Convert to Mm
    for LL in ['length_best','length_pc']:
        df[LL] =  df[LL].map(lambda ff: format_to_n_sigfigs(ff,2))
        
    for LL in [cc for cc in df.columns if cc.startswith('frc')]:
        df[LL]=df[LL].map(lambda ll: '' if pd.isnull(ll) else '%0.2f'%ll)
    df['pop2012']=df['pop2012'].map(lambda pp: '' if pd.isnull(pp) else
                                    str(int(round_to_n_sigfigs(pp/1e6,3)))+' M' if pp>=1e8
                                    else                                     str(int(round_to_n_sigfigs(pp/1e3,3)))+' k' )
    df['Best fit']=df.fitfunc_name.map(lambda ff: ff.replace('sigmoid','Logistic').replace('gompertz','Gompertz').replace('_with_1_jumps','(1J)').replace('_with_2_jumps','(2J)' ).replace('_with_3_jumps','(3J)').replace('_with_4_jumps','(4J)').replace('_with_1_ramp','(1R)'))

    
    df.drop(['pop2012','fitfunc_name'], axis=1, inplace=True)  # no longer needed

    # Export to html and latex
    for tabType in ['SI']:
        tableFn = paths['output']+'countries-table'
        df['frcComplete_MRP_outofsample_ptandrange']=  df.apply(lambda adf: '' if not adf['frcComplete_MRP_outofsample'] else '%s (%s--%s)'%( adf['frcComplete_MRP_outofsample'],adf['frcComplete_MRP_outofsample_5pct'],adf['frcComplete_MRP_outofsample_95pct']), axis=1)
        df['frcComplete_countryfit']=  df.apply(lambda adf: adf['frcComplete_fit']+' ('+adf['Best fit'].replace('stic(','stic+').replace(')','')+')'  , axis=1)

        newheaders=[[u'countryname','','','Country '],
           [ u'length_best','Length','','Total (103 km)'], # Make this an exponent later
           [ u'length_pc','Length','','Per capita (m)'],
           [ u'frcComplete_best','Fraction complete','Best', '   '],
           [ u'frcComplete_MRP_outofsample_ptandrange','Fraction complete','Multilevel model', ' '],
           [ u'frcComplete_countryfit','Fraction complete','Parametric fits','Country-level'],                    
           [ u'frcComplete_lev1_fit','Fraction complete','Parametric fits','From sub-geography'],
           [ u'frcComplete_quintiles_fit','Fraction complete','Parametric fits','From quintiles'],
           ]
        dfx=df[  [vv for vv,aa,bb,cc in newheaders] ]
        firstrow=[aa for vv,aa,bb,cc in newheaders]
        secondrow=[bb for vv,aa,bb,cc in newheaders]
        thirdrow=[cc for vv,aa,bb,cc in newheaders]
        dfx.columns=thirdrow

        firstrow=' & '.join(firstrow)
        secondrow=' & '.join(secondrow)
        thirdrow=' & '.join(thirdrow)
        print firstrow
        print secondrow
        firstrow=firstrow.replace( ' & Fraction complete'*5, r' & \multicolumn{5}{|c|}{Fraction complete}')
        firstrow=firstrow.replace( '& Length '*2, r'& \multicolumn{2}{|c|}{Length}')
        secondrow=secondrow.replace( ' & Parametric fits'*3, r' & \multicolumn{3}{|c|}{Parametric fits}')
        print firstrow
        print secondrow

        #formatters = [lambda x: '%s' %x]+[lambda x: '%.2f' %x]*5+[lambda x: '%.1f' %x]*10
        dfx.to_html(tableFn+'.html', index = False, na_rep='')
        dfx.to_latex(tableFn+'.tex', index = False,  na_rep='', longtable=True) #formatters = formatters,
        with  open(tableFn+'.tex','rt') as f:
            inTable = f.read()

        if texAvailable:
            inTable=inTable.replace('103 km','10$^3$~km').replace(r'\toprule',  r'\toprule'+'\n'+firstrow+r' \\ \cline{4-8} '+'\n'+secondrow+r' \\ \cline{6-8}'  )
        else:
            inTable=inTable.replace('103 km','thousand km').replace(r'\toprule',  r'\toprule'+'\n'+firstrow+r' \\ \cline{4-8} '+'\n'+secondrow+r' \\ \cline{6-8}'  )
        inTable=inTable.replace('llllllll','|l|p{15mm}p{15mm}|l|l|lp{15mm}p{\widthof{quintiles}}|')#,'|l|p{10mm}p{15mm}|l|llp{20mm}p{20mm}p{20mm}|')

        with  open(tableFn+'.tex','wt') as f:
            f.write(inTable)
    #latex.append(inTable)


    dfc=pd.read_pickle(paths['output']+'multilevel_GT_sq_coefficients.pandas')
    ctableFn = paths['output']+'STAN-coefs-table'

    for LL in ['posterior-mean',       u'5th-percentile', u'95th-percentile']:
        dfc[LL] =  dfc[LL].map(lambda ff: format_to_n_sigfigs(ff,3))

    dfc.to_latex(ctableFn+'.tex', index = False,  na_rep='', longtable=False)

def exportDataFile():
    """Produces the final compiled country-level dataset which is posted in the data archive
    Clean up some fields and saves as pandas and tsv"""
    adf = chooseStanorFit()
    adf['length_pc'] = adf.length_best *1.0 / adf.pop2012
    adf['fitfunction']=adf.fitfunc_name.map(lambda ff: ff.replace('sigmoid','Logistic').replace('gompertz','Gompertz').replace('_with_1_jumps','(1J)').replace('_with_2_jumps','(2J)' ).replace('_with_3_jumps','(3J)').replace('_with_4_jumps','(4J)').replace('_with_1_ramp','(1R)'))
    adf['method']=adf.measure_best.apply(lambda x: 'visual' if x=='stan' else 'parametric' if x=='fit' else np.nan)

    colsToDrop = ['frcComplete_Feb2015_MRP_insample', 'frcComplete_Feb2015_MRP_insample_5pct', 'frcComplete_Feb2015_MRP_insample_95pct', 
                  'totSegs_MRP_outofsample', 'totSegs_MRP_outofsample_5pct', 'totSegs_MRP_outofsample_95pct', 'osmSegs_MRP_outofsample', 'osmSegs_MRP_outofsample_5pct', 'osmSegs_MRP_outofsample_95pct', 
                  'frcComplete_Feb2015_fit', 'maxLength_Feb2015', 'length_MRP_insample', 'length_MRP_outofsample', 
                  'fitfunc_pref', 'asymptote', u'finishedDate', u'finishedYear', u'frcComplete_MRP_insample', 
                  'frcComplete_MRP_insample_95pct', 'frcComplete_MRP_insample_5pct', 'frcComplete_model', 'length_model', 'completeAgree', 
                  'measure_best', 'fitfunc_name', 'frcComplete_quintiles_model', 'frcComplete_lev1_model', 'g']
    adf.drop(colsToDrop,axis=1,inplace=True)


    # rename GT to visual and MRP_outofsample to MRP
    renameCols = dict([(cc, cc.replace('GT','visual').replace('MRP_outofsample','MRP')) for cc in adf.columns if 'GT' in cc or 'outofsample' in cc])
    adf.rename(columns=renameCols, inplace=True)
    adf.rename(columns={'maxLength_allvalues':'max_length', 'maxLength':'OSMlength'}, inplace=True)

    # convert lengths to km
    for col in ['OSMlength','max_length','length_fit','length_best']:
        adf[col] = adf[col]/1000.

    colOrder = ['frcComplete_Feb2015_visual','frcComplete_Feb2015_visual_boot','frcComplete_Feb2015_visual_se_boot','frcComplete_Feb2015_visual_5pct_boot',
    'frcComplete_Feb2015_visual_95pct_boot','frcComplete_Feb2015_MRP','frcComplete_Feb2015_MRP_5pct','frcComplete_Feb2015_MRP_95pct',
    'frcComplete_visual','frcComplete_visual_boot','frcComplete_visual_se_boot','frcComplete_visual_5pct_boot','frcComplete_visual_95pct_boot',
    'frcComplete_MRP','frcComplete_MRP_5pct','frcComplete_MRP_95pct',
    'OSMlength','max_length','fitfunction','MSE','length_fit','frcComplete_fit','frcComplete_lev1_fit','frcComplete_quintiles_fit',
    'method','frcComplete_best','length_best','length_pc',
    'GovernanceRegulatory2013','GovernanceCorruption2013','GovernanceEffectiveness2013',
    'GovernanceRuleOfLaw2013','GovernanceVoice2013','GovernanceStability2013',
    'countryname','WBcode','gdpCapitaPPP_WB','gdp2005USD_WB','gdpCapita2005USD_WB','internetUsersper100_2013',
    'landSqKm','literacyPcAdult','mobilePhonesPer100','pop2012','popDensity_sqkm2012','urbanPop2012',
    'roads_datayear','roads_km','roads_pc_paved','roads_excludes_urban_or_local',]

    adf = adf[colOrder]
    adf.to_csv(paths['output']+'countries_compiled.tsv', sep='\t')
    adf.to_pickle(paths['output']+'countries_compiled.pandas')
  
if __name__ == '__main__':
    runmode=None if len(sys.argv)<2 else sys.argv[1].lower()
    forceUpdate = any(['forceupdate' in arg.lower() for arg in sys.argv])
    print 'forceUpdates is %s' % forceUpdate 

    if runmode in [None,'plots']:
        from fits import plotFits, plotFitsDensity
        plotFits('best')
        plotFitsDensity()
        plotCompletenessMeasures()
        figMapCompleteness()
        plotIRFvsOSM()
    if runmode in [None,"tables"]:
        country_summary_table()
    if runmode in [None,'outputfile']:
        exportDataFile()


