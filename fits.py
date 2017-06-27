#!/usr/bin/python
# coding=utf-8

""" 
Fits sigmoid and linear functions to osm_history
"""

import os, sys, datetime
from history_config import *
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use(mpl_backend)
import matplotlib.pyplot as plt
import history_tools as ht

from cpblUtilities.mathgraph import transbg, figureFontSetup, transLegend
from cpblUtilities.utilities import shelfLoad, shelfSave, mergePDFs
from cpblUtilities.parallel import runFunctionsInParallel

if texAvailable: 
    figureFontSetup(uniform=11)
else:
    plt.rcParams.update({'font.size': 11, 'ytick.labelsize': 11, 'xtick.labelsize': 11,'legend.fontsize': 11, 'axes.labelsize': 11, 'figure.figsize': (4.6, 4)})

def loadAggregatedHistory(gadmLevel=0, roadsOnly=True, density=False):
    """
    Loads the aggregated OSM history (by country, sub-national unit and density)
    The required files are provided as part of the SI, or can be recreated using the functions in process_raw.py
    Returns a pandas dataframe
    
    gadmLevel can be 0 (country level), -1 (whole world), or 1 (sub-national units)
    roadsOnly restricts the dataframe to OSM ways that are roads (not pedestrian paths, etc.). See the manuscript for a description of the tags included as "roads."
    density=True also disaggregates by density decile
    
    """
    df = pd.read_hdf(paths['working']+'osmHistory_level'+str(gadmLevel)+'D'+'_density'*density+'.hd5', 'history')
    if gadmLevel==0: 
        dfWorld = pd.read_hdf(paths['working']+'osmHistory_level-1'+'D'+'_density'*density+'.hd5', 'history')
        dfWorld['ISOalpha3'] = 'ALL'
        df = pd.concat([df, dfWorld])
    if roadsOnly: df = df[df.roadFlag==True]
    if 'ISOalpha3' in df.columns: 
        df.loc[(df.ISOalpha3=='KSV'),'ISOalpha3']='XKO' # kludge because Kosovo has non-standard ISO code. The raw data still has KSV, but since updated
        df.loc[(df.ISOalpha3=='SP-'),'ISOalpha3']='XSP' # another kludge
    assert df.index.name=='date'
    df['date']=df.index
    
    # Rescale to help with optimization
    df['xdata'] = date2xdata(df.date)
    df['ydata'] = df.length/yscaleFactor  
    assert df['xdata'].mean()<10000 and df['xdata'].mean()>.1
        
    return(df)

def fit_region(df,fitfunc,hopping=True,METHOD='Nelder-Mead', DEBUG=False,regionname=None):
    """
    Fits a sigmoid (or other fit function defined in this module) curve for one region (normally a country or a sub-national unit)
    df is a dataframe with date and length fields, and an ISOalpha3 column (which should be all the same)
    
    Returns and saves the array of estimated parameters
    """
    from scipy.optimize import basinhopping,minimize  #curve_fit
    
    xdata=df[(pd.notnull(df.xdata)) & (pd.notnull(df.ydata))].xdata
    ydata=df[(pd.notnull(df.xdata)) & (pd.notnull(df.ydata))].ydata
    if regionname is None:
        regionname=df.ISOalpha3.values[0]
    suffix='-'.join([regionname, fitfunc.__name__, METHOD.replace('-',''), 'hopping'*hopping])

    initParams= globals()[fitfunc.__name__].initparams(xdata,ydata)
    bounds=None

    minimizer_kwargs = dict(method=METHOD, options=dict(maxfev=200000,maxiter=20000))
    if hopping:
        temperature=(ydata.max()-ydata.min())*1.0/100 # See docs for basin hopping
        estParams = basinhopping(lambda pp, func=fitfunc, xd=xdata,yd=ydata: func_mean_square_error(xd,yd,func,pp), initParams,  minimizer_kwargs=minimizer_kwargs, 
        T=temperature, niter=500)
    else:
        estParams =     minimize(lambda pp, func=fitfunc, xd=xdata,yd=ydata: func_mean_square_error(xd,yd,func,pp), initParams,bounds=bounds,method=METHOD, options=dict(maxfev=20000,maxiter=20000))#callback=callbackPlot) #None)#"TNC")
    
    MSE=func_mean_square_error(xdata,ydata,fitfunc,estParams.x)

    # Save results
    from cpblUtilities.utilities import shelfSave
    success=False
    if 'success' in estParams and   estParams['success']:
        success=True
    if 'minimization_failures' in  estParams and   'successfully' in estParams.message[0]: 
        success=True
    if success:
        returnVal=dict(estparams=estParams.x,asymptote=fitfunc.asymptote(xdata,ydata,*(estParams.x)),
                       finishedDate=xdata2date(  fitfunc.whenCompleted(xdata,ydata,*(estParams.x))  ),
                       MSE=MSE,AIC=None,success=success,fullreport=estParams,ISOalpha3=regionname,hopping=hopping,fitfunc=fitfunc.__name__,method=METHOD)
        shelfSave(paths['scratch']+'completenessFits/onefit-'+suffix,returnVal)

        return None
    else:
        print('FAILED TO CONVERGE for '+suffix)
        return None

def fitallCountries(gadmLevel=0, forceUpdate=False, parallel=True):
    """
    Fit a various curves to each country
    Plot the results
    Save the fit parameters, AIC and other information to a countries dataframe
    """
    
    outFn = paths['working'] + 'countryFits%d.pandas'%gadmLevel  # data frame with fits and other country info
    if os.path.exists(outFn) and not forceUpdate:
        return pd.read_pickle(outFn)
        
    df=loadAggregatedHistory(gadmLevel=gadmLevel)
    assert date2xdata(df.date.max()) == date2xdata(xdateMAX) 
    assert gadmLevel in [0,1] # not yet implemented for deeper gadm levels, but this is trivial
    lastDate = df['date'].max()-pd.DateOffset(days=14)  # so we can smooth the last two weeks, in case there is a jump right at the end
    
    if gadmLevel==0:  # country-level
        countries = pd.DataFrame(df.groupby(['ISOalpha3']).length.max())
        countries.rename(columns={'length':'maxLength_allvalues'}, inplace=True)
        countries['maxLength'] = df[df['date']>lastDate].groupby(['ISOalpha3']).length.mean().astype(int)
        countries['maxLength'] = countries.maxLength.combine_first(countries.maxLength_allvalues)  # for countries with no data in the last week
    else:
        gadmList = ['id'+str(ii) for ii in range(gadmLevel+1)]
        countries = pd.DataFrame(df.groupby(gadmList).length.max())
        countries.rename(columns={'length':'maxLength_allvalues'}, inplace=True)
        countries['maxLength'] = df[df['date']>lastDate].groupby(gadmList).length.mean().astype(int)

    countries = countries.sort_values(by='maxLength', ascending=False)
    countryISOs = [cc for cc in countries.index.values]  # not really countryISOs, but named for consistency with above
    
    # Loop over various fit functions, with countries in parallel
    DEBUG=True # This is fine for production; it's producing a plot for each country/region
    listoffuncs, listofnames = [], []
    for HOPPING in [False]:#, True]:
        for fitfunc in [sigmoid_with_4_jumps, sigmoid_with_1_jumps,sigmoid_with_3_jumps,sigmoid_with_2_jumps, sigmoid_with_1_ramp, justaline, sigmoid,gompertz, ]:
            for METHOD in ['Nelder-Mead']:#'L-BFGS-B','TNC','COBYLA',,'Powell']:
                print 'Now adding calls for fitting %s in parallel (%s), hopping is %s; method is %s' % (fitfunc, str(parallel),HOPPING,METHOD)
                if gadmLevel==0:
                    for country in countryISOs:
                        suffix='-'.join([country,       fitfunc.__name__,    METHOD.replace('-',''),  'hopping'*HOPPING])
                        if os.path.exists(paths['scratch'] + 'completenessFits/onefit-'+suffix+'.pyshelf') and not(forceUpdate): 
                            print '\tAlready done %s (%s)' % (country ,suffix)
                            continue
                        listoffuncs+=[[fit_region,[df[df.ISOalpha3==country], fitfunc, HOPPING, METHOD, DEBUG]]]
                        listofnames+=[suffix]     
                else: # This is 1000+ times faster than looping and selecting
                    for ids,adf in df.groupby(['id0','id1']):
                        sids='_'.join([str(int(cc)) for cc in ids])
                        cname = 'gadmLev'+str(gadmLevel)+'_'+sids
                        suffix='-'.join([cname,       fitfunc.__name__,    METHOD.replace('-',''),  'hopping'*HOPPING])
                        if os.path.exists(paths['scratch'] + 'completenessFits/onefit-'+suffix+'.pyshelf') and not(forceUpdate): 
                            print '\tAlready done %s (%s)' % (sids ,suffix)
                            continue
                        if len(adf)<10:
                            print('Region %s has only %d timepoints. SKIPPING IT!'%(cname,len(adf)))
                            continue
                        listoffuncs+=[[fit_region,[adf, fitfunc, HOPPING, METHOD, DEBUG,cname]]]
                        listofnames+=[suffix]     

    print('Launching %d estimates...'%len(listoffuncs))
    runFunctionsInParallel(listoffuncs,names=listofnames, parallel=parallel, expectNonzeroExit=True, maxFilesAtOnce=400)

    print('\n\nCompleted all fitting.\n\n  Now aggregating fit results..')
    # Collect all the results and add to the country dataframe
    if gadmLevel==0:
        fitFns = [ff for ff in os.listdir(paths['scratch']+'completenessFits/') if ff.startswith('onefit') and 'gadmLev' not in ff and '-WBreg-' not in ff and 'quintile' not in ff ]# and 'NelderMead' in ff]
    else:
        fitFns = [ff for ff in os.listdir(paths['scratch']+'completenessFits/') if ff.startswith('onefit') and 'gadmLev'+str(gadmLevel) in ff]# and 'NelderMead' in ff]
        
    cfitdict={}
    for fitFn in fitFns:
        countryFit = shelfLoad(paths['scratch']+'completenessFits/'+fitFn)
        if 'success' not in countryFit or countryFit['success'] == False: continue
        assert fitFn.endswith('.pyshelf')
        onefit, ISOcode,funcname, method, hopping = fitFn[:-8].split('-')
        if ISOcode not in cfitdict: cfitdict[ISOcode]=[]
        newpref='-'.join([funcname, method, hopping ])

        cfitdict[ISOcode]+=[[newpref+'_estparams',countryFit['estparams'],],
                             [newpref+'_MSE',countryFit['MSE'],],
                             [newpref+'_asymptote',countryFit['asymptote'],],
                             [newpref+'_finishedDate',countryFit['finishedDate'],],
                            ]+(gadmLevel==0)*[  ['ISOalpha3',ISOcode]
                            ]+(gadmLevel>0)*[  ['countryname','_'.join(ISOcode.split('_')[1:])  ],
                                               ]
    if gadmLevel>0: # Shouldn't we include gadm0 for country case, and ISO in gadm case?
        id0ToISO = dict([(v['id_0'], iso) for iso, v in ht.get_all_GADM_countries().iteritems()])  
        if id0ToISO[216]=='SP-': id0ToISO[216] = 'XSP'  # kludge
        id0ToISO[44] = 'XCA' # kludge, Caspian Sea, subsequently dropped from GADM
        for kk in cfitdict:
            gadms=kk.split('_')[1:]
            cfitdict[kk]+=zip(gadmList,map(int,gadms))
            cfitdict[kk]+=[('ISOalpha3',id0ToISO[int(gadms[0])])]
            
    allfitsdf=pd.DataFrame(  [ dict(vv) for vv in cfitdict.values() ]).set_index('ISOalpha3' if gadmLevel==0 else gadmList)
    countriesWithFits=countries.merge(allfitsdf,  left_index=True,right_index=True)
    
    countriesWithFits.to_pickle(outFn)
    print('Saved aggregate results to %s' % outFn)
    
    return countriesWithFits

def fitallRegionsByDensity(forceUpdate=False, parallel=True):
    """
    Fit a various curves by World Bank region and density bin
    Plot the results
    
    Save the fit parameters, AIC and other information to a densities dataframe
    
    Similar to fitallCountries, but heavily simplified
    """
    outFn = paths['working'] + 'densityRegionFits0.pandas' 
    if os.path.exists(outFn) and not forceUpdate:
        return pd.read_pickle(outFn)
        
    df=loadAggregatedHistory(gadmLevel=0, density=True)
    assert date2xdata(df.date.max()) == date2xdata(xdateMAX) 

    # We have to do this rather than groupby, because WB regions are non-unique
    regionDfs = aggregateByWBregion(df) 

    # Loop over various fit functions, with countries in parallel
    listoffuncs, listofnames = [], []
    METHOD, HOPPING, DEBUG = 'Nelder-Mead', False, True
    for fitfunc in [sigmoid_with_4_jumps, sigmoid_with_1_jumps,sigmoid_with_3_jumps,sigmoid_with_2_jumps, sigmoid_with_1_ramp, justaline, sigmoid,gompertz, ]:
        print 'Now adding calls for fitting %s in parallel (%s), hopping is %s; method is %s' % (fitfunc, str(parallel),HOPPING,METHOD)
        for wbr in regionDfs:
            for dcl in range(0,10)+[99]:
                regionname = 'WBreg-'+wbr+'-dc'+str(dcl)
                suffix='-'.join([regionname, fitfunc.__name__, METHOD.replace('-',''), 'hopping'*HOPPING])
                if os.path.exists(paths['scratch'] + 'completenessFits/onefit-'+suffix+'.pyshelf') and not(forceUpdate): 
                    print '\tAlready done %s, %s (%s)' % (wbr, dcl ,suffix)
                    continue
                listoffuncs+=[[fit_region,[regionDfs[wbr][regionDfs[wbr].densDecile==dcl], fitfunc, HOPPING, METHOD, DEBUG,regionname]]]
                listofnames+=[suffix]     
   
    print('Launching %d estimates...'%len(listoffuncs))
    runFunctionsInParallel(listoffuncs,names=listofnames, parallel=parallel, expectNonzeroExit=True, maxFilesAtOnce=400)

    print('\n\nCompleted all fitting.\n\n  Now aggregating fit results..')
    # Collect all the results and add to the country dataframe
    fitFns = [ff for ff in os.listdir(paths['scratch']+'completenessFits/') if ff.startswith('onefit-WBreg-') and 'gadmLev' not in ff]# and 'NelderMead' in ff]
    cfitdict={}

    # calculate maximum length, so we can calculate frcComplete later on. This will be used in the aggregation step
    maxLengths = {}
    for wbr in regionDfs:
        for decile in range(0,10)+[99]:
            dfx = regionDfs[wbr]  # for convenience, avoid typing
            lastDate = dfx[dfx.densDecile==decile].date.max()-pd.DateOffset(days=14)  # so we can smooth the last two weeks, in case there is a jump right at the end
            maxLengths[wbr+'dc'+str(decile)] = dfx[(dfx['date']>lastDate) & (dfx.densDecile==decile)].length.mean().astype(int)

    for fitFn in fitFns:
        countryFit = shelfLoad(paths['scratch']+'completenessFits/'+fitFn)
        if 'success' not in countryFit or countryFit['success'] == False: continue
        assert fitFn.endswith('.pyshelf') and '--' not in fitFn

        onefit, WBreg, wbr, decile, funcname, method, hopping = fitFn[:-8].split('-')
        if wbr+decile not in cfitdict: cfitdict[wbr+decile]=[]
        newpref='-'.join([funcname, method, hopping ])

        cfitdict[wbr+decile]+=[[newpref+'_estparams',countryFit['estparams'],],
                             [newpref+'_MSE',countryFit['MSE'],],
                             [newpref+'_asymptote',countryFit['asymptote'],],
                             [newpref+'_finishedDate',countryFit['finishedDate'],],
                             ['maxLength', maxLengths[wbr+decile]],
                             ['wbr', wbr], ['decile', int(decile[2:])]]
                            
    allfitsdf=pd.DataFrame(  [ dict(vv) for vv in cfitdict.values() ]).set_index('wbr')
    
    allfitsdf.to_pickle(outFn)
    print('Saved aggregate results to %s' % outFn)
    
    return allfitsdf

def fitallDensityQuintiles(forceUpdate=False, parallel=True):
    """
    Fit a various curves for each country and density quintile
    Plot the results
    Save the fit parameters, AIC and other information to a densities dataframe
    
    Similar to fitallCountries, but heavily simplified
    """
    outFn = paths['working'] + 'countryFits0_density.pandas' 
    if os.path.exists(outFn) and not forceUpdate:
        return pd.read_pickle(outFn)
        
    df=loadAggregatedHistory(gadmLevel=0, density=True)
    assert date2xdata(df.date.max()) == date2xdata(xdateMAX) 

    df = densityDeciles2Quintiles(df).set_index(['ISOalpha3','densQuintile'])
    df.sort_index(inplace=True)  # helps performance with slicing
    countryISOs = df.index.get_level_values(0).unique()
    
    # Loop over various fit functions, with countries in parallel
    listoffuncs, listofnames = [], []
    METHOD, HOPPING, DEBUG = 'Nelder-Mead', False, True
    for fitfunc in [sigmoid_with_4_jumps, sigmoid_with_1_jumps,sigmoid_with_3_jumps,sigmoid_with_2_jumps, sigmoid_with_1_ramp, justaline, sigmoid,gompertz, ]:
        print 'Now adding calls for fitting %s in parallel (%s), hopping is %s; method is %s' % (fitfunc, str(parallel),HOPPING,METHOD)
        for iso in countryISOs:
            for quintile in range(0,5):
                cname = iso+'-quintile'+str(quintile)
                suffix='-'.join([cname, fitfunc.__name__, METHOD.replace('-',''), 'hopping'*HOPPING])
                if os.path.exists(paths['scratch'] + 'completenessFits/onefit-'+suffix+'.pyshelf') and not(forceUpdate): 
                    print '\tAlready done %s, %s (%s)' % (iso, quintile ,suffix)
                    continue
                if (iso,quintile) not in df.index: 
                    print 'No data for %s quintile %d' % (iso, quintile)
                    continue
                listoffuncs+=[[fit_region,[df.ix[(iso,quintile)].reset_index().set_index('date'), fitfunc, HOPPING, METHOD, DEBUG, cname]]]
                listofnames+=[suffix]     
   
    print('Launching %d estimates...'%len(listoffuncs))
    runFunctionsInParallel(listoffuncs,names=listofnames, parallel=parallel, expectNonzeroExit=True, maxFilesAtOnce=400)

    print('\n\nCompleted all fitting.\n\n  Now aggregating fit results..')
    # Collect all the results and add to the country dataframe
    fitFns = [ff for ff in os.listdir(paths['scratch']+'completenessFits/') if 'quintile' in ff]# and 'NelderMead' in ff]
    cfitdict={}

    # calculate maximum length, so we can calculate frcComplete later on. This will be used in the aggregation step
    maxLengths = {}
    for iso in countryISOs:
        for quintile in range(0,5):
            if (iso,quintile) not in df.index: continue
            lastDate = df.ix[(iso, quintile)].date.max()-pd.DateOffset(days=14)  # so we can smooth the last two weeks, in case there is a jump right at the end
            maxLengths[iso+'quintile'+str(quintile)] = df.ix[(iso, quintile)][df.ix[(iso, quintile),'date']>lastDate].length.mean().astype(int)

    for fitFn in fitFns:
        countryFit = shelfLoad(paths['scratch']+'completenessFits/'+fitFn)
        if 'success' not in countryFit or countryFit['success'] == False: continue
        assert fitFn.endswith('.pyshelf') and '--' not in fitFn

        onefit, iso, quintile, funcname, method, hopping = fitFn[:-8].split('-')
        if iso+quintile not in cfitdict: cfitdict[iso+quintile]=[]
        newpref='-'.join([funcname, method, hopping ])

        cfitdict[iso+quintile]+=[[newpref+'_estparams',countryFit['estparams'],],
                             [newpref+'_MSE',countryFit['MSE'],],
                             [newpref+'_asymptote',countryFit['asymptote'],],
                             [newpref+'_finishedDate',countryFit['finishedDate'],],
                             ['maxLength', maxLengths[iso+quintile]],
                             ['ISOalpha3', iso], ['densQuintile', int(quintile.replace('quintile',''))]]
                            
    allfitsdf=pd.DataFrame(  [ dict(vv) for vv in cfitdict.values() ]).set_index(['ISOalpha3','densQuintile'])
    
    allfitsdf.to_pickle(outFn)
    print('Saved aggregate results to %s' % outFn)
    
    return allfitsdf

def densityDeciles2Quintiles(adf):
    from math import floor
    adf = adf[adf.densDecile>=0]  # drop -1 quintile, which is for np.nan
    adf['densQuintile'] = (adf.densDecile/2.).apply(floor)
    # collapse from deciles to quintiles
    adf = adf.groupby(['densQuintile','roadFlag','xdata','date','ISOalpha3'])[['length','ydata']].sum()
    return adf.reset_index()

def chooseBestFit(gadmLevel=0, excludeJustaline=False):
    """
    Picks the best functional form for a given gadmLevel.
    Returns the dataframe (super fast, so no need to save it)
    Pass gadmLevel = 'wbr' to choose the best fit from the density deciles by WB region
      or gadmLevel = 'quintiles' to choose the best fit from the density quintiles by country
    """
    if gadmLevel=='wbr':
        countries = fitallRegionsByDensity()
        assert countries.index.name=='wbr'
    elif gadmLevel=='quintiles':
        countries = fitallDensityQuintiles()
        assert countries.index.names==['ISOalpha3', 'densQuintile']
    else:
        assert isinstance(gadmLevel,int)
        countries = fitallCountries(gadmLevel=gadmLevel)
        assert countries.index.name=='ISOalpha3' or all([ii.startswith('id') for ii in countries.index.names])

    MSEcols = [cc for cc in countries.columns if cc.endswith('_MSE')]  # and 'hopping' not in cc]
    if excludeJustaline: MSEcols = [cc for cc in MSEcols if not('justaline' in cc)]
    countries['fitfunc_pref'] = countries[MSEcols].T.idxmin().str[:-4]  
    toUse = pd.notnull(countries['fitfunc_pref'])
    countries['fitfunc_name'] = countries[toUse].fitfunc_pref.apply(lambda x: x.split('-')[0])
    countries['asymptote']=countries[toUse].apply(lambda adf: adf[adf['fitfunc_pref']+'_asymptote'], axis=1)
    countries['MSE']=countries[toUse].apply(lambda adf: adf[adf['fitfunc_pref']+'_MSE'], axis=1)
    countries['finishedDate']=countries[toUse].apply(lambda adf: adf[adf['fitfunc_pref']+'_finishedDate'], axis=1)
    # Crude approximation to decimal year (but more than good enough for us):
    countries['finishedYear']=countries.finishedDate.map(lambda d: np.nan if pd.isnull(d) else  (float(d.strftime("%j"))-1) / 366 + float(d.strftime("%Y")) )
    countries['length_fit'] = countries.asymptote*yscaleFactor # Same as asymptote, but scaled to km
    countries['fitfunc']=countries[toUse].fitfunc_name.map(lambda ss: globals()[ss])
    
    # Warn where maxLength (i.e., from data) is greater than asymptote
    if (countries.length_fit<countries.asymptote).sum()>0:
        print 'Warning. Following countries have a maxLength (i.e., from data) greater than asymptote'
        print (countries.length_fit-countries.asymptote)[countries.length_fit<countries.asymptote]
    
    # To determine the fraction complete in the fit, we appeal to the fit itself
    def sigmoidCompleteness(adf):
        pp=adf[adf['fitfunc_pref']+'_estparams']
        # returns predicted y value (i) now and (ii) a long time in the future (i.e., the approximated asymptote)
        lastY,infY=adf['fitfunc'].f(np.array([date2xdata(xdateMAX), date2xdata(xdateMAX)*10]), *pp) # Don't need xdata, ydata for this calculation
        return(lastY/infY)
        
    # Without reference to data, frcComplete_model tells us how far along the sigmoid 2015 is (will be nonsense for linear /justaline fits)
    countries['frcComplete_model'] = countries[toUse].apply(sigmoidCompleteness, axis=1)
    countries['length_model'] = countries[toUse].maxLength / countries.frcComplete_model

    # The following (_latestdatafit) tells us where the latest data (OSM length)  is, compared with the 2015/final fit (modeled) value
    countries['frcComplete_fit'] =     countries[toUse].maxLength/countries.length_fit # Should be same as ydata[-1]/asymptote. 

    colsToUse = ['ISOalpha3','decile','maxLength_allvalues','maxLength','fitfunc_pref','fitfunc_name','asymptote','MSE','finishedDate','finishedYear','length_fit','length_model','frcComplete_model','frcComplete_fit']
    colsToUse = [cc for cc in colsToUse if cc in countries.columns]  # some columns won't be in all version (e.g. wb density, different gadmLevels)
    return countries[colsToUse]

def aggregateByWBregion(df):
    """
    Takes the country-level dataframe and returns a dictionary of dataframes
    Each is aggregated to a WB region, with one row per density decile
    """
    # get country ISOs in each continent
    WBregions = ht.country2WBregionLookup()    
    WBregionDict = WBregions.GroupName.drop_duplicates().to_dict()
     
    aggDfs = {}
    for wbr in WBregionDict:
        print('Aggregating World Bank region %s' % WBregionDict[wbr])
        if wbr=='WLD': # whole world
            dfTmp = df
        else:
            dfTmp = df[df.ISOalpha3.isin(WBregions.ISOalpha3.ix[wbr])]
        dfTmp    = dfTmp.groupby(['date','xdata','roadFlag','densDecile'])[['freq','length','ydata']].sum().reset_index()
        dfTmp['WBregion'] = wbr
        dfTmpAll = dfTmp.groupby(['date','xdata','roadFlag'])[['freq','length','ydata']].sum().reset_index()
        dfTmpAll['densDecile'] = 99 # for all deciles, collapsed
        dfTmpAll['WBregion'] = wbr
        
        aggDfs[wbr] = pd.concat([dfTmp, dfTmpAll])
    
    return aggDfs

def plotFits(fitFunc2Plot=None,gadmLevel=0):
    """
    Plots country-specific fits for a given fit function and hopping, and saves as a PDF
    Also plots the best fit for each country
        
    Assumes that the countryFits0 dataframe has been already saved with the parameters 
    (this should be created in fitallCountries)
    
    If fitfunc is None and HOPPING is None, we cycle through all the fits and call this function recursively    
    """
    
    if fitFunc2Plot is None:         # call recursively
        countries = pd.read_pickle(paths['working'] + 'countryFits%d.pandas'%gadmLevel)
        fitList = [cc for cc in countries.columns if cc.endswith('estparams')]
        assert fitList
        for fitName in fitList:
            print "Now calling plotFits recursively for "+fitName[:-10]
            plotFits(fitName[:-10])
        plotFits('best')  # plot function with lowest MSE
        return

    if fitFunc2Plot=='best':
        fitfunc,suffix='best','-best'
    else:
        assert '-' in fitFunc2Plot
        fitF,method,hopping = fitFunc2Plot.split('-')
        fitfunc=globals()[fitF]
        suffix='-'+fitFunc2Plot
    print(' File suffix is '+suffix)
    ISO2cname = ht.country2ISOLookup()['ISOalpha2shortName']
    ISO2cname['ALL'] = 'World'

    df=loadAggregatedHistory(gadmLevel=gadmLevel).set_index('ISOalpha3')  
    df['yPred'] = np.nan
    dfDensity=loadAggregatedHistory(gadmLevel=gadmLevel,density=True)
    dfDensity = densityDeciles2Quintiles(dfDensity).set_index('ISOalpha3')   
    
    # load countries dataframe with parameter estimates (from parametric models)
    countries = pd.read_pickle(paths['working'] + 'countryFits%d.pandas'%gadmLevel)
    countries.sort_values(by='maxLength', ascending=False, inplace=True)
    countryISOs = [cc for cc in countries.index.values if cc in df.index and cc in ISO2cname and cc in dfDensity.index] 
    assert countryISOs
    
    # load best fits, Stan fits and other compiled data
    from analysis import compileCountries
    bestFits=compileCountries()

    # set up plotting
    plt.close('all')
    outList = []  # list of outPDF files
    xlims = (datetime.date(2007,1,1), xdateMAX)
    xticks = [datetime.date(2009,1,1), datetime.date(2012,1,1), datetime.date(2015,1,1)]
    xticksMinor = [datetime.date(yy,1,1) for yy in range(2008,2016)]
    mpl.rcParams['xtick.labelsize'] = 9
    mpl.rcParams['ytick.labelsize'] = 9
    page = -1    
    
    # make the predictions and plot    
    for cc, country in enumerate(countryISOs):    
        if cc%23==0: # new page
            page+=1
            fig, axes = plt.subplots(6, 4, figsize=figSizePage)
        row = int((cc-23*page)/4)
        col = cc-23*page-4*row  # for plotting

        mask = df.index.values==country  # avoids this problem: http://stackoverflow.com/questions/23227171/assignment-to-containers-in-pandas
        if mask.sum()<2: continue # at least one country only has one entry
        if fitfunc=='best':             # choose the function with lowest MSE for that country
            """ We may want another version here, ie other than "best", for which there is a penalty for having more jumps or etc. (but without using AIC)"""
            if pd.isnull(bestFits.loc[country, 'fitfunc_name']): continue
            bfuncname,fitFunc2Plot = bestFits.loc[country, 'fitfunc_name'],        bestFits.loc[country, 'fitfunc_pref']
            df.loc[mask, 'yPred'] = globals()[bfuncname].f(df.loc[mask].set_index('date').xdata,*(countries.loc[country, fitFunc2Plot+'_estparams'])).values
        else:
            try:
                df.loc[mask, 'yPred'] = fitfunc.f(df.loc[mask].set_index('date').xdata,*(countries.loc[country, fitFunc2Plot+'_estparams'])).values
            except (TypeError,IndexError): #ValueError:
                print('   TypeError or IndexError from fitparams: The fit has failed for this one: '+country+' : '+str(fitfunc))
                axes[row, col].set_title(ISO2cname[country].replace('&',' and '), fontsize=11)     
                continue

        if isinstance(df.ix[country], pd.core.series.Series):
            print('  This country ...')
            notsurewhythisisneeded
            plotDf     = pd.DataFrame(df.ix[country]).T.set_index('date')
            plotDfDens = pd.DataFrame(dfDensity.ix[country]).T.set_index('date')
        else:
            plotDf     = df.ix[country].set_index('date')
            plotDfDens = pd.DataFrame(dfDensity.ix[country]).set_index('date')
            
        # only plot the first day of each month for the daily data (unless this would result in fewer than 50 points...this is a workaround for small countries)
        datemask =     plotDf.index.day==1 # | (plotDf.index.month==3) & (plotDf.index.year==2015)
        datemaskDens = plotDfDens.index.day==1 # | (plotDf.index.month==3) & (plotDf.index.year==2015)
        densRatios =   [plotDf.ydata.max() / plotDfDens[plotDfDens.densQuintile==qq].ydata.max() for qq in range(0,5)]  # normalize ydata in each quintile, so the maximum is the same
        
        if datemask.sum()<50: 
            # plot everything, and we probably shouldn't plot the density
            plotDf.ydata.plot(ax=axes[row, col], xlim=xlims, style='.', ms=1, color='k', label='Actual')
        else:
            plotDf[datemask].ydata.plot(ax=axes[row, col], xlim=xlims, style='.', ms=2, color='k', label='Actual')
            for quintile in range(0,5): 
                try:
                    (plotDfDens[(datemaskDens) & (plotDfDens.densQuintile==quintile)].ydata*densRatios[quintile]).plot(ax=axes[row, col], xlim=xlims, style='.', ms=1, color='k', alpha=0.2, label='Density quintiles')
                except: # usually because there are no rows in this country
                    print('Failed plotting density quintile %d for country %s' % (quintile, country))
        plotDf.yPred.plot(ax=axes[row, col], xlim=xlims, color=c5s[2], label='Predicted')
        
        # plot ymax groundtruth. Use Stan results if we have them, otherwise bootstrap. Use color of patch (for now) to indicate the method
        GTmethod = 'stan' if pd.notnull(bestFits.loc[country, 'frcComplete_MRP_outofsample']) else 'none' 
        if GTmethod=='none':
            GT, GT5, GT95, length  = np.nan, np.nan, np.nan, bestFits.loc[country, 'maxLength']
            lengthGT = [np.nan, np.nan]
        else:
            GT, GT5, GT95, length = bestFits.loc[country, ['frcComplete_MRP_outofsample','frcComplete_MRP_outofsample_5pct', 'frcComplete_MRP_outofsample_95pct', 'maxLength']].tolist()
            lengthGT = [float(length)/GT5/yscaleFactor, float(length)/GT95/yscaleFactor, float(length)/GT/yscaleFactor]
            xls = axes[row,col].get_xlim()
            axes[row, col].add_patch(mpl.patches.Rectangle((xls[0],lengthGT[0]),(xls[1]-xls[0]),lengthGT[1]-lengthGT[0], alpha=0.2, color=c5s[1], lw=0))
            axes[row, col].hlines(lengthGT[2], xlims[0], xlims[-1], lw=0.5, alpha=0.7, color=c5s[1])
        if 0:  # horizontal line at visual assessment mean (for testing)
            lengthGT_naive = float(length)/bestFits.loc[country, 'frcComplete_visual'] /yscaleFactor
            axes[row, col].hlines(lengthGT_naive, xlims[0], xlims[-1], linestyle='--', lw=0.5, colors='r')

        # horizontal line at asymptote
        axes[row, col].hlines(countries.loc[country, fitFunc2Plot+'_asymptote'], xlims[0], xlims[-1], colors=c5s[3])
        ymax = max([yy for yy in [lengthGT[0], plotDf.ydata.max(), plotDf.yPred.max(), countries.loc[country,  fitFunc2Plot+'_asymptote']] if pd.notnull(yy)])
        axes[row,col].set_ylim(0, 1.05*ymax)

        axes[row, col].set_title(ISO2cname[country].replace('&',' and '), fontsize=11)      
        axes[row, col].set_yticks([])
        axes[row, col].set_xlabel('')
        
        axes[row, col].set_xticks(xticks)
        axes[row, col].set_xticks(xticksMinor, minor=True)
        
        if row!=5:  # hide labels, except on last row
            axes[row, col].xaxis.set_ticklabels([])
            
        if fitfunc=='best':
            yPos, xPos = axes[row, col].get_ylim()[1]*0.08, axes[row, col].get_xlim()[1]*.999
            axes[row, col].text(xPos, yPos, bfuncname.replace('_','-'), fontsize=6, horizontalalignment='right')    

        if row==0 and col==0:
            # we take the legend handles from [0,0], but plot it in [5,3]
            axes[0,0].plot(0,0,c='w',lw=0,label='Visual assessment:')  # dummy line for legend
            legHandles, legLabels = axes[0, 0].get_legend_handles_labels()
            legHandles = legHandles[0:2]+[legHandles[-2]]+[mpl.lines.Line2D([], [], color=c5s[3])]+[legHandles[-1]]+[mpl.lines.Line2D([], [], color=c5s[1], alpha=0.7)]+[mpl.patches.Patch(color=c5s[1], alpha=0.2)]
            if texAvailable:
                legLabels =  legLabels[0:2]+ [legLabels[-2]]+ ['Asymptote']+[legLabels[-1]]+['Multilevel estimate','95$\%$ CI']
            else:
                legLabels =  legLabels[0:2]+ [legLabels[-2]]+ ['Asymptote']+[legLabels[-1]]+['Multilevel estimate','95pc CI']
            
            axes[5,3].legend(legHandles, legLabels, loc='best', fontsize=8)
            axes[5,3].axis('off') # suppress axes and ticks

        if (row==5 and col==2) or cc==len(countries)-1:  
            # last one on this page
            plt.tight_layout()
            if not(os.path.exists(paths['scratch']+'completenessFitsPlots')): os.mkdir(paths['scratch']+'completenessFitsPlots')
            outFn = paths['scratch']+'completenessFitsPlots/fitted-p'+str(page)+suffix+'.pdf'
            outList.append(outFn)
            print "Saved temporary file %s" % outFn
            plt.savefig(outFn)
            
    # merge the PDFs into one file   
    outFn = paths['output']+'osmFits'+suffix+'.pdf'
    mergePDFs(outList, outFn)
        
    if fitfunc=='best':
        # Now do one for the top 10 countries (for the journal)
        # If we were clever, we would use some functions to avoid redundancy with above...
        dfPedPaths=loadAggregatedHistory(gadmLevel=0, roadsOnly=False)
        dfPedPaths=dfPedPaths[dfPedPaths.roadFlag==False]
        dfPedPaths.set_index('ISOalpha3', inplace=True)

        plt.close('all')
        mpl.rcParams['xtick.labelsize'] = 9
        mpl.rcParams['ytick.labelsize'] = 8
        fig, axes = plt.subplots(3, 4, figsize=figSizeBig)
        
        for cc, country in enumerate(countryISOs[:11]):    
            row = int(cc/4)
            col = cc-4*row  # for plotting

            mask = df.index.values==country  # avoids this problem: http://stackoverflow.com/questions/23227171/assignment-to-containers-in-pandas
            # choose the function with lowest MSE for that country
            bfuncname,fitFunc2Plot = bestFits.loc[country, 'fitfunc_name'],        bestFits.loc[country, 'fitfunc_pref']
            df.loc[mask, 'yPred'] = globals()[bfuncname].f(df.loc[mask].set_index('date').xdata,*(countries.loc[country, fitFunc2Plot+'_estparams'])).values

            plotDf = df.ix[country].set_index('date')
            plotDfPedPaths = dfPedPaths.ix[country].set_index('date')
            
            plotDf.ydata.plot(ax=axes[row, col], xlim=xlims, color=c5s[0], lw=1.5, alpha=0.7, label='Actual')
            plotDf.yPred.plot(ax=axes[row, col], xlim=xlims, color=c5s[2], lw=1.5, alpha=0.7, label='Predicted')
            plotDfPedPaths.ydata.plot(ax=axes[row, col], xlim=xlims, color=c5s[0], lw=0.5, label='Other paths\n(non-roads)')

            # plot ymax groundtruth
            GT, GT5, GT95, length = bestFits.loc[country, ['frcComplete_MRP_outofsample','frcComplete_MRP_outofsample_5pct', 'frcComplete_MRP_outofsample_95pct', 'maxLength']].tolist()
            lengthGT = [float(length)/GT5/yscaleFactor, float(length)/GT95/yscaleFactor,float(length)/GT/yscaleFactor]
            xls = axes[row,col].get_xlim()
            axes[row, col].add_patch(mpl.patches.Rectangle((xls[0],lengthGT[0]),(xls[1]-xls[0]),lengthGT[1]-lengthGT[0], alpha=0.2, lw=0, color=c5s[1]))
            axes[row, col].hlines(lengthGT[2], xlims[0], xlims[-1], lw=0.5, alpha=0.7, color=c5s[1])
            axes[row,col].set_ylim(top=1.05*max(lengthGT[0], plotDf.ydata.max(), plotDf.yPred.max()))

            # horizontal line at ymax
            axes[row, col].hlines(countries.loc[country, fitFunc2Plot+'_asymptote'], xlims[0], xlims[-1], colors=c5s[3])
            if country=='ALL': lengthGT[0] = 0  # kludge
            axes[row,col].set_ylim(top=1.05*max(lengthGT[0], plotDf.ydata.max(), plotDf.yPred.max(), plotDfPedPaths.ydata.max(), countries.loc[country,  fitFunc2Plot+'_asymptote']))

            axes[row, col].set_title(ISO2cname[country].replace('&',' and '), fontsize=11)      
            
            axes[row, col].set_xlabel('')
            axes[row, col].set_xticks(xticks)
            axes[row, col].set_xticks(xticksMinor, minor=True)
            
            axes[row, col].yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
            axes[row, col].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
            if country=='ALL': axes[row, col].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))  # ensures that the labels line up
            
            if col==0:
                assert yscaleFactor == 10**9
                if texAvailable:
                    axes[row, col].set_ylabel('Length (km $10^{6}$)', fontsize=9)
                else:
                    axes[row, col].set_ylabel('Length (million km)', fontsize=9)

        axes[2,3].axis('off') # suppress axes and ticks
        
        # we take the legend handles from [0,0], but plot it underneath [1,2]
        axes[0,0].plot(0,0,c='w',lw=0,label='Visual assessment:')  # dummy line for legend
        legHandles, legLabels = axes[0, 0].get_legend_handles_labels()
        legHandles = legHandles[0:2]+[mpl.lines.Line2D([], [], color=c5s[3])]+[legHandles[-1]]+[mpl.lines.Line2D([], [], color=c5s[1], alpha=0.7)]+[mpl.patches.Patch(color=c5s[1], alpha=0.2)]+[legHandles[-2]]
        if texAvailable:
            legLabels =  legLabels[0:2]+ ['Asymptote']+[legLabels[-1]]+['Multilevel estimate','95$\%$ CI']+[legLabels[-2]]
        else:
            legLabels =  legLabels[0:2]+ ['Asymptote']+[legLabels[-1]]+['Multilevel estimate','95pc CI']+[legLabels[-2]]
           
        axes[1,3].legend(legHandles, legLabels, bbox_to_anchor=(1.1, -0.4), fontsize=8)
        plt.tight_layout()
        outFn = paths['output']+'fitted-top10best'
        plt.savefig(outFn+'.pdf')        
        #plt.savefig(outFn+'.eps')   # generates latex error     
        print "Saved file %s (.pdf - .eps not yet fixed)" % outFn

def plotFitsDensity():
    """
    Plots WB regions by density
    Similar to plotFits, but we have all deciles on a single chart
    """
    # load aggregated dataframe by WB region (for plots)
    regionDfs = aggregateByWBregion(loadAggregatedHistory(gadmLevel=0, density=True))

    WBregions = ht.country2WBregionLookup()    
    WBregionDict = WBregions.GroupName.drop_duplicates().to_dict()

    # load countries dataframe with parameter estimates
    allfits = fitallRegionsByDensity()
    bestfits = chooseBestFit('wbr')
    bestfits.set_index('decile', append=True, inplace=True) # multiindex region, decile
    allfits.set_index('decile', append=True, inplace=True) # multiindex region, decile
    bestfits.sort_values(by='maxLength', ascending=False, inplace=True)
       
    # set up plotting
    plt.close('all')
    outList = []  # list of outPDF files
    xlims = (datetime.date(2007,1,1), xdateMAX)
    xticks = [datetime.date(2009,1,1), datetime.date(2012,1,1), datetime.date(2015,1,1)]
    xticksMinor = [datetime.date(yy,1,1) for yy in range(2008,2016)]
    mpl.rcParams['xtick.labelsize'] = 9
    mpl.rcParams['ytick.labelsize'] = 9

    for wbr in regionDfs:
        fig, axes = plt.subplots(4, 3, figsize=figSizePage)
        fig.suptitle(WBregionDict[wbr].replace('&','and'), fontsize=14)
        for decile in [99]+range(0,10): 
            row = 0 if decile==99 else  int((decile+1)/3)
            col = 0 if decile==99 else (decile+1)-3*row
            pltTitle = 'All deciles' if decile==99 else 'Density decile %d' % (decile+1)

            # Calculate predicted values
            if pd.isnull(bestfits.loc[(wbr, decile), 'fitfunc_name']): continue
            bfuncname,fitFunc2Plot = bestfits.loc[(wbr, decile), 'fitfunc_name'],        bestfits.loc[(wbr, decile), 'fitfunc_pref']
            
            plotDf = (regionDfs[wbr][regionDfs[wbr].densDecile==decile]).set_index('date')
            plotDf['yPred'] = globals()[bfuncname].f(plotDf.xdata,*(allfits.loc[(wbr, decile), fitFunc2Plot+'_estparams'])).values

            # only plot the first day of each month for the daily data (unless this would result in fewer than 50 points...this is a workaround for small countries)
            datemask =     plotDf.index.day==1 # | (plotDf.index.month==3) & (plotDf.index.year==2015)
        
            if datemask.sum()<50: # plot everything
                plotDf.ydata.plot(ax=axes[row, col], xlim=xlims, style='.', ms=1, color='k', label='Actual')
            else:
                plotDf[datemask].ydata.plot(ax=axes[row, col], xlim=xlims, style='.', ms=2, color='k', label='Actual')

            plotDf.yPred.plot(ax=axes[row, col], xlim=xlims, color=c5s[1], lw=1.5, alpha=0.7, label='Predicted')
            
            # horizontal line at ymax
            axes[row, col].hlines(allfits.loc[(wbr, decile), fitFunc2Plot+'_asymptote'], xlims[0], xlims[-1], colors='b')
            axes[row,col].set_ylim(top=1.05*max(plotDf.ydata.max(), plotDf.yPred.max(), allfits.loc[(wbr, decile),  fitFunc2Plot+'_asymptote']))

            axes[row, col].set_title(pltTitle, fontsize=11)      
            axes[row, col].set_yticks([])
            axes[row, col].set_xlabel('')

            axes[row, col].set_xticks(xticks)
            axes[row, col].set_xticks(xticksMinor, minor=True)
            if row!=3: axes[row, col].xaxis.set_ticklabels([]) # hide labels, except on last row
            
            # What function was the best?
            yPos, xPos = axes[row, col].get_ylim()[1]*0.1, axes[row, col].get_xlim()[1]*.99
            axes[row, col].text(xPos, yPos, bfuncname.replace('_','-'), fontsize=6, horizontalalignment='right')    

        # we take the legend handles from [0,0], but plot it in [5,3]
        legHandles = axes[0,0].get_legend_handles_labels()[0][0:2]+[mpl.lines.Line2D([], [], color='b')]
        legLabels = axes[0,0].get_legend_handles_labels()[1][0:2]+['Asymptote']           
        axes[3,2].legend(legHandles, legLabels, loc='best', fontsize=9)
        axes[3,2].axis('off') # suppress axes and ticks

        plt.tight_layout()
        plt.subplots_adjust(top=0.92) # top margin
    
        outFn = paths['scratch']+'completenessFitsPlots/fitted-'+wbr+'.pdf'
        outList.append(outFn)
        print "Saved temporary file %s" % outFn
        plt.savefig(outFn)
        plt.close()
            
    # merge the PDFs into one file   
    outFn = paths['output']+'osmFitsWBregions-bydensity.pdf'
    mergePDFs(outList, outFn)

def func_mean_square_error(xdata_,ydata_, func,params):
     if  pd.isnull(params).any():
         return(np.nan)
     yPred=func.f(xdata_,*params)
     err=np.sum((yPred-ydata_)**2)/len(yPred)
     return(err)

def date2xdata(fromdate):
    if isinstance(fromdate,datetime.datetime):
        import time
        return int(time.mktime(fromdate.timetuple()))/1e9
    elif isinstance(fromdate, pd.Series):
        # IF it's a Pandas series, it has a (vector) shortcut:
        # the to_period is needed so that we convert to seconds resolution, to match the time.mktime() above
        return pd.datetools.to_datetime(fromdate.values).to_period('s').astype(int)/ 1e9 
    else:
        print 'date2xdata did not understand dtype'
        endherenow
        
def xdata2date(xdata):
    assert isinstance(xdata, float)
    import datetime
    if pd.isnull(xdata) or xdata>date2xdata(datetime.datetime(2100,1,1,0,0)) or xdata<date2xdata(datetime.datetime(2000,1,1,0,0)): # scalar only
        return(np.nan)
    return(   datetime.datetime.fromtimestamp(xdata* 1e9 )   )

class justaline:
    @staticmethod
    def f(x, x0, m):
         y = (x>x0)*(x-x0)*m
         return y
    @staticmethod
    def initparams(xdata,ydata):
        return([0,0])
    @staticmethod
    def whenCompleted(xdata,ydata,x0, m):
        return(  np.nan )
    @staticmethod
    def asymptote(xdata,ydata,x0, m):
        """ Returns the predicted asymptote (max) given a particular set of parameters (ie a fit)"""
        return(  np.nan )

class sigmoid:
    @staticmethod
    def f(x, x0, k, maxy):
         y = maxy / (1 + np.exp(-k*(x-x0)))
         return y
    @staticmethod
    def initparams(xdata,ydata):
        return(  [(min(xdata)+max(xdata))/2.0, 10.0/(max(xdata)-min(xdata)), max(ydata) ]  )
    @staticmethod
    def whenCompleted(xdata,ydata,x0, k,maxy):
        """ Estimates x value, given xdata, for when modeled shape, ignoring jumps, is 99% complete """
        return(   -(1/k)*np.log( 1.0/(COMPLETE)-1 ) +x0 )
    @staticmethod
    def asymptote(xdata,ydata,x0, k,maxy):
        """ Returns the predicted asymptote (max) given a particular set of parameters (ie a fit)"""
        return(  maxy )

class sigmoid_with_1_jumps:
    # In some countries (e.g. USA), the xjumps seem to be coming outside the range of the xdata
    # Let's calculate the asymptote as maxy - abs(dy2) if xjump2 > max(xdata)    
    @staticmethod
    def f(x, x0, k,miny,maxy, xjump1,dy1):
         # Caution! dx1 must be constrained to be positive
         if xjump1>date2xdata(xdateMAX): dy1=0
         y = abs(miny) + (x>xjump1)*abs(dy1)     +  (maxy-abs(dy1)-abs(miny)) / (1 + np.exp(-k*(x-x0))) 
         return y
    @staticmethod
    def initparams(xdata,ydata):
        # Find the steepest single step
        dydata=ydata-ydata.shift(1)
        if len(dydata[dydata>0])<4:
            return([np.nan]*6)
        xjump1=xdata.shift(1)[dydata==dydata.max()]
        dy1=(ydata-ydata.shift(1))[dydata==dydata.max()]
        return(  [(min(xdata)+max(xdata))/2.0, 10.0/(max(xdata)-min(xdata)), min(ydata), max(ydata),  xjump1.values[0], dy1.values[0] ])
    @staticmethod
    def whenCompleted(xdata,ydata,x0, k,miny,maxy, xjump1,dy1):
        """ Estimates x value, given xdata, for when modeled shape, ignoring jumps, is 99% complete """
        return(   -(1/k)*np.log( 1.0/(COMPLETE)-1 ) +x0 )
    @staticmethod
    def asymptote(xdata,ydata,x0, k,miny,maxy, xjump1,dy1):
        """ Returns the predicted asymptote (max) given a particular set of parameters (ie a fit)"""
        if xjump1<max(xdata): dy1=0  #and xjump1>=min(xdata)
        #return(  maxy -abs(dy1))
        return maxy
        
class sigmoid_with_2_jumps:
    @staticmethod
    def f(x, x0, k,miny,maxy, xjump1,dy1,  xjump2,dy2):
        # Caution! dy1, dy2, miny must be constrained to be positive. Using abs here is one (odd) way.
        if xjump1>date2xdata(xdateMAX): dy1=0        
        if xjump2>date2xdata(xdateMAX): dy2=0        
        y = abs(miny)+   (x>xjump1)*abs(dy1)+ (x>xjump2)*abs(dy2)     +  (maxy-abs(dy1)-abs(dy2)-abs(miny)) / (1 + np.exp(-k*(x-x0)))
        return y
    @staticmethod
    def initparams(xdata,ydata):
        # Find the steepest single step?
        # Careful... I'm making use of Pandas properties, but xdata could be just a vector, rather than a pd.Series
        dydata=ydata-ydata.shift(1)
        if len(dydata[dydata>0])<4:
            return([np.nan]*8)
        xjump1=xdata.shift(1)[dydata==dydata.max()]
        dy1=(ydata-ydata.shift(1))[dydata==dydata.max()]

        flatter=dydata<dydata.mean()
        boundaries=np.hstack([np.array(False), np.diff(flatter.astype(int)).astype(bool)])  # hstack is because boundaries needs to be the same length as flatter; np.diff will be one short
        try: # These seems to work on every case except non-changing ydata (with one change only)
            iBoundaries=    flatter[boundaries].index # This is where slope changes (crosses) its mean value
            # Here is the index from which (inclusive) we should start masking away:, ie  ind the latest boundary before our jump, and the first following it:
            startMask,endMask =   iBoundaries[iBoundaries<xjump1.index.values[0]].max()   ,     iBoundaries[iBoundaries>xjump1.index.values[0]].min()
            dydata.ix[startMask:endMask]=0 # This zeros out the slope on a contiguous region around our first jump.

            xjump2=xdata.shift(1)[dydata==dydata.max()]
            dy2=(ydata-ydata.shift(1))[dydata==dydata.max()]
        except AttributeError:
            assert sum(boundaries)==1  # This is actually really pathalogical. 
            xjump2=xjump1
            dy2=dy1
        return(  [(min(xdata)+max(xdata))/2.0, 10.0/(max(xdata)-min(xdata)), min(ydata), max(ydata),  xjump1.values[0], dy1.values[0], xjump2.values[0], dy2.values[0]]  )
    @staticmethod
    def whenCompleted(xdata,ydata,x0, k,miny,maxy, xjump1,dy1,  xjump2,dy2):
        """ Estimates x value, given xdata, for when modeled shape, ignoring jumps, is 99% complete """
        return(   -(1/k)*np.log( 1.0/(COMPLETE)-1 ) +x0 )
    @staticmethod
    def asymptote(xdata,ydata,x0, k,miny,maxy, xjump1,dy1,  xjump2,dy2):
        """ Returns the predicted asymptote (max) given a particular set of parameters (ie a fit)"""
        if xjump1<max(xdata): dy1=0  
        if xjump2<max(xdata): dy2=0 
        return maxy
        
class sigmoid_with_3_jumps:
    @staticmethod
    def f(x, x0, k,miny,maxy, xjump1,dy1,  xjump2,dy2,   xjump3,dy3):
        if xjump1>date2xdata(xdateMAX): dy1=0        
        if xjump2>date2xdata(xdateMAX): dy2=0        
        if xjump3>date2xdata(xdateMAX): dy3=0        
        # Caution! dy1, dy2, miny must be constrained to be positive. Using abs here is one (odd) way.
        y = abs(miny)+   (x>xjump1)*abs(dy1)+ (x>xjump2)*abs(dy2)+ (x>xjump3)*abs(dy3)     +  (maxy-abs(dy1)-abs(dy2)-abs(dy3)-abs(miny)) / (1 + np.exp(-k*(x-x0)))
        return y
    @staticmethod
    def initparams(xdata,ydata):
        # Find the steepest single step
        # Careful... I'm making use of Pandas properties, but xdata could be just a vector, rather than a pd.Series
        dydata=ydata-ydata.shift(1)
        if len(dydata[dydata>0])<5:
            return([np.nan]*10)
        xjump1=xdata.shift(1)[dydata==dydata.max()]
        dy1=(ydata-ydata.shift(1))[dydata==dydata.max()]

        flatter=dydata<dydata.mean()
        boundaries=np.hstack([np.array(False), np.diff(flatter.astype(int)).astype(bool)])  # hstack is because boundaries needs to be the same length as flatter; np.diff will be one short
        try: # These seems to work on every case except non-changing ydata (with one change only)
            iBoundaries=    flatter[boundaries].index # This is where slope changes (crosses) its mean value
            # Here is the index from which (inclusive) we should start masking away:, ie  ind the latest boundary before our jump, and the first following it:
            startMask,endMask =   iBoundaries[iBoundaries<xjump1.index.values[0]].max()   ,     iBoundaries[iBoundaries>xjump1.index.values[0]].min()
            dydata.ix[startMask:endMask]=0 # This zeros out the slope on a contiguous region around our first jump.

            xjump2=xdata.shift(1)[dydata==dydata.max()]
            dy2=(ydata-ydata.shift(1))[dydata==dydata.max()]
        except AttributeError:
            assert sum(boundaries) in [1]  # This is actually really pathalogical. 
            xjump2=xjump1
            dy2=dy1

        try: # These seems to work on every case except non-changing ydata (with one change only)
            iBoundaries=    flatter[boundaries].index # This is where slope changes (crosses) its mean value
            # Here is the index from which (inclusive) we should start masking away:, ie  ind the latest boundary before our jump, and the first following it:
            startMask,endMask =   iBoundaries[iBoundaries<xjump2.index.values[0]].max()   ,     iBoundaries[iBoundaries>xjump2.index.values[0]].min()
            dydata.ix[startMask:endMask]=0 # This zeros out the slope on a contiguous region around our first jump.

            xjump3=xdata.shift(1)[dydata==dydata.max()]
            dy3=(ydata-ydata.shift(1))[dydata==dydata.max()]
        except AttributeError:
            assert sum(boundaries) in [1]  # This is actually really pathalogical. 
            xjump3=xjump2
            dy3=dy2
        return(  [(min(xdata)+max(xdata))/2.0, 10.0/(max(xdata)-min(xdata)), min(ydata), max(ydata),  xjump1.values[0], dy1.values[0], xjump2.values[0], dy2.values[0], xjump3.values[0], dy3.values[0]]  )
    @staticmethod
    def whenCompleted(xdata,ydata,x0, k,miny,maxy, xjump1,dy1,  xjump2,dy2,  xjump3,dy3):
        """ Estimates x value, given xdata, for when modeled shape, ignoring jumps, is 99% complete """
        return(   -(1/k)*np.log( 1.0/(COMPLETE)-1 ) +x0 )
    @staticmethod
    def asymptote(xdata,ydata,x0, k,miny,maxy, xjump1,dy1,  xjump2,dy2,  xjump3,dy3):
        """ Returns the predicted asymptote (max) given a particular set of parameters (ie a fit)"""
        # note: strict inequality, because xjump only applies at x>xjump
        if xjump1<max(xdata): dy1=0 #  and xjump1>=min(xdata)
        if xjump2<max(xdata): dy2=0  # and xjump2>=min(xdata)
        if xjump3<max(xdata): dy3=0  #  and xjump3>=min(xdata) 
        return maxy
        
class sigmoid_with_4_jumps:
    @staticmethod
    def f(x, x0, k,miny,maxy, xjump1,dy1,  xjump2,dy2,   xjump3,dy3,  xjump4, dy4):
        if xjump1>date2xdata(xdateMAX): dy1=0        
        if xjump2>date2xdata(xdateMAX): dy2=0        
        if xjump3>date2xdata(xdateMAX): dy3=0        
        if xjump4>date2xdata(xdateMAX): dy4=0        
        # Caution! dy1, dy2, miny must be constrained to be positive. Using abs here is one (odd) way.
        y = abs(miny)+   (x>xjump1)*abs(dy1)+ (x>xjump2)*abs(dy2)+ (x>xjump3)*abs(dy3) + (x>xjump4)*abs(dy4)     +  (maxy-abs(dy1)-abs(dy2)-abs(dy3)-abs(dy4)-abs(miny)) / (1 + np.exp(-k*(x-x0)))
        return y
    @staticmethod
    def initparams(xdata,ydata):
        # Find the steepest single step?
        # Careful... I'm making use of Pandas properties, but xdata could be just a vector, rather than a pd.Series
        dydata=ydata-ydata.shift(1)
        if len(dydata[dydata>0])<6:
            return([np.nan]*12)
        xjump1=xdata.shift(1)[dydata==dydata.max()]
        dy1=(ydata-ydata.shift(1))[dydata==dydata.max()]

        flatter=dydata<dydata.mean()
        boundaries=np.hstack([np.array(False), np.diff(flatter.astype(int)).astype(bool)])  # hstack is because boundaries needs to be the same length as flatter; np.diff will be one short
        try: # These seems to work on every case except non-changing ydata (with one change only)
            iBoundaries=    flatter[boundaries].index # This is where slope changes (crosses) its mean value
            # Here is the index from which (inclusive) we should start masking away:, ie  ind the latest boundary before our jump, and the first following it:
            startMask,endMask =   iBoundaries[iBoundaries<xjump1.index.values[0]].max()   ,     iBoundaries[iBoundaries>xjump1.index.values[0]].min()
            dydata.ix[startMask:endMask]=0 # This zeros out the slope on a contiguous region around our first jump.

            xjump2=xdata.shift(1)[dydata==dydata.max()]
            dy2=(ydata-ydata.shift(1))[dydata==dydata.max()]
        except AttributeError:
            assert sum(boundaries) in [1,2]  # This is actually really pathalogical. 
            xjump2=xjump1
            dy2=dy1

        try: # These seems to work on every case except non-changing ydata (with one change only)
            iBoundaries=    flatter[boundaries].index # This is where slope changes (crosses) its mean value
            # Here is the index from which (inclusive) we should start masking away:, ie  ind the latest boundary before our jump, and the first following it:
            startMask,endMask =   iBoundaries[iBoundaries<xjump2.index.values[0]].max()   ,     iBoundaries[iBoundaries>xjump2.index.values[0]].min()
            dydata.ix[startMask:endMask]=0 # This zeros out the slope on a contiguous region around our first jump.

            xjump3=xdata.shift(1)[dydata==dydata.max()]
            dy3=(ydata-ydata.shift(1))[dydata==dydata.max()]
        except AttributeError:
            assert sum(boundaries) in [1,2]  # This is actually really pathalogical. 
            xjump3=xjump2
            dy3=dy2

        try: # These seems to work on every case except non-changing ydata (with one change only)
            iBoundaries=    flatter[boundaries].index # This is where slope changes (crosses) its mean value
            startMask,endMask =   iBoundaries[iBoundaries<xjump3.index.values[0]].max()   ,     iBoundaries[iBoundaries>xjump3.index.values[0]].min()
            dydata.ix[startMask:endMask]=0 

            xjump4=xdata.shift(1)[dydata==dydata.max()]
            dy4=(ydata-ydata.shift(1))[dydata==dydata.max()]
        except AttributeError:
            assert sum(boundaries) in [1,2]  # This is actually really pathalogical. 
            xjump4=xjump3
            dy4=dy3
        return(  [(min(xdata)+max(xdata))/2.0, 10.0/(max(xdata)-min(xdata)), min(ydata), max(ydata),  xjump1.values[0], dy1.values[0], xjump2.values[0], dy2.values[0], xjump3.values[0], dy3.values[0], xjump4.values[0], dy4.values[0]]  )
    @staticmethod
    def whenCompleted(xdata,ydata,x0, k,miny,maxy, xjump1,dy1,  xjump2,dy2,  xjump3,dy3, xjump4, dy4):
        """ Estimates x value, given xdata, for when modeled shape, ignoring jumps, is 99% complete """
        return(   -(1/k)*np.log( 1.0/(COMPLETE)-1 ) +x0 )
    @staticmethod
    def asymptote(xdata,ydata,x0, k,miny,maxy, xjump1,dy1,  xjump2,dy2,  xjump3,dy3, xjump4, dy4):
        """ Returns the predicted asymptote (max) given a particular set of parameters (ie a fit)"""
        # these all need to be ignored. if xjump1 is in the range, it is implicit in ymax. But if xjump1 >xdataMAX, it is ignored in the f() method above.
        if xjump1<max(xdata): dy1=0 #  and xjump1>=min(xdata)
        if xjump2<max(xdata): dy2=0 #  and xjump2>=min(xdata)
        if xjump3<max(xdata): dy3=0 #  and xjump3>=min(xdata)
        if xjump4<max(xdata): dy4=0 #  and xjump4>=min(xdata)
        #return(  maxy -abs(dy1)-abs(dy2)-abs(dy3)-abs(dy4))
        return maxy

class sigmoid_with_1_ramp:
    @staticmethod
    def f(x, x0, k,maxy, xi,xif,dx):
         # Caution! dx1 must be constrained to be positive
         xf=xi+xif
         y = maxy / (1 + np.exp(-k*(x-x0 + (x>xi)*(x<xf)*abs(dx)*(x-xi)/(xif)  )))  
         return y
    @staticmethod
    def whenCompleted(xdata,ydata,x0, k,maxy, xi,xif,dx):
        """ Estimates x value, given xdata, for when modeled shape, ignoring jumps, is 99% complete """
        return(   -(1/k)*np.log( 1.0/(COMPLETE)-1 ) +x0 )
    @staticmethod
    def initparams(xdata,ydata):
        return(  [min(xdata), 0, max(ydata),  np.mean(xdata),.01,.01] )
    @staticmethod
    def asymptote(xdata,ydata,x0, k,maxy, xi,xif,dx):
        """ Returns the predicted asymptote (max) given a particular set of parameters (ie a fit)"""
        return(  maxy )

class gompertz: 
    @staticmethod
    def f(xdata,a,b,c): #    y(t)=ae^{-be^{-ct}}
        xnorm=(xdata - date2xdata(xdateMIN))/(date2xdata(xdateMAX) - date2xdata(xdateMIN))
        G=a*np.exp(-b*np.exp(-c*xnorm))
        return G
    @staticmethod
    def whenCompleted(xdata,ydata,a,b,c):
        """ Estimates x value, given xdata, for when modeled shape, ignoring jumps, is 99% complete """
        return(   -1/c*  np.log( -1/b*  np.log(COMPLETE) )  *(max(xdata)-min(xdata))    + min(xdata)  )

    @staticmethod
    def asymptote(xdata,ydata,a,b,c):
        """ Returns the predicted asymptote (max) given a particular set of parameters (ie a fit)"""
        return(  a  )  # not b
    @staticmethod
    def initparams(xdata,ydata):
        a=max(ydata)
        b=10
        c=6
        return(a,b,c)

if __name__ == '__main__':
    runmode=None if len(sys.argv)<2 else sys.argv[1].lower()
    forceUpdate = any(['forceupdate' in arg.lower() for arg in sys.argv])
    print 'forceUpdates is %s' % forceUpdate 
 
    if runmode in [None,'fits']:
        fitallCountries(gadmLevel=0, forceUpdate=forceUpdate, parallel=True)
        fitallCountries(gadmLevel=1, forceUpdate=forceUpdate, parallel=True)
        fitallRegionsByDensity(forceUpdate=forceUpdate, parallel=True)
        fitallDensityQuintiles(forceUpdate=forceUpdate, parallel=True)
    if runmode in [None,'plots']:
        plotFits()
        plotFits(gadmLevel=1)
        plotFitsDensity() # note, also done in analysis
