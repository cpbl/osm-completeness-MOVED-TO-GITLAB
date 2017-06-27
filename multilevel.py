#!/usr/bin/python
# coding=utf-8

"""
Fit various multilevel models using Stan

Useful resources:
Quick start guide to Stan for economists: http://nbviewer.jupyter.org/github/QuantEcon/QuantEcon.notebooks/blob/master/IntroToStan_basics_workflow.ipynb
Out of sample prediction: https://groups.google.com/forum/#!searchin/stan-users/prediction/stan-users/tB40xNXP26g/A42w8c9pqBUJ
Poisson model (not multilevel): https://github.com/stan-dev/example-models/blob/master/ARM/Ch.17/17.5_multilevel_poisson.stan

Suggestions from Chris Warshaw:
- look at his same-sex marriage Stan code
- look at Gelman AJPS article (http://onlinelibrary.wiley.com/doi/10.1111/ajps.12004/abstract) and code (https://github.com/gelman/mrp-stan/tree/master/models)
- Matt trick (reparameterize to reduce correlations between variables, and improve convergence. See 114-138 of turnout_complex_post.stan and Stan manual

Choice of Priors: see especially https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations
Weak prior helps the model to converge. Stan manual recommends Cauchy
Polson and Scott (2011): “The half-Cauchy occupies a sensible ‘middle ground’ . . . it performs very well near the origin, but does not lead to drastic compromises in other parts of the parameter space.” (from http://www.stat.columbia.edu/%7Egelman/presentations/wipnew2_handout.pdf)

"""
import os, sys, math
from history_config import *

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use(mpl_backend)
import matplotlib.pyplot as plt

import history_tools as ht
  
from cpblUtilities.mathgraph import cpblScatter, dfPlotWithEnvelope, weightedMeanSE_pandas, figureFontSetup
from cpblUtilities.utilities import shelfLoad, shelfSave
from cpblUtilities.parallel import runFunctionsInParallel

from analysis import aggregateVisualAssessmentPoints

countryVars     = ['gdpCapitaPPP_WB', 'internetUsersper100_2013', 'pop2012', 'GovernanceVoice2013']
countryVarNames = ['log GDP pc', 'internet', 'log pop', 'governance']
stanModels = [ff for ff in os.listdir(paths['bin']) if ff.endswith('stan') and 'GT' in ff and not('cat' in ff)]  # skip the density-as-categorical plot for now

def multilevelModel(stanFn='multilevel_GT_sq',nDraws=1000, forceUpdate=False):
    """
    Uses Stan to fit a multilevel model of the number of OSM segments and frc complete
    Saves the output to working (right now, just the predictions at the country level, but we could save more as well)
    and plots measures of fit, etc.
    nDraws gives the number of iterations that will be used (note: spread across 10 chains, which are parallelized, and thinned to every 5)
    
    If you call this with stanFn=None, it will iterate over all four models. 
    By default, the preferred model is multilevel_GT_sq (which has a squared density term).
    """

    if stanFn is None:
        for stanModel in stanModels: multilevelModel(stanFn=stanModel.replace('.stan',''), forceUpdate=forceUpdate) 
        return
        
    outFn = paths['working'] + 'countriesFit_'+stanFn+'.pickle'
    if os.path.exists(outFn) and not forceUpdate:
        return pd.read_pickle(outFn)

    import pystan
    import matplotlib.pyplot as plt

    # retain old parameters, to restore them afterwards
    oldrcParams = plt.rcParams
    plt.rcParams['text.usetex'] = False
    plt.rcParams['legend.fontsize'] = 7

    # get point-level data
    pointsDf = pd.read_pickle(pointsFn)
    pointsDf.dropna(subset=['totSegs', 'NpresentSegs', 'density', 'weight'], inplace=True)
    pointsDf['Pselection'] = 1./pointsDf.weight  # weight is inverse probability of selection in each country, normalized to sum to 1
    countryDf = ht.loadOtherCountryData()
    fitCountries = [cc for cc in pointsDf.index.unique() if cc in countryDf.index and all([pd.notnull(countryDf.loc[cc, col]) for col in countryVars])]

    #assert len(pointsDf.index.unique()) == len(fitCountries) # check we have no missing data
    pointsDf = pointsDf.ix[fitCountries]  # if we have missing data
    countryDf = countryDf.ix[fitCountries]

    # Rasters use id_0, not iso, so convert to that
    iso2id_0 = ht.get_all_GADM_countries()
    pointsDf['id_0']  = pointsDf.index.map(lambda x: iso2id_0[x]['id_0']) 
    countryDf['id_0'] = countryDf.index.map(lambda x: iso2id_0[x]['id_0'])
    countryDf.sort_values('id_0', inplace=True)
    
    # convert to cid, which is a continuous-range integer, for Stan's benefit 
    id_0ToCid = dict([(id_0, ii+1) for ii, id_0 in enumerate(countryDf.id_0.values)])
    pointsDf['cid']  = pointsDf.id_0.map(lambda x: id_0ToCid[x])
    countryDf['cid'] = countryDf.id_0.map(lambda x: id_0ToCid[x])
    
    # need to estimate parameters and then parametrically calculate conf intervals
    stanData = {'N': len(pointsDf), 'J': len(pointsDf.index.unique()), 'L':1, 'M': 4,
                'obsSegs': pointsDf.NpresentSegs.values, 'totSegs': pointsDf.totSegs.values, 'country': pointsDf.cid.values, 
                'logdensity': stdize(pointsDf.density, True).values,  'wght':    pointsDf.Pselection.values,
                'Z1': stdize(countryDf[countryVars[0]], True).values, 'Z2': stdize(countryDf[countryVars[1]]).values, 
                'Z3': stdize(countryDf[countryVars[2]], True).values,  'Z4': stdize(countryDf[countryVars[3]]).values}
    if 'cat' in stanFn: 
        quintiles = [getDensityDeciles()[ii] for ii in [0,2,4,6,8,10]]
        stanData['L'] = 6 
        pointsDf['quintile'] = pointsDf.density.apply(lambda x: np.digitize(x, quintiles))
        stanData['quintile'] = pointsDf.quintile.values
    elif 'topdec' in stanFn: # just a dummy for the highest decile
        topdecile = getDensityDeciles()[9]
        stanData['L'] = 3
        pointsDf['topdecile'] = pointsDf.density.apply(lambda x: 1 if x>=topdecile else 0)
        stanData['topdecile'] = pointsDf.topdecile.values
    
    print 'Fitting Stan model %s...' %stanFn
    # draws are spread across 10 chains, but thinned to every 5, half used for burn=in. Seed is chosen arbitrarily
    fit = pystan.stan(file=paths['bin']+stanFn+'.stan', data=stanData,model_name=stanFn,iter=nDraws, thin=5, chains=10, seed=5000)
    print 'Done with fit.'

    # add posterior values for frc complete to country dataframe
    countryDf['frcComplete_data']               = np.mean(fit.extract('fc')['fc'], axis=0) 
    countryDf['frcComplete_MRP_insample']       = np.mean(fit.extract('fc_hat')['fc_hat'], axis=0) 
    countryDf['frcComplete_MRP_insample_5pct']  = np.percentile(fit.extract('fc_hat')['fc_hat'], 5, axis=0)  
    countryDf['frcComplete_MRP_insample_95pct'] = np.percentile(fit.extract('fc_hat')['fc_hat'], 95, axis=0) 

    # calculate full distribution of country-level coefficients
    # this is a n x draws matrix for each coefficient, where n is number of countries, and draws is number of draws
    # assumption is that each coefficient matrix is ordered as a polynomial (intercept, linear, squared, cubic, etc. term)
    coeffs = {}
    for coeff in ['beta_psn','beta_bin']:
        coeffs[coeff] = fit.extract(coeff)[coeff]
    
    coeffTable(fit, stanFn)
    # merge in out of sample predictions
    densStd = (pointsDf.density.apply(np.log).mean(),  pointsDf.density.apply(np.log).std())  # to enable standardization
    aggDf = multilevelPrediction(coeffs,id_0ToCid,densStd,stanFn)
    assert 'ALL' not in countryDf.index
    countryDf.loc['ALL','id_0'] = -99 # kludge for world
    countryDf = countryDf.reset_index().set_index('id_0').join(aggDf).set_index('ISOalpha3')
    countryDf.to_pickle(outFn)

    stanPlots(fit, pointsDf, countryDf.drop('ALL'), stanFn)             # plot the mixing and predicted values, etc.
    if stanFn=='multilevel_GT_sq': margEffectsPlots(fit, densStd)       # not implemented for the other specifications
    countryFitStanPlots(pointsDf, id_0ToCid, coeffs, densStd, stanFn)   # diagnostic plots by country

    # clean up - restore original settings
    plt.rcParams = oldrcParams
  
    return countryDf

def getDensityDeciles(forceUpdate=False):
    """Returns the deciles (globally) of density at the grid-cell level
    These are calculated from the Landscan density raster (excluding cells with zero population)
    and are hardcoded here to avoid the user needing the raster installed
    Calculated as  np.percentile(lsarray[lsarray>0],range(0,101,10)) """

    deciles = [1.17,1.37,2.34,3.52,5.08,7.80,12.38,22.14,46.16,147.76,225905.59]
    return deciles

def coeffTable(fit, stanFn):
    """Exports a table of standardized coefficients
    Need to do this as .tex as well"""
    if stanFn!='multilevel_GT_sq': return   # coefficient table only coded for this model (the preferred one)
    
    outFn = paths['output']+stanFn+'_coefficients.pandas'
    
    coeffList = []
    for suf, sufName in [('_bin','Fraction-complete'),('_psn','N-segments')]:
        for betaNum, betaName in [('1', 'intercept'),('2','log-density'),('3','log-density-sq')]:
            posterior = fit.extract('gamma'+betaNum+suf)['gamma'+betaNum+suf]
            pmeans = posterior.mean(axis=0)
            pctiles = np.percentile(posterior, [5,95], axis=0)
            for varNum, varName in enumerate(['intercept']+countryVarNames):
                coeffList.append([sufName,betaName,varName.replace(' ','-'),pmeans[varNum]]+list(pctiles[:,varNum]))
                
    coeffDf = pd.DataFrame(coeffList, columns=['model','coefficient','variable','posterior-mean','5th-percentile','95th-percentile'])
    coeffDf.to_pickle(outFn)

def stanPlots(fit, pointsDf, countryDf,stanFn):
    """Plots of mixing and coefficient distributions, and predicted values"""
    print 'Plotting Stan diagnostics and predictive performance'
    figs = []
    nCoeffs = fit.extract('gamma_psn')['gamma_psn'].shape[2]
    gammaNames = ['gamma'+str(n)+'_'+suf for suf in ['psn','bin'] for n in range(1,nCoeffs+1) ]
    if len(gammaNames)>4:
        for page in range(int(math.ceil(len(gammaNames)/3.))): 
            figs.append(fit.plot(gammaNames[page*3:(page+1)*3]))
    else:
        figs.append(fit.plot(gammaNames))
    
    # Set labels for gamma coefficients
    gammaLabels = ['intercept']+countryVarNames
    plotTitles1 = {'gamma1':'Intercept', 'gamma2':'Log density', 'gamma3':'Log density sq.'}
    plotTitles1b = {'b1':'Intercept', 'b2':'Log density', 'b3':'Log density sq.'}
    plotTitles2 = {'_bin':'Frc complete', '_psn':'N segments'}
    
    for fig in figs:
        for axnum in range(len(fig.get_axes())):
            if 'sq' in stanFn: # change plot labels too
                axTitle = fig.get_axes()[axnum].get_title()
                axTitleNew  = 'Trace plot' if axTitle=='' else 'Country-level: '+plotTitles2[axTitle[-4:]]+'\n'+plotTitles1[axTitle[:6]]
                fig.get_axes()[axnum].set_title(axTitleNew)
            if axnum%2==0: # only do this for left-hand column (odd axes)
                for ii, linelabel in enumerate(gammaLabels):
                    fig.get_axes()[axnum].lines[ii].set_label(linelabel)
                legend = fig.get_axes()[axnum].legend(loc=1)
    
    bnames = ['b'+str(n)+'_'+suf for suf in ['psn','bin'] for n in range(1,nCoeffs+1) ]
    for page in range(int(math.ceil(len(bnames)/3.))):
        figs.append(fit.plot(bnames[page*3:(page+1)*3]))
        if 'sq' in stanFn: # change plot labels too
            for axnum in range(len(figs[-1].get_axes())):
                axTitle = figs[-1].get_axes()[axnum].get_title()
                axTitleNew  = 'Trace plot' if axTitle=='' else 'Grid cell-level: '+plotTitles2[axTitle[-4:]]+'\n'+plotTitles1b[axTitle[:2]]
                figs[-1].get_axes()[axnum].set_title(axTitleNew)
    figs.append(fit.plot(['fc_hat', 'predError_pc']))
    
    # replace _ with - for tex
    for fig in figs:
        for ax in fig.axes:
            if '_' in ax.get_title(): ax.set_title(ax.get_title().replace('_','-'))

    ### PLOTS OF PREDICTED VS SAMPLE VALUES
    Npoints = len(pointsDf)
    pointsDf['predSegs']   = np.mean(fit.extract('totSegs_hat')['totSegs_hat'], axis=0) 
    pointsDf['predFc']     = np.mean(invlogit(fit.extract('theta')['theta']), axis=0) 
    pointsDf['fc']         = pointsDf.NpresentSegs.astype(float) / pointsDf.totSegs
    pointsDf['loggdp']     = countryDf.gdpCapitaPPP_WB.apply(np.log)
    pointsDf['logdensity'] = pointsDf.density.apply(np.log)
    pointsDf['gdpcolor']   = pointsDf.loggdp.apply(lambda x: (x-pointsDf.loggdp.min())*1.0 / (pointsDf.loggdp.max()-pointsDf.loggdp.min()))

    # Plots of predictive performance
    # Colors distinguish countries based on log income
    fig, axes = plt.subplots(3,2)
    scplot = cpblScatter(pointsDf, 'totSegs', 'predSegs', z='gdpcolor', cmap='jet', markersize=5, ax = axes[0,0])
    scplot['markers'].set_edgecolor('face')
    axes[0,0].plot([0,1200],[0,1200])  # 45 degree line
    axes[0,0].set_xlabel('Actual OSM segments')
    axes[0,0].set_ylabel('Predicted OSM segments')
    axes[0,0].set_ylim(bottom=-50)

    scplot = cpblScatter(pointsDf, 'logdensity', 'predSegs', z='gdpcolor', cmap='jet', markersize=5, ax = axes[1,0])
    scplot['markers'].set_edgecolor('face')
    axes[1,0].set_xlabel('Log density')
    axes[1,0].set_ylabel('Predicted OSM segments')
    axes[1,0].set_xlim(left=0)
    axes[1,0].set_ylim(bottom=-50)

    scplot = cpblScatter(pointsDf, 'logdensity', 'predFc', z='gdpcolor', cmap='jet', markersize=5, ax = axes[0,1])
    scplot['markers'].set_edgecolor('face')
    axes[0,1].set_xlabel('Log density')
    axes[0,1].set_ylabel('Predicted frcComplete (points)')

    countryDf['log1minusfrcComplete_data'] = (1-countryDf.frcComplete_data).apply(lambda x: np.nan if x==0 else np.log(x))
    countryDf['log1minusfrcComplete_MRP_outofsample'] = (1-countryDf.frcComplete_MRP_outofsample).apply(lambda x: np.nan if x==0 else np.log(x))
    cpblScatter(countryDf.reset_index(), 'log1minusfrcComplete_data', 'log1minusfrcComplete_MRP_outofsample', cmap='jet', labelfontsize=7, ax = axes[1,1])
    axes[1,1].set_xlabel('log (1-fc) data')
    axes[1,1].set_ylabel('log (1-fc) MRP out of sample')
    axes[1,1].plot([-8,1],[-8,1])
    cpblScatter(countryDf.reset_index(), 'frcComplete_data', 'frcComplete_MRP_outofsample', labels='ISOalpha3', cmap='jet', labelfontsize=6, ax = axes[2,0])
    axes[2,0].set_xlabel('frcComplete_data')
    axes[2,0].set_ylabel('frcComplete_MRP_outofsample')
    
    cpblScatter(countryDf.reset_index(), 'frcComplete_data', 'frcComplete_MRP_insample', labels='ISOalpha3', cmap='jet', labelfontsize=6, ax = axes[2,1])

    for col in range(0,2):
        for row in range(0,3): 
            print axes[row,col].get_ylabel()
            print axes[row,col].get_xlabel()
            if 'frc' in axes[row,col].get_ylabel(): 
                axes[row,col].set_ylim([-0.1,1.1])
            if 'frc' in axes[row,col].get_xlabel(): 
                axes[row,col].set_xlim([-0.1,1.1])
                axes[row,col].plot([0,1],[0,1])
    figs.append(fig)

    ht.compileAllPlots(figs,paths['output']+'stanPlots_'+stanFn)

    # now redo the out of sample plot for production
    fig, ax=plt.subplots(figsize=figSize)
    ax.scatter(countryDf.frcComplete_data.values, countryDf.frcComplete_MRP_outofsample.values, s=10, c=c5s[0])
    ax.set_xlabel('Fraction complete (data)')
    ax.set_ylabel('Fraction complete (multilevel model)')
    ax.plot([0,1],[0,1], c=c5s[0])
    ax.axis('image')  # square plot
    ax.set_xlim([-0.1,1.1])
    ax.set_ylim([-0.1,1.1])
    fig.tight_layout()
    fig.savefig(paths['output']+'fig-frcComplete-prediction.pdf')
    
def margEffectsPlots(fit, densStd):
    """
    Plots marginal effects of country-level coefficients against density
    We plot the marginal effect of a 1-sd change in the coefficient for each point on the density distribution
    And also the baseline effect of all coefficients at their means
    """
    if texAvailable: 
        figureFontSetup(uniform=11)
    else:
        plt.rcParams.update({'font.size': 11, 'ytick.labelsize': 11, 'xtick.labelsize': 11,'legend.fontsize': 11, 'axes.labelsize': 11, 'figure.figsize': (4.6, 4)})

    # extract coefficients. each is a nx5x3 array (n draws, 4 country predictors + intercept, 3 grid-cell coefficients)
    gamma_psn = fit.extract('gamma_psn')['gamma_psn']
    gamma_bin = fit.extract('gamma_bin')['gamma_bin']
    assert gamma_psn.shape[1:] ==  gamma_bin.shape[1:] == (5,3)
    
    # load density raster (this was created in multilevelPrediction)
    print 'Loading rasters from file'
    arrays = np.load(paths['scratch']+'inrasters.npz')
    lsarray = arrays['rast'][:,:,0]  # density only, not gadm
    lsarray = lsarray.reshape(21600*43200)  # so we can apply the function along a single axis

    # calculate predictive impact at each density point
    logdens     = np.arange(np.log(1), np.log(np.nanmax(lsarray)), 0.2)
    logdensStd  = (logdens-densStd[0])/densStd[1]         # standardize the density variable
    plotDf = pd.DataFrame(logdens, columns=['logdens'])

    # calculate distribution of density, to overlay on plot. Note this is approximate because not weighted by area
    densPctiles = np.log(np.percentile(lsarray[lsarray>0], range(0,101)))    

    for cc, cvar in enumerate(['intercept']+countryVars):
        colNames = [pp+ss for pp in ['nSegs_'+cvar, 'fc_'+cvar] for ss in ['','_5pct','_95pct']]
        plotDf = plotDf.join(pd.DataFrame(calcEffect(logdensStd, gamma_psn, gamma_bin, cc).T, columns = colNames))
    
    plt.close('all')
    fig, axes = plt.subplots(1,2, figsize=figSize)
    ax2s = [ax.twinx() for ax in axes]
    plotOrder = [1,3,0,2] # order which variables should be plotted and appear in legend
    varNames = ['intercept'] + [countryVars[ii] for ii in plotOrder]
    legLabels = ['intercept only']+[countryVarNames[ii] for ii in plotOrder]
    for ax,ax2,pref in zip(axes,ax2s,['fc_','nSegs_']):
        for cc, cvar in enumerate(varNames):  
            dfPlotWithEnvelope(plotDf,'logdens', pref+cvar, ylow=pref+cvar+'_5pct',yhigh=pref+cvar+'_95pct', color=c5s[cc], alpha=0.2, label=legLabels[cc], ax=ax)
        ax2.plot(densPctiles, range(0,101), color='0.75')
        ax.plot(0,1,color='0.75',label='Log density CDF')  # to add this to legend
        if 'Segs' in pref:
            ax.set_ylabel('Predicted OSM segments')
            #ax.set_ylim(0,500)
            ax.set_yscale('log')
            ax.legend(loc=4,fontsize=8.25, frameon=False)
            ax.text(-.3, 1, 'B', weight='bold', size=14, transform = ax.transAxes)
        else:
            ax.set_ylabel('Predicted fraction complete')
            ax.set_xticks(np.arange(0,1.1,0.25))
            ax.text(-.3, 1, 'A', weight='bold', size=14, transform = ax.transAxes)
        ax.set_xlim(0,12)
        ax2.set_yticks([])
        ax.set_xticks(range(0,13,4))
        ax.set_xlabel('Log density')
    fig.tight_layout()
    figFn = paths['output']+'fig-marginalEffects.pdf'
    fig.savefig(figFn)
    print 'Saved marginal effects figure to %s' % figFn
    
def calcEffect(logdensStd, gamma_psn, gamma_bin, coeffNum, sdIncrease=True):
    """Calculate effect of a given gamma coefficient for each point on the density distribution
    Effect is the intercept plus a 1 sd DECREASE in the coefficient, evaluated at means of country-level variables
    Returns a 6xn array, with mean/5pc/95pc for each of bin and psn, and n being the number of density points evaluated at"""
    
    # note that we plot the intercept + coefficient, but ignore the uncertainty in the intercept
    sign = 1 if sdIncrease else -1
    if coeffNum==0:
        segsCoeffs = gamma_psn[:,0,:]
        fcCoeffs   = gamma_bin[:,0,:]
    else:
        segsCoeffs = np.mean(gamma_psn[:,0,:],axis=0) + sign*gamma_psn[:,coeffNum,:]
        fcCoeffs   = np.mean(gamma_bin[:,0,:],axis=0) + sign*gamma_bin[:,coeffNum,:]    
    assert segsCoeffs.shape[1] == fcCoeffs.shape[1] == 3

    nSegs  = np.exp(np.polynomial.polynomial.polyval(logdensStd, segsCoeffs.T, True))
    fc    = invlogit(np.polynomial.polynomial.polyval(logdensStd, fcCoeffs.T, True))
    nSegsPctiles = np.percentile(nSegs,[5,95],axis=0)
    fcPctiles    = np.percentile(fc,[5,95], axis=0)
    
    return np.vstack([np.mean(nSegs, axis=0), nSegsPctiles, np.mean(fc, axis=0), fcPctiles])

def countryFitStanPlots(pointsDf,id_0ToCid,coeffs,densStd,stanFn):
    """Country-specific plots of Stan fit. Returns list of full-page figures"""
    print 'Compiling plots of country-level fit'
    assert pointsDf.index.name=='ISOalpha3'

    countryISOs = pointsDf.index.unique()
    iso2name = ht.country2ISOLookup()
    pointsDf['logdensity'] = pointsDf.density.apply(np.log)
    pointsDf['frcComplete']=pointsDf.NpresentSegs.astype(float)/pointsDf.totSegs
    
    # density range in standardized terms
    plotDensities = np.exp(range(12))
    quintiles = None if not('cat' in stanFn) else [getDensityDeciles()[ii] for ii in [0,2,4,6,8,10]]
    topdecile = None if not('topdecile' in stanFn) else getDensityDeciles()[9]
    
    plt.close('all')
    page = -1
    figs = []
    for cc, iso in enumerate(countryISOs):    
        if cc%24==0: # new page
            if cc>0: figs+=[fig]
            page+=1
            fig, axes = plt.subplots(6, 4, figsize=figSizePage)
        row = int((cc-24*page)/4)
        col = cc-24*page-4*row  

        cpblScatter(pointsDf.ix[iso], 'logdensity', 'frcComplete', ax=axes[row, col])
        
        # plot estimated functional form
        id_0 = pointsDf.id_0[iso].unique()
        inPredict  = np.vstack([plotDensities,id_0.repeat(len(plotDensities))])
        outPredict = np.apply_along_axis(predictCell,0,inPredict,coeffs,densStd,id_0ToCid,quintiles,topdecile) 
        predictDf = pd.DataFrame(np.vstack([inPredict,outPredict]).T, columns=['density','id_0','segs','segs_5pct','segs_95pct','fc','fc_5pct','fc_95pct'])
        predictDf['logdensity'] = predictDf.density.apply(np.log)
        dfPlotWithEnvelope(predictDf,'logdensity','fc','fc_5pct','fc_95pct', ax=axes[row, col])
        
        axes[row,col].set_title(iso2name['ISOalpha2shortName'][iso].replace('&',' and '), fontsize=9)  
        axes[row,col].set_ylim(-0.05,1.05)
        axes[row,col].set_xticks([0,5,10])
        axes[row,col].set_yticks([0,0.5,1])
        
        # add coefficients to plot (posterior mean)
        cs = np.mean(coeffs['beta_bin'][:,id_0ToCid[id_0[0]]-1,:], axis=0)
        cs = '\n'.join(['b%d: %.2f'%(ii,cc) for ii, cc in enumerate(cs)])
        axes[row,col].text(-0.5, 0.1, cs, fontsize=4)
            
        if row==5: 
            axes[row,col].set_xticklabels([0,5,10],fontsize=8)
            axes[row,col].set_xlabel('Log density', size=8)
        else:
            axes[row,col].set_xticklabels([])
            
        if col==0:            
            axes[row,col].set_yticklabels([0,0.5,1],size=8)   
            axes[row,col].set_ylabel('Frc complete', size=8)
        else:
            axes[row,col].set_yticklabels([])
            axes[row,col].set_ylabel('')
    
    figs+=[fig]
    ht.compileAllPlots(figs, paths['output']+'stanPlots_countryFit_'+stanFn)

def multilevelPrediction(coeffs,id_0ToCid,densStd,stanFn,forceUpdate=False,parallel=True):
    """Given a multilevel model, calculate the out-of-sample prediction for each grid cell"""
    print 'Doing out of sample predictions for whole world'

    # Load rasters for countries and density
    rastFn = paths['scratch']+'inrasters.npz'
    if not(os.path.exists(rastFn) or forceUpdate):
        print 'Loading rasters from PostGIS'
        rc = ht.rasterCon('landscan')
        lsarray = rc.asarray(bands=3, padEmptyRows=True) # band 3 is density
        rc = ht.rasterCon('gadmraster')
        gaarray = rc.asarray(bands=1, padEmptyRows=True) # band 1 is id_0
        inRasters = np.dstack([lsarray, gaarray])
        assert inRasters.shape==(21600, 43200, 2)
        np.savez_compressed(rastFn, rast=inRasters)
    else:
        print 'Loading rasters from file'
        arrays = np.load(rastFn)
        inRasters = arrays['rast']
    inRasters = inRasters.reshape(21600*43200,2)  # so we can apply the function along a single axis

    # exclude missing data (mainly oceans) and zero density (because we take the log). This can be done inside predictCell, but is faster here
    rasterMask = np.logical_not(np.isnan(inRasters[:,0]) | np.isnan(inRasters[:,1]) | (inRasters[:,0]==0))
    inRasters_masked = inRasters[rasterMask]

    print 'Predicting array values' # apply predictCell to each pair of (density, id_0) values
    quintiles = None if not('cat' in stanFn) else [getDensityDeciles()[ii] for ii in [0,2,4,6,8,10]]
    topdecile = None if not('topdec' in stanFn) else getDensityDeciles()[9]
    
    if parallel:
        nSlices = 60  # arbitrary, just to cut up the array for parallelization
        sliceLen = int(inRasters_masked.shape[0]/nSlices)
    
        funclol, names=[],[]
        for slice in range(0,nSlices):
            funclol+=[[np.apply_along_axis,[predictCell,1,inRasters_masked[slice*sliceLen:(slice+1)*sliceLen],coeffs,densStd,id_0ToCid,quintiles,topdecile]]]
            names+=['predict_slice'+str(slice)]
        if inRasters_masked.shape[0]%nSlices>0: # do the remainder
            funclol+=[[np.apply_along_axis,[predictCell,1,inRasters_masked[(slice+1)*sliceLen:], coeffs,densStd,id_0ToCid,quintiles,topdecile]]]
            names+=['predict_slice'+str(slice+1)]
        predRasters = np.vstack(runFunctionsInParallel(funclol, names=names, parallel=True, maxAtOnce=30)[1])
    else: 
        predRasters = np.apply_along_axis(predictCell,1,inRasters_masked,coeffs,densStd,id_0ToCid,quintiles,topdecile)  
    assert predRasters.shape[0] == inRasters_masked.shape[0]
    
    # Aggregate by country. Easiest to do this in pandas
    print 'Aggregating predictions to country...'
    suffs = ['_MRP_outofsample','_MRP_outofsample_5pct','_MRP_outofsample_95pct']
    predDf = pd.DataFrame(predRasters, columns=[pref+suff for pref in ['totSegs','frcComplete'] for suff in suffs])
    predDf['id_0'] = inRasters_masked[:,1]
    predDf.set_index('id_0',inplace=True)
    for suff in suffs:
        predDf['osmSegs'+suff] = predDf['totSegs'+suff]*predDf['frcComplete'+suff]
    
    aggDf = predDf.groupby(level=0)[[cc for cc in predDf.columns if 'Segs' in cc]].sum()
    # add the whole world, and give it id0=-99
    assert -99 not in aggDf.index
    aggDf.loc[-99] = predDf[[cc for cc in predDf.columns if 'Segs' in cc]].sum()
    for suff in suffs:  # weighted average
        aggDf['frcComplete'+suff] = aggDf['osmSegs'+suff].astype(float)/aggDf['totSegs'+suff]
    
    # Save the expanded numpy array as an image
    # http://stackoverflow.com/questions/10965417/how-to-convert-numpy-array-to-pil-image-applying-matplotlib-colormap
    print 'Saving prediction maps...'
    predRastersFull = np.full((inRasters.shape[0],6), np.nan)
    predRastersFull[rasterMask] = predRasters
    predRastersFull = predRastersFull.reshape(21600, 43200, 6)
    predRastersFull[:,:,0][inRasters[:,0].reshape(21600, 43200)==0] = 0 # replace places with zero density (np.nan) with 0
    ht.arrayToImage(predRastersFull[:,:,0], mpl.cm.Blues, paths['output']+'predictedSegs_map_'+stanFn+'.png',legTitle='number of segments',useLog='plus1',ndColor = [255,255,255,1])
    alphas = np.log(predRastersFull[:,:,0])
    alphas[np.isinf(alphas)]=np.nan  # where population is zero
    alphas[alphas<1]=1 # bottom code lower part of distribution - we don't care whether there are 0.005 or 1.5 roads
    alphas[alphas>3]=3 # top code at 3 (i.e. log 20)
    amin, amax = np.nanmin(alphas), np.nanmax(alphas)
    alphas = (alphas-amin)/np.float((amax-amin))*0.8 + 0.2 # only use the top 80% of the alpha range
    ht.arrayToImage(predRastersFull[:,:,3], mpl.cm.cool, paths['output']+'predictedFrcComplete_map_'+stanFn+'.png',legTitle='fraction complete',ndColor = [255,255,255,1], alphas = alphas)
    
    return aggDf

def predictCell(incell,coeffs,densStd,id_0ToCid,quintiles,topdecile):
    """calculate mean, 5th and 95th percentile for N segs and density   
    Passing quintiles gives the Stan model with density quintiles
    Passing topdecile gives the Stan model with a dummy variables for the top decile
    Otherwise,gives the polynomial fit"""
    
    assert quintiles is None or topdecile is None
    logdensStd = (np.log(incell[0])-densStd[0])/densStd[1]   # standardize
    try:
        cid = id_0ToCid[incell[1]]-1  # minus 1 because cid index is 1-based, coefficient array is 0-based   
    except:
        return np.array([np.nan]*6)  # country is not in groundtruth dataset

    if quintiles is not None:
        cellQuintile = np.digitize(incell[0], quintiles)
        nSegs = np.exp(coeffs['beta_psn'][:,cid,cellQuintile-1].T + logdensStd * coeffs['beta_psn'][:,cid,5])
        fc  = invlogit(coeffs['beta_bin'][:,cid,cellQuintile-1].T + logdensStd * coeffs['beta_bin'][:,cid,5])
    elif topdecile is not None:
        cellTopDecile = 1 if incell[0]>=topdecile else 0
        nSegs = np.exp(coeffs['beta_psn'][:,cid,0].T + cellTopDecile*coeffs['beta_psn'][:,cid,1].T + logdensStd * coeffs['beta_psn'][:,cid,2])
        fc  = invlogit(coeffs['beta_bin'][:,cid,0].T + cellTopDecile*coeffs['beta_bin'][:,cid,1].T + logdensStd * coeffs['beta_bin'][:,cid,2])
    else:
        nSegs = np.exp(np.polynomial.polynomial.polyval(logdensStd,coeffs['beta_psn'][:,cid,:].T, True))
        fc  = invlogit(np.polynomial.polynomial.polyval(logdensStd,coeffs['beta_bin'][:,cid,:].T, True))
    nSegsPctiles = np.percentile(nSegs,[5,95])
    fcPctiles    = np.percentile(fc,[5,95])
    return np.array([np.mean(nSegs), nSegsPctiles[0], nSegsPctiles[1], np.mean(fc), fcPctiles[0], fcPctiles[1]])

def reportStanFits():
    """Calculates the bias from each fit"""
    stanFns    = [paths['working']+'countriesFit_'+sm.replace('.stan','')+'.pickle' for sm in stanModels]
    if not all([os.path.exists(sFn) for sFn in stanFns] ):
        raise Exception('Estimate all your models first before running reportStanFits')
    
    countryDf = aggregateVisualAssessmentPoints()
    meanErrors, MSEs = [], []
    for sFn in stanFns:
        df = pd.read_pickle(sFn)
        df = df.join(countryDf.frcComplete_visual)
        assert all(df.drop('ALL').frcComplete_data.round(8)==df.drop('ALL').frcComplete_visual.round(8))  # check that Stan returned countries in the right order
        meanErrors.append((df.frcComplete_MRP_outofsample-df.frcComplete_data).mean())
        MSEs.append(((df.frcComplete_MRP_outofsample-df.frcComplete_data)**2).mean())
    stanDf = pd.DataFrame(zip(meanErrors,MSEs), index=stanModels, columns=['meanError','MSE'])
    stanDf.sort_values('meanError',inplace=True)
    
    print stanDf

def stdize(adf, takelog=False, byDf=None):
    """Standardizes a covariate; Stan finds it hard when the covariate scale varies by an order of magnitude
    If byDf is another array, standardizes by that array instead (which was needed if we want Stan to do out-of-sample prediction....which we don't anymore."""
    if byDf is None: byDf = adf
    if takelog: 
        adf  = np.log(adf)
        byDf = np.log(byDf)
    stdArray = (adf - byDf.mean()) / byDf.std()
    return stdArray.values if isinstance(stdArray, pd.DataFrame) else stdArray

def invlogit(x):
    return 1. / (1 + np.exp(x*-1))

if __name__ == '__main__':
    runmode=None if len(sys.argv)<2 else sys.argv[1].lower()
    forceUpdate = any(['forceupdate' in arg.lower() for arg in sys.argv])
    print 'forceUpdates is %s' % forceUpdate    
    if runmode in [None,'stan']:
        df = multilevelModel(stanFn=None, forceUpdate=forceUpdate)
        #df = multilevelModel(forceUpdate=forceUpdate)
        reportStanFits()