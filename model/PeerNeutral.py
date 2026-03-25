#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 10:55:40 2021

@author: yitao Hu
"""
from tqdm import tqdm
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from scipy.optimize import lsq_linear
from jqdatasdk import * 
from statsmodels.distributions.empirical_distribution import ECDF
auth("13764432461", "Nfhq12345")

def cpt_raw_rtn(all_raw_factors,days_before = 800):
    """Compute raw returns for each day during the speciied time period for specified code"""
    # normalise stock code
    norm_code = normalize_code(list(all_raw_factors.ts_code))
    all_raw_factors['code'] = norm_code
    # turn trade_date into date format 
    all_raw_factors['trade_date'] = pd.to_datetime(all_raw_factors['trade_date'].astype(str))
    
    
    # specify first and last data date 
    first_data_date = all_raw_factors['trade_date'].min() - datetime.timedelta(days = days_before)
    last_data_date = all_raw_factors['trade_date'].max()
    print('Data Starts:',first_data_date,'Data Ends:',last_data_date)
    
    # import prices data from jq
    # import price data for all stocks in the universe, note we fill paused with Na
    price_ts = get_price(list(np.unique(norm_code)), start_date = first_data_date,
                                  end_date= last_data_date + datetime.timedelta(days = 10),
                                  frequency='daily', 
                                  fields=['close'],fq='pre',fill_paused = False)
    # change the colnames
    price_ts.columns = ['trade_date','code','close']
    
   

    
    # compute daily returns for each stock
    rtn_ts = price_ts.set_index(['trade_date','code']).unstack().pct_change()[1:]
    rtn_ts.columns = rtn_ts.columns.get_level_values(1)

    
    
    return rtn_ts

def cpt_ex_rtn(all_raw_factors, bench_code = ['000905.XSHG'],days_before = 800):
    """Compute excess returns for each day during the speciied time period for specified code"""
    # normalise stock code
    norm_code = normalize_code(list(all_raw_factors.ts_code))
    all_raw_factors['code'] = norm_code
    # turn trade_date into date format 
    all_raw_factors['trade_date'] = pd.to_datetime(all_raw_factors['trade_date'].astype(str))
    
    
    # specify first and last data date 
    first_data_date = all_raw_factors['trade_date'].min() - datetime.timedelta(days = days_before)
    last_data_date = all_raw_factors['trade_date'].max()
    print('Data Starts:',first_data_date,'Data Ends:',last_data_date)
    
    # import prices data from jq
    # import price data for all stocks in the universe, note we fill paused with Na
    price_ts = get_price(list(np.unique(norm_code)), start_date = first_data_date,
                                  end_date= last_data_date + datetime.timedelta(days = 10),
                                  frequency='daily', 
                                  fields=['close'],fq='post',fill_paused = False)
    # change the colnames
    price_ts.columns = ['trade_date','code','close']
    # import stock index data to compute excess return 
    # import price data 
    index_ts = get_price(['000905.XSHG'], start_date = first_data_date,
                                  end_date=last_data_date,
                                  frequency='daily', 
                                  fields=['close'],fq='pre')
    # # change the colnames
    index_ts.columns = ['trade_date','code','close']
    # compute index returns
    index_ts = index_ts.drop('code',axis = 1).set_index('trade_date')
    index_ts = index_ts.pct_change()[1:]
    index_ts.columns = ['bench_rtn']
    index_ts = index_ts.reset_index()

    
    # compute daily returns for each stock
    rtn_ts = price_ts.set_index(['trade_date','code']).unstack().pct_change()[1:].stack()
    rtn_ts.columns =['rtn']
    rtn_ts = rtn_ts.reset_index()
    # merge bench rtn
    rtn_ts = rtn_ts.merge(index_ts,on = 'trade_date',how ='left')
    # compute excess rtn 
    rtn_ts['ex_rtn'] = rtn_ts.rtn - rtn_ts.bench_rtn
    # grab useful col
    exrtn_ts = rtn_ts[['trade_date','code','ex_rtn']]
    # unstack the exrtn ts 
    exrtn_ts = exrtn_ts.set_index(['trade_date','code']).unstack()
    # reset the cols 
    exrtn_ts.columns = exrtn_ts.columns.get_level_values(1)
    import pdb
    pdb.set_trace()
    return exrtn_ts


def cpt_ex_rtn_from_data(all_raw_factors, daily_close_paused_data, index_close_data, code_price_name, index_price_name, pause_flag_name):
    """Compute excess returns for each day during the speciied time period for specified code"""
    all_raw_factors = all_raw_factors.reset_index()
    daily_close_paused_data = daily_close_paused_data.reset_index()
    index_close_data = index_close_data.reset_index()
    codes = all_raw_factors['code'].unique()
    daily_close_paused_data = daily_close_paused_data[daily_close_paused_data.code.map(lambda x: x in codes)]
    daily_close_paused_data.sort_values(['code', 'trade_date'], inplace=True)
    daily_close_paused_data['pct_chg'] = daily_close_paused_data.groupby("code")[code_price_name].apply(lambda x: x.pct_change())
    daily_close_paused_data = daily_close_paused_data[daily_close_paused_data['pct_chg'].notnull()]
    index_close_data.sort_values('trade_date', inplace=True)
    index_close_data['pct_chg_index'] = index_close_data[index_price_name].pct_change()
    index_close_data = index_close_data[index_close_data['pct_chg_index'].notnull()]
    daily_close_paused_data = pd.merge(daily_close_paused_data, index_close_data[['trade_data', 'pct_chg_index']], how='left', on='trade_date')
    # daily_close_paused_data['alpha'] = daily_close_paused_data['pct_chg'] - daily_close_paused_data['pct_chg_index']
    daily_close_paused_data['alpha'] = daily_close_paused_data.apply(lambda x: None if x[pause_flag_name] != 0 else x['pct_chg'] - x['pct_chg_index'], axis=1)
    exrtn_ts = daily_close_paused_data.set_index(['trade_date', 'code'])['alpha'].unstack()
    # exrtn_ts = exrtn_ts.fillna(0)
    return exrtn_ts


def _cpt_cross_peer_corr(roll_ex,n_peer_group = 10 ,corr_type = 'pearson'):
    
    """
    Compute the excess return pair-wise correlation for each stock's n most 
        correlated peers
        Args:
        ----------
            roll_ex: pd.DataFrame 
                Window length by N Matrix where:
                    The index is the timestep to estimate correlation matrix
                    The column names is the stock code 
                    The value is the excess return for each stock
                    - Example:
                ex_rtn                                                  
                code       000006.XSHE 000008.XSHE 000009.XSHE 000012.XSHE 000021.XSHE   
                trade_date                                                               
                2015-01-05   -0.012642   -0.035754         NaN    0.007970    0.017267   
                2015-01-06   -0.043587   -0.005478         NaN   -0.008023    0.025019   
                2015-01-07    0.000210    0.020783         NaN   -0.017572    0.012058   
                2015-01-08   -0.004099    0.039800    0.010532    0.020735    0.025480   
                2015-01-09   -0.005877    0.012195   -0.022005   -0.018619   -0.017290   
                            
             n_peer_group : int 
                 The number of stocks in the peer group
             corr_type: {'pearson', 'kendall', 'spearman'} or callable
                 correlation type -- default is pearson correlation
                * pearson : standard correlation coefficient
                * kendall : Kendall Tau correlation coefficient
                * spearman : Spearman rank correlation
        Returns:
        ----------
            cross_peers : pd.DataFrame
                n_peer_group * N by 3 matrix
                    where the first col is the stock code
                              second col is its peer code
                              third col is its correlation 
                - Example:
                               code         peer      corr
                        0  000006.XSHE  001914.XSHE  0.388203
                        1  000006.XSHE  600820.XSHG  0.384117
                        2  000006.XSHE  000027.XSHE  0.354779
                        3  000006.XSHE  600284.XSHG  0.354304
                        4  000006.XSHE  000537.XSHE  0.347396
    """
    # compute the sample corr mat and drop all nas 
    cross_corr = roll_ex.corr().dropna(how = 'all',axis =1).dropna(how = 'all',axis =0).stack()
    # rename the index
    cross_corr.index = cross_corr.index.set_names(['code1','code2'],level= [0,1])
    # reset the index 
    cross_corr = cross_corr.reset_index()
    # rename the columns
    cross_corr.columns = list(cross_corr.columns[:-1]) + ['corr']
    # get n_peer with largest correlation 
    cross_peers = cross_corr.groupby('code1').\
    apply(lambda corr: corr.set_index('code2').\
          nlargest(n_peer_group+1,'corr')['corr']).reset_index()
    # change the col names 
    cross_peers.columns = ['code','peer','corr']
    return cross_peers


def _group_demean(cross_alpha,cross_peers,universe):
    """Demean every stock by its peers at one timestep"""
    # initialize group means 
    group_mean = pd.DataFrame(data = np.zeros((len(universe),2)),index=universe,columns = ['group_mean','group_std'])
    
    for stock in list(cross_alpha.index):
        # find the intersection of peers and universe
        valid_peers = list(set(cross_peers.loc[stock].peer.values) & set(universe))
        # compute the group mean and group std
        group_mean.loc[stock,'group_mean'] = cross_alpha.loc[valid_peers].mean()[0]
        group_mean.loc[stock,'group_std'] = cross_alpha.loc[valid_peers].std()[0]
        
    # merge group mean and compute demeand alpha
    cross_alpha = cross_alpha.merge(group_mean,left_index=True,right_index=True)
    cross_alpha['zscore_alpha'] = (cross_alpha.iloc[:,0] - cross_alpha.group_mean)/cross_alpha.group_std
    cross_alpha = cross_alpha.reset_index()
    cross_alpha.columns = ['code'] + list(cross_alpha.columns[1:])
    return cross_alpha





def peers_demean(alpha_blend,
                   exrtn_ts,
                   Cal_day_window = 365,
                   n_peer_group = 10,
                   corr_type = 'pearson'):
    """
    Find the most correlated n peers for each stock at each time step,
    Note the timestep of exrtn must cover the timestep of alpha_blend, padded by
        window size
    The real peer is defined as the intersection of n_peer_group and the asset universe
    Then, demean the alpha score with each group. 
        Args:
        ----------
            alpha_blend: MultiIndex pd.DataFrame 
                Where the level 0 index is the timestep 
                     the level 1 index is the stock code
                     the value is the aggreated alpha score from the model
                    - Example:
                                            long_time_rolling_factor
                    trade_date code                                 
                    2016-01-05 000543.XSHE                  0.011816
                               000600.XSHE                  0.013183
                               000685.XSHE                  0.010656
                               000690.XSHE                  0.009828
                               000939.XSHE                  0.010814
                    ...                                          ...
                    2021-11-02 601198.XSHG                  0.003886
                               601456.XSHG                  0.005811
                               601577.XSHG                  0.004381
                               601860.XSHG                  0.003416
                               601997.XSHG                  0.002829 
             exrtn_ts: pd.DataFrame
                 wide excess return matrix
                     where the index is the time step and the column is the 
                     stock code 
                - Example:
                    code        000006.XSHE  000008.XSHE  000009.XSHE  000012.XSHE  000021.XSHE  
                    trade_date                                                                    
                    2015-01-05    -0.012642    -0.035754          NaN     0.007970     0.017267   
                    2015-01-06    -0.043587    -0.005478          NaN    -0.008023     0.025019   
                    2015-01-07     0.000210     0.020783          NaN    -0.017572     0.012058   
                    2015-01-08    -0.004099     0.039800     0.010532     0.020735     0.025480   
                    2015-01-09    -0.005877     0.012195    -0.022005    -0.018619    -0.017290 
                    
             Cal_day_window: int
                 the number of calender days look back, default is 365 or one year 
             n_peer_group : int 
                 The number of stocks in the peer group
             corr_type: {'pearson', 'kendall', 'spearman'} or callable
                 correlation type -- default is pearson correlation
                * pearson : standard correlation coefficient
                * kendall : Kendall Tau correlation coefficient
                * spearman : Spearman rank correlation
        Returns:
        ----------
            alpha_demeaned  : pd.DataFrame
                melten pd Dataframe with peer demeaned alpha score and group mean
                
                - Example:
                          code  long_time_rolling_factor trade_date  group_mean  demean_alpha
                0  000543.XSHE                  0.011816 2016-01-05    0.009996      0.001820
                1  000600.XSHE                  0.013183 2016-01-05    0.011076      0.002107
                2  000685.XSHE                  0.010656 2016-01-05    0.009875      0.000781
                3  000690.XSHE                  0.009828 2016-01-05    0.010929     -0.001101
                4  000939.XSHE                  0.010814 2016-01-05    0.008956      0.001858
    """
    # set up the for loop 
    timesteps = pd.to_datetime(np.unique(alpha_blend.index.get_level_values(0)))
    # initialized the alpha_demeaned
    alpha_demeaned = pd.DataFrame(columns=['code']+list(alpha_blend.columns)+\
                                  ['trade_date','group_mean','zscore_alpha'])
    for timestep in tqdm(timesteps):
        # get cross-sectional alpha score
        cross_alpha = alpha_blend.loc[timestep].reset_index().set_index('code')
        cross_alpha['trade_date'] = timestep
        # define the universe
        universe = list(cross_alpha.index)
        # carve out one year rolling data 
        roll_ex = exrtn_ts[timestep - datetime.timedelta(days = Cal_day_window):timestep][universe]
        # compute cross-sectional peer at each factor update date
        cross_peers = _cpt_cross_peer_corr(roll_ex,n_peer_group,corr_type)
        cross_peers['trade_date'] = timestep
        cross_peers = cross_peers.set_index('code')
        cross_demeaned_alpha = _group_demean(cross_alpha,cross_peers,universe)
        alpha_demeaned = pd.concat([alpha_demeaned,cross_demeaned_alpha])

    return alpha_demeaned
        
def _cpt_cross_peer_corr_smooth(roll_ex1,
                                roll_ex2,
                                roll_ex3,
                                n_peer_group = 10 ,
                                corr_type = 'pearson'):
    
    """
    Compute the excess return pair-wise correlation for each stock's n most 
        correlated peers
        Args:
        ----------
            roll_ex: pd.DataFrame 
                Window length by N Matrix where:
                    The index is the timestep to estimate correlation matrix
                    The column names is the stock code 
                    The value is the excess return for each stock
                    - Example:
                ex_rtn                                                  
                code       000006.XSHE 000008.XSHE 000009.XSHE 000012.XSHE 000021.XSHE   
                trade_date                                                               
                2015-01-05   -0.012642   -0.035754         NaN    0.007970    0.017267   
                2015-01-06   -0.043587   -0.005478         NaN   -0.008023    0.025019   
                2015-01-07    0.000210    0.020783         NaN   -0.017572    0.012058   
                2015-01-08   -0.004099    0.039800    0.010532    0.020735    0.025480   
                2015-01-09   -0.005877    0.012195   -0.022005   -0.018619   -0.017290   
                            
             n_peer_group : int 
                 The number of stocks in the peer group
             corr_type: {'pearson', 'kendall', 'spearman'} or callable
                 correlation type -- default is pearson correlation
                * pearson : standard correlation coefficient
                * kendall : Kendall Tau correlation coefficient
                * spearman : Spearman rank correlation
        Returns:
        ----------
            cross_peers : pd.DataFrame
                n_peer_group * N by 3 matrix
                    where the first col is the stock code
                              second col is its peer code
                              third col is its correlation 
                - Example:
                               code         peer      corr
                        0  000006.XSHE  001914.XSHE  0.388203
                        1  000006.XSHE  600820.XSHG  0.384117
                        2  000006.XSHE  000027.XSHE  0.354779
                        3  000006.XSHE  600284.XSHG  0.354304
                        4  000006.XSHE  000537.XSHE  0.347396
    """
    # define the minimal number of periods to be at least half of roll excess return 
    min_periods_nums = [int(roll_ex1.shape[0]/2),int(roll_ex2.shape[0]/2),int(roll_ex3.shape[0]/2)]
    
    # compute the sample corr mat and drop all nas 
    cross_corr1 = pd.DataFrame(roll_ex1.corr(min_periods=min_periods_nums[0]).fillna(0).stack())
    cross_corr2 = pd.DataFrame(roll_ex2.corr(min_periods=min_periods_nums[1]).fillna(0).stack())
    cross_corr3= pd.DataFrame(roll_ex3.corr(min_periods=min_periods_nums[2]).fillna(0).stack())
    # compute the mean of three correlation
    cross_corr = cross_corr1.merge(cross_corr2,left_index=True,right_index=True).\
        merge(cross_corr3,left_index=True,right_index=True).mean(axis =1)
    # rename the index
    cross_corr.index = cross_corr.index.set_names(['code1','code2'],level= [0,1])
    # reset the index 
    cross_corr = cross_corr.reset_index()
    # rename the columns
    cross_corr.columns = list(cross_corr.columns[:-1]) + ['corr']
    # get n_peer with largest correlation 
    cross_peers = cross_corr.groupby('code1').\
    apply(lambda corr: corr.set_index('code2').\
          nlargest(n_peer_group+1,'corr')['corr']).reset_index()
    # change the col names 
    cross_peers.columns = ['code','peer','corr']
    return cross_peers


def dyna_peer_cutoff(code2peers,corr_cutoff = None,n_group_cutoff = 10,quantile = 0.5):
    """
    Add corr cutoff and lowest num_group cutoff 

    Parameters
    ----------
    code2peers : pd.DataFrame
        uncut code2peers mapping.
    corr_cutoff : float, optional
        corr cut. The default is 0.3.
    quantile: float, optional
        quantile cut. The default is 0.5.
    n_group_cutoff : TYPE, optional
        num_group_cut. The default is 10.

    Returns
    -------
    code2peers: pd.DataFrame
        peer mapping after cutting.

    """
    # get rid of non_tradeable stocks 
    if quantile:
        corr_cutoff = code2peers.quantile(q= quantile).values[0]
    non_tradeable_mask = (code2peers.code == code2peers.peer) & (code2peers['corr']<1.0)
    non_tradeable_stock_list = code2peers[non_tradeable_mask].code.to_list()
    code2peers = code2peers[~code2peers.code.isin(non_tradeable_stock_list)]
    # build the mask
    low_cutoff_mask = (code2peers['corr']<corr_cutoff)
    group_count = code2peers[~low_cutoff_mask].groupby('code')[['corr']].count().sort_values('corr')['corr']
    small_group_target = group_count[group_count<n_group_cutoff].index.to_list()
    small_group_mask = code2peers[~low_cutoff_mask].code.isin(small_group_target)
    
    return code2peers[~low_cutoff_mask][~small_group_mask]


def reverse_corr_cut(one_day_corr,n_peer = 100,n_peer_dec = 10):
    """
    cut the peer size reversely, leave less peer size for highly correlated groups
    """
    mean_group_corr = one_day_corr.groupby('code')['corr'].mean().reset_index().sort_values(['corr'])
    mapping_aft_cut = []
    n_target = mean_group_corr.shape[0]

    decile_size = round(n_target/10)
    for i in range(10):
        decile_code_list = mean_group_corr.iloc[i*decile_size :(i+1)*decile_size].code.to_list()
        corr_aft_cut = one_day_corr[one_day_corr.code.isin(decile_code_list)]
        corr_aft_cut = corr_aft_cut.groupby('code',as_index = False)\
        .apply(lambda x: x.nlargest(n_peer+1,'corr')).reset_index(drop = True)
        mapping_aft_cut.append(corr_aft_cut)
        n_peer = n_peer - n_peer_dec
    one_day_corr_aft_cut = pd.concat(mapping_aft_cut) 
    return one_day_corr_aft_cut

def peers_demean_smoothed(alpha_blend,
                   exrtn_ts,
                   Cal_day_window = [30,90,365],
                   n_peer_group = 10,
                   corr_type = 'pearson'):
    """
    Find the most correlated n peers for each stock at each time step,
    Note the timestep of exrtn must cover the timestep of alpha_blend, padded by
        window size
    The real peer is defined as the intersection of n_peer_group and the asset universe
    Then, demean the alpha score with each group. 
        Args:
        ----------
            alpha_blend: MultiIndex pd.DataFrame 
                Where the level 0 index is the timestep 
                     the level 1 index is the stock code
                     the value is the aggreated alpha score from the model
                    - Example:
                                            long_time_rolling_factor
                    trade_date code                                 
                    2016-01-05 000543.XSHE                  0.011816
                               000600.XSHE                  0.013183
                               000685.XSHE                  0.010656
                               000690.XSHE                  0.009828
                               000939.XSHE                  0.010814
                    ...                                          ...
                    2021-11-02 601198.XSHG                  0.003886
                               601456.XSHG                  0.005811
                               601577.XSHG                  0.004381
                               601860.XSHG                  0.003416
                               601997.XSHG                  0.002829 
             exrtn_ts: pd.DataFrame
                 wide excess return matrix
                     where the index is the time step and the column is the 
                     stock code 
                - Example:
                    code        000006.XSHE  000008.XSHE  000009.XSHE  000012.XSHE  000021.XSHE  
                    trade_date                                                                    
                    2015-01-05    -0.012642    -0.035754          NaN     0.007970     0.017267   
                    2015-01-06    -0.043587    -0.005478          NaN    -0.008023     0.025019   
                    2015-01-07     0.000210     0.020783          NaN    -0.017572     0.012058   
                    2015-01-08    -0.004099     0.039800     0.010532     0.020735     0.025480   
                    2015-01-09    -0.005877     0.012195    -0.022005    -0.018619    -0.017290 
                    
             Cal_day_window: int
                 the number of calender days look back, default is 365 or one year 
             n_peer_group : int 
                 The number of stocks in the peer group
             corr_type: {'pearson', 'kendall', 'spearman'} or callable
                 correlation type -- default is pearson correlation
                * pearson : standard correlation coefficient
                * kendall : Kendall Tau correlation coefficient
                * spearman : Spearman rank correlation
        Returns:
        ----------
            alpha_demeaned  : pd.DataFrame
                melten pd Dataframe with peer demeaned alpha score and group mean
                
                - Example:
                          code  long_time_rolling_factor trade_date  group_mean  demean_alpha
                0  000543.XSHE                  0.011816 2016-01-05    0.009996      0.001820
                1  000600.XSHE                  0.013183 2016-01-05    0.011076      0.002107
                2  000685.XSHE                  0.010656 2016-01-05    0.009875      0.000781
                3  000690.XSHE                  0.009828 2016-01-05    0.010929     -0.001101
                4  000939.XSHE                  0.010814 2016-01-05    0.008956      0.001858
    """
    # set up the for loop 
    timesteps = pd.to_datetime(np.unique(alpha_blend.index.get_level_values(0)))
    # initialized the alpha_demeaned
    alpha_demeaned = pd.DataFrame(columns=['code']+list(alpha_blend.columns)+\
                                  ['trade_date','group_mean','zscore_alpha'])
    for timestep in tqdm(timesteps):
        # get cross-sectional alpha score
        cross_alpha = alpha_blend.loc[timestep].reset_index().set_index('code')
        cross_alpha['trade_date'] = timestep
        # define the universe
        universe = list(cross_alpha.index)
        # carve out one year rolling data 
        roll_ex1 = exrtn_ts[timestep - datetime.timedelta(days = Cal_day_window[0]):timestep][universe]
        roll_ex2 = exrtn_ts[timestep - datetime.timedelta(days = Cal_day_window[1]):timestep][universe]
        roll_ex3 = exrtn_ts[timestep - datetime.timedelta(days = Cal_day_window[2]):timestep][universe]
        # compute cross-sectional peer at each factor update date
        cross_peers = _cpt_cross_peer_corr_smooth(roll_ex1, roll_ex2, roll_ex3,n_peer_group,corr_type)
        cross_peers['trade_date'] = timestep
        cross_peers = cross_peers.set_index('code')
        cross_demeaned_alpha = _group_demean(cross_alpha,cross_peers,universe)
        alpha_demeaned = pd.concat([alpha_demeaned,cross_demeaned_alpha])

    return alpha_demeaned

def multi_fac_peers_demean_smoothed(alpha_fac,
                                    exrtn_ts,
                                    Cal_day_window=[30, 90, 365],
                                    n_peer_group=15,
                                    corr_type='pearson'):
    """
    Find the most correlated n peers for each stock, each factor at each time step,
    Note the timestep of exrtn must cover the timestep of alpha_blend, padded by
        window size
    The real peer is defined as the intersection of n_peer_group and the asset universe
    Then, demean the alpha score with each group. 
    
    Args:
    ----------
    alpha_fac: pd.DataFrame
        T * N by f +2 long matrix 
            alpha values for each factor, each stock at each time 
        --Example: 
            trade_date         code  value_factor  liquidity_factor  
        67448 2017-01-03  000543.XSHE      0.928571          0.528571   
        67449 2017-01-03  000598.XSHE      0.452381          0.671429   
        67450 2017-01-03  000600.XSHE      0.952381          0.857143   
        67451 2017-01-03  000685.XSHE      0.500000          0.614286   
        67452 2017-01-03  000939.XSHE      0.357143          0.189286   
        
               leverage_factor  overall_momentum_factor  growth_factor  
        67448         0.556429                -0.071429       0.857143   
        67449         0.411429                -0.285714       0.446429   
        67450         0.737857                 0.428571       0.267857   
        67451         0.125714                 0.285714       0.750000   
        67452         0.934286                 0.000000       0.696429   
        
               quality_factor  long_time_rolling_factor  
        67448        0.750000                  0.007725  
        67449        0.642857                  0.006536  
        67450        0.839286                  0.006862  
        67451        0.285714                  0.006356  
        67452        0.214286                  0.002046  
    exrtn_ts: pd.DataFrame
        wide excess return matrix, see  peers_demean_smoothed
    Cal_day_window: int
        the number of calender days look back, default is 30,90,365 
    n_peer_group : int 
        The number of stocks in the peer group
    corr_type: {'pearson', 'kendall', 'spearman'} or callable
        correlation type -- default is pearson correlation
       * pearson : standard correlation coefficient
       * kendall : Kendall Tau correlation coefficient
       * spearman : Spearman rank correlation
    """
    # demean each constituent factor
    factor_list = list (alpha_fac.set_index(['trade_date','code']).columns)
    alphas_demeaned={}
    for factor_name in tqdm(factor_list):
        print('Process: ',factor_name)
        alphas_demeaned[factor_name] = peers_demean_smoothed(alpha_fac.set_index(['trade_date','code'])\
                                                                [[factor_name]],
                                                                  exrtn_ts,
                                                                  Cal_day_window=Cal_day_window,
                                                                  n_peer_group= n_peer_group,
                                                                  corr_type=corr_type)
    # get back to a pd.DataFrame
    alpha_demeaned_grid = pd.DataFrame()
    for facname, alpha_data in alphas_demeaned.items():
        alpha_demeaned_sr = alpha_data.set_index(['trade_date','code']).zscore_alpha
        alpha_demeaned_sr.name = facname
        alpha_demeaned_grid[facname] = alpha_demeaned_sr
        
    return alpha_demeaned_grid





def _group_zscore(cross_alpha,cross_peers,universe):
    """Zscore every stock by its peers at one timestep"""
    # initialize group means 
    Zscore_alpha = pd.DataFrame(index = cross_alpha.index,columns=cross_alpha.columns)
    
    for stock in list(cross_alpha.index):
        # find the intersection of peers and universe
        valid_peers = list(set(cross_peers.loc[stock].peer.values) & set(universe))
        peer_mean = cross_alpha.loc[valid_peers].mean()
        peer_std = cross_alpha.loc[valid_peers].apply(lambda fac: np.nanstd(fac))
        Zscore_alpha.loc[stock] = (cross_alpha.loc[stock] - peer_mean)/peer_std    
    return Zscore_alpha




def peers_zscore_smoothed(alpha_blend,
                   exrtn_ts,
                   Cal_day_window = [30,90,365],
                   n_peer_group = 10,
                   corr_type = 'pearson'):
    """
    Find the most correlated n peers for each stock at each time step,
    Note the timestep of exrtn must cover the timestep of alpha_blend, padded by
        window size
    The real peer is defined as the intersection of n_peer_group and the asset universe
    Then, demean the alpha score with each group. 
        Args:
        ----------
            alpha_blend: MultiIndex pd.DataFrame 
                Where the level 0 index is the timestep 
                     the level 1 index is the stock code
                     the columns are the constituents alpha factors 
                    - Example:
                                            long_time_rolling_factor
                    trade_date code                                 
                    2016-01-05 000543.XSHE                  0.011816
                               000600.XSHE                  0.013183
                               000685.XSHE                  0.010656
                               000690.XSHE                  0.009828
                               000939.XSHE                  0.010814
                    ...                                          ...
                    2021-11-02 601198.XSHG                  0.003886
                               601456.XSHG                  0.005811
                               601577.XSHG                  0.004381
                               601860.XSHG                  0.003416
                               601997.XSHG                  0.002829 
             exrtn_ts: pd.DataFrame
                 wide excess return matrix
                     where the index is the time step and the column is the 
                     stock code 
                - Example:
                    code        000006.XSHE  000008.XSHE  000009.XSHE  000012.XSHE  000021.XSHE  
                    trade_date                                                                    
                    2015-01-05    -0.012642    -0.035754          NaN     0.007970     0.017267   
                    2015-01-06    -0.043587    -0.005478          NaN    -0.008023     0.025019   
                    2015-01-07     0.000210     0.020783          NaN    -0.017572     0.012058   
                    2015-01-08    -0.004099     0.039800     0.010532     0.020735     0.025480   
                    2015-01-09    -0.005877     0.012195    -0.022005    -0.018619    -0.017290 
                    
             Cal_day_window: int
                 the number of calender days look back, default is 365 or one year 
             n_peer_group : int 
                 The number of stocks in the peer group
             corr_type: {'pearson', 'kendall', 'spearman'} or callable
                 correlation type -- default is pearson correlation
                * pearson : standard correlation coefficient
                * kendall : Kendall Tau correlation coefficient
                * spearman : Spearman rank correlation
        Returns:
        ----------
            alpha_zscore  : pd.DataFrame
                peer group cross-sectionally zscored 
                
              
    """
    # set up the for loop 
    timesteps = pd.to_datetime(np.unique(alpha_blend.index.get_level_values(0)))
    # initialized the alpha_demeaned
    alpha_zscore= pd.DataFrame()
    for timestep in tqdm(timesteps):
        # get cross-sectional alpha score
        cross_alpha = alpha_blend.loc[timestep].reset_index().set_index('code')
        # define the universe
        universe = list(cross_alpha.index)
        # carve out one year rolling data 
        roll_ex1 = exrtn_ts[timestep - datetime.timedelta(days = Cal_day_window[0]):timestep][universe]
        roll_ex2 = exrtn_ts[timestep - datetime.timedelta(days = Cal_day_window[1]):timestep][universe]
        roll_ex3 = exrtn_ts[timestep - datetime.timedelta(days = Cal_day_window[2]):timestep][universe]
        # compute cross-sectional peer at each factor update date
        cross_peers = _cpt_cross_peer_corr_smooth(roll_ex1, roll_ex2, roll_ex3,n_peer_group,corr_type)
        cross_peers = cross_peers.set_index('code')
        cross_zscore_alpha = _group_zscore(cross_alpha, cross_peers, universe)
        cross_zscore_alpha = cross_zscore_alpha.reset_index()
        cross_zscore_alpha['trade_date'] = timestep
        alpha_zscore = pd.concat([alpha_zscore,cross_zscore_alpha])

    return alpha_zscore


def OLS_rolling_regression_score(raw_factor,
                                window = 100,
                                overall_factor_name ='blend_alpha',
                                 lb = [0, 0, -np.inf, -np.inf, 0, 0],
                                 ub = [np.inf, np.inf, np.inf, 0, np.inf, np.inf]
                                ):
    """
    Compute combined alpha factor with rolling OLS regression
        where the dependent variable is the forward return for the next period.
        Args:
        ----------
            raw_factor:  pd.DataFrame 
                f+3 by T matrix where the last column is the 
                    - Example:
                 trade_date         code  value_factor  liquidity_factor  leverage_factor  \
                0 2014-01-07  000543.XSHE      0.944444          0.666667         0.700833   
                1 2014-01-07  000685.XSHE      0.472222          0.750000         0.169167   
                2 2014-01-07  000690.XSHE      0.722222          0.379167         0.525000   
                3 2014-01-07  000939.XSHE      0.222222          0.362500         0.830833   
                4 2014-01-07  002479.XSHE      0.444444          0.458333         0.189167   
                
                   overall_momentum_factor  growth_factor  quality_factor   fwd_rtn  
                0                -0.083333       0.729167        0.916667 -0.008798  
                1                -0.166667       0.625000        0.375000  0.000000  
                2                 0.166667       0.937500        0.895833 -0.022333  
                3                 0.666667       0.333333        0.395833 -0.038462  
                4                -0.166667       0.354167        0.687500 -0.067961  
                    
             window: int
                 the number of trading weeks to look back, default is 100 or two year 
             overall_factor_name : str
                 Name of the blended alpha
                 
            lb: list 
                lower boundary for each factor coe-fficient
            ub: list 
                upper boundary for each factor coefficient
        
        
        
        """
    factor_list = list(raw_factor.columns[2:-1])
    print(factor_list)
    date_list = list(set(raw_factor['trade_date']))
    date_list.sort()
    new_raw_factor = pd.DataFrame()


    for i, date in tqdm(enumerate(date_list)):
        if i < window:
            raw_factor_mini = raw_factor[
                raw_factor.trade_date == date].copy()
            new_raw_factor = new_raw_factor.append(raw_factor_mini)
            continue
        raw_factor_mini = raw_factor[raw_factor.trade_date == date].copy()
        raw_factor_mini = raw_factor_mini.reset_index(drop=True)
        tmp_date_list = date_list[i - window: i]

        train_data = raw_factor[raw_factor.trade_date.isin(tmp_date_list)]
        x = train_data[factor_list]
        y = train_data.iloc[:,-1]

        model = lsq_linear(x, y, bounds=(lb, ub))
        
        raw_factor_mini[overall_factor_name] = (model.x * raw_factor_mini[factor_list]).sum(axis=1)


        new_raw_factor = new_raw_factor.append(raw_factor_mini)

        
    return new_raw_factor




def _group_ecdf(cross_alpha,cross_peers,universe):
    """ecdf every stock by its peers at one timestep"""
    # initialize group means 
    ecdf_alpha = pd.DataFrame(index = cross_alpha.index,columns=cross_alpha.columns)
    
    for stock in list(cross_alpha.index):
            # find the intersection of peers and universe
            valid_peers = list(set(cross_peers.loc[stock].peer.values) & set(universe))
            cross_ecdf = cross_alpha.loc[valid_peers].apply(lambda fac: PA._ecdf_trans(fac))
            cross_ecdf.index = valid_peers
            ecdf_alpha.loc[stock,:] = cross_ecdf.loc[stock,:]   
    return ecdf_alpha



def peers_ecdf_smoothed(alpha_blend,
                   exrtn_ts,
                   Cal_day_window = [30,90,365],
                   n_peer_group = 10,
                   corr_type = 'pearson'):
    """
    Find the most correlated n peers for each stock at each time step,
    Note the timestep of exrtn must cover the timestep of alpha_blend, padded by
        window size
    The real peer is defined as the intersection of n_peer_group and the asset universe
    Then, demean the alpha score with each group. 
        Args:
        ----------
            alpha_blend: MultiIndex pd.DataFrame 
                Where the level 0 index is the timestep 
                     the level 1 index is the stock code
                     the columns are the constituents alpha factors 
                    - Example:
                                            long_time_rolling_factor
                    trade_date code                                 
                    2016-01-05 000543.XSHE                  0.011816
                               000600.XSHE                  0.013183
                               000685.XSHE                  0.010656
                               000690.XSHE                  0.009828
                               000939.XSHE                  0.010814
                    ...                                          ...
                    2021-11-02 601198.XSHG                  0.003886
                               601456.XSHG                  0.005811
                               601577.XSHG                  0.004381
                               601860.XSHG                  0.003416
                               601997.XSHG                  0.002829 
             exrtn_ts: pd.DataFrame
                 wide excess return matrix
                     where the index is the time step and the column is the 
                     stock code 
                - Example:
                    code        000006.XSHE  000008.XSHE  000009.XSHE  000012.XSHE  000021.XSHE  
                    trade_date                                                                    
                    2015-01-05    -0.012642    -0.035754          NaN     0.007970     0.017267   
                    2015-01-06    -0.043587    -0.005478          NaN    -0.008023     0.025019   
                    2015-01-07     0.000210     0.020783          NaN    -0.017572     0.012058   
                    2015-01-08    -0.004099     0.039800     0.010532     0.020735     0.025480   
                    2015-01-09    -0.005877     0.012195    -0.022005    -0.018619    -0.017290 
                    
             Cal_day_window: int
                 the number of calender days look back, default is 365 or one year 
             n_peer_group : int 
                 The number of stocks in the peer group
             corr_type: {'pearson', 'kendall', 'spearman'} or callable
                 correlation type -- default is pearson correlation
                * pearson : standard correlation coefficient
                * kendall : Kendall Tau correlation coefficient
                * spearman : Spearman rank correlation
        Returns:
        ----------
            alpha_zscore  : pd.DataFrame
                peer group cross-sectionally zscored 
                
              
    """
    # set up the for loop 
    timesteps = pd.to_datetime(np.unique(alpha_blend.index.get_level_values(0)))
    # initialized the alpha_demeaned
    alpha_ecdf= pd.DataFrame()
    for timestep in tqdm(timesteps):
        # get cross-sectional alpha score
        cross_alpha = alpha_blend.loc[timestep].reset_index().set_index('code')
        # define the universe
        universe = list(cross_alpha.index)
        # carve out one year rolling data 
        roll_ex1 = exrtn_ts[timestep - datetime.timedelta(days = Cal_day_window[0]):timestep][universe]
        roll_ex2 = exrtn_ts[timestep - datetime.timedelta(days = Cal_day_window[1]):timestep][universe]
        roll_ex3 = exrtn_ts[timestep - datetime.timedelta(days = Cal_day_window[2]):timestep][universe]
        # compute cross-sectional peer at each factor update date
        cross_peers = _cpt_cross_peer_corr_smooth(roll_ex1, roll_ex2, roll_ex3,n_peer_group,corr_type)
        cross_peers = cross_peers.set_index('code')
        cross_ecdf_alpha = _group_ecdf(cross_alpha, cross_peers, universe)
        
        cross_ecdf_alpha = cross_ecdf_alpha.reset_index()
        cross_ecdf_alpha['trade_date'] = timestep
        alpha_ecdf = pd.concat([alpha_ecdf,cross_ecdf_alpha])
    return alpha_ecdf



def win(x, trim=0.2, limit = 'both'):
    y = x.copy()
    x.dropna()
    if (trim < 0) | (trim > 0.5):
        print("trimming must be reasonable")
    try:
        qtrim_min = x.quantile(trim)
        qtrim_mid = x.quantile(0.5)
        qtrim_max = x.quantile(1-trim)
    except:
        import pdb
        pdb.set_trace()
    if trim >0.5:
        y[x != None] = qtrim_mid
    else:
        if limit=='both':
            y[x < qtrim_min] = qtrim_min
            y[x > qtrim_max] = qtrim_max
        elif limit == 'ub':
            y[x > qtrim_max] = qtrim_max
        elif limit == 'lb':
            y[x < qtrim_min] = qtrim_min
    return y


def stand(z, trim_num,limit = 'both'):
    x = win(z, trim_num,limit)
    try:
        x_mean = np.nanmean(x)
        if len(x) == 0 or np.nan in x:
            print('bug')
    except:
        print('bug')
        print('bug')
    x_std = np.nanstd(x)
    y = (x - x_mean) / x_std
    return y



def std_winsor(z):
    tmp_z = z.copy()
    tmp_z = tmp_z.dropna()
    z_std = np.std(tmp_z)
    min_std = -3 * z_std
    max_std = 3 * z_std
    z[z<min_std] = min_std
    z[z>max_std] = max_std
    z[z==None] = min_std
    return z

def _single_fac_preprocess(fac):
    # preprocess a single factor

    sd_fac = stand(fac, 0.05)  # trim and zscore

    sd_fac = std_winsor(sd_fac) # 3 sigma winsorize 
    # fit the cdf of the factor


    fac_cdf = ECDF(sd_fac)
    #return the fitted values of ecdf 
    fac_cprob = fac_cdf(sd_fac)


    return fac_cprob
    


def _group_preprocess(cross_alpha,cross_peers,universe):
    """
    preprocess every stock within its peers at one timestep, the preprocess 
    is in three steps: 
        First trim the value at both ends at 5% and zscore 
        Second use three sigma winsorization 
        Finally fit a ECDF function"""
    # initialize group means 
    preprocessed_alpha = pd.DataFrame(index = cross_alpha.index,columns=cross_alpha.columns)
    
    for stock in list(cross_alpha.index):
            # find the intersection of peers and universe
            valid_peers = list(set(cross_peers.loc[stock].peer.values) & set(universe))
            # perform the preprocess for each stock's each factor
            cross_preprocessed = cross_alpha.loc[valid_peers].apply(lambda fac: _single_fac_preprocess(fac))
            cross_preprocessed.index = valid_peers
            preprocessed_alpha.loc[stock,:] = cross_preprocessed.loc[stock,:]  
            
    return preprocessed_alpha

def peers_preprocess_smoothed(alpha_blend,
                   exrtn_ts,
                   Cal_day_window = [30,90,365],
                   n_peer_group = 10,
                   corr_type = 'pearson'):
    """
    Find the most correlated n peers for each stock at each time step,
    Note the timestep of exrtn must cover the timestep of alpha_blend, padded by
        window size
    The real peer is defined as the intersection of n_peer_group and the asset universe
    Then, demean the alpha score with each group. 
        Args:
        ----------
            alpha_blend: MultiIndex pd.DataFrame 
                Where the level 0 index is the timestep 
                     the level 1 index is the stock code
                     the columns are the constituents alpha factors 
                    - Example:
                                            long_time_rolling_factor
                    trade_date code                                 
                    2016-01-05 000543.XSHE                  0.011816
                               000600.XSHE                  0.013183
                               000685.XSHE                  0.010656
                               000690.XSHE                  0.009828
                               000939.XSHE                  0.010814
                    ...                                          ...
                    2021-11-02 601198.XSHG                  0.003886
                               601456.XSHG                  0.005811
                               601577.XSHG                  0.004381
                               601860.XSHG                  0.003416
                               601997.XSHG                  0.002829 
             exrtn_ts: pd.DataFrame
                 wide excess return matrix
                     where the index is the time step and the column is the 
                     stock code 
                - Example:
                    code        000006.XSHE  000008.XSHE  000009.XSHE  000012.XSHE  000021.XSHE  
                    trade_date                                                                    
                    2015-01-05    -0.012642    -0.035754          NaN     0.007970     0.017267   
                    2015-01-06    -0.043587    -0.005478          NaN    -0.008023     0.025019   
                    2015-01-07     0.000210     0.020783          NaN    -0.017572     0.012058   
                    2015-01-08    -0.004099     0.039800     0.010532     0.020735     0.025480   
                    2015-01-09    -0.005877     0.012195    -0.022005    -0.018619    -0.017290 
                    
             Cal_day_window: int
                 the number of calender days look back, default is 365 or one year 
             n_peer_group : int 
                 The number of stocks in the peer group
             corr_type: {'pearson', 'kendall', 'spearman'} or callable
                 correlation type -- default is pearson correlation
                * pearson : standard correlation coefficient
                * kendall : Kendall Tau correlation coefficient
                * spearman : Spearman rank correlation
        Returns:
        ----------
            alpha_zscore  : pd.DataFrame
                peer group cross-sectionally zscored 
                
              
    """
    # set up the for loop 
    timesteps = pd.to_datetime(np.unique(alpha_blend.index.get_level_values(0)))
    # initialized the alpha_preprocess
    alpha_preprocess= pd.DataFrame()
    for timestep in tqdm(timesteps):
        # get cross-sectional alpha score
        cross_alpha = alpha_blend.loc[timestep].reset_index().set_index('code')
        # define the universe
        universe = list(cross_alpha.index)
        # carve out one year rolling data 
        roll_ex1 = exrtn_ts[timestep - datetime.timedelta(days = Cal_day_window[0]):timestep][universe]
        roll_ex2 = exrtn_ts[timestep - datetime.timedelta(days = Cal_day_window[1]):timestep][universe]
        roll_ex3 = exrtn_ts[timestep - datetime.timedelta(days = Cal_day_window[2]):timestep][universe]
        # compute cross-sectional peer at each factor update date
        cross_peers = _cpt_cross_peer_corr_smooth(roll_ex1, roll_ex2, roll_ex3,n_peer_group,corr_type)
        cross_peers = cross_peers.set_index('code')
        cross_preprocess_alpha  = _group_preprocess(cross_alpha, cross_peers, universe)
        
        cross_preprocess_alpha  = cross_preprocess_alpha .reset_index()
        cross_preprocess_alpha ['trade_date'] = timestep
        alpha_preprocess = pd.concat([alpha_preprocess,cross_preprocess_alpha])
        
    return alpha_preprocess

def cpt_dyna_peers(all_raw_factor,n_peer_group =20,cut = False):
    """
    Compute the dynamic peer group for each stock at each point of time with
    pairwise correlation

    Parameters
    ----------
    all_raw_factor : pd.DataFrame
        raw_factor.
    n_peer_group : int, optional
        num of largest group size. The default is 20.
    cut : bool, optional
        flag to turn on cut. The default is False.

    Returns
    -------
    peer_mapping : pd.Dataframe
        time,stock, peer indexed mapping.

    """

    # normalise stock code
    norm_code = normalize_code(list(all_raw_factor.ts_code))
    all_raw_factor['code'] = norm_code
    # turn trade_date into date format
    all_raw_factor['trade_date'] = pd.to_datetime(all_raw_factor['trade_date'].astype(str))
    all_raw_factor = all_raw_factor[all_raw_factor.weight>0]
    # compute excess returns
    ex_rtn = cpt_ex_rtn(all_raw_factor)
    ex_rtn.to_pickle(r"C:\Users\Administrator\PycharmProjects\compare_diff\old_alpha.pkl")

    # set up the for loop
    timesteps = all_raw_factor.trade_date.drop_duplicates().to_list()

    peer_mapping = pd.DataFrame()
    for timestep in tqdm(timesteps):
        cross_fac = all_raw_factor[all_raw_factor.trade_date ==timestep]
        # define the universe
        universe = cross_fac.code.to_list()
        # compute dyna peers
        roll_ex1 = ex_rtn[timestep - datetime.timedelta(days = 720):timestep][universe]
        roll_ex2 = ex_rtn[timestep - datetime.timedelta(days = 365):timestep][universe]
        roll_ex3 = ex_rtn[timestep - datetime.timedelta(days = 180):timestep][universe]
        cross_peers = _cpt_cross_peer_corr_smooth(roll_ex1,roll_ex2,roll_ex3,n_peer_group=n_peer_group)
        # conditional on if we gonna cut the the corr
        if cut:
            cross_peers = reverse_corr_cut(cross_peers,n_peer = n_peer_group)
        cross_peers['trade_date'] = timestep
        peer_mapping = peer_mapping.append(cross_peers)
    # map back to ts_code
    norm_code2ts_code = all_raw_factor[['ts_code','code']].drop_duplicates()
    peer_mapping = peer_mapping.merge(norm_code2ts_code,on ='code').\
        merge(norm_code2ts_code,left_on='peer',right_on='code',suffixes=('','_peer'))\
            .drop('code_peer',axis =1)
    peer_mapping = peer_mapping.drop(['code','peer'],axis =1)
    peer_mapping = peer_mapping.sort_values(['trade_date','ts_code','ts_code_peer'])
    #strtify the time
    peer_mapping.trade_date = peer_mapping.trade_date.apply(lambda x: x.strftime('%Y%m%d')).astype(int)

    return peer_mapping


def cpt_dyna_peers_(valid_raw_factor, daily_trading_paused_data, n_peer_group =20, cut = False):
    """
    Compute the dynamic peer group for each stock at each point of time with
    pairwise correlation

    Parameters
    ----------
    valid_raw_factor : pd.DataFrame
        raw_factor.
    daily_trading_paused_data: pd.DataFrame adj close price ,pause flag
    n_peer_group : int, optional
        num of largest group size. The default is 20.
    cut : bool, optional
        flag to turn on cut. The default is False.

    Returns
    -------
    peer_mapping : pd.Dataframe
        time,stock, peer indexed mapping.

    """
    valid_raw_factor['trade_date'] = pd.to_datetime(valid_raw_factor['trade_date'].astype(str))


# df = pd.DataFrame({'ts_code': ['000001.SZ', "000002.SZ", '000001.SZ', "000002.SZ"], 'trade_date': [20230313, 20230313, 20230314, 20230314]})
#
# ex_rtn = cpt_ex_rtn(df)