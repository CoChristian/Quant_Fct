import pandas as pd
import datetime
from tqdm import tqdm
from jqdatasdk import *
auth("13764432461", "Nfhq12345")
import numpy as np

def cpt_ex_rtn(all_raw_factors, bench_code=['000905.XSHG'], days_before=800):
    """Compute excess returns for each day during the speciied time period for specified code"""
    # normalise stock code
    # norm_code = normalize_code(list(all_raw_factors.ts_code))
    # all_raw_factors['code'] = norm_code
    all_raw_factors = all_raw_factors.reset_index()
    norm_code = all_raw_factors['code'].values
    # turn trade_date into date format
    all_raw_factors['trade_date'] = pd.to_datetime(all_raw_factors['trade_date'].astype(str))

    # specify first and last data date
    first_data_date = all_raw_factors['trade_date'].min() - datetime.timedelta(days=days_before)
    last_data_date = all_raw_factors['trade_date'].max()
    print('Data Starts:', first_data_date, 'Data Ends:', last_data_date)

    # import prices data from jq
    # import price data for all stocks in the universe, note we fill paused with Na
    price_ts = get_price(list(np.unique(norm_code)), start_date=first_data_date,
                         end_date=last_data_date + datetime.timedelta(days=10),
                         frequency='daily',
                         fields=['close'], fq='pre')
    # change the colnames

    price_ts.columns = ['trade_date', 'code', 'close']
    print(price_ts['trade_date'].value_counts().sort_index())
    # import stock index data to compute excess return
    # import price data
    index_ts = get_price(['000905.XSHG'], start_date=first_data_date,
                         end_date=last_data_date,
                         frequency='daily',
                         fields=['close'], fq='pre')
    # # change the colnames
    index_ts.columns = ['trade_date', 'code', 'close']
    # compute index returns
    index_ts = index_ts.drop('code', axis=1).set_index('trade_date')
    index_ts = index_ts.pct_change()[1:]
    index_ts.columns = ['bench_rtn']
    index_ts = index_ts.reset_index()

    # compute daily returns for each stock
    rtn_ts = price_ts.set_index(['trade_date', 'code']).unstack().pct_change()[1:].stack()
    rtn_ts.columns = ['rtn']
    rtn_ts = rtn_ts.reset_index()
    # merge bench rtn
    rtn_ts = rtn_ts.merge(index_ts, on='trade_date', how='left')
    # compute excess rtn
    rtn_ts['ex_rtn'] = rtn_ts.rtn - rtn_ts.bench_rtn
    # grab useful col
    exrtn_ts = rtn_ts[['trade_date', 'code', 'ex_rtn']]
    # unstack the exrtn ts
    exrtn_ts = exrtn_ts.set_index(['trade_date', 'code']).unstack()
    # reset the cols
    exrtn_ts.columns = exrtn_ts.columns.get_level_values(1)

    return exrtn_ts


def cpt_ex_rtn_from_data(all_raw_factors, daily_close_paused_data, index_close_data, code_price_name, index_price_name, pause_flag_name):
    """Compute excess returns for each day during the speciied time period for specified code"""
    # import pdb
    # pdb.set_trace()
    all_raw_factors = all_raw_factors.reset_index()
    daily_close_paused_data = daily_close_paused_data.reset_index()
    index_close_data = index_close_data.reset_index()
    codes = all_raw_factors['code'].unique()
    daily_close_paused_data = daily_close_paused_data[daily_close_paused_data.code.map(lambda x: x in codes)]
    daily_close_paused_data.sort_values(['code', 'trade_date'], inplace=True)
    daily_close_paused_data['pct_chg'] = daily_close_paused_data.groupby("code", group_keys=False)[code_price_name].apply(lambda x: x.pct_change())
    daily_close_paused_data = daily_close_paused_data[daily_close_paused_data['pct_chg'].notnull()]
    index_close_data.sort_values('trade_date', inplace=True)
    index_close_data['pct_chg_index'] = index_close_data[index_price_name].pct_change()
    index_close_data = index_close_data[index_close_data['pct_chg_index'].notnull()]
    daily_close_paused_data = pd.merge(daily_close_paused_data, index_close_data[['trade_date', 'pct_chg_index']], how='left', on='trade_date')
    daily_close_paused_data['alpha'] = daily_close_paused_data['pct_chg'] - daily_close_paused_data['pct_chg_index']
    # daily_close_paused_data['alpha'] = daily_close_paused_data.apply(lambda x: None if x[pause_flag_name] != 0 else x['pct_chg'] - x['pct_chg_index'], axis=1)

    exrtn_ts = daily_close_paused_data.set_index(['trade_date', 'code'])['alpha'].unstack()
    # exrtn_ts.to_pickle("exrtn_post_adj.pkl")
    # import pdb
    # pdb.set_trace()
    # exrtn_ts = exrtn_ts.fillna(0)
    return exrtn_ts


def _cpt_cross_peer_corr_smooth(roll_ex1,
                                roll_ex2,
                                roll_ex3,
                                n_peer_group=10,
                                corr_type='pearson'):
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
    min_periods_nums = [int(roll_ex1.shape[0] / 2), int(roll_ex2.shape[0] / 2), int(roll_ex3.shape[0] / 2)]

    # compute the sample corr mat and drop all nas
    cross_corr1 = pd.DataFrame(roll_ex1.corr(min_periods=min_periods_nums[0]).fillna(0).stack())
    cross_corr2 = pd.DataFrame(roll_ex2.corr(min_periods=min_periods_nums[1]).fillna(0).stack())
    cross_corr3 = pd.DataFrame(roll_ex3.corr(min_periods=min_periods_nums[2]).fillna(0).stack())
    # compute the mean of three correlation

    cross_corr = cross_corr1.merge(cross_corr2, left_index=True, right_index=True). \
        merge(cross_corr3, left_index=True, right_index=True).mean(axis=1)

    cross_corrS1 = cross_corr1.merge(cross_corr2, left_index=True, right_index=True)
    cross_corrS2 = cross_corrS1.merge(cross_corr3, left_index=True, right_index=True)
    cross_corr = cross_corrS2.mean(axis=1)

    # rename the index
    cross_corr.index = cross_corr.index.set_names(['code1', 'code2'], level=[0, 1])
    # reset the index
    cross_corr = cross_corr.reset_index()
    # rename the columns
    cross_corr.columns = list(cross_corr.columns[:-1]) + ['corr']
    # get n_peer with largest correlation
    cross_peers = cross_corr.groupby('code1'). \
        apply(lambda corr: corr.set_index('code2'). \
              nlargest(n_peer_group + 1, 'corr')['corr']).reset_index()
    # change the col names
    cross_peers.columns = ['code', 'peer', 'corr']
    return cross_peers

def _cpt_cross_peer_corr_with_industry_smooth(roll_ex1,roll_ex2,roll_ex3,industry_info, industry_names, n_peer_group=10,corr_type='pearson'):
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
            industry_info: pd.dataframe
              stock industry info, code as index, industry code as value
            industry_names:
             multi industry names to cal industry corr
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
    min_periods_nums = [int(roll_ex1.shape[0] / 2), int(roll_ex2.shape[0] / 2), int(roll_ex3.shape[0] / 2)]
    # compute industry corr
    industry_corrs = []
    for industry_name in industry_names:        
        industry_vec_info = pd.get_dummies(industry_info[industry_name])
        industry_corr = pd.DataFrame(industry_vec_info.T.corr().applymap(lambda x: 1 if x == 1 else -1).stack())
#         industry_corr.index = industry_corr.index.set_names(['code1', 'code2'], level=[0, 1])
        industry_corrs.append(industry_corr)
    industry_corr_df = pd.concat(industry_corrs, axis=1)
    industry_corr_all = pd.DataFrame(industry_corr_df.mean(axis=1))
    

    # compute the sample corr mat and drop all nas
    cross_corr1 = pd.DataFrame(roll_ex1.corr(min_periods=min_periods_nums[0]).fillna(0).stack())
    cross_corr2 = pd.DataFrame(roll_ex2.corr(min_periods=min_periods_nums[1]).fillna(0).stack())
    cross_corr3 = pd.DataFrame(roll_ex3.corr(min_periods=min_periods_nums[2]).fillna(0).stack())
    # compute the mean of three correlation

    cross_corr = cross_corr1.merge(cross_corr2, left_index=True, right_index=True). \
        merge(cross_corr3, left_index=True, right_index=True).mean(axis=1)
    cross_corr = pd.DataFrame(cross_corr)

    cross_corr = cross_corr.merge(industry_corr_all, left_index=True, right_index=True).sum(axis=1)

    # rename the index
    cross_corr.index = cross_corr.index.set_names(['code1', 'code2'], level=[0, 1])
    # reset the index
    cross_corr = cross_corr.reset_index()
    # rename the columns
    cross_corr.columns = list(cross_corr.columns[:-1]) + ['corr']
    # get n_peer with largest correlation
    cross_peers = cross_corr.groupby('code1'). \
        apply(lambda corr: corr.set_index('code2'). \
              nlargest(n_peer_group + 1, 'corr')['corr']).reset_index()
    # change the col names
    cross_peers.columns = ['code', 'peer', 'corr']
    return cross_peers

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



def cpt_dyna_peers(all_raw_factor, ex_rtn,  hist_window_size=[180, 360, 720], n_peer_group =20,cut = False, add_industry_info=False, industry_names = []):
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
    all_raw_factor = all_raw_factor.reset_index()

    all_raw_factor['trade_date'] = pd.to_datetime(all_raw_factor['trade_date'].astype(str))
    ex_rtn.index = pd.to_datetime(ex_rtn.index.astype(str))
    # compute excess returns
    # ex_rtn.to_pickle(r"C:\Users\Administrator\PycharmProjects\compare_diff\new_alpha.pkl")
    # set up the for loop
    timesteps = all_raw_factor.trade_date.drop_duplicates().to_list()

    peer_mapping = pd.DataFrame()
    for timestep in tqdm(timesteps):
        cross_fac = all_raw_factor[all_raw_factor.trade_date ==timestep]
        # define the universe
        universe = cross_fac.code.to_list()
        # compute dyna peers
#         industry_info = cross_fac.set_index('code')['GicsIndustryCode']
        
        roll_ex1 = ex_rtn[timestep - datetime.timedelta(days = hist_window_size[2]):timestep][universe]
        roll_ex2 = ex_rtn[timestep - datetime.timedelta(days = hist_window_size[1]):timestep][universe]
        roll_ex3 = ex_rtn[timestep - datetime.timedelta(days = hist_window_size[0]):timestep][universe]
        if add_industry_info:
            cross_peers = _cpt_cross_peer_corr_with_industry_smooth(roll_ex1,roll_ex2,roll_ex3, cross_fac.set_index('code')[industry_names], industry_names, n_peer_group=n_peer_group)
        else:
            cross_peers = _cpt_cross_peer_corr_smooth(roll_ex1,roll_ex2,roll_ex3,n_peer_group=n_peer_group)
        # conditional on if we gonna cut the the corr
        if cut:
            cross_peers = reverse_corr_cut(cross_peers,n_peer = n_peer_group)
        cross_peers['trade_date'] = timestep
        peer_mapping = peer_mapping.append(cross_peers)
    # map back to ts_code
    # norm_code2ts_code = all_raw_factor[['ts_code','code']].drop_duplicates()
    # peer_mapping = peer_mapping.merge(norm_code2ts_code,on ='code').\
    #     merge(norm_code2ts_code,left_on='peer',right_on='code',suffixes=('','_peer'))\
    #         .drop('code_peer',axis =1)
    # peer_mapping = peer_mapping.drop(['code','peer'],axis =1)
    # peer_mapping = peer_mapping.sort_values(['trade_date','ts_code','ts_code_peer'])
    # #strtify the time
    peer_mapping.trade_date = peer_mapping.trade_date.apply(lambda x: x.strftime('%Y%m%d')).astype(int)
    peer_mapping = peer_mapping.set_index(['code', 'trade_date', 'peer'])
    return peer_mapping
