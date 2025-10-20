import pandas as pd
import numpy as np
import preprocess
import group_calc
import factor_analysis

factor_df=pd.read_csv("F:\\quant_factor\\factor_backtest-main\\data\\MAfct_data\\MA10D_fct.csv")
CF_fct = pd.read_csv("F:\\quant_factor\\factor_backtest-main\\data\\MAfct_data\\fct_data\\CF_data.csv")
CF_fct['stock_code'] = CF_fct['code']
CF_fct['datetime'] = pd.to_datetime(CF_fct['trade_date'])
CF_fct['trade_date'] = CF_fct['datetime'].dt.date
# used to cal ret and other fundalmental factor
close_df = pd.read_csv("F:\\quant_factor\\factor_backtest-main\\data\\MAfct_data\\close_fct.csv")
ohlcvm_df = pd.read_csv("F:\\work\\Data\\Database_to_csv\\stock_price_history_total\\stock_price_history251014.csv")
index_data = pd.read_csv("F:\\quant_factor\\factor_backtest-main\\data\\MAfct_data\\index.csv")
status_data = pd.read_csv("F:\\quant_factor\\factor_backtest-main\\data\\MAfct_data\\stock_status.csv")
valuation_data = pd.read_csv("F:\\quant_factor\\factor_backtest-main\\data\\MAfct_data\\valuation.csv")
ohlcvm_df['datetime'] = pd.to_datetime(ohlcvm_df['date'])
ohlcvm_df['trade_date'] = ohlcvm_df['datetime'].dt.date
ohlcvm_df['stock_code'] = ohlcvm_df['code']
START_YEAR = 2015

g_stock = ohlcvm_df.groupby('stock_code',group_keys=False)

# get trade date
indexv1 = index_data[index_data['stock_code'] == '000001.XSHG'].copy()
indexv1 = indexv1.sort_values('trade_date',ascending=True)
indexv1 = indexv1.drop_duplicates(subset=['trade_date'], keep='first')
indexv1['datetime'] = pd.to_datetime(indexv1['trade_date'])
indexv1['year'] = indexv1['datetime'].dt.year
indexv1['month'] = indexv1['datetime'].dt.month
indexv1['week'] = indexv1['datetime'].dt.isocalendar().week
indexv1['trade_date'] = indexv1['datetime'].dt.date


trade_date = pd.DataFrame(indexv1[['trade_date','year','week','month']])
def get_trade_date(df, start_year, period=None):
    df = df[df['year'] >= start_year]
    period_date = None
    if period in ['week', 'month']:
        period_date = df.groupby(['year','week'])['trade_date'].last()
    elif type(period) == int:
        period_date = df[::period]['trade_date']

    return pd.DataFrame(period_date)

# get ret depend on trade date
trade_date = get_trade_date(trade_date, START_YEAR, period='week')


def get_fwd_data(df, trade_date):
    df = df.sort_values('trade_date',ascending=True).copy()
    df = df.drop_duplicates(subset=['trade_date'], keep='first')
    df['open_fwd'] = df['open'].shift(-1)
    df['open_fq'] = df['open'] * df['factor']
    df['close_fq'] = df['close'] * df['factor']
    df['close_fwd'] = df['close'].shift(-1)
    df['high_limit_fwd'] = df['high_limit'].shift(-1)
    df['low_limit_fwd'] = df['low_limit'].shift(-1)
    df['factor_fwd'] = df['factor'].shift(-1)
    df['open_fwd_fq'] = df['open_fwd'] * df['factor_fwd']
    df['close_fwd_fq'] = df['close_fwd'] * df['factor_fwd']
    # get return
    df = pd.merge(df, trade_date, on=['trade_date'], how='right')
    df['period_ret_O'] = (df['open_fq'].shift(-1) / df['open_fwd_fq']) - 1
    return df

ohlcvm_dfv2 = g_stock.apply(get_fwd_data, trade_date)

ret_df = ohlcvm_dfv2.rename(columns={'period_ret_O':'ret'})[['trade_date','stock_code','ret']].copy()
factorCF_df = pd.merge(ret_df, CF_fct, on=['trade_date','stock_code'])
PCF_fct = factorCF_df[['trade_date','stock_code','PCF']].copy()
VCF_fct = factorCF_df[['trade_date','stock_code','VCF']].copy()
ACF_fct = factorCF_df[['trade_date','stock_code','ACF']].copy()
TRCF_fct = factorCF_df[['trade_date','stock_code','TRCF']].copy()
TRCF_fct['TRCF'] = -1 * TRCF_fct['TRCF']
# bp_df = pd.read_csv('./data/pb.csv')
# ret_df=pd.read_csv("F:\\quant_factor\\factor_backtest-main\\data\\MAfct_data\\ret_10D.csv")

ind_data = pd.read_csv("F:\\quant_factor\\factor_backtest-main\\data\\MAfct_data\\industry.csv")
# get industry
ind_datav1 = ind_data[ind_data['type']=='sw_l1'].copy()
ind_datav2 = ind_datav1.rename(columns={'industry_name':'ind_code'})
ind_datav2 = ind_datav2.sort_values('trade_date',ascending=True)
ind_datav2 = ind_datav2.drop_duplicates(subset=['stock_code','trade_date'], keep='first')
ind_datav2['datetime'] = pd.to_datetime(ind_datav2['trade_date'])
ind_datav2['trade_date'] = ind_datav2['datetime'].dt.date
ind_datav2 = ind_datav2[['trade_date','stock_code','ind_code']]

# get mkmtv
get_mk = valuation_data[['trade_date','stock_code','circulating_cap']]
mkmtv_close = pd.merge(get_mk, close_df,on=["trade_date","stock_code"])
mkmtv_close['mktmv'] = (mkmtv_close['circulating_cap'] * mkmtv_close['close']) / 10000
mkmtv_close['datetime'] = pd.to_datetime(mkmtv_close['trade_date'])
mkmtv_close['trade_date'] = mkmtv_close['datetime'].dt.date
mkmtv = mkmtv_close[['trade_date','stock_code','mktmv']].copy()





# mktmv_df = pd.read_csv('./data/mktmv_df.csv')
benchmark = index_data[index_data['stock_code']=='000300.XSHG'].copy()
benchmark['datetime'] = pd.to_datetime(benchmark['trade_date'])
benchmark['trade_date'] = benchmark['datetime'].dt.date
benchmark['open_fwd'] = benchmark['open'].shift(-1)
benchmark = pd.merge(benchmark, trade_date,on=['trade_date'])
benchmark['ret'] = benchmark['open'].shift(-1) / benchmark['open_fwd'] - 1
# benchmark['ret'] = benchmark['retOnopen_10D'].shift(-1)

benchmark = benchmark[['trade_date','ret']]

# benchmark = pd.read_csv('./data/index_ret.csv')
# ind_df=pd.read_csv('./data/ind_df.csv')

# get year
# def get_start_year(df, Start_year, col):
#     df['datetime'] = pd.to_datetime(df['trade_date'])
#     df['year'] = df['datetime'].dt.year
#     df = df[df['year'] >= Start_year]
#     stay_col = ['trade_date','stock_code', col]
#     df = df[stay_col].copy()
#     return df
# Start_year = 2020
# trade_date = trade_date[trade_date['year']>=Start_year]
# trade_date10D = trade_date[::11].copy()
# trade_date10D = pd.DataFrame(trade_date10D['trade_date'])
# factor_df = get_start_year(factor_df, Start_year, 'MA_fctEMA10D')
# g_stock = factor_df.groupby('stock_code', group_keys=False)

# def get_date_factor(df, date_df):
#     factor_df = pd.merge(df, date_df, on=["trade_date"],how='right')
#     return factor_df
#
# factor_df10D = g_stock.apply(get_date_factor, trade_date10D)
# factor_df10Dv1 = factor_df10D.dropna()

# check st
# factor_df10Dv1_addst = pd.merge(factor_df10Dv1, status_data, on=["trade_date", "stock_code"], how='left')
# factor_df10Dv2= factor_df10Dv1_addst[factor_df10Dv1_addst['isst']==0].copy()
# factor_df10Dv2 = factor_df10Dv2[['trade_date','stock_code','MA_fctEMA10D']]
# factor_df10D = factor_df[::11].copy()
# ret_df = get_start_year(ret_df, Start_year, 'ret')
# ret_df10D = ret_df[::11].copy()



# codefunc1=lambda x: '{:0>6}'.format(x)
# factor_df['stock_code']=factor_df['stock_code'].apply(codefunc1)
# bp_df['stock_code']=bp_df['stock_code'].apply(codefunc1)
# ret_df['stock_code']=ret_df['stock_code'].apply(codefunc1)
# mktmv_df['stock_code']=mktmv_df['stock_code'].apply(codefunc1)
# ind_df['stock_code']=ind_df['stock_code'].apply(codefunc1)

# codefunc2 = lambda x: x if np.isnan(x) else str(int(x))
# factor_df['trade_date']=factor_df['trade_date'].apply(codefunc2)
# bp_df['trade_date']=bp_df['trade_date'].apply(codefunc2)
# ret_df['trade_date']=ret_df['trade_date'].apply(codefunc2)
# mktmv_df['trade_date']=mktmv_df['trade_date'].apply(codefunc2)
# ind_df['trade_date']=ind_df['trade_date'].apply(codefunc2)
# benchmark['trade_date'] = benchmark['trade_date'].apply(codefunc2)

#
# # 处理factor因子
# MAD去极值
factor_name = 'TRCF'
factor_df = TRCF_fct.copy()

factor_df = preprocess.del_outlier(factor_df, factor_name, method='mad', n=3)
# 排序标准化
factor_df = preprocess.standardize(factor_df, factor_name, method='rank')
# 同时做市值中性化和行业中性化
factor_df=preprocess.neutralize(factor_df=factor_df,
                                 factor_name=factor_name,
                                 mktmv_df=mkmtv,
                                 industry_df=ind_datav2)
#
# bp_df['bp'] = -bp_df['pb']
# bp_df=bp_df.drop(columns=['pb'])
# # MAD去极值
# bp_df = preprocess.del_outlier(bp_df, 'bp', method='mad', n=3)
# # 排序标准化
# bp_df = preprocess.standardize(bp_df, 'bp', method='rank')
# # 同时做市值中性化和行业中性化
# bp_df=preprocess.neutralize(factor_df=bp_df,
#                                  factor_name='bp',
#                                  mktmv_df=mktmv_df,
#                                  industry_df=ind_df)


mw_group_ret = group_calc.get_group_ret(factor_df, ret_df, factor_name, 5, mkmtv)

mean_result = factor_analysis.newy_west_test(mw_group_ret['H-L'],factor_name)

ic_dct, ic_fig = factor_analysis.analysis_factor_ic(factor_df, ret_df, factor_name)

ew_backtest_df,ew_fig1,ew_fig2=group_calc.analysis_group_ret(factor_df, ret_df, factor_name, n_groups=5, benchmark=benchmark, mktmv_df=mkmtv)

mw_backtest_df,mw_fig1,mw_fig2=group_calc.analysis_group_ret(factor_df, ret_df, factor_name, n_groups=5, mktmv_df=mkmtv)