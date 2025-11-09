#  cal barra ret regression

from barra_cne5_factor import GetData
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

import preprocess
import group_calc
import factor_analysis

index_data = GetData._get_index()
indexv1 = index_data[index_data['stock_code'] == '000001.XSHG'].copy()
indexv1 = indexv1.sort_values('trade_date',ascending=True)
indexv1 = indexv1.drop_duplicates(subset=['trade_date'], keep='first')
indexv1['datetime'] = pd.to_datetime(indexv1['trade_date'])
indexv1['year'] = indexv1['datetime'].dt.year
indexv1['month'] = indexv1['datetime'].dt.month
indexv1['week'] = indexv1['datetime'].dt.isocalendar().week
indexv1['trade_date'] = indexv1['datetime'].dt.date

date_frame = pd.DataFrame(indexv1[['trade_date','year','month','week']])
print("test")

START_YEAR = 2015


total_df = GetData._get_price_()

# get barra factor
# beta
beta_df = pd.read_parquet("F:\\work\\Projectpycharm\\MA_factor\\factor_data\\BARRA\\beta.parquet")# beta
# beta_df['trade_date'] = pd.to_datetime(beta_df['trade_date']).dt.date
btop_df = pd.read_parquet("F:\\work\\Projectpycharm\\MA_factor\\factor_data\\BARRA\\BTOP.parquet") # book to price (value)
# btop_df['trade_date'] = pd.to_datetime(btop_df['trade_date']).dt.date
resvol_df = pd.read_parquet("F:\\work\\Projectpycharm\\MA_factor\\factor_data\\BARRA\\RESVOL.parquet") # residual
# resvol_df['trade_date'] = pd.to_datetime(resvol_df['trade_date']).dt.date
rstr_df = pd.read_parquet("F:\\work\\Projectpycharm\\MA_factor\\factor_data\\BARRA\\RSTR.parquet") # momentum
# rstr_df['trade_date'] = pd.to_datetime(rstr_df['trade_date']).dt.date
size_df = pd.read_parquet("F:\\work\\Projectpycharm\\MA_factor\\factor_data\\BARRA\\LNCAP.parquet") # size
# size_df['trade_date'] = pd.to_datetime(size_df['trade_date']).dt.date
nonlrsize_df = pd.read_parquet("F:\\work\\Projectpycharm\\MA_factor\\factor_data\\BARRA\\NLSIZE.parquet") # non linear size
# nonlrsize_df['trade_date'] = pd.to_datetime(nonlrsize_df['trade_date']).dt.date
leverage_df = pd.read_parquet("F:\\work\\Projectpycharm\\MA_factor\\factor_data\\BARRA\\LEVERVAGE.parquet") # levervage
# leverage_df ['trade_date'] = pd.to_datetime(leverage_df['trade_date']).dt.date
liquidity_df = pd.read_parquet("F:\\work\\Projectpycharm\\MA_factor\\factor_data\\BARRA\\LIQUIDITY.parquet") # liquidity
# liquidity_df ['trade_date'] = pd.to_datetime(liquidity_df['trade_date']).dt.date



# 这一部分要转成parquet 来进行读取

total_df = pd.merge(total_df, beta_df, on=['trade_date','stock_code'], how='left')
total_df = pd.merge(total_df, btop_df, on=['trade_date','stock_code'], how='left')
total_df = pd.merge(total_df, resvol_df, on=['trade_date','stock_code'], how='left')
total_df = pd.merge(total_df, rstr_df, on=['trade_date','stock_code'], how='left')
total_df = pd.merge(total_df, size_df,on=['trade_date','stock_code'], how='left')
total_df = pd.merge(total_df, nonlrsize_df, on=['trade_date','stock_code'], how='left')
total_df = pd.merge(total_df, leverage_df, on=['trade_date','stock_code'], how='left')
total_df = pd.merge(total_df, liquidity_df, on=['trade_date','stock_code'], how='left')




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
    # 需要修改
    df[['BETA','BTOP_pp','RESVOL','RSTR_pp','LNCAP','NLSIZE','LEVERAGE_pp','LIQUIDITY_pp']] = df[['BETA','BTOP_pp','RESVOL','RSTR_pp','LNCAP','NLSIZE','LEVERAGE_pp','LIQUIDITY_pp']].ffill()
    # get return
    df = pd.merge(df, trade_date, on=['trade_date'], how='right')
    # 后续添加判断，如果是日频计算需要改变
    df['period_ret_O'] = (df['open_fq'].shift(-1) / df['open_fwd_fq']) - 1
    df['current_ret'] = (df['open_fwd_fq'] / df['open_fq'].shift()) - 1
    return df

def get_trade_date(df, start_year, period=None):
    df = df[df['year'] >= start_year]
    period_date = None
    if period in ['week', 'month']:
        period_date = df.groupby(['year','week'])['trade_date'].last()
    elif type(period) == int:
        period_date = df[::period]['trade_date']

    return pd.DataFrame(period_date)

trade_date = get_trade_date(date_frame, START_YEAR, period='week')



g_stock = total_df.groupby('stock_code',group_keys=False)
period_total_df = g_stock.apply(get_fwd_data, trade_date)
period_total_dfv2 = period_total_df[period_total_df['paused']==0].copy()



# period_total_dfv3 = period_total_dfv2.dropna(subset=['BETA','BTOP_pp','RESVOL','RSTR_pp','LNCAP','NLSIZE','LEVERAGE_pp','LIQUIDITY_pp','current_ret'])

# get period factor regression
barra_cols = ['BETA','BTOP_pp','RESVOL','RSTR_pp','LNCAP','NLSIZE','LEVERAGE_pp','LIQUIDITY_pp'] # 有些没有中性化和标准化，后续需要修改
g_section = period_total_dfv2.groupby('trade_date', group_keys=False)


ret_dict = {}

def regrs_section_barra(df, ret_dict):
    df = df.dropna(subset=['BETA','BTOP_pp','RESVOL','RSTR_pp','LNCAP','NLSIZE','LEVERAGE_pp','LIQUIDITY_pp','current_ret'])
    if len(df) == 0:
        return
    date = df['trade_date'].values[0]
    X = df[barra_cols]
    y = df['current_ret']
    model_barra = LinearRegression(fit_intercept=False)
    model_barra.fit(X, y)
    ret_dict[date] = model_barra.coef_

g_section.apply(regrs_section_barra, ret_dict)



# get current ret for analyzing ret





# cal portfolio ret
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

def get_fwd_datav2(df, trade_date):
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


g_stock = ohlcvm_df.groupby('stock_code',group_keys=False)

ohlcvm_dfv2 = g_stock.apply(get_fwd_datav2, trade_date)

ret_df = ohlcvm_dfv2.rename(columns={'period_ret_O':'ret'})[['trade_date','stock_code','ret']].copy()
factorCF_df = pd.merge(ret_df, CF_fct, on=['trade_date','stock_code'])
PCF_fct = factorCF_df[['trade_date','stock_code','PCF']].copy()
VCF_fct = factorCF_df[['trade_date','stock_code','VCF']].copy()
ACF_fct = factorCF_df[['trade_date','stock_code','ACF']].copy()
TRCF_fct = factorCF_df[['trade_date','stock_code','TRCF']].copy()
TRCF_fct['TRCF'] = TRCF_fct['TRCF']
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


mw_group_ret = group_calc.get_group_ret(factor_df, ret_df, factor_name, 5, mkmtv)


print("test")


