import pandas as pd
import numpy as np

Data_chart = pd.read_csv("F:\\work\\Data\\Database_to_csv\\stock_price_history_total\\stock_price_history251014.csv")
Valuation_chart = pd.read_csv("F:\\work\\Data\\Database_to_csv\\valuation_total\\valuation251014.csv")
val_data = Valuation_chart.copy()
val_data['datetime'] = pd.to_datetime(val_data['day'])
val_data['trade_date'] = val_data['datetime'].dt.date
mkt_data = val_data[['trade_date','code','circulating_cap']]
fct_data = Data_chart.copy()
fct_data['datetime'] = pd.to_datetime(fct_data['date'])
fct_data['trade_date'] = fct_data['datetime'].dt.date
fct_data['close_fq'] = fct_data['close'] * fct_data['factor']
fct_data = pd.merge(fct_data, mkt_data, on=['trade_date','code'],how='left')
fct_data['turnover'] = (fct_data['volume'] / 10000) / fct_data['circulating_cap']
stock_data = fct_data.groupby('code', group_keys=False)



def cal_ma(df):
    df['ma5'] = df['close_fq'].rolling(5).mean()
    df['ma10'] = df['close_fq'].rolling(10).mean()
    df['ma20'] = df['close_fq'].rolling(20).mean()
    df['ma60'] = df['close_fq'].rolling(60).mean()
    df['ma120'] = df['close_fq'].rolling(120).mean()
    std_ma = np.std(df[['close_fq','ma5','ma10','ma20','ma60','ma120']], axis=1)
    df['PCF'] =-1 * np.log(1 + std_ma)
    df['vma5'] = df['volume'].rolling(5).mean()
    df['vma10'] = df['volume'].rolling(10).mean()
    df['vma20'] = df['volume'].rolling(20).mean()
    df['vma60'] = df['volume'].rolling(60).mean()
    df['vma120'] = df['volume'].rolling(120).mean()
    std_vma = np.std(df[['volume','vma5','vma10','vma20','vma60','vma120']], axis=1)
    df['VCF'] =-1 * np.log(1 + std_vma)
    df['ama5'] = df['money'].rolling(5).mean()
    df['ama10'] = df['money'].rolling(10).mean()
    df['ama20'] = df['money'].rolling(20).mean()
    df['ama60'] = df['money'].rolling(60).mean()
    df['ama120'] = df['money'].rolling(120).mean()
    std_ama = np.std(df[['money','ama5','ama10','ama20','ama60','ama120']], axis=1)
    df['ACF'] =-1 * np.log(1 + std_ama)
    df['tma5'] = df['turnover'].rolling(5).mean()
    df['tma10'] = df['turnover'].rolling(10).mean()
    df['tma20'] = df['turnover'].rolling(20).mean()
    df['tma60'] = df['turnover'].rolling(60).mean()
    df['tma120'] = df['turnover'].rolling(120).mean()
    std_tma = np.std(df[['turnover','tma5','tma10','tma20','tma60','tma120']], axis=1)
    df['TRCF'] =-1 * np.log(1 + std_tma)
    return df

fctCF_data = stock_data.apply(cal_ma)
fct_tocsv = fctCF_data[['trade_date','code','PCF','VCF','ACF','TRCF']]
fct_tocsv.to_csv("F:\\quant_factor\\factor_backtest-main\\data\\MAfct_data\\fct_data\\CF_data.csv", index=False, encoding='utf-8-sig')