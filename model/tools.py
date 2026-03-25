from time import time
import pickle as pkl
import numpy as np
import os
import logging
import functools
import pandas as pd
from matplotlib import pyplot as plt
from joblib import Parallel, delayed


def show_process(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        start = time()
        print('On processing %s' % func.__name__)
        r = func(*args, **kw)
        # print("cost time :", time()-start)
        return r
    return wrapper


def apply_parallel(df_grouped, func, n_jobs=10, **kw):
    res = Parallel(n_jobs=n_jobs)(delayed(func)(group, **kw) for name, group in df_grouped)
    return pd.concat(res)


def jqcode_to_tscode(code):
    if code.split('.')[1] == 'XSHE':
        code = code.split('.')[0] +'.SZ'
    else:
        code = code.split('.')[0] +'.SH'
    return code


def tscode_to_jqcode(code):
    if code.split('.')[1] == 'SZ':
        code = code.split('.')[0] +'.XSHE'
    elif code.split('.')[1] == 'SH':
        code = code.split('.')[0] +'.XSHG'
    return code


def myround(x):
    conds = [x <= 0.15,
             (x > 0.15) & (x <= 0.2),
             (x > 0.2) & (x <= 0.3),
             (x > 0.3) & (x <= 0.4),
             (x > 0.4) & (x <= 0.5),
             (x > 0.5) & (x <= 0.6),
             (x > 0.6) & (x <= 0.7),
             (x > 0.7) & (x <= 0.8),
             x > 0.8]
    funcs = [lambda y: np.ceil(y * 100)/100,
             lambda y: 0.2,
             lambda y: 0.3,
             lambda y: 0.4,
             lambda y: 0.5,
             lambda y: 0.6,
             lambda y: 0.7,
             lambda y: 0.8,
             lambda y: 1.0]
    x = np.piecewise(x, conds, funcs)
    return x


def win(x, trim=0.2, limit = 'both'):
    """
    Winsorize top and/or tail n% data 
    
    Params:
    --------------
    x: pd.Series, data to winsorize 
    
    trim: float, percentage to winsorize
    
    limit: str, one of ['both','ub','lb'], indicating direcatory to winsorize 
    
    Returns:
    ---------------
    
    y: pd.Series,  winsorized Data  
    """
    y = x.copy()
    x.dropna()
    if (trim < 0) | (trim > 0.5):
        print("trimming must be reasonable")
        exit()
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
    """
    1. Winsorize data series
    2. Z_score series
    
    Params:
    --------------
    z: pd.Series, data to std 
    
    trim: float, percentage to winsorize
    
    limit: str, one of ['both','ub','lb'], indicating direcatory to winsorize 
    
    Returns:
    ---------------
    
    y: pd.Series,  std Data  
    
    """
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
    """
    3 sigma winsorize series
    """
    tmp_z = z.copy()
    tmp_z = tmp_z.dropna()
    z_std = np.std(tmp_z)
    min_std = -3 * z_std
    max_std = 3 * z_std
    z[z<min_std] = min_std
    z[z>max_std] = max_std
    z[z==None] = min_std
    return z


def save_obj(file, obj, **kwargs):
    with open(file, 'wb') as f:
        pkl.dump(obj, f)


def load_obj(file, **kwargs):
    with open(file, 'rb') as f:
        obj = pkl.load(f)
    return obj


def dict_prod(source, target):
    assert source.keys() == target.keys(), f'len source{len(source)}, len target {len(target)}'
    return sum(source.get(key, 0) * target.get(key, 0) for key in source.keys() | target.keys())


def data_process(opt_monthly_data_path, raw_factor_path, trade_monthly_data_path):
    opt_monthly_data = pd.read_csv(opt_monthly_data_path)
    trade_monthly_data = pd.read_csv(trade_monthly_data_path)
    raw_factor = pd.read_csv(raw_factor_path, sep=',')
    return opt_monthly_data, trade_monthly_data, raw_factor


def get_code_share_position(wgt_opt_df, date=None):
    if date:
        wgt_opt_df = wgt_opt_df[wgt_opt_df.trade_date == date]
    num = len(wgt_opt_df)
    position = {}
    wgt_opt_df_columns = wgt_opt_df.columns
    assert 'ts_code' in wgt_opt_df_columns, wgt_opt_df_columns
    assert 'round_shares' in wgt_opt_df_columns, wgt_opt_df_columns
    for i in range(num):
        tmp = wgt_opt_df.iloc[i]
        position[tmp['ts_code']] = tmp['round_shares']
    return position

def compute_weight_by_position(position, close_price, all_cash):
    weight = {}
    all_position_price = dict_prod(position, close_price)
    print(f'总市值 {all_cash}, 持仓市值 {all_position_price}, 现金{all_cash-all_position_price}')
    for key, val in position.items():
        weight[key] = (val * close_price[key]) / all_cash
    assert sum(weight.values()) >= 0.9999999999, print(sum(weight.values()))
    return weight

def get_col_val(df, col_name):
    if type(col_name) == str:
        df = df[['ts_code', col_name]]
        factor_score = {}
        for i in range(len(df)):
            tmp = df.iloc[i]
            code = tmp['ts_code']
            score = tmp[col_name]
            factor_score[code] = score
    elif type(col_name) == list:
        df = df[['ts_code'] + col_name]
        factor_score = {}
        for i in range(len(df)):
            tmp = df.iloc[i]
            code = tmp['ts_code']
            score = sum([tmp[fac] for fac in col_name])
            factor_score[code] = score
    return factor_score


def trans_order2everbright(order):
    order = order[['ts_code', 'name', 'order_share_diff']].copy()
    code = norm_code(order['ts_code'])
    order.loc[:, 'ts_code'] = code
    order_sale_part = order[order.order_share_diff < 0].copy()
    order_buy_part = order[order.order_share_diff > 0].copy()
    sale_all = sum(order_sale_part['order_share_diff'])
    buy_all = sum(order_buy_part['order_share_diff'])
    logging.info("sale order shares sum {}".format(sale_all))
    logging.info("buy order shares sum {}".format(buy_all))
    tmp = order_sale_part['order_share_diff'].to_numpy()
    order_sale_part.loc[:, 'order_share_diff'] = np.abs(tmp)
    order_sale_part.loc[:, 'weight'] = norm_weight(order_sale_part['order_share_diff'])
    order_buy_part.loc[:, 'weight'] = norm_weight(order_buy_part['order_share_diff'])
    order_sale_part = norm_columns(order_sale_part, sale_all)
    order_buy_part = norm_columns(order_buy_part, buy_all)
    return order_sale_part, order_buy_part, sale_all, buy_all


def trans_order2guotai(order):
    code = norm_code_guojun(order['ts_code'])
    order.loc[:, 'ts_code'] = code
    order = order.rename(columns={'order_share_diff': 'quantity', 'ts_code': 'code'})
    pre_hold_code = order[order.pre_hold_flag != 0]
    pre_hold_code_without_trade = pre_hold_code[pre_hold_code.quantity >= 0]
    pre_hold_code_without_trade = pre_hold_code_without_trade[['code', 'quantity']].copy()
    order = order[['code', 'quantity']].copy()
    order_sale_part = order[order.quantity < 0].copy()
    order_buy_part = order[order.quantity > 0].copy()
    if not pre_hold_code_without_trade.empty:
        pre_hold_code_without_trade.quantity = 0
        order_sale_part = order_sale_part.append(pre_hold_code_without_trade, ignore_index=True, sort=False)
        assert len(order_sale_part) == len(pre_hold_code), '卖出持仓数目与上期不符'
    sale_all = sum(order_sale_part['quantity'])
    buy_all = sum(order_buy_part['quantity'])
    logging.info("sale order shares sum {}".format(sale_all))
    logging.info("buy order shares sum {}".format(buy_all))
    tmp = order_sale_part['quantity'].to_numpy()
    order_sale_part.loc[:, 'quantity'] = np.abs(tmp)
    return order_sale_part, order_buy_part, sale_all, buy_all


def norm_code_guojun(serise):
    func = lambda x: '.'.join(x.split('.')[::-1])
    serise = serise.apply(func)
    serise = serise.to_list()
    return serise


def norm_code(serise):
    func = lambda x: str(x.split(r'.')[0])
    serise = serise.apply(func)
    serise = serise.to_list()
    return serise


def norm_weight(serise):
    sum_ = sum(serise)
    wgt = 100 * serise.to_numpy()/sum_
    return wgt


def norm_columns(order, all_shares):
    shares = order['weight'].to_numpy() * all_shares/100
    order = {
        '证券代码': order['ts_code'].to_numpy(),
        '证券名称': order['name'].to_numpy(),
        '权重': order['weight'].to_numpy(),
        '买入价位': '卖一价',
        '卖出价位': '买一价',
        '备注': shares
    }
    order = pd.DataFrame(order)
    return order


def update_trading_data():
    sr1 = r'./data/zz500/source/000905.SH_trading_data.txt'
    sr2 = r'./data/zz500/source/000905.SH_update_trading_data.txt'
    func1 = lambda x: pd.read_csv(x, sep='\t')
    df1 = func1(sr1)
    df2 = func1(sr2)
    max_date = max(df1.trade_date.to_list())
    df2 = df2[df2.trade_date > max_date]
    df2 = df2[["ts_code", "trade_date", "close", "close_hfq", "pct_chg_hfq", "pb", "total_mv", "circ_mv"]]
    print('原本的长度 :', len(df1))
    df = df1.append((df2), sort=False)
    df = df.sort_values(['ts_code', 'trade_date'])
    print('len before :', len(df))
    df = df.drop_duplicates(subset=['ts_code', 'trade_date'])
    print('len after :', len(df))
    dates = df['trade_date'].drop_duplicates().to_list()
    dates.sort()
    print('after update last dates is ', dates[-10:])
    df.to_csv(sr1, sep='\t', index=False)


def update_paused():
    sr1 = r'./data/zz500/source/000905.SH_trading_data_new.txt'
    sr2 = r'./tmp/zz500_factor_daily/raw_factor_peer_group.csv'
    func1 = lambda x: pd.read_csv(x, sep='\t')
    func2 = lambda x: pd.read_csv(x, sep=',')
    df1 = func1(sr1)
    paused = df1.apply(lambda x: 2 if abs(x['close'] / x['pre_close'] - 1) >= 0.095 else 0, axis=1)
    df1['paused'] = paused
    df1 = df1[['ts_code', 'trade_date', 'paused']]
    df2 = func2(sr2)
    df2 = df2.merge(df1, how='left', on=['ts_code', 'trade_date'])
    df2 = df2.fillna({'paused': 3})
    df2.to_csv(r'./tmp/zz500_factor_daily/raw_factor_new.csv', index=False)
    print(df1.columns)


def rolling_back():
    func1 = lambda x: pd.read_csv(x, sep='\t')
    func2 = lambda x: pd.read_csv(x, sep=',')
    root = r'./tmp/zz500_factor_week2/'
    dirs = os.listdir(root)
    for f in dirs:
        fp = os.path.join(root, f)
        if '.csv' in fp:
            df = func2(fp)
            if 'trade_date' in df.columns:
                len_bf = len(df)
                print(df.isnull().any())
                print(f'len of judge before {len(df)}')
                df = df[df.trade_date <= 20200308]
                len_af = len(df)
                print(f'len of  after {len(df)}')
                print(df.isnull().any())
                if len_af != len_bf:
                    df.to_csv(fp, index=False)


def get_hs300_return():
    import tushare as ts
    TOKEN = '4352ff55035eaf86db50e7e60e272d5898c65aadb2707dd63c40e3cc'  ## Jason'soken
    ts.set_token(TOKEN)
    pro = ts.pro_api()
    df = pro.index_daily(ts_code='399300.SZ', start_date='20050101', end_date='20200313')
    df = df[['ts_code', 'trade_date', 'pct_chg', 'close']]
    df = df.sort_values(['ts_code', 'trade_date'])
    return df


def plot_strategy_curve():
    ret_fp = r'./result/zz500_week2_weighted_add_penalty_from2016/mrawret.csv'
    mrawret = pd.read_csv(ret_fp)
    max_date = str(min(mrawret['trade_date'].to_list()))
    hs300_ret = get_hs300_return()
    hs300_ret = hs300_ret[hs300_ret.trade_date >= max_date]
    hs300_ret['trade_date'] = pd.to_datetime(hs300_ret.trade_date, format='%Y%m%d')
    hs300_ret_rate = hs300_ret.pct_chg.to_numpy()
    hs300_cum = []
    tmp = 3000000
    for i in hs300_ret_rate:
        tmp *= (1+i/100)
        hs300_cum.append(tmp)
    hs300_ret['hs300_cum'] = hs300_cum
    amount = mrawret.loc[0, 'factor_model_cum']
    mDate_ret = [str(mrawret.loc[x, 'trade_date']) for x in range(mrawret.shape[0])]
    mrawret['mrawret'] = mDate_ret
    mrawret['mrawret'] = pd.to_datetime(mrawret['mrawret'])
    fig, ax = plt.subplots()
    ax.plot(mrawret['mrawret'], mrawret.factor_model_cum / amount, color="red", linewidth=1.25,
            label="Factor Model")
    ax.plot(mrawret['mrawret'], mrawret.index_cum / amount, color="blue", linewidth=1.25, label="zz500")
    ax.plot(hs300_ret['trade_date'], hs300_ret.hs300_cum / amount, color="green", linewidth=1.25, label="hs300")
    ax.legend(loc=2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.set_title('The Cumulative Return for the Factor Model and the Index Strategy')
    plt.show()
    fig.savefig('./img.png', dpi=100)


def main():
    pass


if __name__ == '__main__':
    update_trading_data()
