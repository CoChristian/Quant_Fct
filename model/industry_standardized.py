from statsmodels.distributions.empirical_distribution import ECDF
import statsmodels.api as sm
import numpy as np
from statsmodels.regression.rolling import RollingOLS
import hashlib
import json
import time
import datetime
from statsmodels.distributions.empirical_distribution import ECDF
import pandas as pd
def win(x, trim, limit='both'):
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
        qtrim_max = x.quantile(1 - trim)
    except:
        import pdb
        pdb.set_trace()
    if trim > 0.5:
        y[x != None] = qtrim_mid
    else:
        if limit == 'both':
            y[x < qtrim_min] = qtrim_min
            y[x > qtrim_max] = qtrim_max
        elif limit == 'ub':
            y[x > qtrim_max] = qtrim_max
        elif limit == 'lb':
            y[x < qtrim_min] = qtrim_min
    return y
def stand(z, trim_num, limit='both'):
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
    x = win(z, trim_num, limit)
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
    min_std = -7 * z_std
    max_std = 7 * z_std
    z[z < min_std] = min_std #对数据截尾
    z[z > max_std] = max_std
    z[z == None] = min_std
    return z
def sd_win_sort(raw_fac, limit=0.05, sort_func=ECDF, reverse=False):
    """
    Perform  5% trime, zscore and 3 sigma winsorization and Ecdf sort on a group of a single factor
    """

    idx = raw_fac.index
    sd_fac = stand(raw_fac, limit)##标准化
    if reverse:
        sd_fac = - sd_fac
    sd_win_fac = std_winsor(sd_fac)
    fac_cdf_clf = sort_func(sd_win_fac)
    fac_cdf = fac_cdf_clf(sd_win_fac)

    fac_cdf_series = pd.Series(fac_cdf, index=idx)
    return fac_cdf_series
def industry_standardized_factor(data, feature_neutral_infos):
    # index_names = data.index.names
    data = data.reset_index()
    for feature_info in feature_neutral_infos:
        limit_value = feature_info['limit_value']
        reverse = feature_info['reverse']
        output_name = feature_info['output_name']
        featuren_name = feature_info['feature_name']
        sort_func = feature_info['sort_func']
        industry_name = feature_info['industry_name']
        data[output_name] = data.groupby(['trade_date', industry_name])[featuren_name].apply(
            lambda x: sd_win_sort(x, limit=limit_value, sort_func=ECDF, reverse=reverse))
    # data = data.set_index(index_names)
    return data
