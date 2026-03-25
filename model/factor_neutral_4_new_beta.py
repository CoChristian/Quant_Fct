from func_operator import get_data_from_multi_source, pipline_sum_data_weight_weight, save_data_to_table, concat_data_process
from statsmodels.distributions.empirical_distribution import ECDF
from model.indicator_operator import FactorCompute
import pandas as pd
import numpy as np


def add_valid_tag(data, invalid_infos):
    invalid_tag = pd.Series([False for j in range(len(data))], index=data.index)

    for invalid_info in invalid_infos:

        feature_name = invalid_info['feature_name']
        type_ = invalid_info['type']
        feature_value = invalid_info['feature_value']
        if type_ == "equal":
            invalid_tag |= data[feature_name] == feature_value
        elif type_ == "big":
            invalid_tag |= data[feature_name] > feature_value
        elif type_ == "less":
            invalid_tag |= data[feature_name] < feature_value
        elif type_ == "big_equal":
            invalid_tag |= data[feature_name] >= feature_value
        elif type_ == "less_equal":
            invalid_tag |= data[feature_name] <= feature_value
        elif type_ == "not_equal":
            invalid_tag |= data[feature_name] != feature_value
        elif type_ == "not_top":
            data['rank'] = data.groupby('trade_date')[feature_name].rank(ascending=False)
            invalid_tag |= data["rank"].map(lambda x: x>feature_value)
        else:
            pass

    data['valid_tag'] = ~invalid_tag
    return data


def industry_standardized_factor(data, feature_neutral_infos):
    index_names = data.index.names
    data = data.reset_index()
    
    for feature_info in feature_neutral_infos:
        limit_value = feature_info['limit_value']
        reverse = feature_info['reverse']
        output_name = feature_info['output_name']
        featuren_name = feature_info['feature_name']
        sort_func = feature_info['sort_func']
        industry_name = feature_info['industry_name']
        data[industry_name] = data[industry_name].fillna(20)
        data[industry_name] = data[industry_name].map(int)
        data[industry_name] = data[industry_name].map(lambda x: 45 if x == 50 else x)

        neutral_data = data.groupby(['trade_date', industry_name]).apply(
            lambda x: sd_win_sort(x, feature_name=featuren_name, limit=limit_value, sort_func=sort_func, reverse=reverse))
        neutral_data = neutral_data.droplevel(['trade_date', industry_name])
        data[output_name] = neutral_data


    data = data.set_index(index_names)
    return data

def win(x, x_, trim=0.2, limit='both'):
    """
    Winsorize top and/or tail n% data

    Params:
    --------------
    x: pd.Series, valid data to winsorize
    x_: pd.Series, no valid data to winsorize
    trim: float, percentage to winsorize

    limit: str, one of ['both','ub','lb'], indicating direcatory to winsorize

    Returns:
    ---------------

    y: pd.Series,  winsorized Data
    """
    y = x.copy()
    y_ = x_.copy()
    x.dropna()
    x_.dropna()
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
        y_[x_ != None] = qtrim_mid
    else:
        if limit == 'both':
            y[x < qtrim_min] = qtrim_min
            y[x > qtrim_max] = qtrim_max
            y_[x_ < qtrim_min] = qtrim_min
            y_[x_ > qtrim_max] = qtrim_max
        elif limit == 'ub':
            y[x > qtrim_max] = qtrim_max
            y_[x_ > qtrim_max] = qtrim_max

        elif limit == 'lb':
            y[x < qtrim_min] = qtrim_min
            y_[x_ < qtrim_min] = qtrim_min

    return y, y_


def stand(z, z_, trim_num, limit='both'):
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
    x, x_ = win(z, z_, trim_num, limit)
    try:
        x_mean = np.nanmean(x)
        if len(x) == 0 or np.nan in x:
            print('bug')
    except:
        print('bug')
        print('bug')
    x_std = np.nanstd(x)
    y = (x - x_mean) / x_std
    y_ = (x_ - x_mean) / x_std
    return y, y_

def std_winsor(z, z_):
    """
    3 sigma winsorize series
    """
    tmp_z = z.copy()
    tmp_z = tmp_z.dropna()

    z_std = np.std(tmp_z)
    min_std = -3 * z_std
    max_std = 3 * z_std
    z[z < min_std] = min_std
    z[z > max_std] = max_std
    z[z == None] = min_std

    z_[z_ < min_std] = min_std
    z_[z_ > max_std] = max_std
    z_[z_ == None] = min_std

    return z, z_


def sd_win_sort(raw_fac, feature_name, limit=0.05, sort_func=ECDF, reverse=False):
    """
    Perform  5% trime, zscore and 3 sigma winsorization and Ecdf sort on a group of a single factor
    """
    valid_fac = raw_fac[raw_fac['valid_tag']][feature_name]
    novalid_fac = raw_fac[~raw_fac['valid_tag']][feature_name]
    fac_idx = raw_fac.index
    valid_fac_idx = valid_fac.index
    novalid_fac_idx = novalid_fac.index
    sd_valid_fac, sd_novalid_fac = stand(valid_fac, novalid_fac, limit)
    if reverse:
        sd_valid_fac = -sd_valid_fac
        sd_novalid_fac = -sd_novalid_fac
    sd_valid_win_fac, sd_novalid_win_fac = std_winsor(sd_valid_fac, sd_novalid_fac)
    fac_cdf_clf = sort_func(sd_valid_win_fac)
    valid_fac_cdf = fac_cdf_clf(sd_valid_win_fac)
    novalid_fac_cdf = fac_cdf_clf(sd_novalid_win_fac)
    valid_fac_cdf_series = pd.Series(valid_fac_cdf, index=valid_fac_idx)
    novalid_fac_cdf_series = pd.Series(novalid_fac_cdf, index=novalid_fac_idx)
    fac_cdf_series = pd.concat([valid_fac_cdf_series, novalid_fac_cdf_series])

    return fac_cdf_series.loc[fac_idx]

def data_process(data, tgt_factors):
    data = data.fillna(-999)
    return data[tgt_factors]


class IndustryNeutralWeeklyFactor(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.start_date = param_info['start_date']
        self.end_date = param_info['end_date']
        self.source_data_infos = param_info['source_data_infos']

        self.invalid_infos = param_info['invalid_infos']
        self.neutral_infos = param_info['neutral_infos']
        self.sum_infos = param_info['sum_infos']
        self.output_factors = param_info['output_factors']
        self.save_info = param_info['save_info']
        self.operators = [
            {
                "func": get_data_from_multi_source,
                "param": {
                    "data_source_infos": self.source_data_infos,
                    "start_date": self.start_date,
                    "end_date": self.end_date
                },
                "input_data": {},
                "output": ['weekly_indicator']
            },

            {
                "func": add_valid_tag,
                "param": {
                    "invalid_infos": self.invalid_infos

                },
                "input_data": {"data": "weekly_indicator"},
                "output": ['weekly_indicator']
            },
            {
                "func": industry_standardized_factor,
                "param": {
                    "feature_neutral_infos": self.neutral_infos,
                },
                "input_data": {"data": "weekly_indicator"},
                "output": ["weekly_indicator"]
            },
            {
                "func": pipline_sum_data_weight_weight,
                "param": {
                    "sum_infos": self.sum_infos,
                },
                "input_data": {'data': "weekly_indicator"},
                "output": ["weekly_industry_neural_factor"]
            },
            {
                "func": data_process,
                "param": {"tgt_factors": self.output_factors},
                "input_data": {"data": "weekly_industry_neural_factor"},
                "output": ["weekly_industry_neural_factor"]

            },
            {
                "func": save_data_to_table,
                "param": {"engine": self.save_info['engine'], "table": self.save_info['table'], "if_exists": self.save_info.get("if_exists", "append")},
                "input_data": {"data": "weekly_industry_neural_factor"},
                "output": ["weekly_industry_neural_factor"]
            }
        ]
        self.output_vars = ["weekly_industry_neural_factor"]