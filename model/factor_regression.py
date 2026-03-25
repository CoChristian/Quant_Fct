import pandas as pd

from indicator_operator import FactorCompute, get_hist_data_4_factor_compute, timer, memorize, merge_data, save_data_to_table
from factor_neutral import get_data_from_multi_source, transfer_data_to_valid_and_not_valid
from func_operator import stand, std_winsor
from statsmodels.distributions.empirical_distribution import ECDF
from tqdm import tqdm
import numpy as np
from scipy.optimize import lsq_linear


def regression_4_score(all_factor_data, start_date, end_date, window_size, factors, r_name, factor_constr_lb, factor_constr_ub, overall_factor_name):
    if 'trade_date' not in all_factor_data.columns:
        all_factor_data = all_factor_data.reset_index()
    # import pdb
    # pdb.set_trace()
    trade_dates = all_factor_data['trade_date'].unique()

    trade_dates = sorted(trade_dates)
    overall_scores = []
    params = []
    for i, date in enumerate(tqdm(trade_dates)):
        if i >= window_size and date > start_date and date <= end_date:
            hist_dates = trade_dates[i-window_size: i]
            hist_factor_data = all_factor_data[all_factor_data.trade_date.map(lambda x: x in hist_dates)].copy()
            tmp_factor = all_factor_data[all_factor_data.trade_date == date].copy()
            hist_factor_data = hist_factor_data[hist_factor_data[r_name].notnull()]
            x = hist_factor_data[factors]
            y = hist_factor_data[r_name]

            lb = factor_constr_lb
            ub = factor_constr_ub
            try:
                model = lsq_linear(x, y, bounds=(lb, ub))
            except Exception as e:
                import pdb
                pdb.set_trace()
                pass
            tmp_factor[overall_factor_name] = (model.x * tmp_factor[factors]).sum(axis=1)
            param_info = dict(zip(factors, model.x))
            param_info.update({'trade_date': date})
            params.append(param_info)
            overall_scores.append(tmp_factor)
    param_df = pd.DataFrame(params)
    overall_score_df = pd.concat(overall_scores)

    return param_df, overall_score_df


def process_not_valid_data_and_merge(overall_score, invalid_data, output_score_name):
    if "trade_date" not in overall_score.columns:
        overall_score = overall_score.reset_index()
    if "trade_date" not in invalid_data.columns:
        invalid_data = invalid_data.reset_index()
    invalid_data[output_score_name] = -999
    all_data_score = pd.concat([overall_score, invalid_data], axis=0)
    # import pdb
    # pdb.set_trace()
    return all_data_score[['code', 'trade_date', output_score_name]]


def generate_one_term_return(data, one_week_momentum_name, output_name, limit=0, is_3_sigma_std=False, is_ecdf=False):
    """
    讲历史的数据向前shift 1周，作为 y
    :param data:
    :param one_week_momentum_name:
    :param output_name:
    :return:
    """

    data = data.sort_index(level=['code', 'trade_date'])
    data[output_name] = data.groupby(level='code')[one_week_momentum_name].shift(-1)

    if limit > 0:
        sd_fac = stand(data[output_name], limit)
    else:
        sd_fac = data[output_name].values
    if is_3_sigma_std:
        sd_win_fac = std_winsor(sd_fac)
    else:
        sd_win_fac = sd_fac
    if is_ecdf:
        fac_cdf_clf = ECDF(sd_win_fac)
        fac_cdf = fac_cdf_clf(sd_win_fac)
    else:
        fac_cdf = sd_win_fac
    data[output_name] = fac_cdf

    return data

def process_one_term_return(data, output_name, limit=0, is_3_sigma_std=False, is_ecdf=False):
    if limit > 0:
        sd_fac = stand(data[output_name], limit)
    else:
        sd_fac = data[output_name].values
    if is_3_sigma_std:
        sd_win_fac = std_winsor(sd_fac)
    else:
        sd_win_fac = sd_fac
    if is_ecdf:
        fac_cdf_clf = ECDF(sd_win_fac)
        fac_cdf = fac_cdf_clf(sd_win_fac)
    else:
        fac_cdf = sd_win_fac
    data[output_name] = fac_cdf

    return data

class RollingRegressionScore(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.start_date = param_info['start_date']
        self.end_date = param_info['end_date']
        self.source_data_infos = param_info['source_data_infos']
        self.invalid_infos = param_info['invalid_infos']
        self.window_size = param_info['window_size']
        self.independent_variable_infos = param_info['independent_variable_infos']
        self.dependent_variable = param_info['dependent_variable']
        self.output_score_name = param_info['output_score_name']
        self.save_info_4_score = param_info['save_info_4_score']
        self.save_info_4_param = param_info['save_info_4_param']

        self.operators = [
            {
                "func": get_data_from_multi_source,
                "param": {
                    "data_source_infos": self.source_data_infos,
                    "start_date": self.start_date,
                    "end_date": self.end_date,
                    "join_method": 'inner'
                },
                "input_data": {},
                "output": ['factor']
            },
            {
                "func": generate_one_term_return,
                "param": {
                    "one_week_momentum_name": "MomentumWeeks1",
                    "output_name": "OneTermReturn",
                },
                "input_data": {'data': 'factor'},
                "output": ['factor']
            },
            {
                "func": transfer_data_to_valid_and_not_valid,
                "param": {
                    "invalid_infos": self.invalid_infos

                },
                "input_data": {"data": "factor"},
                "output": ['valid_factor', 'invalid_factor']
            },
            {
                "func": regression_4_score,
                "param": {
                    "window_size": self.window_size,
                    "start_date": self.start_date,
                    "end_date": self.end_date,
                    "factors": [_['feature_name'] for _ in self.independent_variable_infos],
                    "r_name": self.dependent_variable,
                    "factor_constr_lb":  [_['lower_bound'] for _ in self.independent_variable_infos],
                    "factor_constr_ub": [_['upper_bound'] for _ in self.independent_variable_infos],
                    "overall_factor_name": self.output_score_name,
                },
                "input_data": {"all_factor_data": "valid_factor"},
                "output": ["params", "overall_score"],
            },
            # {
            #     "func": process_not_valid_data_and_merge,
            #     "param": {
            #         "output_score_name": self.output_score_name,
            #     },
            #     "input_data": {
            #         "overall_score": "overall_score",
            #         "invalid_data": "invalid_factor"
            #     },
            #     "output": ["all_data_score"]
            # },
            {
                "func": save_data_to_table,
                "param": {"engine": self.save_info_4_score['engine'], "table": self.save_info_4_score['table'],
                          "if_exists": self.save_info_4_score.get("if_exists", "append")},
                "input_data": {"data": "overall_score"},
                "output": ["overall_score"]
            },
            {
                "func": save_data_to_table,
                "param": {"engine": self.save_info_4_param['engine'], "table": self.save_info_4_param['table'],
                          "if_exists": self.save_info_4_param.get("if_exists", "append")},
                "input_data": {"data": "params"},
                "output": ["params"]
            },
        ]
        self.output_vars = ["overall_score", "params"]



