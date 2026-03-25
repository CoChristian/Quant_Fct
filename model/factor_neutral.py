import pandas as pd
import math
from model.indicator_operator import FactorCompute, get_hist_data_4_factor_compute, timer, memorize, merge_data, \
    standard_and_merge_data, save_data_to_table
from peer_neutral import *
from statsmodels.distributions.empirical_distribution import ECDF
from func_operator import *






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
            # {'func': get_hist_data_4_factor_compute,
            #  'param': {
            #      "read_engine": "",
            #      "save_engine": "",
            #      "start_date": 0,
            #      "end_date": 0,
            #      "table": "all_data_test_all_mkt_indicator",
            #      # "table": "all_trading_data_monthly",
            #      "field": [ ],
            #      "index": ['trade_date', "code"],
            #      "hist_year": 0,
            #      "name_dict": {}},
            #  "input_data": {},
            #  "output": ['weekly_indicator']},
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
            # {
            #     "func": cal_mv_weight,
            #     "param": {
            #         "circ_mv_name": "CirculatingMarketCap",
            #         "total_mv_name": "MarketCap",
            #         "raw_weight_name": "CSISmallcap500Weight",
            #         "output_name": "CSISmallcap500MvWeight"
            #     },
            #     "input_data": {"data": "weekly_indicator"},
            #     "output": ['weekly_indicator']
            # },
            # {
            #     "func": sum_data_with_weight,
            #     "param": {
            #         "features": ["short_time_momentum", "long_time_momentum"],
            #         "weights": [-1, 1],
            #         "output_name": "long_minus_short"
            #     },
            #     "input_data": {"data": "weekly_indicator"},
            #     "output": ["weekly_indicator"]
            # },
            # {
            #     "func": merge_data,
            #     "param": {
            #     },
            #     "input_data": {"1": "weekly_indicator", "2": "weight"},
            #     "output": ['weekly_indicator']
            # },
            {
                "func": transfer_data_to_valid_and_not_valid,
                "param": {
                    "invalid_infos": self.invalid_infos

                },
                "input_data": {"data": "weekly_indicator"},
                "output": ['valid_weekly_indicator', 'not_valid_weekly_indicator']
            },
#             {
#                 "func": gen_mktcap_bin,
#                 "param": {
#                     "mv_name": "LogMktCap"

#                 },
#                 "input_data": {"data": "valid_weekly_indicator"},
#                 "output": ['valid_weekly_indicator']
#             },            
            {
                "func": industry_standardized_factor,
                "param": {
                    "feature_neutral_infos": self.neutral_infos,
                },
                "input_data": {"data": "valid_weekly_indicator"},
                "output": ["valid_weekly_indicator"]
            },
            {
                "func": pipline_sum_data_weight_weight,
                "param": {
                    "sum_infos": self.sum_infos,
                },
                "input_data": {'data': "valid_weekly_indicator"},
                "output": ["valid_weekly_indicator"]
            },
            {
                "func": concat_data_process,
                "param": {"tgt_factors": self.output_factors},
                "input_data": {"valid_data": "valid_weekly_indicator", "invalid_data": "not_valid_weekly_indicator"},
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

class IndustryNeutralWeeklyGroupFactor(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.start_date = param_info['start_date']
        self.end_date = param_info['end_date']
        self.source_data_infos = param_info['source_data_infos']

        self.invalid_infos = param_info['invalid_infos']
        self.neutral_infos = param_info['neutral_infos']
        self.group_infos = param_info['group_infos']
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
                "func": transfer_data_to_valid_and_not_valid,
                "param": {
                    "invalid_infos": self.invalid_infos

                },
                "input_data": {"data": "weekly_indicator"},
                "output": ['valid_weekly_indicator', 'not_valid_weekly_indicator']
            },
         
            {
                "func": group_feature,
                "param": {
                    "group_infos": self.group_infos,
                },
                "input_data": {"data": "valid_weekly_indicator"},
                "output": ["valid_weekly_indicator"]
            },
            {
                "func": pipline_sum_data_weight_weight,
                "param": {
                    "sum_infos": self.sum_infos,
                },
                "input_data": {'data': "valid_weekly_indicator"},
                "output": ["valid_weekly_indicator"]
            },
            {
                "func": industry_standardized_factor,
                "param": {
                    "feature_neutral_infos": self.neutral_infos,
                },
                "input_data": {"data": "valid_weekly_indicator"},
                "output": ["valid_weekly_indicator"]
            },
            {
                "func": concat_data_process,
                "param": {"tgt_factors": self.output_factors},
                "input_data": {"valid_data": "valid_weekly_indicator", "invalid_data": "not_valid_weekly_indicator"},
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
        
        
# def transfer_code_to_valid_invalid(factor_data, weight_name, nan_flag_name, end_flag_name, st_flag_name, listed_flag_name,
#                                   pause_flag_name):
#     valid_tag = factor_data[weight_name]> 0
#     valid_factor_data = factor_data[valid_tag]
#     invalid_factor_data = factor_data[~valid_tag]
#     return valid_factor_data, invalid_factor_data


class PeerFactor(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.peer_param = param_info['peer_param']
        self.corr_history_window_size = self.peer_param.get("corr_history_window_size", [180, 360, 720])
        self.corr_type = self.peer_param.get("corr_type", "pearson")
        self.n_peer_group = self.peer_param.get("n_peer_group", 20)
        self.add_industry_info = self.peer_param.get("add_industry_info", False)
        self.industry_names = self.peer_param.get("industry_names", [])
        self.raw_rtn = self.peer_param.get("raw_rtn", False)
        self.benchmark_code = self.peer_param.get("benchmark_code", "000905.XSHG")
        self.start_date = param_info['start_date']
        self.end_date = param_info['end_date']
        self.source_data_infos = param_info['source_data_infos']

        self.invalid_infos = param_info['invalid_infos']
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
                "func": transfer_data_to_valid_and_not_valid,
                "param": {
                    "invalid_infos": self.invalid_infos

                },
                "input_data": {"data": "weekly_indicator"},
                "output": ['valid_weekly_indicator', 'invalid_weekly_indicator', ]
            },

            {
                "func": get_hist_data_4_factor_compute,
                'param': {
                    "read_engine": "mysql+pymysql://root:swisschina@192.168.110.5:3306/factor_research_full_a_share",
                    "save_engine": "mysql+pymysql://root:swisschina@192.168.110.5:3306/factor_research_full_a_share",
                    "start_date": self.start_date,
                    "end_date": self.end_date,
                    # "table": "all_mkt_indicator",
                    "table": "daily_trading_data",
                    "field": ['trade_date', "code", "close", "paused"],
                    "index": ['trade_date', "code"],
                    "hist_year": math.ceil(self.corr_history_window_size[2]/360),
                    "name_dict": {}},
                "input_data": {},
                "output": ["daily_close_price"]
            },


            # {
            #     "func": get_hist_data_4_factor_compute,
            #     'param': {
            #         "read_engine": "mysql+pymysql://develop:haikuan_2025@localhost:3306/factor_research_full_a_share",
            #         "save_engine": "mysql+pymysql://develop:haikuan_2025@localhost:3306/factor_research_full_a_share",
            #         "start_date": 0,
            #         "end_date": 0,
            #         # "table": "all_mkt_indicator",
            #         "table": "daily_trading_data",
            #         "field": ['trade_date', "code", "close", 'paused'],
            #         "index": ['trade_date', "code"],
            #         "hist_year": (math.ceil(self.corr_history_window_size[2] / 360)),
            #         "name_dict": {}},
            #     "input_data": {},
            #     "output": ["daily_close_price"]
            # },
            {
                'func': get_hist_data_4_factor_compute,
                'param': {
                    "read_engine": "mysql+pymysql://root:swisschina@192.168.110.5:3306/factor_research_full_a_share",
                    "save_engine": "mysql+pymysql://root:swisschina@192.168.110.5:3306/factor_research_full_a_share",
                    "start_date": self.start_date,
                    "end_date": self.end_date,
                    "table": "index_level",
                    "field": ["trade_date", "code", 'close'],
                    "hist_year": 2,
                    "other_filter_info": {"field": "code", "type": "equal", "param": self.benchmark_code},
                    "name_dict": {}},
                "input_data": {},
                "output": ['benchmark_price']
            },
            {
                'func': cpt_ex_rtn_from_data,
                'param': {
                    "code_price_name": "close",
                    "index_price_name": "close",
                    "pause_flag_name": "paused"
                },
                "input_data": {
                    "all_raw_factors": "valid_weekly_indicator",
                    "daily_close_paused_data": "daily_close_price",
                    "index_close_data": "benchmark_price"
                },
                "output": ['ex_rtn']
            },
            # {
            #     'func': cpt_ex_rtn,
            #     'param': {
            #         "days_before": self.corr_history_window_size[2]+50,
            #     },
            #     "input_data": {
            #         "all_raw_factors": "valid_weekly_indicator",
            #     },
            #     "output": ['ex_rtn']
            # },

            {
                'func': cpt_dyna_peers,
                'param': {
                    "hist_window_size": self.corr_history_window_size,
                    "n_peer_group": self.n_peer_group,
                    "add_industry_info": self.add_industry_info,
                    "industry_names": self.industry_names
                },
                "input_data": {
                    "all_raw_factor": "valid_weekly_indicator",
                    "ex_rtn": "ex_rtn",
                },
                "output": ['peer_mapping']
            },
            {
                "func": save_data_to_table,
                "param": {"engine": self.save_info['engine'], "table": self.save_info['table'],
                          "if_exists": self.save_info.get("if_exists", "append"),
                          },
                "input_data": {"data": "peer_mapping"},
                "output": ["peer_mapping"]
            }
        ]
        self.output_vars = ['peer_mapping']





class PeerNeutralWeeklyFactor(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.start_date = param_info['start_date']
        self.end_date = param_info['end_date']
        self.source_data_infos = param_info['source_data_infos']
        self.peer_data_infos = param_info['peer_data_info']
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
                "func": get_data_from_multi_source,
                "param": {
                    "data_source_infos": self.peer_data_infos,
                    "start_date": self.start_date,
                    "end_date": self.end_date
                },
                "input_data": {},
                "output": ['peer_mapping']
            },
            {
                "func": transfer_data_to_valid_and_not_valid,
                "param": {
                    "invalid_infos": self.invalid_infos

                },
                "input_data": {"data": "weekly_indicator"},
                "output": ['valid_weekly_indicator', 'not_valid_weekly_indicator', ]
            },
            {
                # "func": peer_standardized_factor,
                "func": fast_peer_standardized_factor_v2,
                "param": {
                    "feature_neutral_infos": self.neutral_infos,
                },
                "input_data": {
                    "peer_mapping": "peer_mapping",
                    "valid_data": "valid_weekly_indicator"
                },
                "output": ["valid_weekly_indicator"]
            },
            {
                "func": pipline_sum_data_weight_weight,
                "param": {
                    "sum_infos": self.sum_infos,
                },
                "input_data": {'data': "valid_weekly_indicator"},
                "output": ["valid_weekly_indicator"]
            },
            {
                "func": concat_data_process,
                "param": {"tgt_factors": self.output_factors},
                "input_data": {"valid_data": "valid_weekly_indicator", "invalid_data": "not_valid_weekly_indicator"},
                "output": ["weekly_peer_neural_factor"]
            },
            {
                "func": save_data_to_table,
                "param": {"engine": self.save_info['engine'], "table": self.save_info['table'],
                          "if_exists": self.save_info.get("if_exists", "append")},
                "input_data": {"data": "weekly_peer_neural_factor"},
                "output": ["weekly_peer_neural_factor"]
            }
        ]
        self.output_vars = ["weekly_peer_neural_factor"]


class IndustryNeutralWeeklyFilterFactor(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.start_date = param_info['start_date']
        self.end_date = param_info['end_date']
        self.source_data_infos = param_info['source_data_infos']

        self.invalid_infos = param_info['invalid_infos']
        self.neutral_infos_4_filter = param_info['neutral_infos_4_filter']
        self.sum_infos_4_filter = param_info['sum_infos_4_filter']
        self.invalid_infos_4_type = param_info['invalid_infos_4_type']
        self.neutral_infos = param_info['neutral_infos']
        self.sum_infos = param_info['sum_infos']
        self.output_factors = param_info.get('output_factors', [])
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
                "func": transfer_data_to_valid_and_not_valid,
                "param": {
                    "invalid_infos": self.invalid_infos

                },
                "input_data": {"data": "weekly_indicator"},
                "output": ['valid_weekly_indicator', 'not_valid_weekly_indicator']
            },
            {
                "func": industry_standardized_factor,
                "param": {
                    "feature_neutral_infos": self.neutral_infos_4_filter,
                },
                "input_data": {"data": "valid_weekly_indicator"},
                "output": ["valid_weekly_indicator"]
            },
            {
                "func": pipline_sum_data_weight_weight,
                "param": {
                    "sum_infos": self.sum_infos_4_filter,
                },
                "input_data": {'data': "valid_weekly_indicator"},
                "output": ["valid_weekly_indicator"]
            },
            {
                "func": transfer_data_to_valid_and_not_valid,
                "param": {
                    "invalid_infos": self.invalid_infos_4_type

                },
                "input_data": {"data": "valid_weekly_indicator"},
                "output": ['weekly_indicator_tgt_type', 'weekly_indicator_non_tgt_type']
            },
            {
                "func": industry_standardized_factor,
                "param": {
                    "feature_neutral_infos": self.neutral_infos,
                },
                "input_data": {"data": "weekly_indicator_tgt_type"},
                "output": ["weekly_indicator_tgt_type"]
            },
            {
                "func": pipline_sum_data_weight_weight,
                "param": {
                    "sum_infos": self.sum_infos,
                },
                "input_data": {'data': "weekly_indicator_tgt_type"},
                "output": ["weekly_indicator_tgt_type"]
            },
            {
                "func": get_tgt_factor_from_data,
                "param": {
                    "tgt_factors": self.output_factors,
                },
                "input_data": {'data': "weekly_indicator_tgt_type"},
                "output": ["weekly_indicator_tgt_type"]
            },            
            {
                "func": save_data_to_table,
                "param": {"engine": self.save_info['engine'], "table": self.save_info['table'], "if_exists": self.save_info.get("if_exists", "append")},
                "input_data": {"data": "weekly_indicator_tgt_type"},
                "output": ["weekly_indicator_tgt_type"]
            }
        ]
        self.output_vars = ["weekly_indicator_tgt_type"]