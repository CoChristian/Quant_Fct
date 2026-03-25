from indicator_operator import *
from test_func_operator import *
from factor_regression import generate_one_term_return
# from test_func_operator import multi_divide_two_variable
@timer
@memorize
def get_hist_data_4_factor_compute_no_trade_date_condition(read_engine, save_engine, table, field=['trade_date', 'code'], name_dict={},
                                   index=['trade_date', 'code'], hist_year=0, start_date=None, end_date=None,
                                   other_filter_info=None):
    """
    读取特定数据
    :param read_engine
    :param save_engine
    :param table:
    :param field:
    :param name_dict: 将的字段映射为新的字段
    :param index, 输出的 index
    :param hist_year: 需要 历史数据，如果为0表示不需要start_date以前的历史数据， 如果为正表示需要 hist_year 年份的历史数据，如果为-1表示需要所有的历史数据
    :param start_date: 筛选数据的开始日期
    :param end_date: 筛选数据的结束日期
    :param other_filter_info: 其他筛选条件
    :return: 读取的数据
    """
#     if hist_year < 0:
#         trade_date_condition = [{'field': 'trade_date',
#                                  'type': 'less_equal',
#                                  'param': end_date}]
#     else:
#         trade_date_condition = [{'field': 'trade_date',
#                                  'type': 'between',
#                                  'param': [start_date - hist_year * 10000, end_date]}]
    trade_date_condition = []
    if other_filter_info:
        trade_date_condition.append(other_filter_info)
    query_info = {'method': 'select',
                  'sheet_name': table,
                  'tgt_field': {'way': 'show', 'field': field},
                  'conditions': trade_date_condition}
    sql_api_clf = create_sql_api(read_engine=read_engine, save_engine=save_engine)
    raw_fac = sql_api_clf.read_data_from(query_info)
    # if "start_date" in field:
    #     raw_fac['trade_date'] =raw_fac['start_date'].map(lambda x: int(x.strftime("%Y%m%d")))
    raw_fac = raw_fac.rename(name_dict, axis=1)

    raw_fac = raw_fac.set_index(index)

    return raw_fac


class IndustryWeeklyGroupFactor(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.start_date = param_info['start_date']
        self.end_date = param_info['end_date']
        self.source_data_infos = param_info['source_data_infos']

        self.invalid_infos = param_info['invalid_infos']
#         self.neutral_infos = param_info['neutral_infos']
        self.industry_group_infos = param_info['industry_group_infos']
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
                "func": industry_group_factor,
                "param": {
                    "feature_infos": self.industry_group_infos,
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
                "output": ["weekly_industry_group_factor"]

            },
            {
                "func": save_data_to_table,
                "param": {"engine": self.save_info['engine'], "table": self.save_info['table'], "if_exists": self.save_info.get("if_exists", "append")},
                "input_data": {"data": "weekly_industry_group_factor"},
                "output": ["weekly_industry_group_factor"]
            }
        ]
        self.output_vars = ["weekly_industry_group_factor"]

class IndustryNeutralFactorThroughStepWiseChosenIndicator(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.start_date = param_info['start_date']
        self.end_date = param_info['end_date']
        self.source_data_infos = param_info['source_data_infos']
        self.invalid_infos = param_info['invalid_infos']
        self.save_info = param_info['save_info']
        self.chosen_indicator_info = param_info['chosen_indicator_info']
        self.industry_name = param_info['industry_name']
        self.output_factor_name = param_info['output_factor_name']
        self.output_factors = param_info['output_factors']
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
                "func": industry_stanadard_indicator_sum,
                'param': {
                    "chosen_indicator_info": self.chosen_indicator_info,
                    "industry_name": self.industry_name,
                    "output_factor_name": self.output_factor_name,
                },
                'input_data': {'data': "valid_weekly_indicator"},
                "output": ['valid_weekly_indicator'],
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

        
class FixedWeightFactorDifferentFactor(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.start_date = param_info['start_date']
        self.end_date = param_info['end_date']
        self.source_data_infos = param_info['source_data_infos']
        self.invalid_infos = param_info['invalid_infos']
        self.chosen_indicator_info = param_info['chosen_indicator_info']

        # self.factor_names = param_info['factor_names']
        self.save_info = param_info['save_info']
        # self.output_score_name = param_info['output_score_name']
        self.gen_fixed_weight_score_func = param_info['gen_fixed_weight_score_func']
        self.operators = [
            {
                "func": get_data_from_multi_source,
                "param": {
                    "data_source_infos": self.source_data_infos,
                    "start_date": self.start_date,
                    "end_date": self.end_date
                },
                "input_data": {},
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
                "func": self.gen_fixed_weight_score_func,
                "param": {
                    # "factor_names": self.factor_names,
                    "start_date": self.start_date,
                    "end_date": self.end_date,
                    # "output_score_name": self.output_score_name,
                    "chosen_indicator_info": self.chosen_indicator_info,

                },
                "input_data": {"data": "valid_factor"},
                "output": ["overall_score"],
            },
            {
                "func": save_data_to_table,
                "param": {"engine": self.save_info['engine'], "table": self.save_info['table'],
                          "if_exists": self.save_info.get("if_exists", "append")},
                "input_data": {"data": "overall_score"},
                "output": ["overall_score"]
            }
        ]

        self.output_vars = ["overall_score"]
        
        
class IndustryNeutralWeeklyGroupFactor(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.start_date = param_info['start_date']
        self.end_date = param_info['end_date']
        self.source_data_infos = param_info['source_data_infos']

        self.invalid_infos = param_info['invalid_infos']
#         self.neutral_infos = param_info['neutral_infos']
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
                "func": transfer_data_to_valid_and_not_valid,
                "param": {
                    "invalid_infos": self.invalid_infos

                },
                "input_data": {"data": "weekly_indicator"},
                "output": ['valid_weekly_indicator', 'not_valid_weekly_indicator']
            },
         
            {
                "func": industry_standardized_group_factor,
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
        
class PeerNeutralWeeklyGroupFactor(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.start_date = param_info['start_date']
        self.end_date = param_info['end_date']
        self.source_data_infos = param_info['source_data_infos']
        self.peer_data_infos = param_info['peer_data_info']

        self.invalid_infos = param_info['invalid_infos']
#         self.neutral_infos = param_info['neutral_infos']
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
                "output": ['valid_weekly_indicator', 'not_valid_weekly_indicator']
            },
         
            {
                "func": fast_peer_standardized_group_factor,
                "param": {
                    "feature_neutral_infos": self.neutral_infos,
                },
                "input_data": {"valid_data": "valid_weekly_indicator", "peer_mapping": "peer_mapping"},
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
                "param": {"engine": self.save_info['engine'], "table": self.save_info['table'], "if_exists": self.save_info.get("if_exists", "append")},
                "input_data": {"data": "weekly_peer_neural_factor"},
                "output": ["weekly_peer_neural_factor"]
            }
        ]
        self.output_vars = ["weekly_peer_neural_factor"]
        
        
class ResearchRptDeepSeekScore(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)

        self.operators = [
            {
                'func': get_hist_data_4_factor_compute_no_trade_date_condition,
                 'param': {
                     "read_engine": None,
                     "save_engine": None,
                     "start_date": None,
                     "end_date": None,
                     "table": "rpt_forecast_detail_from_gj",
                     "field": ['stock_code', 'entry_date', '_id'],
                     "index": ["_id"],
                     "name_dict": {}},
                 "input_data": {},
                 "output": ['research_rpt_check_in_date']
            },
            {
                'func': get_hist_data_4_factor_compute_no_trade_date_condition,
                 'param': {
                     "read_engine": None,
                     "save_engine": None,
                     "start_date": None,
                     "end_date": None,
                     "table": "rpt_forecast_emotion_score_from_gj",
                     "field": ['score', '_id'],
                     "index": ["_id"],
                     "name_dict": {}},
                 "input_data": {},
                 "output": ['research_rpt_score']
            },
            {
                'func': get_hist_data_4_factor_compute_no_trade_date_condition,
                'param': {
                    "read_engine": "mysql+pymysql://develop:haikuan_2025@localhost:3306/factor_compute_new",
                    "save_engine": None,
                    "start_date": None,
                    "end_date": None,
                    "table": "opt2trade",
                    "field": [],
                    "hist_year": 1,
                    "index": ["opt_date"],
                    "name_dict": {}},
                "input_data": {},
                "output": ['opt2trade']
            },

            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},

            },
            # {
            #     "class": OptToTrade,
            #     "output_name_mapping": {"opt_to_trade": "opt_to_trade"},
            #
            # },
            {
                "func": cal_research_report_emotion_score,
                "param": {"start_date": param_info['common']['start_date'], 'end_date': param_info['common']['end_date']},
                "input_data": {"research_report_time_info": "research_rpt_check_in_date",
                               "research_report_score_info": "research_rpt_score",
                               "opt_2_trade" : "opt2trade"},
                "output": ["research_report_hist_score"],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "zero"},
                "input_data": {"data": "research_report_hist_score", "index": "factor_index"},
                "output": ["research_report_hist_deep_seek_score"],
            },
            {
                "func": save_data_to_table,
                "param": {"engine": param_info['common']['save_engine'], "table": "research_report_hist_deep_seek_score",
                          "if_exists": param_info.get("insert_way", "append")},
                "input_data": {"data": "research_report_hist_deep_seek_score"},
                "output": ["research_report_hist_deep_seek_score"]
            },
        ]
        self.output_vars = ["research_report_hist_deep_seek_score"]
        
class RptJORScore(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                'func': get_hist_data_4_factor_compute_no_trade_date_condition,
                 'param': {
                     "read_engine": None,
                     "save_engine": None,
                     "start_date": None,
                     "end_date": None,
                     "table": "balance_stk",
                     "field": ['code', 'trade_date'],
                     "index": ["code", 'trade_date'],
                     "name_dict": {}},
                 "input_data": {},
                 "output": ['balance_stk']
            },
            {
                'func': get_hist_data_4_factor_compute_no_trade_date_condition,
                 'param': {
                     "read_engine": None,
                     "save_engine": None,
                     "start_date": None,
                     "end_date": None,
                     "table": "large_shareholder_share_change_stk",
                     "field": ['code', 'trade_date'],
                     "index": ["code", 'trade_date'],
                     "name_dict": {}},
                 "input_data": {},
                 "output": ['large_shareholder_share_change_stk']
            },
            {
                'func': get_hist_data_4_factor_compute_no_trade_date_condition,
                 'param': {
                     "read_engine": None,
                     "save_engine": None,
                     "start_date": None,
                     "end_date": None,
                     "table": "capital_change_stk",
                     "field": ['code', 'trade_date'],
                     "index": ["code", 'trade_date'],
                     "name_dict": {}},
                 "input_data": {},
                 "output": ['capital_change_stk']
            }, 
            {
                'func': get_hist_data_4_factor_compute_no_trade_date_condition,
                 'param': {
                     "read_engine": None,
                     "save_engine": None,
                     "start_date": None,
                     "end_date": None,
                     "table": "limited_shares_stk",
                     "field": ['code', 'trade_date'],
                     "index": ["code", 'trade_date'],
                     "name_dict": {}},
                 "input_data": {},
                 "output": ['limited_shares_stk']
            },
            {
                'func': get_hist_data_4_factor_compute_no_trade_date_condition,
                 'param': {
                     "read_engine": None,
                     "save_engine": None,
                     "start_date": None,
                     "end_date": None,
                     "table": "limited_shares_unlock_stk",
                     "field": ['code', 'trade_date'],
                     "index": ["code", 'trade_date'],
                     "name_dict": {}},
                 "input_data": {},
                 "output": ['limited_shares_unlock_stk']
            },
            {
                'func': get_hist_data_4_factor_compute_no_trade_date_condition,
                 'param': {
                     "read_engine": None,
                     "save_engine": None,
                     "start_date": None,
                     "end_date": None,
                     "table": "unlock_shares_data",
                     "field": ['code', 'trade_date'],
                     "index": ["code", 'trade_date'],
                     "name_dict": {}},
                 "input_data": {},
                 "output": ['unlock_shares_data']
            },
            {
                'func': get_hist_data_4_factor_compute_no_trade_date_condition,
                 'param': {
                     "read_engine": None,
                     "save_engine": None,
                     "start_date": None,
                     "end_date": None,
                     "table": "xr_xd_stk",
                     "field": ['code', 'trade_date'],
                     "index": ["code", 'trade_date'],
                     "name_dict": {}},
                 "input_data": {},
                 "output": ['xr_xd_stk']
            },  
            {
                "func": transfer_finanical_data_for_jor,
                "param": {},
                "input_data": {
                    "report_data": "balance_stk",
                },
                "output": ['balance_stk'],  
            },
            {
                "func": transfer_finanical_data_for_jor,
                "param": {},
                "input_data": {
                    "report_data": "large_shareholder_share_change_stk",
                },
                "output": ['large_shareholder_share_change_stk'],  
            },          
            {
                "func": transfer_finanical_data_for_jor,
                "param": {},
                "input_data": {
                    "report_data": "capital_change_stk",
                },
                "output": ['capital_change_stk'],  
            }, 
            {
                "func": transfer_finanical_data_for_jor,
                "param": {},
                "input_data": {
                    "report_data": "limited_shares_stk",
                },
                "output": ['limited_shares_stk'],  
            },
            {
                "func": transfer_finanical_data_for_jor,
                "param": {},
                "input_data": {
                    "report_data": "limited_shares_unlock_stk",
                },
                "output": ['limited_shares_unlock_stk'],  
            },          
            {
                "func": transfer_finanical_data_for_jor,
                "param": {},
                "input_data": {
                    "report_data": "unlock_shares_data",
                },
                "output": ['unlock_shares_data'],  
            },  
            {
                "func": transfer_finanical_data_for_jor,
                "param": {},
                "input_data": {
                    "report_data": "xr_xd_stk",
                },
                "output": ['xr_xd_stk'],  
            }, 
            {
                'func': get_hist_data_4_factor_compute,
                 'param': {
                     "read_engine": None,
                     "save_engine": None,
                     "start_date": None,
                     "end_date": None,
                     "table": "daily_trading_data",
                     "field": ['code', 'trade_date', 'open', 'pre_close', 'low'],
                     "index": ["trade_date", 'code'],
                     "name_dict": {},
                     "hist_year": 2,
                 },
                 "input_data": {},
                 "output": ['daily_trading_data']
            },            
            {
                'func': get_hist_data_4_factor_compute,
                 'param': {
                     "read_engine": None,
                     "save_engine": None,
                     "start_date": None,
                     "end_date": None,
                     "table": "index_level",
                     "field": ['code', 'trade_date', 'open', 'low', 'close'],
                     "index": ["trade_date", 'code'],
                     "name_dict": {},
                     "hist_year": 2,

                     "other_filter_info": {'field': 'code',
                                 'type': 'equal',
                                 'param': "000905.XSHG"},

                 },
                 "input_data": {},
                 "output": ['csi500_data']
            },             
           
            {
                "class": OptToTrade,
                "output_name_mapping": {"opt_to_trade": "opt_to_trade"},

            },
            {
                "func": cal_report_jor_score,
                "param": {"rolling_window": param_info['rolling_window'], "output_name": "FinanicalRptJORScore"},
                "input_data": {"report_data": "balance_stk", "code_price_data": "daily_trading_data",  "index_price_data" : "csi500_data", "opt_2_trade": "opt_to_trade"},
                "output": ["finanical_rpt_jor_score"],
            },
            {
                "func": cal_report_jor_score,
                "param": {"rolling_window": param_info['rolling_window'], "output_name": "ShareholderShareChangeJORScore"},
                "input_data": {"report_data": "large_shareholder_share_change_stk", "code_price_data": "daily_trading_data",  "index_price_data" : "csi500_data", "opt_2_trade": "opt_to_trade"},
                "output": ["shareholder_share_change_jor_score"],
            },            
            {
                "func": cal_report_jor_score,
                "param": {"rolling_window": param_info['rolling_window'], "output_name": "CapitalChangeJORScore"},
                "input_data": {"report_data": "capital_change_stk", "code_price_data": "daily_trading_data",  "index_price_data" : "csi500_data", "opt_2_trade": "opt_to_trade"},
                "output": ["capital_change_jor_score"],
            },
            {
                "func": cal_report_jor_score,
                "param": {"rolling_window": param_info['rolling_window'], "output_name": "LimitedShareJORScore"},
                "input_data": {"report_data": "limited_shares_stk", "code_price_data": "daily_trading_data",  "index_price_data" : "csi500_data", "opt_2_trade": "opt_to_trade"},
                "output": ["limited_share_jor_score"],
            },
            {
                "func": cal_report_jor_score,
                "param": {"rolling_window": param_info['rolling_window'], "output_name": "LimitedShareUnlockJORScore"},
                "input_data": {"report_data": "limited_shares_unlock_stk", "code_price_data": "daily_trading_data",  "index_price_data" : "csi500_data", "opt_2_trade": "opt_to_trade"},
                "output": ["limited_share_unlock_jor_score"],
            },
            {
                "func": cal_report_jor_score,
                "param": {"rolling_window": param_info['rolling_window'], "output_name": "UnlockSharesJORScore"},
                "input_data": {"report_data": "unlock_shares_data", "code_price_data": "daily_trading_data",  "index_price_data" : "csi500_data", "opt_2_trade": "opt_to_trade"},
                "output": ["unlock_shares_jor_score"],
            },
            {
                "func": cal_report_jor_score,
                "param": {"rolling_window": param_info['rolling_window'], "output_name": "XrXdJORScore"},
                "input_data": {"report_data": "xr_xd_stk", "code_price_data": "daily_trading_data",  "index_price_data" : "csi500_data", "opt_2_trade": "opt_to_trade"},
                "output": ["xr_xd_jor_score"],
            },
            {
                "func": merge_data,
                "param": {},
                "input_data": {"1": "finanical_rpt_jor_score", "2": "shareholder_share_change_jor_score",  "3" : "capital_change_jor_score", "4": "limited_share_jor_score", '5': "limited_share_unlock_jor_score", "6": "unlock_shares_jor_score", "7": "xr_xd_jor_score"},
                "output": ["jor_score"],
            },
            {
                "func": save_data_to_table,
                "param": {"engine": param_info['common']['save_engine'], "table": 'jor_score',
                          "if_exists": param_info.get("insert_way", "append")},
                "input_data": {"data": "jor_score"},
                "output": ["jor_score"]
            },
        ]
        self.output_vars = ["jor_score"]
    
        
class ResearchRptJORScore(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                'func': get_hist_data_4_factor_compute_no_trade_date_condition,
                 'param': {
                     "read_engine": None,
                     "save_engine": None,
                     "start_date": None,
                     "end_date": None,
                     "table": "rpt_forecast_detail_from_gj",
                     "field": ['stock_code', 'entry_date', '_id', "create_date"],
                     "index": ["_id"],
                     "name_dict": {}},
                 "input_data": {},
                 "output": ['research_rpt_check_in_date']
            },
            {
                "func": std_rpt_data,
                "param": {},
                "input_data": {
                    "report_data": "research_rpt_check_in_date",
                },
                "output": ['research_rpt_check_in_date'],
                
            },
            {
                'func': get_hist_data_4_factor_compute,
                 'param': {
                     "read_engine": None,
                     "save_engine": None,
                     "start_date": None,
                     "end_date": None,
                     "table": "daily_trading_data",
                     "field": ['code', 'trade_date', 'open', 'pre_close', 'low'],
                     "index": ["trade_date", 'code'],
                     "name_dict": {},
                     "hist_year": 2,
                 },
                 "input_data": {},
                 "output": ['daily_trading_data']
            },            
            {
                'func': get_hist_data_4_factor_compute,
                 'param': {
                     "read_engine": None,
                     "save_engine": None,
                     "start_date": None,
                     "end_date": None,
                     "table": "index_level",
                     "field": ['code', 'trade_date', 'open', 'low', 'close'],
                     "index": ["trade_date", 'code'],
                     "name_dict": {},
                     "other_filter_info": {'field': 'code',
                                 'type': 'equal',
                                 'param': "000905.XSHG"},
                     "hist_year": 2,

                 },
                 "input_data": {},
                 "output": ['csi500_data']
            },             
           
            {
                "class": OptToTrade,
                "output_name_mapping": {"opt_to_trade": "opt_to_trade"},

            },
            {
                "func": cal_report_jor_score,
                "param": {"rolling_window": param_info['rolling_window'], "output_name": "ResearchRptJORScore"},
                "input_data": {"report_data": "research_rpt_check_in_date", "code_price_data": "daily_trading_data",  "index_price_data" : "csi500_data", "opt_2_trade": "opt_to_trade"},
                "output": ["rpt_jor_data"],
            },
            {
                "func": save_data_to_table,
                "param": {"engine": param_info['common']['save_engine'], "table": 'research_report_jor_score',
                          "if_exists": param_info.get("insert_way", "append")},
                "input_data": {"data": "rpt_jor_data"},
                "output": ["rpt_jor_data"]
            },

        ]
        self.output_vars = ["rpt_jor_data"]
        
        
class BetaFactor(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.if_exists=param_info.get("insert_way", "append")
        self.benchmark_code = self.param_info.get("benchmark_code", "000905.XSHG")
        self.save_table = self.param_info['save_table']
        self.operators = [
            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},
            },
            {
                "class": MarketBeta,
                "param": {"window_size": 60, "benchmark_code": self.benchmark_code},
                "output_name_mapping": {"market_beta_{}_60".format(self.benchmark_code.replace(".", "")): "market_beta_{}_60".format(self.benchmark_code.replace(".", ""))},
            },
            {
                "class": MarketBeta,
                "param": {"window_size": 40, "benchmark_code": self.benchmark_code},
                "output_name_mapping": {"market_beta_{}_40".format(self.benchmark_code.replace(".", "")): "market_beta_{}_40".format(self.benchmark_code.replace(".", ""))},
            },            
            {
                "class": MarketBeta,
                "param": {"window_size": 20, "benchmark_code": self.benchmark_code},
                "output_name_mapping": {"market_beta_{}_20".format(self.benchmark_code.replace(".", "")): "market_beta_{}_20".format(self.benchmark_code.replace(".", ""))},
            },
            {
                "class": MarketBeta,
                "param": {"window_size": 10, "benchmark_code": self.benchmark_code},
                "output_name_mapping": {"market_beta_{}_10".format(self.benchmark_code.replace(".", "")): "market_beta_{}_10".format(self.benchmark_code.replace(".", ""))},
            },
            {
                "class": MarketBeta,
                "param": {"window_size": 5, "benchmark_code": self.benchmark_code},
                "output_name_mapping": {"market_beta_{}_5".format(self.benchmark_code.replace(".", "")): "market_beta_{}_5".format(self.benchmark_code.replace(".", ""))},
            },
            {
                "func": standard_and_merge_data,
                "param": {},
                "input_data":
                    {
                        "factor_index": "factor_index",
                        "1": "market_beta_{}_60".format(self.benchmark_code.replace(".", "")),
                        "2": "market_beta_{}_40".format(self.benchmark_code.replace(".", "")),
                        "3": "market_beta_{}_20".format(self.benchmark_code.replace(".", "")),
                        "4": "market_beta_{}_10".format(self.benchmark_code.replace(".", "")),
                        "5": "market_beta_{}_5".format(self.benchmark_code.replace(".", "")),

                    },
                "output": ["beta_factor"],
            },
            {
                "func": save_data_to_table,
                "param": {"engine": param_info['common']['save_engine'], "table": self.save_table,
                          "if_exists": self.if_exists},
                "input_data": {"data": "beta_factor"},
                "output": ["beta_data"]
            },
        ]
        self.output_vars = ["beta_data"]
        

class EearningForecastValueGrowthFactor(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.if_exists=param_info.get("insert_way", "append")
        self.operators = [
            {
                'func': get_hist_data_4_factor_compute,
                 'param': {
                     "read_engine":  "mysql+pymysql://develop:haikuan_2025@localhost:3306/factor_compute_new",
                     "save_engine": None,
                     "start_date": None,
                     "end_date": None,
                     "table": "roll_consistent_earning_forecast_weekly",
                     "field": [],
                     "hist_year": 1,
                     "name_dict": {}
                 },
                 "input_data": {},
                 "output": ['roll_consistent_earning_forecast_weekly']
             },
            {
                'func': get_hist_data_4_factor_compute,
                 'param': {
                     "read_engine":  "mysql+pymysql://develop:haikuan_2025@localhost:3306/factor_compute_new",
                     "save_engine": None,
                     "start_date": None,
                     "end_date": None,
                     "table": "all_data_test_all_mkt_indicator",
                     "field": ['code', 'trade_date', 'MarketCap'],
                     "hist_year": 1,
                     "name_dict": {}
                 },
                 "input_data": {},
                 "output": ['market_cap']
             },
            {
                "func": merge_data,
                "param": {},
                "input_data": {"1": "roll_consistent_earning_forecast_weekly", "2": "market_cap"},
                "output": ["merged_data"]
            }, 
            {
                "func": divide_two_variable,
                "param": {'first_var_name': 'RollNetProfitConsistentMean', 'second_var_name': 'MarketCap', 'output_name': 'RollForecastMeanEP'},
                "input_data": {"data": "merged_data"},
                "output": ["roll_forecast_mean_ep"]
            },                        
            {
                "func": divide_two_variable,
                "param": {'first_var_name': 'RollNetAssetConsistentMean', 'second_var_name': 'MarketCap', 'output_name': 'RollForecastMeanBP'},
                "input_data": {"data": "merged_data"},
                "output": ["roll_forecast_mean_bp"]
            },              
            {
                "func": divide_two_variable,
                "param": {'first_var_name': 'RollMainRevenueConsistentMean', 'second_var_name': 'MarketCap', 'output_name': 'RollForecastMeanSP'},
                "input_data": {"data": "merged_data"},
                "output": ["roll_forecast_mean_sp"]
            },
            {
                "func": divide_two_variable,
                "param": {'first_var_name': 'RollNetProfitConsistentMedian', 'second_var_name': 'MarketCap', 'output_name': 'RollForecastMedianEP'},
                "input_data": {"data": "merged_data"},
                "output": ["roll_forecast_median_ep"]
            },
            {
                "func": divide_two_variable,
                "param": {'first_var_name': 'RollNetAssetConsistentMedian', 'second_var_name': 'MarketCap', 'output_name': 'RollForecastMedianBP'},
                "input_data": {"data": "merged_data"},
                "output": ["roll_forecast_median_bp"]
            },
            {
                "func": divide_two_variable,
                "param": {'first_var_name': 'RollMainRevenueConsistentMedian', 'second_var_name': 'MarketCap', 'output_name': 'RollForecastMedianSP'},
                "input_data": {"data": "merged_data"},
                "output": ["roll_forecast_median_sp"]
            },
            {
                "func": cal_data_hist_trend,
                "param": {'feature': 'RollNetProfitConsistentMean', 'hist_window': 12, 'min_hist_window': 6},
                "input_data": {"data": "merged_data"},
                "output": ["roll_net_profit_consistent_mean_trend"]
            },
            {
                "func": cal_data_hist_trend,
                "param": {'feature': 'RollMainRevenueConsistentMean', 'hist_window': 12, 'min_hist_window': 6},
                "input_data": {"data": "merged_data"},
                "output": ["roll_main_revenue_consistent_mean_trend"]
            },
            {
                "func": cal_data_hist_trend,
                "param": {'feature': 'RollNetProfitConsistentMedian', 'hist_window': 12, 'min_hist_window': 6},
                "input_data": {"data": "merged_data"},
                "output": ["roll_net_profit_consistent_median_trend"]
            },
            {
                "func": cal_data_hist_trend,
                "param": {'feature': 'RollMainRevenueConsistentMedian', 'hist_window': 12, 'min_hist_window': 6},
                "input_data": {"data": "merged_data"},
                "output": ["roll_main_revenue_consistent_median_trend"]
            },
            {
                "func": merge_data,
                "param": {},
                "input_data": {
                    "1": "roll_forecast_mean_ep", 
                    "2": "roll_forecast_mean_bp",
                    '3': 'roll_forecast_mean_sp',
                    "4": "roll_forecast_median_ep", 
                    "5": "roll_forecast_median_bp",
                    '6': 'roll_forecast_median_sp', 
                    "7": "roll_net_profit_consistent_mean_trend", 
                    "8": "roll_main_revenue_consistent_mean_trend",
                    '9': 'roll_net_profit_consistent_median_trend',
                    "10": "roll_main_revenue_consistent_median_trend",                   
                },
                "output": ["roll_forecast_value_growth_factor"]
            },
            {
                "func": save_data_to_table,
                "param": {"engine": param_info['common']['save_engine'], "table": "roll_forecast_value_growth_factor",
                          "if_exists": self.if_exists},
                "input_data": {"data": "roll_forecast_value_growth_factor"},
                "output": ["roll_forecast_value_growth_factor"]
            },            
        ]
        self.output_vars = ["roll_forecast_value_growth_factor"]
            
            
        
class RollConsistentEearningForecast(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.if_exists=param_info.get("insert_way", "append")
#         self.benchmark_code = self.param_info.get("benchmark_code", "000905.XSHG")
#         self.save_table = self.param_info['save_table']
        self.operators = [
            {
                'func': get_hist_data_4_factor_compute,
                 'param': {
                     "read_engine": None,
                     "save_engine": None,
                     "start_date": None,
                     "end_date": None,
                     "table": "stock_eraning_forecast_detail",
                     "field": ['trade_date', 'code', '报告期', '研究机构名称', '综合值计算标记', '预测净利润_万元', '预测基准股本_万股', '预测每股收益_基本', '预测每股收益_换算', '预测每股收益_摊薄', '预测每股收益_稀释', '每股净资产', '预测主营业务收入_万元'],
                     "hist_year": 1,
                     "name_dict": {}
                 },
                 "input_data": {},
                 "output": ['stock_eraning_forecast_detail']
             },
#             {
#                 'func': get_hist_data_4_factor_compute,
#                  'param': {
#                      "read_engine": "mysql+pymysql://develop:haikuan_2025@localhost:3306/factor_compute_new",
#                      "save_engine": None,
#                      "start_date": None,
#                      "end_date": None,
#                      "table": "opt2trade",
#                      "field": [],
#                      'index': ['opt_date'],
#                      "hist_year": 1,
#                      "name_dict": {}
#                  },
#                  "input_data": {},
#                  "output": ['opt2trade']
#              },
            {
                "class": OptToTrade,
                "output_name_mapping": {"opt_to_trade": "opt2trade"},

            },
            
            {
                'func': get_hist_data_4_factor_compute,
                 'param': {
                     "read_engine": None,
                     "save_engine": None,
                     "start_date": None,
                     "end_date": None,
                     "table": "balance_stk",
                     "field": ['code', 'trade_date', 'end_date'],
                     "hist_year": 2,
                     "name_dict": {}
                 },
                 "input_data": {},
                 "output": ['financial_date_data']
             },
            
            {
                "func": cal_consistent_earning_forecast,
                "param": {"features": ['NetProfit', 'NetAsset', 'MainRevenue']},
                "input_data": {"data": "stock_eraning_forecast_detail", "opt2trade": "opt2trade"},
                "output": ["consistent_earning_forecast"]
            },
#             {
#                 "func": save_data_to_table,
#                 "param": {"engine": param_info['common']['save_engine'], "table": "consistent_earning_forecast",
#                           "if_exists": self.if_exists},
#                 "input_data": {"data": "consistent_earning_forecast"},
#                 "output": ["consistent_earning_forecast"]
#             }, 
            
            {
                "func": roll_consistent_forecast,
                "param": {"features": ['NetProfitConsistentMean', 'NetAssetConsistentMean', 'MainRevenueConsistentMean', 'NetProfitConsistentMedian',  'NetAssetConsistentMedian', 'MainRevenueConsistentMedian']},
                "input_data": {"financial_date_data": "financial_date_data", "consistent_data": "consistent_earning_forecast"},
                "output": ["roll_consistent_earning_forecast"]
            },            

            {
                "class": FactorIndex,
                "input_name_mapping": {},
                "output_name_mapping": {"factor_index": "factor_index"},
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": ""},
                "input_data": {"data": "roll_consistent_earning_forecast", "index": "factor_index"},
                "output": ["roll_consistent_earning_forecast_weekly"],
            },
            {
                "func": save_data_to_table,
                "param": {"engine": param_info['common']['save_engine'], "table": "roll_consistent_earning_forecast_weekly",
                          "if_exists": self.if_exists},
                "input_data": {"data": "roll_consistent_earning_forecast_weekly"},
                "output": ["roll_consistent_earning_forecast_weekly"]
            }, 
            {
                "func": cal_org_inner_forecast_trend,
                "param": {"features": ['NetProfit', 'NetAsset', 'MainRevenue']},
                "input_data": {"data": "stock_eraning_forecast_detail", "opt2trade": "opt2trade"},
                "output": ["org_inner_forecast_trend"]
            },            
            {
                "func": align_data_to_index,
                "param": {"fill_method": ""},
                "input_data": {"data": "org_inner_forecast_trend", "index": "factor_index"},
                "output": ["org_inner_forecast_trend_weekly"],
            },

            {
                "func": save_data_to_table,
                "param": {"engine": param_info['common']['save_engine'], "table": "org_inner_forecast_trend_weekly",
                          "if_exists": self.if_exists},
                "input_data": {"data": "org_inner_forecast_trend_weekly"},
                "output": ["org_inner_forecast_trend_weekly"]
            },
        ]
        self.output_vars = [ 'org_inner_forecast_trend_weekly']

        
class ResearchReportAlternativeFactor(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.if_exists=param_info.get("insert_way", "append")
        self.operators = [
            {
                'func': get_hist_data_4_factor_compute,
                 'param': {
                     "read_engine": None,
                     "save_engine": None,
                     "start_date": None,
                     "end_date": None,
                     "table": "stock_eraning_forecast_detail",
                     "field": ['trade_date', 'code', '研究机构名称', '预测日期', '分析师名称', '综合值计算标记'],
                     "hist_year": 2,
                     "name_dict": {}
                 },
                 "input_data": {},
                 "output": ['stock_eraning_forecast_detail']
             },
            
#             {
#                 'func': get_hist_data_4_factor_compute,
#                  'param': {
#                      "read_engine": "mysql+pymysql://develop:haikuan_2025@localhost:3306/factor_compute_new",
#                      "save_engine": None,
#                      "start_date": None,
#                      "end_date": None,
#                      "table": "opt2trade",
#                      "field": [],
#                      'index': ['opt_date'],
#                      "hist_year": 0,
#                      "name_dict": {}
#                  },
#                  "input_data": {},
#                  "output": ['opt2trade']
#              },
            {
                "class": OptToTrade,
                "output_name_mapping": {"opt_to_trade": "opt2trade"},

            },
            {
                'func': get_hist_data_4_factor_compute,
                 'param': {
                     "read_engine": None,
                     "save_engine": None,
                     "start_date": None,
                     "end_date": None,
                     "table": "balance_stk",
                     "field": ['code', 'trade_date'],
                     "hist_year": 2,
                     "name_dict": {}
                 },
                 "input_data": {},
                 "output": ['balance_stk']
             },
            {
                'func': get_hist_data_4_factor_compute,
                 'param': {
                     "read_engine": None,
                     "save_engine": None,
                     "start_date": None,
                     "end_date": None,
                     "table": "fin_forecast_stk",
                     "field": ['code', 'trade_date'],
                     "hist_year": 2,
                     "name_dict": {}
                 },
                 "input_data": {},
                 "output": ['fin_forecast_stk']
             },
            {
                'func': get_hist_data_4_factor_compute,
                 'param': {
                     "read_engine": None,
                     "save_engine": None,
                     "start_date": None,
                     "end_date": None,
                     "table": "performance_letters_stk",
                     "field": ['code', 'trade_date'],
                     "hist_year": 2,
                     "name_dict": {}
                 },
                 "input_data": {},
                 "output": ['performance_letters_stk']
             },
            {
                'func': merge_data_axis0,
                'param': {},
                'input_data': {
                    '0': 'balance_stk',
                    '1': 'fin_forecast_stk',
                    '2': 'performance_letters_stk',
                },
                'output': ['financial_date_data'],
            },
            {
                "func": count_research_report,
                "param": {},
                "input_data": {"research_report_data": "stock_eraning_forecast_detail", "opt2trade": "opt2trade"},
                "output": ["research_report_count_data"]
            },
            {
                "class": FactorIndex,
                "input_name_mapping": {},
                "output_name_mapping": {"factor_index": "factor_index"},
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": ""},
                "input_data": {"data": "research_report_count_data", "index": "factor_index"},
                "output": ["research_report_count_data"],
            },
            {
                "func": save_data_to_table,
                "param": {"engine": param_info['common']['save_engine'], "table": "research_report_count_factor",
                          "if_exists": self.if_exists},
                "input_data": {"data": "research_report_count_data"},
                "output": ["research_report_count_data"]
            },
            
            {
                "func": cal_research_report_financial_report_pub_date_delay,
                "param": {"min_delay_days": [0, -1, -2, -3]},
                "input_data": {
                    "financial_date_data": "financial_date_data", 
                    "research_report_data": "stock_eraning_forecast_detail",
                    'opt2trade': 'opt2trade',
                },
                "output": ["pub_delay_data"]
            },            
#             {
#                 "func": merge_data,
#                 "param": {},
#                 "input_data": {
#                     "1": "research_report_count_data", "2": "pub_delay_data"},
#                 "output": ["alternative_factor"],
#             },   
            {
                "class": FactorIndex,
                "input_name_mapping": {},
                "output_name_mapping": {"factor_index": "factor_index"},
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": ""},
                "input_data": {"data": "pub_delay_data", "index": "factor_index"},
                "output": ["alternative_factor"],
            },
            {
                "func": save_data_to_table,
                "param": {"engine": param_info['common']['save_engine'], "table": "research_report_alterlative_factor",
                          "if_exists": self.if_exists},
                "input_data": {"data": "alternative_factor"},
                "output": ["alternative_factor"]
            }
        ]
        self.output_vars = [ 'alternative_factor']
        
        
class ForecastRealPriceDiff(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.if_exists=param_info.get("insert_way", "append")
        self.operators = [
            {
                'func': get_hist_data_4_factor_compute_no_trade_date_condition,
                 'param': {
                     "read_engine": None,
                     "save_engine": None,
                     "start_date": None,
                     "end_date": None,
                     "table": "rpt_forecast_detail_from_gj",
                     "field": ['stock_code', 'entry_date', 'author_name', 'organ_id', 'report_id', 'target_price_ceiling', 'target_price_floor', 'current_price', 'create_date'],
                     "index": ["report_id"],
                     "name_dict": {}},
                 "input_data": {},
                 "output": ['forecast_price_data']
            },
            {
                'func': get_hist_data_4_factor_compute,
                 'param': {
                     "read_engine": None,
                     "save_engine": None,
                     "start_date": None,
                     "end_date": None,
                     "table": "daily_trading_data",
                     "field": ['code', 'trade_date', 'close'],
                     
                     'hist_year': 2,
                     "name_dict": {}},
                 "input_data": {},
                 "output": ['hfq_price']
            },            
            {
                'func': get_hist_data_4_factor_compute,
                 'param': {
                     "read_engine": None,
                     "save_engine": None,
                     "start_date": None,
                     "end_date": None,
                     "table": "daily_trading_data_unadjusted",
                     "field": ['code', 'trade_date', 'close'],
                     
                     'hist_year': 2,

                     "name_dict": {}},
                 "input_data": {},
                 "output": ['nfq_price']
            },             
#             {
#                 'func': get_hist_data_4_factor_compute,
#                  'param': {
#                      "read_engine": "mysql+pymysql://develop:haikuan_2025@localhost:3306/factor_compute_new",
#                      "save_engine": None,
#                      "start_date": None,
#                      "end_date": None,
#                      "table": "opt2trade",
#                      "field": [],
#                      'index': ['opt_date'],
#                      "hist_year": 0,
#                      "name_dict": {}
#                  },
#                  "input_data": {},
#                  "output": ['opt2trade']
#              },
            {
                "class": OptToTrade,
                "output_name_mapping": {"opt_to_trade": "opt2trade"},

            },
            {
                'func': cal_consistent_tgt_price_indicator,
                'param': {                  
                    
                },
                'input_data': {
                    'tgt_price_data': 'forecast_price_data',
                    'nfq_price': 'nfq_price',
                    'hfq_price': 'hfq_price',
                    'opt_2_trade': 'opt2trade'
                },
                'output': ['consistent_forecast_real_diff']
            },
            {
                'func': cal_consistent_target_return_indicator,
                'param': {                  
                    
                },
                'input_data': {
                    'tgt_price_data': 'forecast_price_data',
                    'nfq_price': 'nfq_price',
                    'opt_2_trade': 'opt2trade'
                },
                'output': ['consistent_target_return_indicator']
            },
            
            {
                "class": FactorIndex,
                "input_name_mapping": {"data": 'consistent_forecast_real_diff'},
                "output_name_mapping": {"factor_index": "factor_index"},
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": ""},
                "input_data": {"data": "consistent_forecast_real_diff", "index": "factor_index"},
                "output": ["consistent_forecast_real_diff_weekly"],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": ""},
                "input_data": {"data": "consistent_target_return_indicator", "index": "factor_index"},
                "output": ["consistent_target_return_indicator_weekly"],
            },            
#             {
#                 'func': time_series_std_data,
#                 'param': {                  
#                     "features": ['ConsistentTgtRealPriceDiff', 'ConsistentAuthorStdTgtRealPriceDiff', 'ConsistentOrgStdTgtRealPriceDiff'],
#                     'window_size': 24,
#                 },
#                 'input_data': {
#                     'data': 'consistent_forecast_real_diff_weekly',
#                 },
#                 'output': ['consistent_forecast_real_diff_time_series_std_weekly']
#             },
#             {
#                 "class": FactorIndex,
#                 "input_name_mapping": {},
#                 "output_name_mapping": {"factor_index": "factor_index"},
#             },
#             {
#                 "func": align_data_to_index,
#                 "param": {"fill_method": ""},
#                 "input_data": {"data": "consistent_forecast_real_diff_time_series_std_weekly", "index": "factor_index"},
#                 "output": ["consistent_forecast_real_diff_time_series_std_weekly"],
#             },
            
            {
                "func": save_data_to_table,
                "param": {"engine": param_info['common']['save_engine'], "table": "consistent_forecast_real_diff_weekly",
                          "if_exists": self.if_exists},
                "input_data": {"data": "consistent_forecast_real_diff_weekly"},
                "output": ["consistent_forecast_real_diff_weekly"]
            },            
            {
                "func": save_data_to_table,
                "param": {"engine": param_info['common']['save_engine'], "table": "consistent_target_return_indicator_weekly",
                          "if_exists": self.if_exists},
                "input_data": {"data": "consistent_target_return_indicator_weekly"},
                "output": ["consistent_target_return_indicator_weekly"]
            },             
        ]
        self.output_vars = [ 'consistent_forecast_real_diff_weekly', 'consistent_target_return_indicator_weekly']

        
class FinancialRptExceedingExpectation(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.if_exists=param_info.get("insert_way", "append")
        self.operators = [
            {
                'func': get_hist_data_4_factor_compute,
                 'param': {
                     "read_engine": None,
                     "save_engine": None,
                     "start_date": None,
                     "end_date": None,
                     "table": "stock_eraning_forecast_detail",
                     "field": ['综合值计算标记', 'trade_date', '预测净利润_万元', '预测基准股本_万股', '预测每股收益_基本', '预测每股收益_换算', '预测每股收益_摊薄', '预测每股收益_稀释', '每股净资产', '预测主营业务收入_万元', 'code', '报告期', '研究机构名称'],
                     "index": ["code", 'trade_date'],
                     'hist_year': 2,

                     "name_dict": {}},
                 "input_data": {},
                 "output": ['stock_eraning_forecast_detail'],
            },
            {
                'func': get_hist_data_4_factor_compute,
                 'param': {
                     "read_engine": None,
                     "save_engine": None,
                     "start_date": None,
                     "end_date": None,
                     "table": "income_q",
                     "field": ['code', 'trade_date', 'end_date', 'np_parent_company_owners'],
                     
                     'hist_year': 1,
                     "name_dict": {}},
                 "input_data": {},
                 "output": ['income_q']
            },            
            {
                'func': get_hist_data_4_factor_compute,
                 'param': {
                     "read_engine": None,
                     "save_engine": None,
                     "start_date": None,
                     "end_date": None,
                     "table": "performance_letters_stk",
                     "field": ['code', 'trade_date', 'end_date', 'np_parent_company_owners'],
                     "other_filter_info": [{"field": "report_type", "type": "equal", "param": 0}],
                     'hist_year': 1,
                     "name_dict": {}},
                 "input_data": {},
                 "output": ['performance_letters_stk']
            }, 
            {
                'func': get_hist_data_4_factor_compute,
                 'param': {
                     "read_engine": None,
                     "save_engine": None,
                     "start_date": None,
                     "end_date": None,
                     "table": "fin_forecast_stk",
                     "field": ['code', 'trade_date', 'end_date', 'profit_min', 'profit_max'],
                     
                     'hist_year': 1,
                     "name_dict": {}},
                 "input_data": {},
                 "output": ['fin_forecast_stk']
            },
            {
                'func': process_research_rpt_data,
                 'param': {},
                 "input_data": {'research_rpt_detail': 'stock_eraning_forecast_detail'},
                 "output": ['stock_eraning_forecast_detail_std']
            },            
            {
                'func': process_fin_forecast,
                 'param': {},
                 "input_data": {'data': 'fin_forecast_stk'},
                 "output": ['fin_forecast_stk_std']
            },
            {
                'func': cal_quarter_feature,
                 'param': {'feature': "np_parent_company_owners"},
                 "input_data": {'data': 'fin_forecast_stk_std', 'financial_rpt': 'income_q'},
                 "output": ['quarter_fin_forecast_stk_std']
            },
            {
                'func': cal_quarter_feature,
                 'param': {'feature': "np_parent_company_owners"},
                 "input_data": {'data': 'fin_forecast_stk_std', 'financial_rpt': 'income_q'},
                 "output": ['quarter_fin_forecast_stk_std']
            },
            {
                'func': cal_quarter_feature,
                 'param': {'feature': "np_parent_company_owners"},
                 "input_data": {'data': 'performance_letters_stk', 'financial_rpt': 'income_q'},
                 "output": ['quarter_performance_letters_stk']
            }, 
            {
                'func': merge_data_axis0,
                'param': {},
                'input_data': {
                    '0': 'income_q',
                    '1': 'quarter_fin_forecast_stk_std',
                    '2': 'quarter_performance_letters_stk',
                },
                'output': ['financial_rpt_data'],
            },
            {
                "class": FactorIndex,
                "input_name_mapping": {"data": 'consistent_forecast_real_diff'},
                "output_name_mapping": {"factor_index": "factor_index"},
            },
            {
                'func': generate_financial_rpt_exceeding_expectation,
                'param': {'feature_name': "FinancialRpt"},
                'input_data': {
                    'financial_rpt': 'income_q',
                    'research_rpt_detail': 'stock_eraning_forecast_detail_std',
                    'factor_index': 'factor_index',
                },
                'output': ['financial_rpt_exceeding_expectation'],
            },
            {
                'func': generate_financial_rpt_exceeding_expectation,
                'param': {'feature_name': "PerformanceLetters"},
                'input_data': {
                    'financial_rpt': 'quarter_performance_letters_stk',
                    'research_rpt_detail': 'stock_eraning_forecast_detail_std',
                    'factor_index': 'factor_index',
                },
                'output': ['performance_letters_exceeding_expectation'],
            },
            {
                'func': generate_financial_rpt_exceeding_expectation,
                'param': {'feature_name': "FinForecast"},
                'input_data': {
                    'financial_rpt': 'quarter_fin_forecast_stk_std',
                    'research_rpt_detail': 'stock_eraning_forecast_detail_std',
                    'factor_index': 'factor_index',
                },
                'output': ['fin_forecast_exceeding_expectation'],
            },
            {
                'func': generate_financial_rpt_exceeding_expectation,
                'param': {'feature_name': "AllFinancialRpt"},
                'input_data': {
                    'financial_rpt': 'financial_rpt_data',
                    'research_rpt_detail': 'stock_eraning_forecast_detail_std',
                    'factor_index': 'factor_index',
                },
                'output': ['all_fin_rpt_exceeding_expectation'],
            }, 
            {
                "func": merge_data,
                "param": {},
                "input_data": {"1": "financial_rpt_exceeding_expectation", "2": "performance_letters_exceeding_expectation",  "3" : "fin_forecast_exceeding_expectation", "4": "all_fin_rpt_exceeding_expectation"},
                "output": ["exceeding_expectation_data"]
            },
            {
                "func": save_data_to_table,
                "param": {"engine": param_info['common']['save_engine'], "table": "exceeding_expectation",
                          "if_exists": self.if_exists},
                "input_data": {"data": "exceeding_expectation_data"},
                "output": ["exceeding_expectation_data"]
            },   
            
        ]
        self.output_vars = [ 'exceeding_expectation_data']            
           

            
class BookToPriceResidual(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.if_exists=param_info.get("insert_way", "append")

        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "all_data_test_all_mkt_indicator",
                 "field": ['trade_date', 'code', 'BookToPrice', 'ROE', 'SWL1IndustryCode', 'RevenueLRC3', 'RevenueYoy', 'NetIncomeLRC3', 'NetIncomeYoy'],
                 "hist_year": 2,
                 "name_dict": {}},
             "input_data": {},
             "output": ['factor_data']},
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "roll_forecast_value_growth_factor",
                 "field": ['trade_date', 'code', 'RollNetProfitConsistentMeanTrend', 'RollMainRevenueConsistentMeanTrend'],
                 "hist_year": 2,
                 "name_dict": {}},
             "input_data": {},
             "output": ['org_intra_forecast_trend']},
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "org_inner_forecast_trend_weekly",
                 "field": ['trade_date', 'code', 'NetProfitOrgInnerTrend', 'NetAssetOrgInnerTrend', 'MainRevenueOrgInnerTrend'],
                 "hist_year": 2,
                 "name_dict": {}},
             "input_data": {},
             "output": ['org_inter_forecast_trend']},

            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "research_report_hist_deep_seek_score",
                 "field": ['trade_date', 'code', 'TitleEmotionScoreMean', 'TitleEmotionScoreTrend'],
                 "hist_year": 2,
                 "name_dict": {}},
             "input_data": {},
             "output": ['title_emotion_score_info']},
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "research_report_count_factor",
                 "field": ['trade_date', 'code', 'ReportCountTrend', 'ReportLastYearCount'],
                 "hist_year": 2,
                 "name_dict": {}},
             "input_data": {},
             "output": ['report_count_info']},

            {
                "func": merge_data,
                "param": {},
                "input_data": {
                    "1": "factor_data",
                    "2": "org_intra_forecast_trend",
                    "3": "org_inter_forecast_trend",
                    "4": "title_emotion_score_info",
                    "5": "report_count_info",
                },
                "output": ["value_growth_data"]
            },
            {
                "func": merge_growth_indicators,
                "param": {
                    "growth_indicator_infos": [
                        {'feature': 'RevenueLRC3', 'null_value': 0.5, 'reverse': False},
                        {'feature': 'RevenueYoy', 'null_value': 0.5, 'reverse': False},
                        {'feature': 'NetIncomeLRC3', 'null_value': 0.5, 'reverse': False},
                        {'feature': 'NetIncomeYoy', 'null_value': 0.5, 'reverse': False},
                        {'feature': 'RollNetProfitConsistentMeanTrend', 'null_value': 0.5, 'reverse': False},
                        {'feature': 'RollMainRevenueConsistentMeanTrend', 'null_value': 0.5, 'reverse': False},
                        {'feature': 'NetProfitOrgInnerTrend', 'null_value': 0.5, 'reverse': False},
                        {'feature': 'NetAssetOrgInnerTrend', 'null_value': 0.5, 'reverse': False},
                        {'feature': 'MainRevenueOrgInnerTrend', 'null_value': 0.5, 'reverse': False},
                        {'feature': 'TitleEmotionScoreMean', 'null_value': 0.5, 'reverse': False},
                        {'feature': 'TitleEmotionScoreTrend', 'null_value': 0.5, 'reverse': False},
                        {'feature': 'ReportCountTrend', 'null_value': 0.5, 'reverse': False},
                        {'feature': 'ReportLastYearCount', 'null_value': 0, 'reverse': False},
                    ]
                },
                "input_data": {
                    "data": "value_growth_data",
                },
                "output": ["value_growth_data_"]
            },
            {
                'func': cal_value_indicator_residual,
                'param': {},
                'input_data': {
                    'value_growth_factor': 'value_growth_data_',
                },
                "output": ["bp_residual"]
        },
            {
                "func": save_data_to_table,
                "param": {"engine": param_info['common']['save_engine'], "table": "bp_residual",
                          "if_exists": self.if_exists},
                "input_data": {"data": "bp_residual"},
                "output": ["bp_residual"]
            },
        ]
        self.output_vars = ["bp_residual"]
        
            
class EarningsToPrice(FactorCompute):
    """

    """

    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": PriceToEarnings,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"price_to_earnings": "price_to_earnings"},
                "output": ["price_to_earnings"],
            },
            {
                "func": cal_reciprocal,
                "param": {"value_name": "PriceToEarnings", 'output_name': self.__class__.__name__},
                "input_data": {'data': "price_to_earnings"},
                "output": ["earnings_to_price"]
            },
        ]
        self.output_vars = ["earnings_to_price"]
        
        
class ValueFactorDeviation(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.if_exists=param_info.get("insert_way", "append")
        self.operators = [
            {
                "class": CashOverMktCap,
                "param": {},
                "output_name_mapping": {"cash_over_market_cap": "cash_over_market_cap"},
            },
            {
                "class": BookToPrice,
                "param": {},
                "output_name_mapping": {"book_to_price": "book_to_price"},
            },
            
            {
                "class": RevenueOverMktCap,
                "param": {},
                "output_name_mapping": {"revenue_over_market_cap": "revenue_over_market_cap"},
            },
            {
                "class": EarningsToPrice,
                "param": {},
                "output_name_mapping": {"earnings_to_price": "earnings_to_price"},
            },
            {
                "func": cal_feature_deviation,
                "param": {"feature": "EarningsToPrice", 'window': 12},
                "input_data": {'data': "earnings_to_price"},
                "output": ["earnings_to_price_deviation"]
            }, 
            {
                "func": cal_feature_deviation,
                "param": {"feature": "CashOverMktCap", 'window': 12},
                "input_data": {'data': "cash_over_market_cap"},
                "output": ["cash_over_market_cap_deviation"]
            },
            {
                "func": cal_feature_deviation,
                "param": {"feature": "BookToPrice", 'window': 12},
                "input_data": {'data': "book_to_price"},
                "output": ["book_to_price_deviation"]
            }, 
            {
                "func": cal_feature_deviation,
                "param": {"feature": "RevenueOverMktCap", 'window': 12},
                "input_data": {'data': "revenue_over_market_cap"},
                "output": ["revenue_over_market_cap_deviation"]
            }, 
            {
                "func": merge_data,
                "param": {},
                "input_data": {"1": "earnings_to_price_deviation", "2": "cash_over_market_cap_deviation",  "3" : "book_to_price_deviation", "4": "revenue_over_market_cap_deviation"},
                "output": ["value_factor_deviation"]
            },            
            {
                "func": save_data_to_table,
                "param": {"engine": param_info['common']['save_engine'], "table": "value_factor_deviation",
                          "if_exists": self.if_exists},
                "input_data": {"data": "value_factor_deviation"},
                "output": ["value_factor_deviation"]
            },            
        ]
        self.output_vars = ["value_factor_deviation"]

# class AdministrationExpenses(FactorCompute):
#     """

#     """
#     def __init__(self, param_info, input_name_mapping, output_name_mapping):
#         super().__init__(param_info, input_name_mapping, output_name_mapping)
#         self.operators = [
#             {'func': get_hist_data_4_factor_compute,
#              'param': {
#                  "read_engine": None,
#                  "save_engine": None,
#                  "start_date": None,
#                  "end_date": None,
#                  "table": "income_stk",
#                  "field": ["trade_date",  "code", 'administration_expense', 'end_date'],
#                  "index": ['trade_date',  'code', 'end_date'],

#                  "hist_year": -1,
#                  "name_dict": {"administration_expense": self.__class__.__name__}},
#              "input_data": {},
#              "output": ['administration_expense']},
#         ]
#         self.output_vars = ["administration_expense"]

# class SaleExpenses(FactorCompute):
#     """

#     """
#     def __init__(self, param_info, input_name_mapping, output_name_mapping):
#         super().__init__(param_info, input_name_mapping, output_name_mapping)
#         self.operators = [
#             {'func': get_hist_data_4_factor_compute,
#              'param': {
#                  "read_engine": None,
#                  "save_engine": None,
#                  "start_date": None,
#                  "end_date": None,
#                  "table": "income_stk",
#                  "field": ["trade_date",  "code", 'administration_expense', 'end_date'],
#                  "index": ['trade_date',  'code', 'end_date'],

#                  "hist_year": -1,
#                  "name_dict": {"administration_expense": self.__class__.__name__}},
#              "input_data": {},
#              "output": ['administration_expense']},
#         ]
#         self.output_vars = ["administration_expense"]


def fillna_with_zero(data):
    data = data.fillna(0)
    return data
class BalanceAssetToMv(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.if_exists = param_info.get("insert_way", "append")
        self.features = param_info.get("features")
        self.operators = [
            {
                'func': get_hist_data_4_factor_compute,
                 'param': {
                     "read_engine": None,
                     "save_engine": None,
                     "start_date": None,
                     "end_date": None,
                     "table": "balance_stk",
                     "field": ["trade_date",  "code",'end_date'] + self.features,
                     "index": ['trade_date',  'code', 'end_date'],

                     "hist_year": -1,
                 },
                 "input_data": {},
                 "output": ['balance_stk_data']
            },
#             {
#                 'func': fillna_with_zero,
#                  'param': {},
#                 "input_data": {'data': 'balance_stk_data'},
#                 "output": ['balance_stk_data_std']
#             },
            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "balance_stk_data", "index": "factor_index"},
                "output": ["balance_stk_data_weekly"],
            },
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "valuation_q",
                 "field": ['trade_date', 'code', 'market_cap'],
                 "hist_year": 2,
                 "name_dict": {"market_cap": "MV_JQ"}},
             "input_data": {},
             "output": ['market_cap_jq']},
            {'func': std_mkt_cp,
             'param': {"value_name": "MV_JQ", "output_name": "MV"},
             "input_data": {"data": "market_cap_jq"},
             "output": ['market_cap']
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": 'market_cap', "index": "factor_index"},
                "output": ['market_cap_weekly'],
            },            
            {
                "func": merge_data,
                "param": {},
                "input_data": {"1": "balance_stk_data_weekly", "2": "market_cap_weekly"},
                "output": ["merged_data"]
            }, 
            {
                "func": divide_two_variable_4_zero_with_multi_features,
                "param": {
                    "first_var_names": self.features,
                    "second_var_name": "MV",
                },
                "input_data": {"data": "merged_data"},
                "output": ["balance_asset_to_mv"],
            },
            {
                "func": save_data_to_table,
                "param": {"engine": param_info['common']['save_engine'], "table": "balance_asset_to_mv",
                          "if_exists": self.if_exists},
                "input_data": {"data": "balance_asset_to_mv"},
                "output": ["balance_asset_to_mv"]
            },            
            
        ]
        self.output_vars = ['balance_asset_to_mv']      

        
class IncomeItemToMv(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.if_exists = param_info.get("insert_way", "append")
        self.features = param_info.get("features")
        self.operators = [
            {
                'func': get_hist_data_4_factor_compute,
                 'param': {
                     "read_engine": None,
                     "save_engine": None,
                     "start_date": None,
                     "end_date": None,
                     "table": "income_q",
                     "field": ["trade_date",  "code",'end_date'] + self.features,
                     "index": ['trade_date',  'code', 'end_date'],

                     "hist_year": -1,
                 },
                 "input_data": {},
                 "output": ['income_q_data']
            },
#             {
#                 'func': fillna_with_zero,
#                  'param': {},
#                 "input_data": {'data': 'income_q_data'},
#                 "output": ['income_q_data_std']
#             },
            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "income_q_data", "index": "factor_index"},
                "output": ["income_q_data_weekly"],
            },
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "valuation_q",
                 "field": ['trade_date', 'code', 'market_cap'],
                 "hist_year": 2,
                 "name_dict": {"market_cap": "MV_JQ"}},
             "input_data": {},
             "output": ['market_cap_jq']},
            {'func': std_mkt_cp,
             'param': {"value_name": "MV_JQ", "output_name": "MV"},
             "input_data": {"data": "market_cap_jq"},
             "output": ['market_cap']
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": 'market_cap', "index": "factor_index"},
                "output": ['market_cap_weekly'],
            },            
            {
                "func": merge_data,
                "param": {},
                "input_data": {"1": "income_q_data_weekly", "2": "market_cap_weekly"},
                "output": ["merged_data"]
            }, 
            {
                "func": divide_two_variable_4_zero_with_multi_features,
                "param": {
                    "first_var_names": self.features,
                    "second_var_name": "MV",
                },
                "input_data": {"data": "merged_data"},
                "output": ["income_q_to_mv"],
            },
            {
                "func": save_data_to_table,
                "param": {"engine": param_info['common']['save_engine'], "table": "income_q_to_mv",
                          "if_exists": self.if_exists},
                "input_data": {"data": "income_q_to_mv"},
                "output": ["income_q_to_mv"]
            },             
        ]
        self.output_vars = ['income_q_to_mv']  
        

class CashFlowItemToMv(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.if_exists = param_info.get("insert_way", "append")
        self.features = param_info.get("features")
        self.operators = [
            {
                'func': get_hist_data_4_factor_compute,
                 'param': {
                     "read_engine": None,
                     "save_engine": None,
                     "start_date": None,
                     "end_date": None,
                     "table": "cash_flow_q",
                     "field": ["trade_date",  "code",'end_date'] + self.features,
                     "index": ['trade_date',  'code', 'end_date'],

                     "hist_year": -1,
                 },
                 "input_data": {},
                 "output": ['cash_flow_q_data']
            },
#             {
#                 'func': fillna_with_zero,
#                  'param': {},
#                 "input_data": {'data': 'cash_flow_q_data'},
#                 "output": ['cash_flow_q_data_std']
#             },
            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "cash_flow_q_data", "index": "factor_index"},
                "output": ["cash_flow_q_weekly"],
            },
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "valuation_q",
                 "field": ['trade_date', 'code', 'market_cap'],
                 "hist_year": 2,
                 "name_dict": {"market_cap": "MV_JQ"}},
             "input_data": {},
             "output": ['market_cap_jq']},
            {'func': std_mkt_cp,
             'param': {"value_name": "MV_JQ", "output_name": "MV"},
             "input_data": {"data": "market_cap_jq"},
             "output": ['market_cap']
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": 'market_cap', "index": "factor_index"},
                "output": ['market_cap_weekly'],
            },            
            {
                "func": merge_data,
                "param": {},
                "input_data": {"1": "cash_flow_q_weekly", "2": "market_cap_weekly"},
                "output": ["merged_data"]
            }, 
            {
                "func": divide_two_variable_4_zero_with_multi_features,
                "param": {
                    "first_var_names": self.features,
                    "second_var_name": "MV",
                },
                "input_data": {"data": "merged_data"},
                "output": ["cash_flow_to_mv"],
            },
            {
                "func": save_data_to_table,
                "param": {"engine": param_info['common']['save_engine'], "table": "cash_flow_to_mv",
                          "if_exists": self.if_exists},
                "input_data": {"data": "cash_flow_to_mv"},
                "output": ["cash_flow_to_mv"]
            },             
        ]
        self.output_vars = ['cash_flow_to_mv']
        
class FinancialIndicator(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.if_exists = param_info.get("insert_way", "append")
        self.divide_infos = param_info.get("divide_infos")
        self.operators = [
            {
                'func': get_hist_data_4_factor_compute,
                 'param': {
                     "read_engine": None,
                     "save_engine": None,
                     "start_date": None,
                     "end_date": None,
                     "table": "balance_stk",
                     "field": [],
                     "index": ['trade_date',  'code', 'end_date'],

                     "hist_year": -1,
                 },
                 "input_data": {},
                 "output": ['balance_stk_data']
            },
            {
                'func': get_hist_data_4_factor_compute,
                 'param': {
                     "read_engine": None,
                     "save_engine": None,
                     "start_date": None,
                     "end_date": None,
                     "table": "income_q",
                     "field": [],
                     "index": ['trade_date',  'code', 'end_date'],

                     "hist_year": -1,
                 },
                 "input_data": {},
                 "output": ['income_q']
            },
            {
                'func': get_hist_data_4_factor_compute,
                 'param': {
                     "read_engine": None,
                     "save_engine": None,
                     "start_date": None,
                     "end_date": None,
                     "table": "cash_flow_q",
                     "field": [],
                     "index": ['trade_date',  'code', 'end_date'],

                     "hist_year": -1,
                 },
                 "input_data": {},
                 "output": ['cash_flow_q']
            },
            {
                "func": merge_data,
                "param": {},
                "input_data": {"1": "balance_stk_data", "2": "income_q", "3": "cash_flow_q"},
                "output": ["financial_raw_data"]
            },            
            {
                "func": multi_divide_two_variable,
                "param": {
                    "divide_infos": self.divide_infos,
                },
                "input_data": {"data": "financial_raw_data"},
                "output": ["financial_indicator"],
            },
            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},
            },            
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": 'financial_indicator', "index": "factor_index"},
                "output": ['financial_indicator_weekly'],
            }, 
            {
                "func": save_data_to_table,
                "param": {"engine": param_info['common']['save_engine'], "table": "financial_indicator",
                          "if_exists": self.if_exists},
                "input_data": {"data": "financial_indicator_weekly"},
                "output": ["financial_indicator_weekly"]
            },
        ]
        self.output_vars = ['financial_indicator_weekly']
        

        
class ExpensesToMv(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.if_exists = param_info.get("insert_way", "append")
        self.operators = [
            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},
            },
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "income_stk",
                 "field": ["trade_date",  "code", 'administration_expense', 'end_date', "sale_expense", "rd_expenses"],
                 "index": ['trade_date',  'code', 'end_date'],

                 "hist_year": -1,
                 "name_dict": {
                     "administration_expense": "AdministrationExpenses",
                     "sale_expense": "SaleExpenses",
                     "rd_expenses": "RdExpenses"
                 }
             },
             "input_data": {},
             "output": ['expenses_data']},

            {
                "func": cal_factor_ttm,
                "param": {"factor_name": "RdExpenses"},
                "input_data": {"data": "expenses_data", },
                "output": ["rd_expenses_ttm"]
            },
            {
                "func": cal_factor_ttm,
                "param": {"factor_name": "AdministrationExpenses"},
                "input_data": {"data": "expenses_data", },
                "output": ["administration_expenses_ttm"]
            },
            {
                "func": cal_factor_ttm,
                "param": {"factor_name": "SaleExpenses"},
                "input_data": {"data": "expenses_data", },
                "output": ["sale_expenses_ttm"]
            }, 
            {
                "func": merge_data,
                "param": {},
                "input_data": {"1": "rd_expenses_ttm", "2": "administration_expenses_ttm", "3" :"sale_expenses_ttm"},
                "output": ["expenses_ttm"],
            },
            
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "expenses_ttm", "index": "factor_index"},
                "output": ["expenses_ttm_weekly"],
            },
            {
                "class": MarketCap,
                "param": {},
                "output_name_mapping": {"market_cap": "market_cap"},
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": 'market_cap', "index": "factor_index"},
                "output": ['market_cap_weekly'],
            },
            {
                "func": merge_data,
                "param": {},
                "input_data": {"1": "expenses_ttm_weekly", "2": "market_cap_weekly"},
                "output": ["merged_data"]
            },
            {
                "func": divide_two_variable_4_zero,
                "param": {
                    "first_var_name": "RdExpensesTTM",
                    "second_var_name": "MarketCap",
                    "output_name": "RdExpensesTTMToMarketCap"
                },
                "input_data": {"data": "merged_data"},
                "output": ["rd_expenses_to_mv"],
            },
            {
                "func": divide_two_variable_4_zero,
                "param": {
                    "first_var_name": "AdministrationExpensesTTM",
                    "second_var_name": "MarketCap",
                    "output_name": "AdministrationExpensesToMarketCap"
                },
                "input_data": {"data": "merged_data"},
                "output": ["administration_expenses_to_mv"],
            },
            {
                "func": divide_two_variable_4_zero,
                "param": {
                    "first_var_name": "SaleExpensesTTM",
                    "second_var_name": "MarketCap",
                    "output_name": "SaleExpensesTTMToMarketCap"
                },
                "input_data": {"data": "merged_data"},
                "output": ["sale_expenses_to_mv"],
            },
            {
                "func": merge_data,
                "param": {},
                "input_data": {"1": "rd_expenses_to_mv", "2": "administration_expenses_to_mv", "3": "sale_expenses_to_mv"},
                "output": ["expenses_to_mv"]
            },            
            {
                "func": save_data_to_table,
                "param": {"engine": param_info['common']['save_engine'], "table": "expenses_to_mv",
                          "if_exists": self.if_exists},
                "input_data": {"data": "expenses_to_mv"},
                "output": ["expenses_to_mv"]
            }, 
            
        ]
        self.output_vars = ['expenses_to_mv']

class FactorMomentumPerformanceAttributionWithDifferentFeatures(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.start_date = param_info['start_date']
        self.end_date = param_info['end_date']
        self.source_data_infos = param_info['source_data_infos']
        self.invalid_infos = param_info['invalid_infos']
        self.long_window_size = param_info['long_window_size']
        self.short_window_size = param_info['short_window_size']
#         self.is_norm = param_info.get('is_norm', False)
#         self.industry_name = param_info.get("industry_name", "GicsIndustryName")
        self.perf_att_func = param_info.get("perf_att_func", weekly_performance_attribution_with_different_factors)
        self.factors_4_performance_attribution = param_info['factors_4_performance_attribution']
        self.save_info = param_info['save_info']
        self.r_process_param = {}
        self.operators = [
            {
                "func": get_data_from_multi_source,
                "param": {
                    "data_source_infos": self.source_data_infos,
                    "start_date": self.start_date,
                    "end_date": self.end_date
                },
                "input_data": {},
                "output": ['factor']
            },
            {
                "func": generate_one_term_return,
                "param": {
                    "one_week_momentum_name": "MomentumWeeks1",
                    "output_name": "OneTermReturn",
                    "limit": self.r_process_param.get("limit", 0),
                    "is_3_sigma_std": self.r_process_param.get("is_3_sigma_std", False),
                    "is_ecdf": self.r_process_param.get("is_ecdf", False)
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
                "func": self.perf_att_func,
                "param": {
                    "date_to_factors": self.factors_4_performance_attribution,
                    "long_window_size": self.long_window_size,
                    "short_window_size": self.short_window_size,
                    'start_date': self.start_date,
                    "end_date": self.end_date,
                },
                "input_data": {"factor_data": "valid_factor"},
                "output": ["all_data_score"],
            },
            {
                "func": save_data_to_table,
                "param": {"engine": self.save_info['engine'], "table": self.save_info['table'],
                          "if_exists": self.save_info.get("if_exists", "append")},
                "input_data": {"data": "all_data_score"},
                "output": ["all_data_score"]
            }
        ]
        self.output_vars = ["all_data_score"]            
        
        
class FactorMomentumPerformanceAttributionMultilayer(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.start_date = param_info['start_date']
        self.end_date = param_info['end_date']
        self.source_data_infos = param_info['source_data_infos']
        self.invalid_infos = param_info['invalid_infos']
        self.long_window_size = param_info['long_window_size']
        self.short_window_size = param_info['short_window_size']
        self.is_norm = param_info.get('is_norm', False)
        self.industry_name = param_info.get("industry_name", "GicsIndustryName")
        self.layer_tag = param_info['layer_tag']
        self.perf_att_func = param_info["perf_att_func"]

        self.gen_score_func = param_info["gen_score_func"]

        self.save_info = param_info['save_info']
        self.factors_4_performance_attribution = param_info['factors_4_performance_attribution']
        self.factors_4_score = param_info['factors_4_score']
        self.factor_directions = param_info.get("factor_directions", [1 for _ in self.factors_4_score])
        self.r_process_param = param_info.get("r_process_param", {})

        self.operators = [
            {
                "func": get_data_from_multi_source,
                "param": {
                    "data_source_infos": self.source_data_infos,
                    "start_date": self.start_date,
                    "end_date": self.end_date
                },
                "input_data": {},
                "output": ['factor']
            },
            {
                "func": generate_one_term_return,
                "param": {
                    "one_week_momentum_name": "MomentumWeeks1",
                    "output_name": "OneTermReturn",
                    "limit": self.r_process_param.get("limit", 0),
                    "is_3_sigma_std": self.r_process_param.get("is_3_sigma_std", False),
                    "is_ecdf": self.r_process_param.get("is_ecdf", False)
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
                "func": self.perf_att_func,
                "param": {
                    "factors": self.factors_4_performance_attribution,
                    "industry": self.industry_name,
                    "layer_tag": self.layer_tag,
                    'l2': 1000000,
                },
                "input_data": {"factor_data": "valid_factor"},
                "output": ["factor_premium", 'processed_data'],
            },
            {
                "func": self.gen_score_func,
                "param": {
                    "start_date": self.start_date,
                    "end_date": self.end_date,
                    "factors": self.factors_4_score,
                    "factor_directions": self.factor_directions,
                    "long_window_size": self.long_window_size,
                    "short_window_size": self.short_window_size,
                    "layer_tag": self.layer_tag,
                    "is_norm": self.is_norm,
                },
                "input_data": {
                    "multilayer_factor_premium_df": "factor_premium",
                    "processed_factor_df": "processed_data"
                },
                "output": ["all_data_score"]
            },
            {
                "func": save_data_to_table,
                "param": {"engine": self.save_info['engine'], "table": self.save_info['table'],
                          "if_exists": self.save_info.get("if_exists", "append")},
                "input_data": {"data": "all_data_score"},
                "output": ["all_data_score"]
            }
        ]

        self.output_vars = ["all_data_score"]

        