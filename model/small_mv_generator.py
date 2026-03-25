import pandas as pd
from tqdm import tqdm

from indicator_operator import FactorCompute, save_data_to_table
from factor_neutral import get_data_from_multi_source, transfer_data_to_valid_and_not_valid


def small_mv_weight(factor_data, code_count):

    factor_data['mv_rank'] = factor_data.groupby('trade_date')["MarketCap"].rank()
    small_mv_factor_data = factor_data[factor_data['mv_rank'] <= code_count]
    small_mv_factor_data['weight'] = 1/code_count
    return small_mv_factor_data


class SmallMvFactor(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.start_date = param_info['start_date']
        self.end_date = param_info['end_date']
        self.source_data_infos = param_info['source_data_infos']
        self.invalid_infos = param_info['invalid_infos']
        self.save_info = param_info['save_info']
        self.code_count = param_info['code_count']
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
                "func": small_mv_weight,
                "param": {
                    "code_count": self.code_count,
                },
                "input_data": {"factor_data": "valid_factor"},
                "output": ["small_mv_weight"],
            },
            {
                "func": save_data_to_table,
                "param": {"engine": self.save_info['engine'], "table": self.save_info['table'],
                          "if_exists": self.save_info.get("if_exists", "append")},
                "input_data": {"data": "small_mv_weight"},
                "output": ["small_mv_weight"]
            }
        ]
        self.output_vars = ["small_mv_weight"]

