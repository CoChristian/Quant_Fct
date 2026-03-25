#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 17:14:24 2022
data process pipeline no class, not recursive
@author: lianrui
"""


import pandas as pd
from functools import wraps
from tqdm import tqdm

tqdm.pandas()
import statsmodels.api as sm
import numpy as np
from statsmodels.regression.rolling import RollingOLS
import hashlib
import json
from model import SQL_api
from func_operator import *
import time
import datetime
# from factor_neutral import transfer_data_to_valid_and_not_valid


class FactorCompute(object):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        self.param_info = param_info
        self.input_name_mapping = input_name_mapping
        self.raw_output_name_mapping = output_name_mapping
        self.data_cash = {}
        self.operators = [
        ]
        self.input_vars = []
        self.output_vars = []
        self.tag = self.__class__.__name__

    def transfer_operators_2_func_pipline(self, operators, pre_fix, pre_input_data_mapping, pre_output_name_mapping):
        func_infos = []
        for operator in operators:
            if "func" in operator:

                func_infos.append(operator)
            elif "class" in operator:
                class_name = operator['class'].__name__
                # print(class_name)
                param = operator.get('param', {})
                test_clf = operator['class'](param, {}, {})
                class_operators = test_clf.operators
                class_output_vars = test_clf.output_vars

                input_name_mapping = operator.get('input_name_mapping', dict())
                raw_output_name_mapping = operator.get('output_name_mapping', dict())

                output_name_mapping = {}
                for output_var in class_output_vars:
                    output_name_mapping.update({output_var : raw_output_name_mapping.get(output_var, output_var)})
                # if output_name_mapping is {}:
                #     output_name_mapping = {_: _ for _ in class_output_vars}

                sub_funcs = self.transfer_operators_2_func_pipline(class_operators, class_name, input_name_mapping, output_name_mapping)
                func_infos.extend(sub_funcs)
        update_func_infos = []
        for func_info in func_infos:
            update_func_info = func_info.copy()
            input_data = func_info['input_data']
            update_input_data = {}
            for param_name, input_name in input_data.items():
                if param_name in pre_input_data_mapping:
                    update_input_data.update({param_name: pre_input_data_mapping[param_name]})
                elif input_name in pre_output_name_mapping:
                    update_input_data.update({param_name: pre_output_name_mapping[input_name]})
                else:
                    update_input_data.update({param_name: "{}_{}".format(pre_fix, input_name)})

            update_func_info.update({"input_data": update_input_data})
            output_names = func_info['output']

            update_func_info.update({"output": [
                pre_output_name_mapping[_] if _ in pre_output_name_mapping else "{}_{}".format(pre_fix, _) for _ in
                output_names]})
            update_func_infos.append(update_func_info)
        return update_func_infos

    def compute(self):
        self.output_name_mapping = {}
        for output_var in self.output_vars:
            self.output_name_mapping.update({output_var: self.raw_output_name_mapping.get(output_var, output_var)})
        self.funcs = self.transfer_operators_2_func_pipline(self.operators, self.tag, self.input_name_mapping, self.output_name_mapping) # 更新了参数input 和 output 的变量名，加入了class_name, 生成调用的工程流程
        # print(self.param_info)
        #
        # for func in self.funcs:
        #     print(func)
        for func_info in self.funcs:
            print(func_info)
            func = func_info['func']
            # print("func {}".format(func.__name__))
            param = func_info['param']
            common_param = self.param_info.get("common", {})
            param = update_param(param, common_param)
            special_param = self.param_info.get("special", {})

            if "tag" in func_info:
                input_param = special_param.get(func_info['tag'], {})
                param = update_param(param, input_param)
            input_data = func_info['input_data']

            for param_name, param_name_in_cash in input_data.items():
                # assert param_name_in_cash in self.data_cash, print("{} not in data cash".format(param_name_in_cash))
                if param_name_in_cash in self.data_cash:
                    param.update({param_name: self.data_cash[param_name_in_cash]})
                else:
                    print("{} not in data cash".format(param_name_in_cash))
                    param.update({param_name: None})
            # if func.__name__ == "multiply_two_variable":
            #     import pdb
            #     pdb.set_trace()

            result = func(**param)

            output_names = func_info['output']
            if type(result) is tuple:
                assert len(result) == len(output_names)
                self.data_cash.update(dict(zip(output_names, result)))

            else:
                assert len(output_names) == 1
                self.data_cash.update({output_names[0]: result})
        return {output_var: self.data_cash[self.output_name_mapping.get(output_var, output_var)] for output_var in self.output_vars}


class OptToTrade(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': gen_opt_2_trade,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "freq": None,
                 "start_date": None,
                 "end_date": None,},
             "input_data": {},
             "output": ['opt_to_trade']},

        ]
        self.output_vars = ["opt_to_trade"]

class UnadjClosePrice(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "daily_trading_data_unadjusted",
                 "field": ["trade_date", 'code', 'close'],
                 "hist_year": 2,
                 "name_dict": {"close": self.__class__.__name__}},
             "input_data": {},
             "output": ['unadj_close_price']},

        ]
        self.output_vars = ["unadj_close_price"]


class AdjFactor(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "daily_trading_data",
                 "field": ['trade_date', 'code', 'factor'],
                 "hist_year": 2,
                 "name_dict": {"factor": self.__class__.__name__}},
             "input_data": {},
             "output": ['adj_factor']},
        ]
        self.output_vars = ["adj_factor"]


class AdjClosePrice(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": UnadjClosePrice,
                 'param': {},
                "input_name_mapping": {},

                "output_name_mapping": {"unadj_close_price": "unadj_close_price"},
                 "output": ["unadj_close_price"],
                "tag": "cal_unadj_close_price"
            },
            {
                "class": AdjFactor,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"adj_factor": "adj_factor"},
                "output": ["adj_factor"],
                "tag": "cal_adj_factor"
            },
            {
                "func": merge_data,
                "param": {},
                "input_data": {"1": "unadj_close_price", "2": "adj_factor"},
                "output": ["merged_data"]
            },
            {
                "func": multiply_two_variable,
                "param": {"first_var_name": "UnadjClosePrice", "second_var_name": "AdjFactor",  "output_name" : self.__class__.__name__},
                "input_data": {"data": "merged_data"},
                "output": ["adj_close_price"],
            }

        ]
        self.output_vars = ["adj_close_price"]


class FactorIndex(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)

        self.operators = [
            {
                "func": get_fac_idx,
                "param": {"start_date": None, "end_date": None, "freq": "default", "read_engine": None, "save_engine": None},
                "input_data": {"data": ""},
                "output": ["factor_index"],
            }
        ]
        self.output_vars = ['factor_index']

class FactorIndexOnline(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)

        self.operators = [
            {
                "func": get_fac_idx_online,
                "param": {"start_date": None, "end_date": None, "freq": "default", "read_engine": None, "save_engine": None},
                "input_data": {"data": ""},
                "output": ["factor_index"],
            }
        ]
        self.output_vars = ['factor_index']


class DailyIndex(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)

        self.operators = [
            {
                "func": get_fac_idx,
                "param": {"start_date": None, "end_date": None, "freq": "daily", "read_engine": None, "save_engine": None},
                "input_data": {"data": ""},
                "output": ["daily_index"],
            }
        ]
        self.output_vars = ['daily_index']


class AdjClosePriceWeekly(FactorCompute):
    """weekly sampled AdjClosePrice as a continous factor"""
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": AdjClosePrice,
                "output_name_mapping": {"adj_close_price": "adj_close_price"},
            },
            {
                "class": FactorIndex,
                "input_name_mapping": {"data": "adj_close_price"},
                "output_name_mapping": {"factor_index": "factor_index"},
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "adj_close_price", "index": "factor_index"},
                "output": ["adj_close_price_weekly"],
            },
        ]
        self.output_vars = ["adj_close_price_weekly"]




class MarketCap(FactorCompute):
    """
        MktCap
    """
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "valuation_q",
                 "field": ['trade_date', 'code', 'market_cap'],
                 "hist_year": 2,
                 "name_dict": {"market_cap": "{}_JQ".format(self.__class__.__name__)}},
             "input_data": {},
             "output": ['market_cap_jq']},
            {'func': std_mkt_cp,
             'param': {"value_name": "{}_JQ".format(self.__class__.__name__), "output_name": self.__class__.__name__},
             "input_data": {"data": "market_cap_jq"},
             "output": ['market_cap']},
        ]
        self.output_vars = ["market_cap"]


class CirculatingMarketCap(FactorCompute):
    """
        MktCap
    """
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "valuation_q",
                 "field": ['trade_date', 'code', 'circulating_market_cap'],
                 "hist_year": 2,
                 "name_dict": {"circulating_market_cap": "{}_JQ".format(self.__class__.__name__)}},
             "input_data": {},
             "output": ['circulating_market_cap_jq']},
            {'func': std_mkt_cp,
             'param': {"value_name": "{}_JQ".format(self.__class__.__name__), "output_name": self.__class__.__name__},
             "input_data": {"data": "circulating_market_cap_jq"},
             "output": ['circulating_market_cap']},
        ]
        self.output_vars = ["circulating_market_cap"]


class LogMktCap(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": MarketCap,
                "output_name_mapping": {"market_cap": "market_cap"},
            },
            {
                "func": cal_market_log,
                "param": {"value_name": "MarketCap", "output_name": self.__class__.__name__},
                "input_data": {"data": "market_cap"},
                "output": ["log_mkt_cap"]
            }
        ]
        self.output_vars = ['log_mkt_cap']



class NonLinearSize(FactorCompute):
    """
   A class that computes the Non-Linear Size Factor, a subfactor of the size factor.
    """
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": LogMktCap,
                "output_name_mapping": {"log_mkt_cap": "log_mkt_cap"},
            },
            {
                "func": cal_nonlinear_size,
                "param": {"value_name":"LogMktCap", "window_size": 252, "output_name": self.__class__.__name__},
                "input_data": {"data": "log_mkt_cap"},
                "output": ["nonlinear_size"],
            }
        ]
        self.output_vars = ['nonlinear_size']






class PreAdjClosePrice(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": AdjClosePrice,
                "output_name_mapping": {"adj_close_price": "adj_close_price"},
            },
            {
                "func": merge_data,
                "param": {},
                "input_data": {"1": "adj_close_price"},
                "output": ["merged_data"]
            },
            {
                "func": cal_pre_day_price,
                "param": {"value_name": "AdjClosePrice",   "output_name" : self.__class__.__name__},
                "input_data": {"data": "merged_data"},
                "output": ["pre_adj_close_price"]
            },
        ]
        self.output_vars = ['pre_adj_close_price']

@timer
def cal_pct_chg_hfq(adj_close_price_name, pre_adj_close_price_name, data, output_name):

    pct_chg_hfq = divide_two_variable(adj_close_price_name, pre_adj_close_price_name, output_name, data).applymap(lambda x: (x-1)*100)

    return pct_chg_hfq


class PctChgHfqDaily(FactorCompute):
    """Return = (AdjClose - AdjPreClose)/AdjPreClose"""
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": AdjClosePrice,
                "output_name_mapping": {"adj_close_price": "adj_close_price"},
            },
            {
                "class": PreAdjClosePrice,
                "output_name_mapping": {"pre_adj_close_price": "pre_adj_close_price"},
            },
            {
                "func": merge_data,
                "param": {},
                "input_data": {"1": "adj_close_price", "2": "pre_adj_close_price"},
                "output": ["merged_data"]
            },
            {
                "func": cal_pct_chg_hfq,
                "param": {"adj_close_price_name": "AdjClosePrice", "pre_adj_close_price_name": "PreAdjClosePrice",  "output_name" : self.__class__.__name__},
                "input_data": {"data": "merged_data"},
                "output": ["pct_chg_hfq_daily"],
            }

        ]
        self.output_vars = ["pct_chg_hfq_daily"]


#### 量价指标 #####
@timer
def cal_momentum(value_name, window_size, data, output_name):

    data[output_name] = data[value_name].groupby(level='code').progress_apply(lambda x: x / x.shift(window_size) - 1)
    return data[[output_name]]


class MomentumWeeks(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.window_size = param_info['window_size']
        self.operators = [
            {
                "class": AdjClosePriceWeekly,
                "output_name_mapping": {"adj_close_price_weekly": "adj_close_price_weekly"},
            },
            {
                "func": cal_momentum,
                "param": {"value_name": "AdjClosePrice", "window_size": self.window_size,  "output_name" : "{}{}".format(self.__class__.__name__, self.window_size)},
                "input_data": {"data": "adj_close_price_weekly"},
                "output": ["momentum_weeks_{}".format(self.window_size)],
            }
        ]
        self.output_vars = ["momentum_weeks_{}".format(self.window_size)]


class LongMinusShort(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.long_window_size = param_info['long_window_size']
        self.short_window_size = param_info['short_window_size']
        self.operators = [
            {
                "class": MomentumWeeks,
                "param": {"window_size": self.long_window_size},
                "output_name_mapping": {"momentum_weeks_{}".format(self.long_window_size):  "momentum_weeks_{}".format(self.long_window_size)},
            },
            {
                "class": MomentumWeeks,
                "param": {"window_size": self.short_window_size},
                "output_name_mapping": {"momentum_weeks_{}".format(self.short_window_size): "momentum_weeks_{}".format(
                    self.short_window_size)},
            },
            {
                "func": merge_data,
                "param": {},
                "input_data": {"1": "momentum_weeks_{}".format(self.long_window_size),
                               "2": "momentum_weeks_{}".format(self.short_window_size)},
                "output": ["merged_data"],
            },
            {
                "func": minus_two_variable,
                "param": {"first_var_name": "MomentumWeeks{}".format(self.long_window_size),
                          "second_var_name": "MomentumWeeks{}".format(self.short_window_size),
                          "output_name": self.__class__.__name__},
                "input_data": {"data": "merged_data"},
                "output": ["long_minus_short"],
            }
        ]
        self.output_vars = ["long_minus_short"]


@timer
def cal_rsi(rolling_n, pct_chg_hfq_name, output_name, data):
    pct_chg_hfq = data[pct_chg_hfq_name]
    up_tag = pct_chg_hfq.map(lambda x: x>0)
    down_tag = pct_chg_hfq.map(lambda x: x<0)
    up_pct_chg_hfq = up_tag*pct_chg_hfq
    down_pct_chg_hfq = down_tag*pct_chg_hfq
    up_sum = up_pct_chg_hfq.groupby(level='code').progress_apply(lambda x: x.sort_index(level='trade_date').rolling(rolling_n).sum())
    up_sum.name = 'up_sum'
    down_sum = down_pct_chg_hfq.groupby(level='code').progress_apply(lambda x: x.sort_index(level='trade_date').rolling(rolling_n).sum())
    down_sum = down_sum.map(abs)

    down_sum.name = "down_sum"
    up_down_sum = pd.concat([up_sum, down_sum], axis=1)
    up_down_sum[output_name] = up_down_sum['up_sum'] / (
                up_down_sum['up_sum'] + up_down_sum['down_sum'])
    up_down_sum[output_name] = up_down_sum[output_name] * 100
    up_down_sum[output_name] = up_down_sum[output_name].replace(np.inf, 0)
    return up_down_sum[[output_name]]


class RSI(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.window_size = self.param_info.get("window_size", 10)
        self.operators = [
            {
                "class": PctChgHfqDaily,
                "output_name_mapping": {"pct_chg_hfq_daily": "pct_chg_hfq_daily"},
            },
            {
                "func": cal_rsi,
                "param": {"rolling_n": self.window_size,
                          "pct_chg_hfq_name": "PctChgHfqDaily",
                          "output_name" : "{}{}Days".format(self.__class__.__name__, self.window_size)
                          },
                "input_data": {"data": "pct_chg_hfq_daily"},
                "output": ["rsi_{}_days".format(self.window_size)],
            }
        ]
        self.output_vars = ["rsi_{}_days".format(self.window_size)]


@timer
def cal_rolling_std(value_name, window_size, data, output_name):
    data[output_name] = data.sort_index(level=['code', 'trade_date'])[value_name].groupby(level='code').progress_apply(
        lambda x: x.rolling(window_size).std(ddof=0))
    data = data.reset_index().set_index(['trade_date', 'code'])
    return data[[output_name]]


class Volatility(FactorCompute):
    """n-trading day rolling volitility. where default n = 60,
        pctchghfq none value changed 2 zero
    Volatility60Days = PctChgHfq.rolling(60).std()"""
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.window_size = param_info.get("window_size", 60)
        self.operators = [
            {
                "class": PctChgHfqDaily,
                "output_name_mapping": {"pct_chg_hfq_daily": "pct_chg_hfq_daily"},
            },
            {
                "func": merge_data,
                "param": {},
                "input_data": {"1": "pct_chg_hfq_daily"},
                "output": ["merged_data"]
            },
            {
                "func": cal_rolling_std,
                "param": {"value_name": "PctChgHfqDaily", "window_size": self.window_size,  "output_name" : "{}{}Days".format(self.__class__.__name__, self.window_size)},
                "input_data": {"data": "merged_data"},
                "output": ["volatility_{}_days".format(self.window_size)],
            }

        ]
        self.output_vars = ["volatility_{}_days".format(self.window_size)]


class TurnoverRatio(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "valuation_q",
                 "field": ["trade_date", 'code', 'turnover_ratio'],
                 "hist_year": 2,
                 "name_dict": {"turnover_ratio": self.__class__.__name__}},
             "input_data": {},
             "output": ['turnover_ratio']},

        ]
        self.output_vars = ["turnover_ratio"]

    """Daily TurnOverRatio as a continous factor"""


@timer
def cal_share_turnover(value_name, month_count, data, output_name):
    days = month_count*21
    share_turnover = data[value_name].groupby(level='code').progress_apply(
        lambda x: x.sort_index(level='trade_date').rolling(days).sum())/ month_count
    share_turnover = share_turnover.map(np.log)
    share_turnover.name=output_name
    share_turnover = share_turnover.reset_index().set_index(['trade_date', 'code'])
    share_turnover = share_turnover.replace(np.inf, np.nan).replace(-np.inf, np.nan)
    return share_turnover


class STOM(FactorCompute):
    """
    Monthly share turnover
    turnover.rolling(21).apply(np.log(sum(x)))
    """
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": TurnoverRatio,
                "output_name_mapping": {"turnover_ratio": "turnover_ratio"}
            },
            {
                "func": cal_share_turnover,
                "param": {
                    "value_name": "TurnoverRatio",
                    "output_name": self.__class__.__name__,
                    "month_count": 1,
                },
                "input_data": {"data": "turnover_ratio"},
                "output": ["stom"],
            }
        ]
        self.output_vars = ["stom"]



class STOQ(FactorCompute):
    """
    Quarterly share turnover
    turnover.rolling(63).apply(np.log(sum(x)))
    """
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": STOM,
                "output_name_mapping": {"stom": "stom"}
            },
            {
                "func": cal_stoq,
                "param": {
                    "value_name": "STOM",
                    "output_name": self.__class__.__name__,
                },
                "input_data": {"data": "stom"},
                "output": ["stoq"],
            }
        ]
        self.output_vars = ["stoq"]




class STOA(FactorCompute):
    """
    Quarterly share turnover
    turnover.rolling(63).apply(np.log(sum(x)))
    """
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": STOM,
                "output_name_mapping": {"stom": "stom"}
            },
            {
                "func": cal_stoa,
                "param": {
                    "value_name": "STOM",
                    "output_name": self.__class__.__name__,
                },
                "input_data": {"data": "stom"},
                "output": ["stoa"],
            }
        ]
        self.output_vars = ["stoa"]


### benchmark
class BenchmarkPrice(FactorCompute):
    """Return the benchamrk index level as a continous factor"""
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.benchmark_code = self.param_info.get("benchmark_code", "000905.XSHG")

        self.operators = [
            {
                'func': get_hist_data_4_factor_compute,
                 'param': {
                     "read_engine": None,
                     "save_engine": None,
                     "start_date": None,
                     "end_date": None,
                     "table": "index_level",
                     "field": ["trade_date",  "code",'close'],
                     "hist_year": 2,
                     "other_filter_info": {"field": "code", "type": "equal", "param": self.benchmark_code},
                     "name_dict": {"close": "{}{}".format(self.__class__.__name__, self.benchmark_code.replace(".", ""))}},
                 "input_data": {},
                 "output": ['benchmark_price_{}'.format(self.benchmark_code.replace(".", ""))]
            },
        ]
        self.output_vars = ['benchmark_price_{}'.format(self.benchmark_code.replace(".", ""))]


    

class BenchmarkPctChg(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.benchmark_code = self.param_info.get("benchmark_code", "000905.XSHG")

        self.operators = [
            {
                "class": BenchmarkPrice,
                "param": {"benchmark_code": self.benchmark_code},
                "output_name_mapping": {'benchmark_price_{}'.format(self.benchmark_code.replace(".", "")): 'benchmark_price_{}'.format(self.benchmark_code.replace(".", ""))}
            },
            {
                "func": merge_data,
                "param": {},
                "input_data": {"1":  'benchmark_price_{}'.format(self.benchmark_code.replace(".", "")),},
                "output": ["merged_data"]
            },
            {
                "func": cal_benchmark_pct_chg,
                "param": {
                    "value_name": "{}{}".format("BenchmarkPrice", self.benchmark_code.replace(".", "")),
                    "output_name": "{}{}".format(self.__class__.__name__, self.benchmark_code.replace(".", ""))},
                "input_data": {"data": "merged_data"},
                "output": ['benchmark_pct_chg_{}'.format(self.benchmark_code.replace(".", ""))],
            }

        ]
        self.output_vars = ['benchmark_pct_chg_{}'.format(self.benchmark_code.replace(".", ""))]


### beta ####


class MarketBetaFromGenerateWeight(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.benchmark_code = self.param_info.get("benchmark_code", "csi500")
        self.window_size = self.param_info.get("window_size", 252)    

        self.operators = [
            {
                "class": PctChgHfqDaily,
                "output_name_mapping": {"pct_chg_hfq_daily": "pct_chg_hfq_daily"}
            },
#             {
#                 "class": IndexWeight,
#                 "param": {"index_code": self.benchmark_code},
#                 "output_name_mapping": {"{}_weight_weekly".format(self.benchmark_code): "{}_weight_weekly".format(self.benchmark_code)},
#             },
            {
                'func': get_hist_data_4_factor_compute,
                'param': {
                    "read_engine": None,
                    "save_engine": None,
                    "start_date": None,
                    "end_date": None,
                    "table": "{}_weight".format(self.benchmark_code),
                    # "table": "real_index_weight",
                    "field": ['trade_date', 'code', "weight"],
                    "hist_year": -1,
                    "name_dict": {"weight": "{}RawWeight".format(self.benchmark_code.upper())}},
                "input_data": {},
                "output": ['{}_monthly_weight'.format(self.benchmark_code)]
            },

            {
                "func": get_fac_idx,
                "param": {"start_date": None, "end_date": None, "freq": "default", "read_engine": None, "save_engine": None,},
                "input_data": {"data": '{}_monthly_weight'.format(self.benchmark_code)},
                "output": ["factor_index"],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "zero"},
                "input_data": {"data": '{}_monthly_weight'.format(self.benchmark_code), "index": "factor_index"},
                "output": ['{}_monthly_weight_weekly'.format(self.benchmark_code)],
            },
            {
                "class": CirculatingMarketCap,
                "output_name_mapping": {"circulating_market_cap": "circulating_market_cap"},
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": 'circulating_market_cap', "index": "factor_index"},
                "output": ['circulating_market_cap_weekly'],
            },
            {
                "class": MarketCap,
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
                "input_data": {"1": '{}_monthly_weight_weekly'.format(self.benchmark_code),
                               "2": "circulating_market_cap_weekly",
                               "3": "market_cap_weekly"},
                "output": ['{}_monthly_weight_mv_weekly'.format(self.benchmark_code)]
            },
            {
                "func": gen_mv_based_weight_flag,
                "param": {
                    "circ_mv_name": "CirculatingMarketCap",
                    "total_mv_name": "MarketCap",
                    "raw_weight_name": "{}RawWeight".format(self.benchmark_code.upper()),
                    "output_name": "{}MonthlyMvWeight".format(self.benchmark_code.upper()),
                    "output_raw_weight": True,

                },
                "input_data": {"data": '{}_monthly_weight_mv_weekly'.format(self.benchmark_code)},
                "output": ['{}_monthly_upt_weight_weekly'.format(self.benchmark_code)],
            },
            {
                'func': cal_benchmark_pct_chg_from_index_weight,
                 'param': {
                     
                     "weight_name": "{}MonthlyMvWeight".format(self.benchmark_code.upper()),
                 },
                 "input_data": {"code_pct_chg_hfq": "pct_chg_hfq_daily", "index_weight": "{}_monthly_upt_weight_weekly".format(self.benchmark_code),
                               },
                 "output": ['benchmark_pct_chg_{}'.format(self.benchmark_code.replace(".", ""))],                
            },
            {
                "func": cal_market_beta,
                "param": {
                    "output_name": "{}{}{}".format(self.__class__.__name__, self.benchmark_code.replace(".", ""), self.window_size),
                    "window_size": self.window_size
                },
                "input_data": {"code_pct_chg_hfq": "pct_chg_hfq_daily", "benchmark_pct_chg": 'benchmark_pct_chg_{}'.format(self.benchmark_code.replace(".", ""))},
                "output": ["market_beta_{}_{}".format(self.benchmark_code.replace(".", ""), self.window_size)]
            },                        
        ]
        self.output_vars = ["market_beta_{}_{}".format(self.benchmark_code.replace(".", ""), self.window_size)]


class MarketBeta(FactorCompute):
    """CAPM market beta estimated by 252 rolling regression"""
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.benchmark_code = self.param_info.get("benchmark_code", "000905.XSHG")
        self.window_size = self.param_info.get("window_size", 252)

        self.operators = [
            {
                "class": BenchmarkPctChg,
                "param": {"benchmark_code": self.benchmark_code},
                "output_name_mapping": {'benchmark_pct_chg_{}'.format(self.benchmark_code.replace(".", "")): 'benchmark_pct_chg_{}'.format(self.benchmark_code.replace(".", ""))}
            },
            {
                "class": PctChgHfqDaily,
                "output_name_mapping": {"pct_chg_hfq_daily": "pct_chg_hfq_daily"}
            },
            {
                "func": cal_market_beta,
                "param": {
                    "output_name": "{}{}{}".format(self.__class__.__name__, self.benchmark_code.replace(".", ""), self.window_size),
                    "window_size": self.window_size
                },
                "input_data": {"code_pct_chg_hfq": "pct_chg_hfq_daily", "benchmark_pct_chg": 'benchmark_pct_chg_{}'.format(self.benchmark_code.replace(".", ""))},
                "output": ["market_beta_{}_{}".format(self.benchmark_code.replace(".", ""), self.window_size)]
            },
        ]
        self.output_vars = ["market_beta_{}_{}".format(self.benchmark_code.replace(".", ""), self.window_size)]



##### financial factor
class TotalOperatingRevenue(FactorCompute):
    """total opertaing revenue as a continous factor"""
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "income_stk",
                 "field": ["trade_date",  "code", 'total_operating_revenue', "end_date"],
                 "index": ['trade_date',  'code', 'end_date'],
                 "hist_year": -1,
                 "name_dict": {"total_operating_revenue": self.__class__.__name__}},
             "input_data": {},
             "output": ['total_operating_revenue']},
        ]
        self.output_vars = ["total_operating_revenue"]


class OperatingProfit(FactorCompute):
    """total opertaing revenue as a continous factor"""
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "income_stk",
                 "field": ["trade_date",  "code", 'operating_profit', "end_date"],
                 "index": ['trade_date',  'code', 'end_date'],
                 "hist_year": -1,
                 "name_dict": {"operating_profit": self.__class__.__name__}},
             "input_data": {},
             "output": ['operating_profit']},
        ]
        self.output_vars = ["operating_profit"]


class TotalCompositeIncomeQuarterly(FactorCompute):
    """

    """
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "income_q",
                 "field": ["trade_date",  "code",'total_composite_income', "end_date"],
                 "index": ['trade_date',  'code', 'end_date'],
                 "hist_year": -1,
                 "name_dict": {"total_composite_income": self.__class__.__name__}},
             "input_data": {},
             "output": ['total_composite_income_quarterly']},
        ]
        self.output_vars = ["total_composite_income_quarterly"]


class TotalOperatingRevenueQuarterly(FactorCompute):
    """

    """
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "income_q",
                 "field": ["trade_date",  "code", 'total_operating_revenue', 'end_date'],
                 "index": ['trade_date',  'code', 'end_date'],
                 "hist_year": -1,
                 "name_dict": {"total_operating_revenue": self.__class__.__name__}},
             "input_data": {},
             "output": ['total_operating_revenue_quarterly']},
        ]
        self.output_vars = ["total_operating_revenue_quarterly"]


class OperatingProfitQuarterly(FactorCompute):
    """

    """
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "indicator_q",
                 "field": ["trade_date",  "code", 'operating_profit', 'end_date'],
                 "index": ['trade_date',  'code', 'end_date'],
                 "hist_year": -1,
                 "name_dict": {"operating_profit": self.__class__.__name__}},
             "input_data": {},
             "output": ['operating_profit_quarterly']},
        ]
        self.output_vars = ["operating_profit_quarterly"]


class OperatingTaxSurcharges(FactorCompute):
    """

    """
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "income_stk",
                 "field": ["trade_date",  "code", 'Operating_Tax_Surcharges', 'end_date'],
                 "index": ['trade_date',  'code', 'end_date'],

                 "hist_year": -1,
                 "name_dict": {"Operating_Tax_Surcharges": self.__class__.__name__}},
             "input_data": {},
             "output": ['operating_tax_surcharges']},
        ]
        self.output_vars = ["operating_tax_surcharges"]


class OperatingCost(FactorCompute):
    """

    """
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "income_stk",
                 "field": ["trade_date",  "code", 'operating_cost', 'end_date'],
                 "index": ['trade_date',  'code', 'end_date'],

                 "hist_year": -1,
                 "name_dict": {"operating_cost": self.__class__.__name__}},
             "input_data": {},
             "output": ['operating_cost']},
        ]
        self.output_vars = ["operating_cost"]


class SaleExpense(FactorCompute):
    """

    """
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "income_stk",
                 "field": ["trade_date",  "code", 'sale_expense', 'end_date'],
                 "index": ['trade_date',  'code', 'end_date'],

                 "hist_year": -1,
                 "name_dict": {"sale_expense": self.__class__.__name__}},
             "input_data": {},
             "output": ['sale_expense']},
        ]
        self.output_vars = ["sale_expense"]


class AdministrationExpense(FactorCompute):
    """

    """
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "income_stk",
                 "field": ["trade_date",  "code", 'administration_expense', 'end_date'],
                 "index": ['trade_date',  'code', 'end_date'],

                 "hist_year": -1,
                 "name_dict": {"administration_expense": self.__class__.__name__}},
             "input_data": {},
             "output": ['administration_expense']},
        ]
        self.output_vars = ["administration_expense"]


class InterestExpense(FactorCompute):
    """

    """
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "income_stk",
                 "field": ["trade_date",  "code", 'interest_expense', 'end_date'],
                 "index": ['trade_date',  'code', 'end_date'],

                 "hist_year": -1,
                 "name_dict": {"interest_expense": self.__class__.__name__}},
             "input_data": {},
             "output": ['interest_expense']},
        ]
        self.output_vars = ["interest_expense"]


class InterestExpenseQuarterly(FactorCompute):
    """

    """
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "income_q",
                 "field": ["trade_date",  "code", 'interest_expense', 'end_date'],
                 "index": ['trade_date',  'code', 'end_date'],

                 "hist_year": -1,
                 "name_dict": {"interest_expense": self.__class__.__name__}},
             "input_data": {},
             "output": ['interest_expense_quarterly']},
        ]
        self.output_vars = ["interest_expense_quarterly"]


class CommissionExpense(FactorCompute):
    """

    """
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "income_stk",
                 "field": ["trade_date",  "code", 'commission_expense', 'end_date'],
                 "index": ['trade_date',  'code', 'end_date'],

                 "hist_year": -1,
                 "name_dict": {"commission_expense": self.__class__.__name__}},
             "input_data": {},
             "output": ['commission_expense']},
        ]
        self.output_vars = ["commission_expense"]


class RdExpenses(FactorCompute):
    """

    """
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "income_stk",
                 "field": ["trade_date",  "code", 'rd_expenses', 'end_date'],
                 "index": ['trade_date',  'code', 'end_date'],

                 "hist_year": -1,
                 "name_dict": {"rd_expenses": self.__class__.__name__}},
             "input_data": {},
             "output": ['rd_expenses']},
        ]
        self.output_vars = ["rd_expenses"]


class AssetImpairmentLoss(FactorCompute):
    """

    """
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "income_stk",
                 "field": ["trade_date",  "code", 'asset_impairment_loss', 'end_date'],
                 "index": ['trade_date',  'code', 'end_date'],

                 "hist_year": -1,
                 "name_dict": {"asset_impairment_loss": self.__class__.__name__}},
             "input_data": {},
             "output": ['asset_impairment_loss']},
        ]
        self.output_vars = ["asset_impairment_loss"]


class OtherEarnings(FactorCompute):
    """

    """
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "income_stk",
                 "field": ["trade_date",  "code", 'other_earnings', 'end_date'],
                 "index": ['trade_date',  'code', 'end_date'],

                 "hist_year": -1,
                 "name_dict": {"other_earnings": self.__class__.__name__}},
             "input_data": {},
             "output": ['other_earnings']},
        ]
        self.output_vars = ["other_earnings"]


class IncomeTax(FactorCompute):
    """

    """
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "income_stk",
                 "field": ["trade_date",  "code", 'income_tax', 'end_date'],
                 "index": ['trade_date',  'code', 'end_date'],

                 "hist_year": -1,
                 "name_dict": {"income_tax": self.__class__.__name__}},
             "input_data": {},
             "output": ['income_tax']},
        ]
        self.output_vars = ["income_tax"]


class TotalProfit(FactorCompute):
    """

    """
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "income_stk",
                 "field": ["trade_date",  "code", 'total_profit', 'end_date'],
                 "index": ['trade_date',  'code', 'end_date'],

                 "hist_year": -1,
                 "name_dict": {"total_profit": self.__class__.__name__}},
             "input_data": {},
             "output": ['total_profit']},
        ]
        self.output_vars = ["total_profit"]




class EarningBeforeInterestAndTaxes(FactorCompute):
    """Compute EBIT using top down approach"""

    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": TotalOperatingRevenue,
                 'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"total_operating_revenue": "total_operating_revenue"},
                 "output": ["total_operating_revenue"],
            },
            {
                "class": OperatingTaxSurcharges,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"operating_tax_surcharges": "operating_tax_surcharges"},
                "output": ["operating_tax_surcharges"],
            },
            {
                "class": OperatingCost,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"operating_cost": "operating_cost"},
                "output": ["operating_cost"],
            },
            {
                "class": SaleExpense,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"sale_expense": "sale_expense"},
                "output": ["sale_expense"],
            },
            {
                "class": AdministrationExpense,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"administration_expense": "administration_expense"},
                "output": ["administration_expense"],
            },
            {
                "class": InterestExpense,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"interest_expense": "interest_expense"},
                "output": ["interest_expense"],
            },
            {
                "class": CommissionExpense,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"commission_expense": "commission_expense"},
                "output": ["commission_expense"],
            },
            {
                "class": RdExpenses,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"rd_expenses": "rd_expenses"},
                "output": ["rd_expenses"],
            },
            {
                "class": AssetImpairmentLoss,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"asset_impairment_loss": "asset_impairment_loss"},
                "output": ["asset_impairment_loss"],
            },
            {
                "class": OtherEarnings,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"other_earnings": "other_earnings"},
                "output": ["other_earnings"],
            },
            {
                "func": merge_data,
                "param": {},
                "input_data": {
                    "1": "total_operating_revenue",
                    "2": "operating_tax_surcharges",
                    "3": "operating_cost",
                    "4": "sale_expense",
                    "5": "administration_expense",
                    "6": "interest_expense",
                    "7": "commission_expense",
                    "8": "rd_expenses",
                    "9": "asset_impairment_loss",
                    "10": "other_earnings"
                },
                "output": ["merged_data"]
            },
            {
                "func": cal_ebit,
                "param": {
                    "total_operating_revenue_name": "TotalOperatingRevenue",
                    "operating_tax_surcharges_name": "OperatingTaxSurcharges",
                    "operating_cost_name": "OperatingCost",
                    "sale_expense_name": "SaleExpense",
                    "administration_expense_name": "AdministrationExpense",
                    "interest_expense_name": "InterestExpense",
                    "commission_expense_name": "CommissionExpense",
                    "rd_expenses_name": "RdExpenses",
                    "asset_impairment_loss_name": "AssetImpairmentLoss",
                    "other_earnings_name": "OtherEarnings",
                    "output_name": self.__class__.__name__},
                "input_data": {"data": "merged_data"},
                "output": ["earning_before_interest_and_taxes"],
            }
        ]
        self.output_vars = ["earning_before_interest_and_taxes"]



class TaxRate(FactorCompute):
    """Compute tax rate"""

    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": IncomeTax,
                 'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"income_tax": "income_tax"},
                 "output": ["income_tax"],
            },
            {
                "class": TotalProfit,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"total_profit": "total_profit"},
                "output": ["total_profit"],
            },
            {
                "func": merge_data,
                "param": {},
                "input_data": {
                    "1": "income_tax",
                    "2": "total_profit",
                },
                "output": ["merged_data"]
            },
            {
                "func": cal_tax_rate,
                "param": {
                    "income_tax_name": "IncomeTax",
                    "total_profit_name": "TotalProfit",
                    "output_name": self.__class__.__name__},
                "input_data": {"data": "merged_data"},
                "output": ["tax_rate"],
            }
        ]
        self.output_vars = ["tax_rate"]



class CashEquivalents(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "balance_stk",
                 "field": ["trade_date",  "code", 'cash_equivalents', 'end_date'],
                 "index": ['trade_date',  'code', 'end_date'],
                 "hist_year": -1,
                 "name_dict": {"cash_equivalents": self.__class__.__name__}},
             "input_data": {},
             "output": ['cash_equivalents']},
        ]
        self.output_vars = ["cash_equivalents"]


class TotalAssets(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "balance_stk",
                 "field": ["trade_date",  "code", 'total_assets', 'end_date'],
                 "index": ['trade_date',  'code','end_date',],
                 "hist_year": -1,
                 "name_dict": {"total_assets": self.__class__.__name__}},
             "input_data": {},
             "output": ['total_assets']},
        ]
        self.output_vars = ["total_assets"]


class TotalLiability(FactorCompute):
    """Total Liab as a continous factor"""
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "balance_stk",
                 "field": ["trade_date",  "code", 'total_liability', 'end_date'],
                 "index": ['trade_date',  'code','end_date',],
                 "hist_year": -1,
                 "name_dict": {"total_liability": self.__class__.__name__}},
             "input_data": {},
             "output": ['total_liability']},
        ]
        self.output_vars = ["total_liability"]


class TotalOwnerEquities(FactorCompute):
    """Total Equity as a continous factor"""
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "balance_stk",
                 "field": ["trade_date",  "code", 'total_owner_equities', 'end_date'],
                 "index": ['trade_date',  'code','end_date',],
                 "hist_year": -1,
                 "name_dict": {"total_owner_equities": self.__class__.__name__}},
             "input_data": {},
             "output": ['total_owner_equities']},
        ]
        self.output_vars = ["total_owner_equities"]


class EquitiesParentCompanyOwners(FactorCompute):
    """Total Equity for parent company as a continous factor"""
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "balance_stk",
                 "field": ["trade_date",  "code", 'equities_parent_company_owners', 'end_date'],
                 "index": ['trade_date',  'code','end_date',],

                 "hist_year": -1,
                 "name_dict": {"equities_parent_company_owners": self.__class__.__name__}},
             "input_data": {},
             "output": ['equities_parent_company_owners']},
        ]
        self.output_vars = ["equities_parent_company_owners"]


class OtherEquityTools(FactorCompute):
    """OtherEquityTools as a continous factor"""
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "balance_stk",
                 "field": ["trade_date",  "code",  ''],
                 "hist_year": -1,
                 "name_dict": {"": self.__class__.__name__}},
             "input_data": {},
             "output": ['']},
        ]
        self.output_vars = ["equities_parent_company_owners"]


class PreferredSharesEquity(FactorCompute):
    """Total PreferredSharesEquity as a continous factor"""
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "balance_stk",
                 "field": ["trade_date",  "code", 'preferred_shares_equity', 'end_date'],
                 "index": ['trade_date',  'code','end_date',],
                 "hist_year": -1,
                 "name_dict": {"preferred_shares_equity": self.__class__.__name__}},
             "input_data": {},
             "output": ['preferred_shares_equity']},
        ]
        self.output_vars = ["preferred_shares_equity"]


class TotalCurrentAssets(FactorCompute):
    """Total CurrentLiability as a continous factor"""
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "balance_stk",
                 "field": ["trade_date",  "code", 'total_current_assets', 'end_date'],
                 "index": ['trade_date', 'code', 'end_date', ],

                 "hist_year": -1,
                 "name_dict": {"total_current_assets": self.__class__.__name__}},
             "input_data": {},
             "output": ['total_current_assets']},
        ]
        self.output_vars = ["total_current_assets"]


class TotalCurrentLiability(FactorCompute):
    """Total CurrentLiabilities as a continous factor"""
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "balance_stk",
                 "field": ["trade_date",  "code", 'total_current_liability', 'end_date'],
                 "index": ['trade_date',  'code','end_date',],
                 "hist_year": -1,
                 "name_dict": {"total_current_liability": self.__class__.__name__}},
             "input_data": {},
             "output": ['total_current_liability']},
        ]
        self.output_vars = ["total_current_liability"]


class ShorttermLoan(FactorCompute):
    """Shortterm Loan as a continous factor"""
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "balance_stk",
                 "field": ["trade_date",  "code", 'shortterm_loan', 'end_date'],
                 "index": ['trade_date', 'code', 'end_date', ],

                 "hist_year": -1,
                 "name_dict": {"shortterm_loan": self.__class__.__name__}},
             "input_data": {},
             "output": ['shortterm_loan']},
        ]
        self.output_vars = ["shortterm_loan"]


class NonCurrentLiabilityInOneYear(FactorCompute):
    """NonCurrentLiabilityInOneYear as a continous factor"""
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "balance_stk",
                 "field": ["trade_date",  "code", 'non_current_liability_in_one_year', 'end_date'],
                 "index": ['trade_date', 'code', 'end_date', ],

                 "hist_year": -1,
                 "name_dict": {"non_current_liability_in_one_year": self.__class__.__name__}},
             "input_data": {},
             "output": ['non_current_liability_in_one_year']},
        ]
        self.output_vars = ["non_current_liability_in_one_year"]


class TotalNonCurrentLiability(FactorCompute):
    """Total NonCurrentLiability as a continous factor"""
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "balance_stk",
                 "field": ["trade_date",  "code", 'total_non_current_liability', 'end_date'],
                 "index": ['trade_date', 'code', 'end_date', ],

                 "hist_year": -1,
                 "name_dict": {"total_non_current_liability": self.__class__.__name__}},
             "input_data": {},
             "output": ['total_non_current_liability']},
        ]
        self.output_vars = ["total_non_current_liability"]




class OperatingCash(FactorCompute):
    """
    OperatingCash = (CurrentAssets + CashEquivalents) - (CurrentLiabilities - ShortTermLoan - NonCurrentLiabilityInOneYear)
    """

    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": TotalCurrentAssets,
                 'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"total_current_assets": "total_current_assets"},
                 "output": ["total_current_assets"],
            },
            {
                "class": CashEquivalents,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"cash_equivalents": "cash_equivalents"},
                "output": ["cash_equivalents"],
            },
            {
                "class": TotalCurrentLiability,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"total_current_liability": "total_current_liability"},
                "output": ["total_current_liability"],
            },
            {
                "class": ShorttermLoan,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"shortterm_loan": "shortterm_loan"},
                "output": ["shortterm_loan"],
            },
            {
                "class": NonCurrentLiabilityInOneYear,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"non_current_liability_in_one_year": "non_current_liability_in_one_year"},
                "output": ["non_current_liability_in_one_year"],
            },
            {
                "func": merge_data,
                "param": {},
                "input_data": {
                    "1": "total_current_assets",
                    "2": "cash_equivalents",
                    "3": "total_current_liability",
                    "4": "shortterm_loan",
                    "5": "non_current_liability_in_one_year"
                },
                "output": ["merged_data"]
            },
            {
                "func": cal_operating_cash,
                "param": {
                    "total_current_assets_name": "TotalCurrentAssets", 
                    "cash_equivalents_name": "CashEquivalents", 
                    "total_current_liability_name": "TotalCurrentLiability",
                    "shortterm_loan_name": "ShorttermLoan", 
                    "non_current_liability_in_one_year_name": "NonCurrentLiabilityInOneYear",
                    "output_name": self.__class__.__name__},
                "input_data": {"data": "merged_data"},
                "output": ["operating_cash"],
            }
        ]
        self.output_vars = ["operating_cash"]


#### CashFlow Factors
class NetOperateCashFlow(FactorCompute):
    """Net Operating CashFlow as a continous factor"""
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "cash_flow_stk",
                 "field": ["trade_date",  "code", 'net_operate_cash_flow', 'end_date'],
                 "index": ['trade_date', 'code', 'end_date', ],

                 "hist_year": -1,
                 "name_dict": {"net_operate_cash_flow": self.__class__.__name__}},
             "input_data": {},
             "output": ['net_operate_cash_flow']},
        ]
        self.output_vars = ["net_operate_cash_flow"]

class NetOperateCashFlowQuarterly(FactorCompute):
    """Net Operating CashFlow as a continous factor"""
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "cash_flow_q",
                 "field": ["trade_date",  "code", 'net_operate_cash_flow', 'end_date'],
                 "index": ['trade_date', 'code', 'end_date', ],

                 "hist_year": -1,
                 "name_dict": {"net_operate_cash_flow": self.__class__.__name__}},
             "input_data": {},
             "output": ['net_operate_cash_flow_quarterly']},
        ]
        self.output_vars = ["net_operate_cash_flow_quarterly"]

        
class NetInvestCashFlow(FactorCompute):
    """Net Investing CashFlow as a continous factor"""
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "cash_flow_stk",
                 "field": ["trade_date",  "code", 'net_invest_cash_flow', 'end_date'],
                 "index": ['trade_date', 'code', 'end_date', ],

                 "hist_year": -1,
                 "name_dict": {"net_invest_cash_flow": self.__class__.__name__}},
             "input_data": {},
             "output": ['net_invest_cash_flow']},
        ]
        self.output_vars = ["net_invest_cash_flow"]


class IntangibleAssetsAmortization(FactorCompute):
    """Amortization as a continous factor"""
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "cash_flow_stk",
                 "field": ["trade_date",  "code", 'intangible_assets_amortization', 'end_date'],
                 "index": ['trade_date', 'code', 'end_date', ],
                 "hist_year": -1,
                 "name_dict": {"intangible_assets_amortization": self.__class__.__name__}},
             "input_data": {},
             "output": ['intangible_assets_amortization']},
        ]
        self.output_vars = ["intangible_assets_amortization"]


class FixedAssetsDepreciation(FactorCompute):
    """Depreciation as a continous factor"""
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "cash_flow_stk",
                 "field": ["trade_date",  "code", 'fixed_assets_depreciation', 'end_date'],
                 "index": ['trade_date', 'code', 'end_date', ],

                 "hist_year": -1,
                 "name_dict": {"fixed_assets_depreciation": self.__class__.__name__}},
             "input_data": {},
             "output": ['fixed_assets_depreciation']},
        ]
        self.output_vars = ["fixed_assets_depreciation"]


class DefferredExpenseAmortization(FactorCompute):
    """DeferredExpenseAmortization as a continous factor"""
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "cash_flow_stk",
                 "field": ["trade_date",  "code", 'defferred_expense_amortization', 'end_date'],
                 "index": ['trade_date', 'code', 'end_date', ],

                 "hist_year": -1,
                 "name_dict": {"defferred_expense_amortization": self.__class__.__name__}},
             "input_data": {},
             "output": ['defferred_expense_amortization']},
        ]
        self.output_vars = ["defferred_expense_amortization"]


class FixIntanOtherAssetAcquiCash(FactorCompute):
    """购建固定资产、无形资产和其他长期资产支付的现金 """

    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "cash_flow_stk",
                 "field": ["trade_date",  "code", 'fix_intan_other_asset_acqui_cash', 'end_date'],
                 "index": ['trade_date', 'code', 'end_date', ],

                 "hist_year": -1,
                 "name_dict": {"fix_intan_other_asset_acqui_cash": self.__class__.__name__}},
             "input_data": {},
             "output": ['fix_intan_other_asset_acqui_cash']},
        ]
        self.output_vars = ["fix_intan_other_asset_acqui_cash"]


class NetInvestCashFlowQuaterly(FactorCompute):
    """Net Investing CashFlow as a continous factor"""
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "cash_flow_q",
                 "field": ["trade_date",  "code", 'net_invest_cash_flow', 'end_date'],
                 "index": ['trade_date', 'code', 'end_date', ],

                 "hist_year": -1,
                 "name_dict": {"net_invest_cash_flow": self.__class__.__name__}},
             "input_data": {},
             "output": ['net_invest_cash_flow_quarterly']},
        ]
        self.output_vars = ["net_invest_cash_flow_quarterly"]




class FCFFTopDown(FactorCompute):
    """
    自由现金流 free cash flow for the firm
    EBIT(1 - TaxRate) + IntangibleAmortization + Depreciation
    + DeferredExpenseAmortization - CapitalExpense
    -(OperatingCash - OperatingCash.shift(1))
    """
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": EarningBeforeInterestAndTaxes,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"earning_before_interest_and_taxes": "earning_before_interest_and_taxes"},
                "output": ["earning_before_interest_and_taxes"],
            },
            {
                "class": TaxRate,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"tax_rate": "tax_rate"},
                "output": ["tax_rate"],
            },
            {
                "class": IntangibleAssetsAmortization,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"intangible_assets_amortization": "intangible_assets_amortization"},
                "output": ["intangible_assets_amortization"],
            },
            {
                "class": FixedAssetsDepreciation,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"fixed_assets_depreciation": "fixed_assets_depreciation"},
                "output": ["fixed_assets_depreciation"],
            },
            {
                "class": DefferredExpenseAmortization,
                "output_name_mapping": {"defferred_expense_amortization": "defferred_expense_amortization"},

            },
            {
                "class": FixIntanOtherAssetAcquiCash,
                "output_name_mapping": {"fix_intan_other_asset_acqui_cash": "fix_intan_other_asset_acqui_cash"},

            },
            {
                "class": OperatingCash,
                "output_name_mapping": {"operating_cash": "operating_cash"},

            },
            {
                "func": merge_data,
                "param": {},
                "input_data": {
                    "1": "earning_before_interest_and_taxes",
                    "2": "tax_rate",
                    "3": "intangible_assets_amortization",
                    "4": "fixed_assets_depreciation",
                    "5": "defferred_expense_amortization",
                    "6": "fix_intan_other_asset_acqui_cash",
                    "7": "operating_cash"
                },
                "output": ["merged_data"]
            },
            {
                "func": cal_fcff_top_down,
                "param": {
                    "earning_before_interest_and_taxes_name": "EarningBeforeInterestAndTaxes",
                    "tax_rate_name": "TaxRate",
                    "intangible_assets_amortization_name": "IntangibleAssetsAmortization",
                    "fixed_assets_depreciation_name": "FixedAssetsDepreciation",
                    "defferred_expense_amortization_name": "DefferredExpenseAmortization",
                    "fix_intan_other_asset_acqui_cash_name": "FixIntanOtherAssetAcquiCash",
                    "operating_cash_name": "OperatingCash",
                    "output_name": self.__class__.__name__},
                "input_data": {"data": "merged_data"},
                "output": ["fcff_top_down"],
            },
            # {
            #     "class": FactorIndex,
            #     "output_name_mapping": {"factor_index": "factor_index"},
            #
            # },
            # {
            #     "func": align_data_to_index,
            #     "param": {"fill_method": "ffill"},
            #     "input_data": {"data": "fcff_top_down", "index": "factor_index"},
            #     "output": ["fcff_top_down"],
            # },
        ]
        self.output_vars = ["fcff_top_down"]



### status ####




class STFlagNameHistory(FactorCompute):
    """st_flag from history name"""

    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                'func': get_hist_data_4_factor_compute,
                 'param': {
                     "read_engine": None,
                     "save_engine": None,
                     "start_date": None,
                     "end_date": None,
                     "table": "name_history_stk",
                     "field": ["trade_date",  "code", 'new_name', "start_date"],
                     "index": ['trade_date',  'code'],

                     "hist_year": -1,
                     "name_dict": {}},
                 "input_data": {},
                 "output": ['hist_name']
            },
            {
                "func": cal_st_based_on_hist_name,
                "param": {"value_name": "new_name", "output_name": self.__class__.__name__},
                "input_data": {"data": "hist_name"},
                "output": ["st_flag_name_history_raw"],
            },
            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},

            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "st_flag_name_history_raw", "index": "factor_index"},
                "output": ["st_flag_name_history"],
            },
        ]
        self.output_vars = ["st_flag_name_history"]


class STFlagNetProfit(FactorCompute):
    """st_flag based on net income data,  """
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                'func': get_hist_data_4_factor_compute,
                 'param': {
                     "read_engine": None,
                     "save_engine": None,
                     "start_date": None,
                     "end_date": None,
                     "table": "income_stk",
                     "field": ["trade_date", "end_date",  "code", 'net_profit', "total_operating_revenue"],
                     "index": ['trade_date',  'code', 'end_date'],

                     "hist_year": -1,
                     "name_dict": {"net_profit": "NetProfit", "total_operating_revenue": "TotalOperatingRevenue"}},
                 "input_data": {},
                 "output": ['net_profit']
            },
            {
                "func": cal_st_flag_based_on_net_profit,
                "param": {"value_name": "NetProfit",  "output_name": self.__class__.__name__},
                "input_data": {"data": "net_profit"},
                "output": ["st_flag_net_profit"],
            },
            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},

            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "st_flag_net_profit", "index": "factor_index"},
                "output": ["st_flag_net_profit_weekly"],
            },
        ]
        self.output_vars = ["st_flag_net_profit_weekly"]


class STFlagNetProfitRevenue(FactorCompute):
    """st_flag based on net income data,  """
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                'func': get_hist_data_4_factor_compute,
                 'param': {
                     "read_engine": None,
                     "save_engine": None,
                     "start_date": None,
                     "end_date": None,
                     "table": "income_stk",
                     "field": ["trade_date", "end_date",  "code", 'net_profit', "total_profit", "total_operating_revenue"],
                     "index": ['trade_date',  'code', 'end_date'],

                     "hist_year": -1,
                     "name_dict": {"net_profit": "NetProfit", "total_operating_revenue": "TotalOperatingRevenue", "total_profit": "TotalProfit"}},
                 "input_data": {},
                 "output": ['net_profit']
            },
            {
                'func': get_hist_data_4_factor_compute,
                 'param': {
                     "read_engine": None,
                     "save_engine": None,
                     "start_date": None,
                     "end_date": None,
                     "table": "indicator_q",
                     "field": ["trade_date", "end_date",  "code", 'adjusted_profit'],
                     "index": ['trade_date',  'code', 'end_date'],

                     "hist_year": -1,
                     "name_dict": {"adjusted_profit": "AdjustedProfit"}},
                 "input_data": {},
                 "output": ['adjusted_profit']
            },
            {
                'func': get_hist_data_4_factor_compute,
                 'param': {
                     "read_engine": None,
                     "save_engine": None,
                     "start_date": None,
                     "end_date": None,
                     "table": "balance_stk",
                     "field": ["trade_date", "end_date",  "code", 'equities_parent_company_owners'],
                     "index": ['trade_date',  'code', 'end_date'],

                     "hist_year": -1,
                     "name_dict": {"equities_parent_company_owners": "NetAsset"}},
                 "input_data": {},
                 "output": ['net_asset']
            },   
            
            {
                "func": cal_st_flag_based_on_profit_revenue,
                "param": {"output_name": self.__class__.__name__},
                "input_data": {"net_profit": "net_profit", "adjusted_profit": "adjusted_profit", "net_asset": "net_asset"},
                "output": ["st_flag_net_profit_revenue"],
            },
            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},

            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "st_flag_net_profit_revenue", "index": "factor_index"},
                "output": ["st_flag_net_profit_revenue_weekly"],
            },
        ]
        self.output_vars = ["st_flag_net_profit_revenue_weekly"]

class STFlag(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": STFlagNameHistory,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"st_flag_name_history": "st_flag_name_history"},
                "output": ["st_flag_name_history"],
            },
            {
                "class": STFlagNetProfit,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"st_flag_net_profit_weekly": "st_flag_net_profit_weekly"},
                "output": ["st_flag_net_profit_weekly"],
            },
            {
                "func": merge_data,
                "param": {},
                "input_data": {"1": "st_flag_name_history", "2": "st_flag_net_profit_weekly"},
                "output": ["merged_data"]
            },
            {
                "func": or_two_variable,
                "param": {"first_var_name": "STFlagNameHistory",
                          "second_var_name": "STFlagNetProfit",
                          "output_name": self.__class__.__name__},
                "input_data": {"data": "merged_data"},
                "output": ["st_flag"],
            }
        ]
        self.output_vars = ['st_flag']

class STFlagV2(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},
            },
            {
                "class": STFlagNameHistory,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"st_flag_name_history": "st_flag_name_history"},
                "output": ["st_flag_name_history"],
            },
            {
                "class": STFlagNetProfitRevenue,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"st_flag_net_profit_revenue_weekly": "st_flag_net_profit_revenue_weekly"},
                "output": ["st_flag_net_profit_revenue_weekly"],
            },
            {
                "func": merge_data,
                "param": {},
                "input_data": {"1": "st_flag_name_history", "2": "st_flag_net_profit_revenue_weekly"},
                "output": ["merged_data"]
            },
            {
                "func": or_two_variable,
                "param": {"first_var_name": "STFlagNameHistory",
                          "second_var_name": "STFlagNetProfitRevenue",
                          "output_name": self.__class__.__name__},
                "input_data": {"data": "merged_data"},
                "output": ["st_flag"],
            },
            {
                "func": standard_and_merge_data,
                "param": {},
                "input_data":
                    {
                        "factor_index": "factor_index",
                        "1": "st_flag_name_history",
                        "2": "st_flag_net_profit_revenue_weekly",
                        "18": "st_flag",
                    },
                "output": ["st_flag_v2"],
            },
        ]
        self.output_vars = ['st_flag_v2']


class PauseFlag(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                'func': get_hist_data_4_factor_compute,
                 'param': {
                     "read_engine": None,
                     "save_engine": None,
                     "start_date": None,
                     "end_date": None,
                     "table": "daily_trading_data",
                     "field": ["trade_date",  "code", 'paused',],
                     "index": ['trade_date',  'code'],

                     "hist_year": -1,
                     "name_dict": {"paused": self.__class__.__name__}},
                 "input_data": {},
                 "output": ['pause_flag']
            },
        ]
        self.output_vars = ['pause_flag']




class EndFlag(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                'func': get_hist_data_4_factor_compute,
                'param': {
                    "read_engine": None,
                    "save_engine": None,
                    "start_date": None,
                    "end_date": None,
                    "table": "name_history_stk",
                    "field": ["trade_date", "code", 'new_name', "start_date"],
                    "index": ['trade_date', 'code'],

                    "hist_year": -1,
                    "name_dict": {}},
                "input_data": {},
                "output": ['hist_name']
            },
            {
                'func': get_hist_data_4_factor_compute,
                 'param': {
                     "read_engine": None,
                     "save_engine": None,
                     "start_date": None,
                     "end_date": None,
                     "table": "stock_universe",
                     "field": ["trade_date",  "code", 'end_date',],
                     "index": ['trade_date',  'code'],
                     "hist_year": 2,
                     "name_dict": {}},
                 "input_data": {},
                 "output": ['stock_info']
            },
            {
                'func': cal_end_flag_based_on_delisting_time,
                'param': {
                    "output_name": "{}Delisting".format(self.__class__.__name__)
                },
                "input_data": {"data": "stock_info"},
                "output": ['end_flag_delisting_time']
            },
            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},

            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "end_flag_delisting_time", "index": "factor_index"},
                "output": ["end_flag_delisting_time_weekly"],
            },
            {
                'func': cal_end_flag_based_on_hist_name,
                'param': {
                    "value_name": "new_name",
                    "output_name": "{}HistName".format(self.__class__.__name__)
                },
                "input_data": {"data": "hist_name"},
                "output": ['end_flag_hist_name']
            },
            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},

            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "end_flag_hist_name", "index": "factor_index"},
                "output": ["end_flag_hist_name_weekly"],
            },
            {
                "func": merge_data,
                "param": {},
                "input_data": {"1": "end_flag_delisting_time_weekly", "2": "end_flag_hist_name_weekly"},
                "output": ["merged_data"]
            },
            {
                "func": or_two_variable,
                "param": {"first_var_name": "{}Delisting".format(self.__class__.__name__),
                          "second_var_name": "{}HistName".format(self.__class__.__name__),
                          "output_name": self.__class__.__name__},
                "input_data": {"data": "merged_data"},
                "output": ["end_flag"],
            }
        ]
        self.output_vars = ['end_flag']

@timer
def cal_listed_flag(mini_list_days, output_name, data):
    # import pdb
    # pdb.set_trace()
    data = data.reset_index()
    data['trade_date_'] = pd.to_datetime(data['trade_date'], format='%Y%m%d')
    data['list_days'] = (data['trade_date_'] - data['start_date']).map(lambda x: x.days)
    data[output_name] = data['list_days'].map(lambda x: x<mini_list_days)
    return data.set_index(['trade_date', 'code'])[[output_name, "list_days"]]


class ListedFlag(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                'func': get_hist_data_4_factor_compute,
                 'param': {
                     "read_engine": None,
                     "save_engine": None,
                     "start_date": None,
                     "end_date": None,
                     "table": "stock_universe",
                     "field": ["trade_date",  "code", 'start_date',],
                     "index": ['trade_date',  'code'],
                     "hist_year": 2,
                     "name_dict": {}},
                 "input_data": {},
                 "output": ['stock_info']
            },
            {
                'func': cal_listed_flag,
                'param': {
                    "mini_list_days": 400,
                    "output_name": self.__class__.__name__
                },
                "input_data": {"data": "stock_info"},
                "output": ['listed_flag']
            },
        ]
        self.output_vars = ['listed_flag']
       

    

        

class NanFlag(FactorCompute):
    """missing key financial value as a categorical factor"""
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},

            },
            {
                "class": FCFFTopDown,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"fcff_top_down": "fcff_top_down"},
                "output": ["fcff_top_down"],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "fcff_top_down", "index": "factor_index"},
                "output": ["fcff_top_down"],
            },
            {
                "class": TotalLiability,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"total_liability": "total_liability"},
                "output": ["total_liability"],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "total_liability", "index": "factor_index"},
                "output": ["total_liability"],
            },
            {
                "class": TotalAssets,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"total_assets": "total_assets"},
                "output": ["total_assets"],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "total_assets", "index": "factor_index"},
                "output": ["total_assets"],
            },
            {
                "class": NetOperateCashFlow,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"net_operate_cash_flow": "net_operate_cash_flow"},
                "output": ["net_operate_cash_flow"],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "net_operate_cash_flow", "index": "factor_index"},
                "output": ["net_operate_cash_flow"],
            },
            {
                "class": TotalOperatingRevenue,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"total_operating_revenue": "total_operating_revenue"},
                "output": ["total_operating_revenue"],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "total_operating_revenue", "index": "factor_index"},
                "output": ["total_operating_revenue"],
            },
            {
                "class": OperatingProfitQuarterly,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"operating_profit_quarterly": "operating_profit_quarterly"},
                "output": ["operating_profit_quarterly"],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "operating_profit_quarterly", "index": "factor_index"},
                "output": ["operating_profit_quarterly"],
            },
            {
                "class": TotalNonCurrentLiability,
                "param": {},
                "input_name_mapping": {},
                "output_name_mapping": {"total_non_current_liability": "total_non_current_liability"},
                "output": ["total_non_current_liability"],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "total_non_current_liability", "index": "factor_index"},
                "output": ["total_non_current_liability"],
            },
            {
                "class": PreferredSharesEquity,
                "param": {},
                "input_name_mapping": {},
                "output_name_mapping": {"preferred_shares_equity": "preferred_shares_equity"},
                "output": ["preferred_shares_equity"],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "zero"},
                "input_data": {"data": "preferred_shares_equity", "index": "factor_index"},
                "output": ["preferred_shares_equity"],
            },
            {
                "class": GicsIndustry,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"gics_industry": "gics_industry"},
                "output": ["gics_industry"],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "gics_industry", "index": "factor_index"},
                "output": ["gics_industry"],
            },
            # {
            #     "func": align_data_to_index,
            #     "param": {"fill_method": "ffill"},
            #     "input_data": {"data": "gics_industry", "index": "factor_index"},
            #     "output": ["gics_industry"],
            # },

            {
                "func": merge_data,
                "param": {},
                "input_data": {"1": "fcff_top_down", "2": "total_liability", "3": "total_assets",
                               "4": "net_operate_cash_flow", "5": "total_operating_revenue",
                               "6": "operating_profit_quarterly", "7": "total_non_current_liability",
                               "8": "preferred_shares_equity", "9": "gics_industry"},
                "output": ["merged_data"]
            },

            {
                "func": cal_nan_flag,
                "param": {"fcff_top_down_name": "FCFFTopDown", "total_liability_name": "TotalLiability", "total_assets_name": "TotalLiability",
                    "net_operate_cash_flow_name": "NetOperateCashFlow", "total_operating_revenue_name": "TotalOperatingRevenue",
                    "operating_profit_quarterly_name": "OperatingProfitQuarterly",
                    "total_non_current_liability_name": "TotalNonCurrentLiability",
                    "preferred_shares_equity_name": "PreferredSharesEquity", "gics_industry_name": "GicsIndustryName",
                     "output_name": self.__class__.__name__},
                "input_data": {
                    "data": "merged_data",
                    },
                "output": ["nan_flag"]
            },



        ]
        self.output_vars = ['nan_flag']

class CashDividendLastYear(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                'func': get_hist_data_4_factor_compute,
                 'param': {
                     "read_engine": None,
                     "save_engine": None,
                     "start_date": None,
                     "end_date": None,
                     "table": "xr_xd_stk",
                     "field": ["trade_date", "end_date",  "code", 'distributed_share_base_board', "bonus_ratio_rmb", "bonus_type"],
                     "index": ['trade_date',  'code'],

                     "hist_year": -1,
                     "name_dict": {}
                 },
                 "input_data": {},
                 "output": ['dividend_data'],
            },
                
            {
                'func': get_hist_data_4_factor_compute,
                 'param': {
                     "read_engine": None,
                     "save_engine": None,
                     "start_date": None,
                     "end_date": None,
                     "table": "valuation_q",
                     "field": ["trade_date", "capitalization",  "code",],
                     "index": ['trade_date',  'code'],
                     "hist_year": -1,
                     "name_dict": {}
                 },
                 "input_data": {},
                 "output": ['capital_data']
            },
            {
                "func": cal_cash_dividend_last_year,
                "param": {
                    "output_name": self.__class__.__name__
                },
                "input_data": {
                    "data": "dividend_data",
                    "capital_data": "capital_data",
                },
                "output": ["cash_dividend_last_year"]
            },
#             {
#                 "class": FactorIndex,
#                 "output_name_mapping": {"factor_index": "factor_index"},
#             },
#             {
#                 "func": align_data_to_index,
#                 "param": {"fill_method": "ffill"},
#                 "input_data": {"data": "cash_dividend_last_year", "index": "factor_index"},
#                 "output": ["cash_dividend_last_year"],
#             },
        ]
        self.output_vars = ['cash_dividend_last_year']



        
        
class DividendYield(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": CashDividendLastYear,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"cash_dividend_last_year": "cash_dividend_last_year"},
                "output": ["cash_dividend_last_year"],
            },
            {
                "func": cal_cash_dividend_history,
                "param": {"history_year_count": 3},
                "input_data": {"cash_dividend_last_year_data": "cash_dividend_last_year"},
                "output": ["cash_dividend_3_years"],
            },
            {
                "class": MarketCap,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"market_cap": "market_cap"},
                "output": ["market_cap"],
            },
            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},

            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "market_cap", "index": "factor_index"},
                "output": ["market_cap"],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "cash_dividend_3_years", "index": "factor_index"},
                "output": ["cash_dividend_3_years"],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "cash_dividend_last_year", "index": "factor_index"},
                "output": ["cash_dividend_last_year"],
            },
            {
                "func": merge_data,
                "param": {},
                "input_data": {"1": "cash_dividend_last_year", "2": "market_cap", '3': 'cash_dividend_3_years'},
                "output": ["merged_data"]
            },

            {
                "func": divide_two_variable,
                "param": {"first_var_name": "CashDividendLastYear",
                          "second_var_name": "MarketCap",
                          "output_name": "{}LastYear".format(self.__class__.__name__)},
                "input_data": {"data": "merged_data"},
                "output": ["dividend_yield_last_year"],
            },
            {
                "func": divide_two_variable,
                "param": {"first_var_name": "CashDividend3Years",
                          "second_var_name": "MarketCap",
                          "output_name": "{}3Years".format(self.__class__.__name__)},
                "input_data": {"data": "merged_data"},
                "output": ["dividend_yield_3_years"],
            },
            {
                "func": merge_data,
                "param": {},
                "input_data": {"1": "dividend_yield_last_year", "2": "dividend_yield_3_years"},
                "output": ["dividend_yield"]
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "zero"},
                "input_data": {"data": "dividend_yield", "index": "factor_index"},
                "output": ["dividend_yield"],
            },
        ]
        self.output_vars = ['dividend_yield']

        

class CashOverMktCap(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": FCFFTopDown,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"fcff_top_down": "fcff_top_down"},
                "output": ["fcff_top_down"],
            },
            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},

            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "fcff_top_down", "index": "factor_index"},
                "output": ["fcff_top_down"],
            },
            {
                "class": MarketCap,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"market_cap": "market_cap"},
                "output": ["market_cap"],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "market_cap", "index": "factor_index"},
                "output": ["market_cap"],
            },
            {
                "func": merge_data,
                "param": {},
                "input_data": {"1": "fcff_top_down", "2": "market_cap"},
                "output": ["merged_data"]
            },

            {
                "func": divide_two_variable,
                "param": {"first_var_name": "FCFFTopDown",
                          "second_var_name": "MarketCap",
                          "output_name": self.__class__.__name__},
                "input_data": {"data": "merged_data"},
                "output": ["cash_over_market_cap"],
            },

        ]
        self.output_vars = ['cash_over_market_cap']


class RevenueOverMktCap(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": TotalOperatingRevenue,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"total_operating_revenue": "total_operating_revenue"},
                "output": ["total_operating_revenue"],
            },
            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},

            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "total_operating_revenue", "index": "factor_index"},
                "output": ["total_operating_revenue"],
            },
            {
                "class": MarketCap,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"market_cap": "market_cap"},
                "output": ["market_cap"],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "market_cap", "index": "factor_index"},
                "output": ["market_cap"],
            },
            {
                "func": merge_data,
                "param": {},
                "input_data": {"1": "total_operating_revenue", "2": "market_cap"},
                "output": ["merged_data"]
            },

            {
                "func": divide_two_variable,
                "param": {"first_var_name": "TotalOperatingRevenue",
                          "second_var_name": "MarketCap",
                          "output_name": self.__class__.__name__},
                "input_data": {"data": "merged_data"},
                "output": ["revenue_over_market_cap"],
            },

        ]
        self.output_vars = ['revenue_over_market_cap']

class RevenueOverMktCapQuarterly(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": TotalOperatingRevenueQuarterly,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"total_operating_revenue_quarterly": "total_operating_revenue_quarterly"},
                "output": ["total_operating_revenue_quarterly"],
            },
            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},

            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "total_operating_revenue_quarterly", "index": "factor_index"},
                "output": ["total_operating_revenue_quarterly"],
            },
            {
                "class": MarketCap,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"market_cap": "market_cap"},
                "output": ["market_cap"],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "market_cap", "index": "factor_index"},
                "output": ["market_cap"],
            },
            {
                "func": merge_data,
                "param": {},
                "input_data": {"1": "total_operating_revenue_quarterly", "2": "market_cap"},
                "output": ["merged_data"]
            },

            {
                "func": divide_two_variable,
                "param": {"first_var_name": "TotalOperatingRevenueQuarterly",
                          "second_var_name": "MarketCap",
                          "output_name": self.__class__.__name__},
                "input_data": {"data": "merged_data"},
                "output": ["revenue_over_market_cap_quarterly"],
            },

        ]
        self.output_vars = ['revenue_over_market_cap_quarterly']
        

# class NetIncomeLRC3Trend(FactorCompute):
#     def __init__(self, param_info, input_name_mapping, output_name_mapping):
#         super().__init__(param_info, input_name_mapping, output_name_mapping)
#         self.operators = [
#             {
#                 "class": TotalCompositeIncomeQuarterly,
#                 'param': {},
#                 "input_name_mapping": {},
#                 "output_name_mapping": {"total_composite_income_quarterly": "total_composite_income_quarterly"},
#                 "output": ["total_composite_income_quarterly"],
#             },
#             {
#                 "func": cal_all_code_quarterly_trend,
#                 "param": {"value_name": "TotalCompositeIncomeQuarterly",
#                           "output_name": self.__class__.__name__[:-5]},
#                 "input_data": {"data": "total_composite_income_quarterly"},
#                 "output": ["net_income_lr_c3_daily"],
#             },
#             {
#                 "func": cal_all_code_quarterly_trend,
#                 "param": {"value_name": "NetIncomeLRC3",
#                           "output_name": self.__class__.__name__,
#                          "hist_quarter_count": 4},
#                 "input_data": {"data": "net_income_lr_c3_daily"},
#                 "output": ["net_income_lr_c3_trend_daily"],
#             },            
#             {
#                 "class": FactorIndex,
#                 "output_name_mapping": {"factor_index": "factor_index"},

#             },
#             {
#                 "func": align_data_to_index,
#                 "param": {"fill_method": "ffill"},
#                 "input_data": {"data": "net_income_lr_c3_trend_daily", "index": "factor_index"},
#                 "output": ["net_income_lr_c3_trend"],
#             },
#         ]
#         self.output_vars = ['net_income_lr_c3_trend']

class NetIncomeLRC3HistTrend(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": TotalCompositeIncomeQuarterly,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"total_composite_income_quarterly": "total_composite_income_quarterly"},
                "output": ["total_composite_income_quarterly"],
            },
            {
                "func": cal_all_code_quarterly_trend,
                "param": {"value_name": "TotalCompositeIncomeQuarterly",
                          "output_name": self.__class__.__name__[:-9]},
                "input_data": {"data": "total_composite_income_quarterly"},
                "output": ["net_income_lr_c3_daily"],
            },
            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},

            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "net_income_lr_c3_daily", "index": "factor_index"},
                "output": ["net_income_lr_c3"],
            },
            
            {
                "func": cal_all_code_hist_corr,
                "param": {"value_name": "NetIncomeLRC3",
                          "output_name": self.__class__.__name__,
                         "hist_week_count": 50},
                "input_data": {"data": "net_income_lr_c3"},
                "output": ["net_income_lr_c3_trend"],
            },            

        ]
        self.output_vars = ['net_income_lr_c3_trend']


class NetIncomeLRC3(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": TotalCompositeIncomeQuarterly,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"total_composite_income_quarterly": "total_composite_income_quarterly"},
                "output": ["total_composite_income_quarterly"],
            },
            {
                "func": cal_all_code_quarterly_trend,
                "param": {"value_name": "TotalCompositeIncomeQuarterly",
                          "output_name": self.__class__.__name__},
                "input_data": {"data": "total_composite_income_quarterly"},
                "output": ["net_income_lr_c3_daily"],
            },
            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},

            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "net_income_lr_c3_daily", "index": "factor_index"},
                "output": ["net_income_lr_c3"],
            },
        ]
        self.output_vars = ['net_income_lr_c3']

# class RevenueLRC3Trend(FactorCompute):
#     def __init__(self, param_info, input_name_mapping, output_name_mapping):
#         super().__init__(param_info, input_name_mapping, output_name_mapping)
#         self.operators = [
#             {
#                 "class": TotalOperatingRevenueQuarterly,
#                 'param': {},
#                 "input_name_mapping": {},
#                 "output_name_mapping": {"total_operating_revenue_quarterly": "total_operating_revenue_quarterly"},
#                 "output": ["total_operating_revenue_quarterly"],
#             },
#             {
#                 "func": cal_all_code_quarterly_trend,
#                 "param": {"value_name": "TotalOperatingRevenueQuarterly",
#                           "output_name": self.__class__.__name__[:-5]},
#                 "input_data": {"data": "total_operating_revenue_quarterly"},
#                 "output": ["revenue_lr_c3_daily"],
#             },
#             {
#                 "func": cal_all_code_quarterly_trend,
#                 "param": {"value_name": "RevenueLRC3",
#                           "output_name": self.__class__.__name__,
#                          "hist_quarter_count": 4},
#                 "input_data": {"data": "revenue_lr_c3_daily"},
#                 "output": ["revenue_lr_c3_trend_daily"],
#             },
#             {
#                 "class": FactorIndex,
#                 "output_name_mapping": {"factor_index": "factor_index"},

#             },
#             {
#                 "func": align_data_to_index,
#                 "param": {"fill_method": "ffill"},
#                 "input_data": {"data": "revenue_lr_c3_trend_daily", "index": "factor_index"},
#                 "output": ["revenue_lr_c3_trend"],
#             },
#         ]
#         self.output_vars = ['revenue_lr_c3_trend']
        
class RevenueLRC3HistTrend(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": TotalOperatingRevenueQuarterly,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"total_operating_revenue_quarterly": "total_operating_revenue_quarterly"},
                "output": ["total_operating_revenue_quarterly"],
            },
            {
                "func": cal_all_code_quarterly_trend,
                "param": {"value_name": "TotalOperatingRevenueQuarterly",
                          "output_name": self.__class__.__name__[:-9]},
                "input_data": {"data": "total_operating_revenue_quarterly"},
                "output": ["revenue_lr_c3_daily"],
            },
            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},

            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "revenue_lr_c3_daily", "index": "factor_index"},
                "output": ["revenue_lr_c3"],
            },
            
            {
                "func": cal_all_code_hist_corr,
                "param": {"value_name": "RevenueLRC3",
                          "output_name": self.__class__.__name__,
                         "hist_week_count": 50},
                "input_data": {"data": "revenue_lr_c3"},
                "output": ["revenue_lr_c3_trend"],
            },            

        ]
        self.output_vars = ['revenue_lr_c3_trend']
        
class RevenueLRC3(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": TotalOperatingRevenueQuarterly,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"total_operating_revenue_quarterly": "total_operating_revenue_quarterly"},
                "output": ["total_operating_revenue_quarterly"],
            },
            {
                "func": cal_all_code_quarterly_trend,
                "param": {"value_name": "TotalOperatingRevenueQuarterly",
                          "output_name": self.__class__.__name__},
                "input_data": {"data": "total_operating_revenue_quarterly"},
                "output": ["revenue_lr_c3_daily"],
            },
            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},

            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "revenue_lr_c3_daily", "index": "factor_index"},
                "output": ["revenue_lr_c3"],
            },
        ]
        self.output_vars = ['revenue_lr_c3']

# class NetIncomeYoyTrend(FactorCompute):
#     """
#         yoy increase in Net Income
#     """
#     def __init__(self, param_info, input_name_mapping, output_name_mapping):
#         super().__init__(param_info, input_name_mapping, output_name_mapping)
#         self.operators = [
#             {
#                 "class": TotalCompositeIncomeQuarterly,
#                 'param': {},
#                 "input_name_mapping": {},
#                 "output_name_mapping": {"total_composite_income_quarterly": "total_composite_income_quarterly"},
#                 "output": ["total_composite_income_quarterly"],
#             },
#             {
#                 "func": cal_all_code_quarter_2_yoy,
#                 "param": {"value_name": "TotalCompositeIncomeQuarterly",
#                           "output_name": self.__class__.__name__[:-5]},
#                 "input_data": {"data": "total_composite_income_quarterly"},
#                 "output": ["net_income_yoy_daily"],
#             },
#             {
#                 "func": cal_all_code_quarterly_trend,
#                 "param": {"value_name": "NetIncomeYoy",
#                           "output_name": self.__class__.__name__,
#                          "hist_quarter_count": 4},
#                 "input_data": {"data": "net_income_yoy_daily"},
#                 "output": ["net_income_yoy_trend_daily"],
#             },
#             {
#                 "class": FactorIndex,
#                 "output_name_mapping": {"factor_index": "factor_index"},

#             },
#             {
#                 "func": align_data_to_index,
#                 "param": {"fill_method": "ffill"},
#                 "input_data": {"data": "net_income_yoy_trend_daily", "index": "factor_index"},
#                 "output": ["net_income_yoy_trend"],
#             },
#         ]
#         self.output_vars = ['net_income_yoy_trend']


class NetIncomeYoyHistTrend(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": TotalCompositeIncomeQuarterly,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"total_composite_income_quarterly": "total_composite_income_quarterly"},
                "output": ["total_composite_income_quarterly"],
            },
            {
                "func": cal_all_code_quarter_2_yoy,
                "param": {"value_name": "TotalCompositeIncomeQuarterly",
                          "output_name": self.__class__.__name__[:-9]},
                "input_data": {"data": "total_composite_income_quarterly"},
                "output": ["net_income_yoy_daily"],
            },
            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},

            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "net_income_yoy_daily", "index": "factor_index"},
                "output": ["net_income_yoy"],
            },
            
            {
                "func": cal_all_code_hist_corr,
                "param": {"value_name": "NetIncomeYoy",
                          "output_name": self.__class__.__name__,
                         "hist_week_count": 50},
                "input_data": {"data": "net_income_yoy"},
                "output": ["net_income_yoy_trend"],
            },            

        ]
        self.output_vars = ['net_income_yoy_trend']
        
        
class NetIncomeYoy(FactorCompute):
    """
        yoy increase in Net Income
    """
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": TotalCompositeIncomeQuarterly,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"total_composite_income_quarterly": "total_composite_income_quarterly"},
                "output": ["total_composite_income_quarterly"],
            },
            {
                "func": cal_all_code_quarter_2_yoy,
                "param": {"value_name": "TotalCompositeIncomeQuarterly",
                          "output_name": self.__class__.__name__},
                "input_data": {"data": "total_composite_income_quarterly"},
                "output": ["net_income_yoy_daily"],
            },
            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},

            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "net_income_yoy_daily", "index": "factor_index"},
                "output": ["net_income_yoy"],
            },
        ]
        self.output_vars = ['net_income_yoy']

# class RevenueYoyTrend(FactorCompute):
#     """
#         yoy increase in Net Income
#     """
#     def __init__(self, param_info, input_name_mapping, output_name_mapping):
#         super().__init__(param_info, input_name_mapping, output_name_mapping)
#         self.operators = [
#             {
#                 "class": TotalOperatingRevenueQuarterly,
#                 'param': {},
#                 "input_name_mapping": {},
#                 "output_name_mapping": {"total_operating_revenue_quarterly": "total_operating_revenue_quarterly"},
#                 "output": ["total_operating_revenue_quarterly"],
#             },
#             {
#                 "func": cal_all_code_quarter_2_yoy,
#                 "param": {"value_name": "TotalOperatingRevenueQuarterly",
#                           "output_name": self.__class__.__name__[:-5]},
#                 "input_data": {"data": "total_operating_revenue_quarterly"},
#                 "output": ["revenue_yoy_daily"],
#             },
#             {
#                 "func": cal_all_code_quarterly_trend,
#                 "param": {"value_name": "NetIncomeYoy",
#                           "output_name": self.__class__.__name__,
#                          "hist_quarter_count": 4},
#                 "input_data": {"data": "revenue_yoy_daily"},
#                 "output": ["revenue_yoy_trend_daily"],
#             },
#             {
#                 "class": FactorIndex,
#                 "output_name_mapping": {"factor_index": "factor_index"},

#             },
#             {
#                 "func": align_data_to_index,
#                 "param": {"fill_method": "ffill"},
#                 "input_data": {"data": "revenue_yoy_trend_daily", "index": "factor_index"},
#                 "output": ["revenue_yoy_trend"],
#             },
#         ]
#         self.output_vars = ['revenue_yoy_trend']
   
class RevenueYoyHistTrend(FactorCompute):
    """
        yoy increase in Net Income trend
    """
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": TotalOperatingRevenueQuarterly,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"total_operating_revenue_quarterly": "total_operating_revenue_quarterly"},
                "output": ["total_operating_revenue_quarterly"],
            },
            {
                "func": cal_all_code_quarter_2_yoy,
                "param": {"value_name": "TotalOperatingRevenueQuarterly",
                          "output_name": self.__class__.__name__[:-9]},
                "input_data": {"data": "total_operating_revenue_quarterly"},
                "output": ["revenue_yoy_daily"],
            },
            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "revenue_yoy_daily", "index": "factor_index"},
                "output": ["revenue_yoy"],
            },
            {
                "func": cal_all_code_hist_corr,
                "param": {"value_name": "RevenueYoy",
                          "output_name": self.__class__.__name__,
                         "hist_week_count": 50},
                "input_data": {"data": "revenue_yoy"},
                "output": ["revenue_yoy_trend"],
            }, 
        ]
        self.output_vars = ['revenue_yoy_trend']

class RevenueYoy(FactorCompute):
    """
        yoy increase in Net Income
    """
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": TotalOperatingRevenueQuarterly,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"total_operating_revenue_quarterly": "total_operating_revenue_quarterly"},
                "output": ["total_operating_revenue_quarterly"],
            },
            {
                "func": cal_all_code_quarter_2_yoy,
                "param": {"value_name": "TotalOperatingRevenueQuarterly",
                          "output_name": self.__class__.__name__},
                "input_data": {"data": "total_operating_revenue_quarterly"},
                "output": ["revenue_yoy_daily"],
            },
            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},

            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "revenue_yoy_daily", "index": "factor_index"},
                "output": ["revenue_yoy"],
            },
        ]
        self.output_vars = ['revenue_yoy']

class ROEYoy(FactorCompute):
    """
        yoy increase as roe
    """
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "indicator_q",
                 "field": ['trade_date', 'end_date', 'code', 'roe'],
                 "index": ['trade_date', 'code', 'end_date'],
                 "hist_year": 2,
                 "name_dict": {"roe": "ROE"}},
             "input_data": {},
             "output": ['roe']
            },
            {
                "func": cal_all_code_quarter_2_diff,
                "param": {"value_name": "ROE",
                          "output_name": self.__class__.__name__},
                "input_data": {"data": "roe"},
                "output": ["roe_yoy_daily"],
            },
            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},

            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "roe_yoy_daily", "index": "factor_index"},
                "output": ["roe_yoy"],
            },
#             {
#                 "func": align_data_2_index_merge,
#                 "param": {"merge_rate": param_info["merge_rate"], "name": "roe_yoy"},
#                 "input_data": {"data": "roe_yoy_daily", "index": "factor_index"},
#                 "output": ["roe_yoy"],
#             },
        ]
        self.output_vars = ['roe_yoy']

        
class ROELRC3(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "indicator_q",
                 "field": ['trade_date', 'end_date', 'code', 'roe'],
                 "index": ['trade_date',  'code', 'end_date'],
                 "hist_year": 2,
                 "name_dict": {"roe": "ROE"}},
             "input_data": {},
             "output": ['roe']
            },
            {
                "func": cal_all_code_quarterly_trend_4_roe,
                "param": {"value_name": "ROE",
                          "output_name": self.__class__.__name__},
                "input_data": {"data": "roe"},
                "output": ["roe_lr_c3_daily"],
            },
            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},

            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "roe_lr_c3_daily", "index": "factor_index"},
                "output": ["roe_lr_c3"],
            },
#             {
#                 "func": align_data_2_index_merge,
#                 "param": {"merge_rate": param_info["merge_rate"], "name": "roe_lr_c3"},
#                 "input_data": {"data": "roe_lr_c3_daily", "index": "factor_index"},
#                 "output": ["roe_lr_c3"],
#             },
        ]
        self.output_vars = ['roe_lr_c3']

        
class OperatingProfitFromPerformanceLetters(FactorCompute):
    """

    """
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "performance_letters_stk",
                 "field": ["trade_date",  "code", 'operating_profit', 'end_date'],
                 "index": ['trade_date',  'code','end_date',],
                 "hist_year": -1,
                 "other_filter_info": {"field": "report_type", "type": "equal", "param": 0},
                 "name_dict": {"operating_profit": self.__class__.__name__}},
             "input_data": {},
             "output": ['operating_profit_from_performance_letters']},
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "income_stk",
                 "field": ["trade_date", "code", 'operating_profit', 'end_date'],
                 "index": ['trade_date', 'code', 'end_date', ],
                 "hist_year": -1,
                 "name_dict": {"operating_profit": "OperatingProfit"}},
             "input_data": {},
             "output": ['operating_profit']},
            {'func': cal_factor_from_performance_letter_quarterly,
             'param': {
                 "factor_name": "OperatingProfit",},
             "input_data": {
                 "factor_data": "operating_profit",
                 "factor_data_from_performance_letter": "operating_profit_from_performance_letters"
             },
             "output": ['operating_profit_from_performance_letters_with_quarterly']},
        ]
        self.output_vars = ["operating_profit_from_performance_letters_with_quarterly"]
        
        
class OperatingProfitYoy(FactorCompute):
    """
        yoy increase as operating profit
    """
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": OperatingProfit,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"operating_profit": "operating_profit"},
                "output": ["operating_profit"],
            },
            {
                "class": OperatingProfitQuarterly,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"operating_profit_quarterly": "operating_profit_quarterly"},
                "output": ["operating_profit_quarterly"],
            },

            {
                "class": OperatingProfitFromPerformanceLetters,
                "param": {},
                "input_name_mapping": {},
                "output_name_mapping": {"operating_profit_from_performance_letters_with_quarterly": "operating_profit_from_performance_letters_with_quarterly"},
                "output": ["operating_profit_from_performance_letters_with_quarterly"],
            },
            {
                "func": cal_all_code_yoy_with_performance_letters,
                "param": {"factor_name": "OperatingProfit",
                          "output_name": self.__class__.__name__},
                "input_data": {"factor_data": "operating_profit",
                               "factor_data_quarterly": "operating_profit_quarterly",
                               "factor_data_performance_letters": "operating_profit_from_performance_letters_with_quarterly"},
                "output": ["operating_profit_yoy_daily", "operating_profit_yoy_quarterly_daily"],
            },
            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},

            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "operating_profit_yoy_daily", "index": "factor_index"},
                "output": ["operating_profit_yoy"],
            },
#             {
#                 "func": align_data_2_index_merge,
#                 "param": {"merge_rate": param_info["merge_rate"], "name": "operating_profit_yoy"},
#                 "input_data": {"data": "operating_profit_yoy_daily", "index": "factor_index"},
#                 "output": ["operating_profit_yoy"],
#             },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "operating_profit_yoy_quarterly_daily", "index": "factor_index"},
                "output": ["operating_profit_yoy_quarterly"],
            },
#             {
#                 "func": align_data_2_index_merge,
#                 "param": {"merge_rate": param_info["merge_rate"], "name": "operating_profit_yoy_quarterly"},
#                 "input_data": {"data": "operating_profit_yoy_quarterly_daily", "index": "factor_index"},
#                 "output": ["operating_profit_yoy_quarterly"],
#             },
        ]
        self.output_vars = ['operating_profit_yoy', 'operating_profit_yoy_quarterly', 'operating_profit_yoy_quarterly_daily']


class OperatingProfitLRC3(FactorCompute):
    """
        quarterly trend as operating profit
    """

    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": OperatingProfitQuarterly,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"operating_profit_quarterly": "operating_profit_quarterly"},
                "output": ["operating_profit_quarterly"],
            },

            {
                "class": OperatingProfitFromPerformanceLetters,
                "param": {},
                "input_name_mapping": {},
                "output_name_mapping": {
                    "operating_profit_from_performance_letters_with_quarterly": "operating_profit_from_performance_letters_with_quarterly"},
                "output": ["operating_profit_from_performance_letters_with_quarterly"],
            },
            {
                "func": cal_all_code_trend_with_performance_letters,
                "param": {"factor_name": "OperatingProfit",
                          "output_name": self.__class__.__name__},
                "input_data": {
                               "factor_data_quarterly": "operating_profit_quarterly",
                               "factor_data_performance_letters": "operating_profit_from_performance_letters_with_quarterly"},
                "output": ["operating_profit_lr_c3_daily"],
            },
            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},

            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "operating_profit_lr_c3_daily", "index": "factor_index"},
                "output": ["operating_profit_lr_c3"],
            },
#             {
#                 "func": align_data_2_index_merge,
#                 "param": {"merge_rate": param_info["merge_rate"], "name": "operating_profit_lr_c3"},
#                 "input_data": {"data": "operating_profit_lr_c3_daily", "index": "factor_index"},
#                 "output": ["operating_profit_lr_c3"],
#             },
        ]
        self.output_vars = ['operating_profit_lr_c3']
        
class OperatingRevenueFromPerformanceLetters(FactorCompute):
    """

    """
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "performance_letters_stk",
                 "field": ["trade_date",  "code", 'total_operating_revenue', 'operating_revenue', 'end_date'],
                 "index": ['trade_date',  'code','end_date',],
                 "hist_year": -1,
                 "other_filter_info": {"field": "report_type", "type": "equal", "param": 0},
                 "name_dict": {}},
             "input_data": {},
             "output": ['operating_revenue_from_performance_letters_unprocess']},
            {'func': process_operationg_revenue_from_performance_letters,
             'param': {},
             "input_data": {"performance_letters_data": "operating_revenue_from_performance_letters_unprocess"},
             "output": ['operating_revenue_from_performance_letters']},            
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "income_stk",
                 "field": ["trade_date", "code", 'total_operating_revenue', 'end_date'],
                 "index": ['trade_date', 'code', 'end_date', ],
                 "hist_year": -1,
                 "name_dict": {"total_operating_revenue": "TotalOperatingRevenue"}},
             "input_data": {},
             "output": ['toal_operating_revenue']},
            {'func': cal_factor_from_performance_letter_quarterly,
             'param': {
                 "factor_name": "TotalOperatingRevenue",},
             "input_data": {
                 "factor_data": "toal_operating_revenue",
                 "factor_data_from_performance_letter": "operating_revenue_from_performance_letters"
             },
             "output": ['operating_revenue_from_performance_letters_with_quarterly']},
        ]
        self.output_vars = ["operating_revenue_from_performance_letters_with_quarterly"]
       


    
class OperatingRevenue(FactorCompute):
    """total opertaing revenue as a continous factor"""
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "income_stk",
                 "field": ["trade_date",  "code", 'operating_revenue', "end_date"],
                 "index": ['trade_date',  'code', 'end_date'],
                 "hist_year": -1,
                 "name_dict": {"operating_revenue": self.__class__.__name__}},
             "input_data": {},
             "output": ['operating_revenue']},
        ]
        self.output_vars = ["operating_revenue"]
        
        
class OperatingRevenueQuarterly(FactorCompute):
    """

    """
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "income_q",
                 "field": ["trade_date",  "code", 'operating_revenue', 'end_date'],
                 "index": ['trade_date',  'code', 'end_date'],
                 "hist_year": -1,
                 "name_dict": {"operating_revenue": self.__class__.__name__}},
             "input_data": {},
             "output": ['operating_revenue_quarterly']},
        ]
        self.output_vars = ["operating_revenue_quarterly"]   

        

        
        
class TotalOperatingRevenueYoy(FactorCompute):
    """
        yoy increase as operating profit
    """
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": TotalOperatingRevenue,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"total_operating_revenue": "total_operating_revenue"},
                "output": ["total_operating_revenue"],
            },
            {
                "class": TotalOperatingRevenueQuarterly,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"total_operating_revenue_quarterly": "total_operating_revenue_quarterly"},
                "output": ["total_operating_revenue_quarterly"],
            },

            {
                "class": OperatingRevenueFromPerformanceLetters,
                "param": {},
                "input_name_mapping": {},
                "output_name_mapping": {"operating_revenue_from_performance_letters_with_quarterly": "operating_revenue_from_performance_letters_with_quarterly"},
                "output": ["operating_revenue_from_performance_letters_with_quarterly"],
            },
            {
                "func": cal_all_code_yoy_with_performance_letters,
                "param": {"factor_name": "TotalOperatingRevenue",
                          "output_name": self.__class__.__name__},
                "input_data": {"factor_data": "total_operating_revenue",
                               "factor_data_quarterly": "total_operating_revenue_quarterly",
                               "factor_data_performance_letters": "operating_revenue_from_performance_letters_with_quarterly"},
                "output": ["total_operating_revenue_yoy_daily", "total_operating_revenue_yoy_quarterly_daily"],
            },
            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},

            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "total_operating_revenue_yoy_daily", "index": "factor_index"},
                "output": ["total_operating_revenue_yoy"],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "total_operating_revenue_yoy_quarterly_daily", "index": "factor_index"},
                "output": ["total_operating_revenue_yoy_quarterly"],
            },
#             {
#                 "func": align_data_2_index_merge,
#                 "param": {"merge_rate": param_info["merge_rate"], "name": "total_operating_revenue_yoy"},
#                 "input_data": {"data": "total_operating_revenue_yoy_daily", "index": "factor_index"},
#                 "output": ["total_operating_revenue_yoy"],
#             },
#             {
#                 "func": align_data_2_index_merge,
#                 "param": {"merge_rate": param_info["merge_rate"], "name": "total_operating_revenue_yoy_quarterly"},
#                 "input_data": {"data": "total_operating_revenue_yoy_quarterly_daily", "index": "factor_index"},
#                 "output": ["total_operating_revenue_yoy_quarterly"],
#             },            
        ]
        self.output_vars = ['total_operating_revenue_yoy', 'total_operating_revenue_yoy_quarterly']

        
class OperatingRevenueLRC3(FactorCompute):
    """
        quarterly trend as operating revenue
    """

    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": TotalOperatingRevenueQuarterly,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"total_operating_revenue_quarterly": "total_operating_revenue_quarterly"},
                "output": ["total_operating_revenue_quarterly"],
            },

            {
                "class": OperatingRevenueFromPerformanceLetters,
                "param": {},
                "input_name_mapping": {},
                "output_name_mapping": {
                    "operating_revenue_from_performance_letters_with_quarterly": "operating_revenue_from_performance_letters_with_quarterly"},
                "output": ["operating_revenue_from_performance_letters_with_quarterly"],
            },
            {
                "func": cal_all_code_trend_with_performance_letters,
                "param": {"factor_name": "TotalOperatingRevenue",
                          "output_name": self.__class__.__name__},
                "input_data": {
                               "factor_data_quarterly": "total_operating_revenue_quarterly",
                               "factor_data_performance_letters": "operating_revenue_from_performance_letters_with_quarterly"},
                "output": ["total_operating_revenue_lr_c3_daily"],
            },
            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},

            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "total_operating_revenue_lr_c3_daily", "index": "factor_index"},
                "output": ["total_operating_revenue_lr_c3"],
            },
#             {
#                 "func": align_data_2_index_merge,
#                 "param": {"merge_rate": param_info["merge_rate"],  "name": "total_operating_revenue_lr_c3"},
#                 "input_data": {"data": "total_operating_revenue_lr_c3_daily", "index": "factor_index"},
#                 "output": ["total_operating_revenue_lr_c3"],
#             }, 
        ]
        self.output_vars = ['total_operating_revenue_lr_c3']        
        
class NOCFYoy(FactorCompute):
    """
        yoy increase in Net Income
    """
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": NetOperateCashFlow,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"net_operate_cash_flow": "net_operate_cash_flow"},
                "output": ["net_operate_cash_flow"],
            },
            {
                "func": cal_all_code_quarter_2_yoy,
                "param": {"value_name": "NetOperateCashFlow",
                          "output_name": self.__class__.__name__},
                "input_data": {"data": "net_operate_cash_flow"},
                "output": ["net_operate_cash_flow_yoy_daily"],
            },
            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},

            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "net_operate_cash_flow_yoy_daily", "index": "factor_index"},
                "output": ["net_operate_cash_flow_yoy"],
            },
#             {
#                 "func": align_data_2_index_merge,
#                 "param": {"merge_rate": 0, "name": "net_operate_cash_flow_yoy"},
#                 "input_data": {"data": "net_operate_cash_flow_yoy_daily", "index": "factor_index"},
#                 "output": ["net_operate_cash_flow_yoy"],
#             },
        ]
        self.output_vars = ['net_operate_cash_flow_yoy']
        
class NOCFLRC3(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": NetOperateCashFlow,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"net_operate_cash_flow": "net_operate_cash_flow"},
                "output": ["net_operate_cash_flow"],
            },
            {
                "func": cal_all_code_quarterly_trend,
                "param": {"value_name": "NetOperateCashFlow",
                          "output_name": self.__class__.__name__},
                "input_data": {"data": "net_operate_cash_flow"},
                "output": ["net_operate_cash_flow_lr_c3_daily"],
            },
            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},

            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "net_operate_cash_flow_lr_c3_daily", "index": "factor_index"},
                "output": ["net_operate_cash_flow_lr_c3"],
            },
#             {
#                 "func": align_data_2_index_merge,
#                 "param": {"merge_rate": 0, "name": "net_operate_cash_flow_lr_c3"},
#                 "input_data": {"data": "net_operate_cash_flow_lr_c3_daily", "index": "factor_index"},
#                 "output": ["net_operate_cash_flow_lr_c3"],
#             },
        ]
        self.output_vars = ['net_operate_cash_flow_lr_c3']

        
###value factor####


class PriceToEarnings(FactorCompute):
    """
        PE Ratio
    """
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "valuation_q",
                 "field": ['trade_date', 'code', 'pe_ratio'],
                 "hist_year": 2,
                 "name_dict": {"pe_ratio": self.__class__.__name__}},
             "input_data": {},
             "output": ['price_to_earnings']},
        ]
        self.output_vars = ["price_to_earnings"]


class PriceToBook(FactorCompute):
    """
        MktCap
    """

    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "valuation_q",
                 "field": ['trade_date', 'code', 'pb_ratio'],
                 "hist_year": 2,
                 "name_dict": {"pb_ratio": self.__class__.__name__}},
             "input_data": {},
             "output": ['price_to_book']},
        ]
        self.output_vars = ["price_to_book"]


class BookToPrice(FactorCompute):
    """

    """

    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": PriceToBook,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"price_to_book": "price_to_book"},
                "output": ["price_to_book"],
            },
            {
                "func": cal_reciprocal,
                "param": {"value_name": "PriceToBook", 'output_name': self.__class__.__name__},
                "input_data": {'data': "price_to_book"},
                "output": ["book_to_price"]
            }
        ]
        self.output_vars = ["book_to_price"]



class BookValue(FactorCompute):
    """
    Book Value of Equity:
    BookValue = MktCap/PriceToBook
    """
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": MarketCap,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"market_cap": "market_cap"},
                "output": ["market_cap"],
            },
            {
                "class": PriceToBook,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"price_to_book": "price_to_book"},
                "output": ["price_to_book"]

            },
            {
                "func": merge_data,
                "param": {},
                "input_data": {"1": "market_cap", "index": "price_to_book"},
                "output": ["merged_data"],
            },
            {
                "func": divide_two_variable,

                "param": {"first_var_name": "MarketCap", "second_var_name": "PriceToBook", "output_name": self.__class__.__name__},
                "input_data": {"data": "merged_data"},
                "output": ["book_value"],
            },
        ]
        self.output_vars = ['book_value']





class BookLeverage(FactorCompute):
    """
    BookLeverage = (BookEquity +PreferredSharesEquity + NonCurrentLiability )/BookEquity
    """
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},
            },
            {
                "class": BookValue,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"book_value": "book_value"},
                "output": ["book_value"],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "book_value", "index": "factor_index"},
                "output": ["book_value"],
            },
            {
                "class": PreferredSharesEquity,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"preferred_shares_equity": "preferred_shares_equity"},
                "output": ["preferred_shares_equity"],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "zero"},
                "input_data": {"data": "preferred_shares_equity", "index": "factor_index"},
                "output": ["preferred_shares_equity"],
            },
            {
                "class": TotalNonCurrentLiability,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"total_non_current_liability": "total_non_current_liability"},
                "output": ["total_non_current_liability"],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "total_non_current_liability", "index": "factor_index"},
                "output": ["total_non_current_liability"],
            },
            {
                "class": DebtOverAssets,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"debt_over_assets": "debt_over_assets"},
                "output": ["debt_over_assets"],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "debt_over_assets", "index": "factor_index"},
                "output": ["debt_over_assets"],
            },
            {
                "class": GicsIndustry,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"gics_industry": "gics_industry"},
                "output": ["gics_industry"],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "gics_industry", "index": "factor_index"},
                "output": ["gics_industry"],
            },
            {
                "func": merge_data,
                "param": {},
                "input_data": {"1": "book_value", "2": "preferred_shares_equity", "3": "total_non_current_liability",
                               "4": "debt_over_assets", "5": "gics_industry"},
                "output": ["merged_data"]
            },
            {
                "func": cal_blev,
                "param": {
                    "book_value_name": "BookValue",
                    "preferred_shares_equity_name": "PreferredSharesEquity",
                    "total_non_current_liability_name": "TotalNonCurrentLiability",
                    "industry_name": "GicsIndustryName",
                    "debt_over_assets_name": "DebtOverAssets",
                    "output_name": self.__class__.__name__
                },
                "input_data": {"data": "merged_data"},
                "output": ["book_leverage"],
            },

        ]
        self.output_vars = ['book_leverage']




class MarketLeverage(FactorCompute):
    """
    MktLeverage = (MktCap +PreferredSharesEquity + NonCurrentLiability )/MktCap
    """
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},
            },
            {
                "class": MarketCap,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"market_cap": "market_cap"},
                "output": ["market_cap"],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "market_cap", "index": "factor_index"},
                "output": ["market_cap"],
            },
            {
                "class": PreferredSharesEquity,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"preferred_shares_equity": "preferred_shares_equity"},
                "output": ["preferred_shares_equity"],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "zero"},
                "input_data": {"data": "preferred_shares_equity", "index": "factor_index"},
                "output": ["preferred_shares_equity"],
            },
            {
                "class": TotalNonCurrentLiability,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"total_non_current_liability": "total_non_current_liability"},
                "output": ["total_non_current_liability"],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "total_non_current_liability", "index": "factor_index"},
                "output": ["total_non_current_liability"],
            },
            {
                "class": DebtOverAssets,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"debt_over_assets": "debt_over_assets"},
                "output": ["debt_over_assets"],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "debt_over_assets", "index": "factor_index"},
                "output": ["debt_over_assets"],
            },
            {
                "class": GicsIndustry,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"gics_industry": "gics_industry"},
                "output": ["gics_industry"],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "gics_industry", "index": "factor_index"},
                "output": ["gics_industry"],
            },
            {
                "func": merge_data,
                "param": {},
                "input_data": {"1": "market_cap", "2": "preferred_shares_equity", "3": "total_non_current_liability",
                               "4": "debt_over_assets", "5": "gics_industry"},
                "output": ["merged_data"]
            },
            {
                "func": cal_mlev,
                "param": {
                    "market_cap_name": "MarketCap",
                    "preferred_shares_equity_name": "PreferredSharesEquity",
                    "total_non_current_liability_name": "TotalNonCurrentLiability",
                    "industry_name": "GicsIndustryName",
                    "debt_over_assets_name" : "DebtOverAssets",
                    "output_name": self.__class__.__name__
                },
                "input_data": {"data": "merged_data"},
                "output": ["market_leverage"],
            },



        ]
        self.output_vars = ['market_leverage']


class DebtOverAssets(FactorCompute):
    """
    Debt Over Total Assets
    """

    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},
            },
            {
                "class": TotalLiability,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"total_liability": "total_liability"},
                "output": ["total_liability"],
            },
            {
                "class": TotalAssets,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"total_assets": "total_assets"},
                "output": ["total_assets"],
            },
            {
                "func": merge_data,
                "param": {},
                "input_data": {"1": "total_liability", "2": "total_assets"},
                "output": ["merged_data"]
            },
            {
                "func": divide_two_variable,
                "param": {
                    "first_var_name": "TotalLiability",
                    "second_var_name": "TotalAssets",
                    "output_name": self.__class__.__name__
                },
                "input_data": {"data": "merged_data"},
                "output": ["debt_over_assets"],
            },

        ]
        self.output_vars = ['debt_over_assets']


class ROA(FactorCompute):
    """
        MktCap
    """

    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "indicator_q",
                 "field": ['trade_date', 'code', 'roa'],
                 "hist_year": 2,
                 "name_dict": {"roa": self.__class__.__name__}},
             "input_data": {},
             "output": ['roa']},
        ]
        self.output_vars = ["roa"]


class ROE(FactorCompute):
    """
        MktCap
    """

    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "indicator_q",
                 "field": ['trade_date', 'code', 'roe'],
                 "hist_year": 2,
                 "name_dict": {"roe": self.__class__.__name__}},
             "input_data": {},
             "output": ['roe']},
        ]
        self.output_vars = ["roe"]


class NOCFOverDebt(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},
            },
            {
                "class": NetOperateCashFlow,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"net_operate_cash_flow": "net_operate_cash_flow"},
                "output": ["net_operate_cash_flow"],
            },

            {
                "class": TotalLiability,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"total_liability": "total_liability"},
                "output": ["total_liability"],
            },
            {
                "func": merge_data,
                "param": {},
                "input_data": {"1": "net_operate_cash_flow", "2": "total_liability"},
                "output": ["merged_data"]
            },
            {
                "func": divide_two_variable,
                "param": {
                    "first_var_name": "NetOperateCashFlow",
                    "second_var_name": "TotalLiability",
                    "output_name": self.__class__.__name__
                },
                "input_data": {"data": "merged_data"},
                "output": ["nocf_over_debt"],
            },

        ]
        self.output_vars = ['nocf_over_debt']


class NOCFOverDebtQuarterly(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},
            },
            {
                "class": NetOperateCashFlowQuarterly,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"net_operate_cash_flow_quarterly": "net_operate_cash_flow_quarterly"},
                "output": ["net_operate_cash_flow_quarterly"],
            },

            {
                "class": TotalLiability,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"total_liability": "total_liability"},
                "output": ["total_liability"],
            },
            {
                "func": merge_data,
                "param": {},
                "input_data": {"1": "net_operate_cash_flow_quarterly", "2": "total_liability"},
                "output": ["merged_data"]
            },
            {
                "func": divide_two_variable,
                "param": {
                    "first_var_name": "NetOperateCashFlowQuarterly",
                    "second_var_name": "TotalLiability",
                    "output_name": self.__class__.__name__
                },
                "input_data": {"data": "merged_data"},
                "output": ["nocf_over_debt_quarterly"],
            },

        ]
        self.output_vars = ['nocf_over_debt_quarterly']
        
        
#### industry data #####
class GicsIndustry(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "gics_industry",
                 "field": ["trade_date",  "code", 'industry_code', 'industry_name'],
                 "index": ['trade_date',  'code'],
                 "hist_year": 2,
                 "name_dict": {"industry_code": "{}Code".format(self.__class__.__name__), "industry_name": "{}Name".format(self.__class__.__name__)}},
             "input_data": {},
             "output": ['gics_industry']},
            {
                "func": std_gics_industry,
                'param': {},
                'input_data': {'data': 'gics_industry'},
                'output': ['gics_industry']
            }
        ]
        self.output_vars = ["gics_industry"]


class SWL1Industry(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "daily_industry_data",
                 "field": ["trade_date",  "code", 'sw_l1_industry_code', 'sw_l1_industry_name'],
                 "index": ['trade_date',  'code'],
                 "hist_year": 2,
                 "name_dict": {"sw_l1_industry_code": "{}Code".format(self.__class__.__name__), "sw_l1_industry_name": "{}Name".format(self.__class__.__name__)}},
             "input_data": {},
             "output": ['sw_l1_industry']},
        ]
        self.output_vars = ["sw_l1_industry"]

class SWL2Industry(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "daily_industry_data",
                 "field": ["trade_date",  "code", 'sw_l2_industry_code', 'sw_l2_industry_name'],
                 "index": ['trade_date',  'code'],
                 "hist_year": 2,
                 "name_dict": {"sw_l2_industry_code": "{}Code".format(self.__class__.__name__), "sw_l2_industry_name": "{}Name".format(self.__class__.__name__)}},
             "input_data": {},
             "output": ['sw_l2_industry']},
        ]
        self.output_vars = ["sw_l2_industry"]

class SWIndustryFactor(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.if_exists=param_info.get("insert_way", "append")
        self.operators = [
            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},
            },
            {
                "class": SWL1Industry,
                "param": {},
                "output_name_mapping": {"sw_l1_industry": "sw_l1_industry"},
            },

            {
                "class": SWL2Industry,
                "param": {},
                "output_name_mapping": {"sw_l2_industry": "sw_l2_industry"},
            },
            {
                "func": standard_and_merge_data,
                "param": {},
                "input_data":
                    {
                        "factor_index": "factor_index",
                        "1": "sw_l1_industry",
                        "2": "sw_l2_industry",
                    },
                 "output": ["sw_industry_factor"]
            },
                
            {
                "func": save_data_to_table,
                "param": {"engine": param_info['common']['save_engine'], "table": "sw_industry_factor",
                          "if_exists": self.if_exists},
                "input_data": {"data": "sw_industry_factor"},
                "output": ["sw_industry_factor"]
            }
        ]
        self.output_vars = ["sw_industry_factor"]
        
#####financial_forecast ######
            
class ResearchReport(FactorCompute):
    """
    研报数据
    """
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)

        self.save_info = param_info['save_info']

        self.operators = [
            {
                'func': get_hist_data_4_factor_compute,
                'param': {
                    "read_engine": None,
                    "save_engine": None,
                    "start_date": None,
                    "end_date": None,
                    "table": "research_report",
                    "field": ['trade_date', 'code'],
                    "hist_year": 2,
                    "name_dict": {}},
                "input_data": {},
                "output": ["research_report_data"]
            },
            {
                "class": FactorIndexOnline,
                "input_name_mapping": {"data": "research_report_data"},
                "output_name_mapping": {"factor_index": "factor_index"},
                
            },
            {
                "func": resample_data_to_index,
                "param": {"fill_method": "backfill"},
                "input_data": {"data": "research_report_data", "index": "factor_index"},
                "output": ["weekly_research_report_data"],
            },
            {
                "func": gen_research_report_weekly_count,
                "param": {},
                "input_data": {'data': "weekly_research_report_data"},
                "output": ['weekly_research_report_count_data']
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "zero"},
                "input_data": {"data": "weekly_research_report_count_data", "index": "factor_index"},
                "output": ["weekly_research_report_count_data"],
            },
            {
                "func": gen_research_report_hist_count,
                "param": {},
                "input_data": {'data': "weekly_research_report_count_data"},
                "output": ['hist_weekly_research_report_count_data']
            },
            {
                "func": save_data_to_table,
                "param": {"engine": self.save_info['engine'], "table": self.save_info['table'],
                          "if_exists": self.save_info.get("if_exists", "append")},
                "input_data": {"data": "hist_weekly_research_report_count_data"},
                "output": ["hist_weekly_research_report_count_data"]
            }
        ]
        self.output_vars = ['hist_weekly_research_report_count_data']

##### RD factor######        
        
class RdExpenses(FactorCompute):
    """total rd expense as a continous factor"""
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": None,
                 "save_engine": None,
                 "start_date": None,
                 "end_date": None,
                 "table": "income_stk",
                 "field": ["trade_date",  "code", 'rd_expenses', "end_date"],
                 "index": ['trade_date',  'code', 'end_date'],
                 "hist_year": -1,
                 "name_dict": {"rd_expenses": self.__class__.__name__}},
             "input_data": {},
             "output": ['rd_expenses']},
        ]
        self.output_vars = ["rd_expenses"]
        
class RdRate(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},
            },
            {
                "class": RdExpenses,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"rd_expenses": "rd_expenses"},
                "output": ["rd_expenses"],
            },

            {
                "class": TotalOperatingRevenue,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"total_operating_revenue": "total_operating_revenue"},
                "output": ["total_operating_revenue"],
            },
            {
                "func": cal_factor_ttm,
                "param": {"factor_name": "RdExpenses"},
                "input_data": {"data": "rd_expenses", },
                "output": ["rd_expenses_ttm"]
            },
            {
                "func": cal_factor_ttm,
                "param": {"factor_name": "TotalOperatingRevenue"},
                "input_data": {"data": "total_operating_revenue", },
                "output": ["total_operating_revenue_ttm"]
            },
            {
                "func": merge_data,
                "param": {},
                "input_data": {"1": "rd_expenses_ttm", "2": "total_operating_revenue_ttm"},
                "output": ["merged_data"]
            },
            {
                "func": divide_two_variable_4_zero,
                "param": {
                    "first_var_name": "RdExpensesTTM",
                    "second_var_name": "TotalOperatingRevenueTTM",
                    "output_name": self.__class__.__name__
                },
                "input_data": {"data": "merged_data"},
                "output": ["rd_rate"],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "rd_rate", "index": "factor_index"},
                "output": ["rd_rate"],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "merged_data", "index": "factor_index"},
                "output": ["merged_data"],
            },
        ]
        self.output_vars = ['rd_rate', 'merged_data']
        
##### event factor######

class FinForecastWeeklyTag(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.window_size = param_info.get("window_size", 16)
        self.operators = [
            
            {
                'func': get_hist_data_4_factor_compute,
                'param': {
                    "read_engine": None,
                    "save_engine": None,
                    "start_date": None,
                    "end_date": None,
                    "table": "fin_forecast_stk",
                    "field": [],
                    "hist_year": -1,
                    "name_dict": {}},
                "input_data": {},
                "output": ["fin_forecast_data"]
            },
            {
                "class": FactorIndex,
                "input_name_mapping": {"data": "fin_forecast_data"},
                "output_name_mapping": {"factor_index": "factor_index"},
                
            },
            {
                "func": resample_data_to_index,
                "param": {"fill_method": "backfill"},
                "input_data": {"data": "fin_forecast_data", "index": "factor_index"},
                "output": ["weekly_fin_forecast_data"],
            },
            {
                "func": gen_weekly_fin_forecast_tag,
                "param": {},
                "input_data": {"data": "weekly_fin_forecast_data", },
                "output": ["weekly_good_fin_forecast_tag", 'weekly_poor_fin_forecast_tag'],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "zero"},
                "input_data": {"data": "weekly_good_fin_forecast_tag", "index": "factor_index"},
                "output": ["weekly_good_fin_forecast_tag"],
            },
            {
                "func": gen_hist_event_tag,
                "param": {"tag_name": "FinanceGoodPredTag", "hist_tag": "HistFinanceGoodPredTag", "window_size": self.window_size},
                "input_data": {"data": "weekly_good_fin_forecast_tag"},
                "output": ["hist_fin_good_forecast_data"],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "zero"},
                "input_data": {"data": "weekly_poor_fin_forecast_tag", "index": "factor_index"},
                "output": ["weekly_poor_fin_forecast_tag"],
            },
            {
                "func": gen_hist_event_tag,
                "param": {"tag_name": "FinancePoorPredTag", "hist_tag": "HistFinancePoorPredTag",
                          "window_size": self.window_size},
                "input_data": {"data": "weekly_poor_fin_forecast_tag"},
                "output": ["hist_fin_poor_forecast_data"],
            },
            {
                "func": merge_data,
                "param": {},
                "input_data": {"1": "hist_fin_good_forecast_data", "2": "hist_fin_poor_forecast_data"},
                "output": ["hist_fin_forecast_data"]
            },
        ]
        self.output_vars = ['hist_fin_forecast_data'] 

def gen_share_pledge_tag(data):
    data['PledgeTag'] = data['unpledged_number'].isnull()
    finance_poor_pred_tag = data.groupby(level=['code', 'trade_date'])['PledgeTag'].sum().map(lambda x: 1 if x>0 else 0).to_frame()
    return finance_poor_pred_tag


class SharePledgeWeeklyTag(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.window_size = param_info.get("window_size", 12)
        self.operators = [
            {
                'func': get_hist_data_4_factor_compute,
                'param': {
                    "read_engine": None,
                    "save_engine": None,
                    "start_date": None,
                    "end_date": None,
                    "table": "shares_pledge_stk",
                    "field": [],
                    "hist_year": -1,
                    "name_dict": {}},
                "input_data": {},
                "output": ["shares_pledge_data"]
             },
            {
                "class": FactorIndex,
                "input_name_mapping": {"data": "shares_pledge_data"},
                "output_name_mapping": {"factor_index": "factor_index"},
                
            },
            {
                "func": resample_data_to_index,
                "param": {"fill_method": "backfill"},
                "input_data": {"data": "shares_pledge_data", "index": "factor_index"},
                "output": ["weekly_shares_pledge_data"],
            },
            {
                "func": gen_share_pledge_tag,
                "param": {},
                "input_data": {"data": "weekly_shares_pledge_data", },
                "output": ["weekly_shares_pledge_data"],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "zero"},
                "input_data": {"data": "weekly_shares_pledge_data", "index": "factor_index"},
                "output": ["weekly_shares_pledge_data"],
            },
            {
                "func": gen_hist_event_tag,
                "param": {"tag_name": "PledgeTag", "hist_tag": "HistPledgeTag",
                          "window_size": self.window_size},
                "input_data": {"data": "weekly_shares_pledge_data"},
                "output": ["hist_shares_pledge_tag_data"],
            },
        ]
        self.output_vars = ['hist_shares_pledge_tag_data']

def gen_bonus_tag(data):
    ##送股数据###

    data = data.reset_index()
    data = data[data['implementation_bonusnote'].notnull()]
    implementation_bonusnote_data = data.groupby(['code', 'trade_date'])['implementation_bonusnote'].sum().reset_index()
    implementation_bonusnote_data['ShareBonusTag'] = implementation_bonusnote_data['implementation_bonusnote'].map(lambda x: '送' in x and "不分配不转增" not in x)
    share_bonus_tag = implementation_bonusnote_data.groupby(['code', 'trade_date'])['ShareBonusTag'].sum().map(lambda x: 1 if x>0 else 0).to_frame()
    return share_bonus_tag


class ShareBonusWeeklyTag(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.window_size = param_info.get("window_size", 120)
        self.operators = [
            {
                'func': get_hist_data_4_factor_compute,
                'param': {
                    "read_engine": None,
                    "save_engine": None,
                    "start_date": None,
                    "end_date": None,
                    "table": "xr_xd_stk",
                    "field": [],
                    "hist_year": -1,
                    "name_dict": {}},
                "input_data": {},
                "output": ["bonus_data"]
             },
            {
                "class": FactorIndex,
                "input_name_mapping": {"data": "bonus_data"},
                "output_name_mapping": {"factor_index": "factor_index"},
                
            },
            {
                "func": resample_data_to_index,
                "param": {"fill_method": "backfill"},
                "input_data": {"data": "bonus_data", "index": "factor_index"},
                "output": ["weekly_bonus_data"],
            },
            {
                "func": gen_bonus_tag,
                "param": {},
                "input_data": {"data": "weekly_bonus_data"},
                "output": ["weekly_bonus_data"],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "zero"},
                "input_data": {"data": "weekly_bonus_data", "index": "factor_index"},
                "output": ["weekly_bonus_data"],
            },
            {
                "func": gen_hist_event_tag,
                "param": {"tag_name": "ShareBonusTag", "hist_tag": "HistShareBonusTag",
                          "window_size": self.window_size},
                "input_data": {"data": "weekly_bonus_data"},
                "output": ["hist_bonus_data"],
            },
        ]
        self.output_vars = ['hist_bonus_data']


def process_limited_share_unlock_data(data):
    data = data.reset_index()
    data['actual_unlimited_date'] = data.actual_unlimited_date.map(lambda x: int(str(x).replace('-', '')))
    data['pub_date'] = data['trade_date']
    data['trade_date'] = data['actual_unlimited_date']
    return data.set_index(['code', 'trade_date'])

def gen_limited_share_unlock_tag(data):
    ##解禁股限售数据###
    data['ShareRewardUnlimitTag'] = data['limited_reason'].map(lambda x: 1 if x == "股权激励" else 0)
#     data['NoShareRewardUnlimitTag'] = data['limited_reason'].map(lambda x: 1 if x != "股权激励" else 0)
    share_reward_unlimit_tag = data.groupby(level=['code', 'trade_date'])['ShareRewardUnlimitTag'].sum().map(lambda x: 1 if x>0 else 0).to_frame()
    return share_reward_unlimit_tag

class LimitedSharesUnlockWeeklyTag(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.short_window_size = param_info.get("show_window_size", 3)
        self.long_window_size = param_info.get("show_window_size", 17)
        self.operators = [
            {
                'func': get_hist_data_4_factor_compute,
                'param': {
                    "read_engine": None,
                    "save_engine": None,
                    "start_date": None,
                    "end_date": None,
                    "table": "limited_shares_unlock_stk",
                    "field": [],
                    "hist_year": -1,
                    "name_dict": {}},
                "input_data": {},
                "output": ["limited_shares_unlock_data"]
             },
            {
                "func": process_limited_share_unlock_data,
                "param": {},
                "input_data": {"data": "limited_shares_unlock_data", },
                "output": ["limited_shares_unlock_data"],
            },
            {
                "class": FactorIndex,
                "input_name_mapping": {"data": "limited_shares_unlock_data"},
                "output_name_mapping": {"factor_index": "factor_index"},
                
            },
            {
                "func": resample_data_to_index,
                "param": {"fill_method": "backfill", 'drop_max': True},
                "input_data": {"data": "limited_shares_unlock_data", "index": "factor_index"},
                "output": ["weekly_limited_shares_unlock_data"],
            },
            {
                "func": gen_limited_share_unlock_tag,
                "param": {},
                "input_data": {"data": "weekly_limited_shares_unlock_data",},
                "output": ["share_reward_unlimit_tag"],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "zero"},
                "input_data": {"data": "share_reward_unlimit_tag", "index": "factor_index"},
                "output": ["share_reward_unlimit_tag"],
            },
            {
                "func": gen_hist_event_tag,
                "param": {"tag_name": "ShareRewardUnlimitTag", "hist_tag": "HistShareRewardUnlimitTag",
                          "window_size": self.short_window_size},
                "input_data": {"data": "share_reward_unlimit_tag"},
                "output": ["hist_short_share_reward_unlimit_tag_data"],
            },

            {
                "func": gen_hist_event_tag,
                "param": {"tag_name": "ShareRewardUnlimitTag", "hist_tag": "HistLongShareRewardUnlimitTag",
                          "window_size": self.long_window_size, "shift_window_size": self.short_window_size+1},
                "input_data": {"data": "share_reward_unlimit_tag"},
                "output": ["hist_long_share_reward_unlimit_tag_data"],
            },
            {
                "func": merge_data,
                "param": {},
                "input_data": {"1": "hist_short_share_reward_unlimit_tag_data", "2": "hist_long_share_reward_unlimit_tag_data"},
                "output": ["hist_share_reward_limited_shares_unlock_data"]
            },
        ]
        self.output_vars = ['hist_share_reward_limited_shares_unlock_data']


def gen_repurchase_tag(data):
    data['RepurchasePlanTag'] = data['proc'].map(lambda x: x in ['预案', '提议'])
    repurchase_plan_tag = data.groupby(level=['code', 'trade_date'])['RepurchasePlanTag'].sum().map(lambda x: 1 if x>0 else 0).to_frame()
    return repurchase_plan_tag

class RepurchaseWeeklyTag(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.window_size = param_info.get("window_size", 24)
        self.operators = [
            {
                'func': get_hist_data_4_factor_compute,
                'param': {
                    "read_engine": None,
                    "save_engine": None,
                    "start_date": None,
                    "end_date": None,
                    "table": "repurchase_data",
                    "field": [],
                    "hist_year": -1,
                    "name_dict": {}},
                "input_data": {},
                "output": ["repurchase_data"]
             },
            {
                "class": FactorIndex,
                "input_name_mapping": {"data": "repurchase_data"},
                "output_name_mapping": {"factor_index": "factor_index"},
                
            },
            {
                "func": resample_data_to_index,
                "param": {"fill_method": "backfill"},
                "input_data": {"data": "repurchase_data", "index": "factor_index"},
                "output": ["weekly_repurchase_data"],
            },

            {
                "func": gen_repurchase_tag,
                "param": {},
                "input_data": {"data": "weekly_repurchase_data",},
                "output": ["weekly_repurchase_data"],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "zero"},
                "input_data": {"data": "weekly_repurchase_data", "index": "factor_index"},
                "output": ["weekly_repurchase_data"],
            },
            {
                "func": gen_hist_event_tag,
                "param": {"tag_name": "RepurchasePlanTag", "hist_tag": "HistRepurchasePlanTag",
                          "window_size": self.window_size},
                "input_data": {"data": "weekly_repurchase_data"},
                "output": ["hist_repurchase_data"],
            },
        ]
        self.output_vars = ['hist_repurchase_data']

def gen_frozen_tag(data):
    data['UnfrozenTag'] = data['frozen_reason'].map(lambda x: 1 if "解除冻结" in x else 0)
    data['FrozenTag'] = data['frozen_reason'].map(lambda x: 1 if "解除冻结" not in x else 0)
    return data[data['UnfrozenTag'] > 0][['UnfrozenTag']], data[data['FrozenTag'] > 0][['FrozenTag']]

class SharesFrozenWeeklyTag(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.window_size = param_info.get("window_size", 20)
        self.operators = [
            {
                'func': get_hist_data_4_factor_compute,
                'param': {
                    "read_engine": None,
                    "save_engine": None,
                    "start_date": None,
                    "end_date": None,
                    "table": "shares_frozen_stk",
                    "field": [],
                    "hist_year": -1,
                    "name_dict": {}},
                "input_data": {},
                "output": ["shares_frozen_data"]
             },
            {
                "class": FactorIndex,
                "input_name_mapping": {"data": "shares_frozen_data"},
                "output_name_mapping": {"factor_index": "factor_index"},
                
            },
            {
                "func": resample_data_to_index,
                "param": {},
                "input_data": {"data": "shares_frozen_data", "index": "factor_index"},
                "output": ["weekly_shares_frozen_data"],
            },
            {
                "func": gen_frozen_tag,
                "param": {},
                "input_data": {"data": "weekly_shares_frozen_data",},
                "output": ["unfrozen_tag", "frozen_tag"],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "zero"},
                "input_data": {"data": "unfrozen_tag", "index": "factor_index"},
                "output": ["weekly_unfrozen_tag"],
            },
            {
                "func": gen_hist_event_tag,
                "param": {"tag_name": "UnfrozenTag", "hist_tag": "HistUnfrozenTag",
                          "window_size": self.window_size},
                "input_data": {"data": "weekly_unfrozen_tag"},
                "output": ["hist_unfrozen_tag_data"],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "zero"},
                "input_data": {"data": "frozen_tag", "index": "factor_index"},
                "output": ["weekly_frozen_tag"],
            },
            {
                "func": gen_hist_event_tag,
                "param": {"tag_name": "FrozenTag", "hist_tag": "HistFrozenTag",
                          "window_size": self.window_size},
                "input_data": {"data": "weekly_frozen_tag"},
                "output": ["hist_frozen_tag_data"],
            },
            {
                "func": merge_data,
                "param": {},
                "input_data": {"1": "hist_unfrozen_tag_data", "2": "hist_frozen_tag_data",
                               },
                "output": ["hist_frozen_unfrozen_tag_data"]
            },
        ]
        self.output_vars = ['hist_frozen_unfrozen_tag_data']

def gen_share_change_number(data):
    data['direction'] = data['type'].map(lambda x: 1 if x == 0 else -1)
    data['change_number'] = data['change_number']*data['direction']
    share_change_number = data.groupby(['code', 'trade_date']).sum()[['change_number']]
    return share_change_number[['change_number']]


class LargeShareholderShareChangeWeeklyTag(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.window_size = param_info.get("window_size", 24)


        self.operators = [
            {
                'func': get_hist_data_4_factor_compute,
                'param': {
                    "read_engine": None,
                    "save_engine": None,
                    "start_date": None,
                    "end_date": None,
                    "table": "large_shareholder_share_change_stk",
                    "field": [],
                    "hist_year": -1,
                    "name_dict": {}},
                "input_data": {},
                "output": ["large_shareholder_share_change_data"]
             },
            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},
            },
            {
                "func": resample_data_to_index,
                "param": {},
                "input_data": {"data": "large_shareholder_share_change_data", "index": "factor_index"},
                "output": ["weekly_large_shareholder_share_change_data"],
            },
            {
                "func": gen_share_change_number,
                "param": {},
                "input_data": {"data": "weekly_large_shareholder_share_change_data",},
                "output": ["weekly_large_shareholder_share_change_data"],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "zero"},
                "input_data": {"data": "weekly_large_shareholder_share_change_data", "index": "factor_index"},
                "output": ["weekly_large_shareholder_share_change_data"],
            },
            {
                "func": gen_hist_event_tag,
                "param": {"tag_name": "change_number", "hist_tag": "HistShareChangePlusTag",
                          "window_size": self.window_size},
                "input_data": {"data": "weekly_large_shareholder_share_change_data"},
                "output": ["hist_share_change_plus_tag_data"],
            },

        ]
        self.output_vars = ['hist_share_change_plus_tag_data']


class EventWeeklyFactor(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)

        self.save_info = param_info['save_info']
        self.operators = [
            # {
            #     "class": ResearchReport,
            #     "param": {},
            #     "output_name_mapping": {"hist_weekly_research_report_count_data": "hist_weekly_research_report_count_data"},
            # },
            {
                "class": FinForecastWeeklyTag,
                "param": {},
                "output_name_mapping": {"hist_fin_forecast_data": "hist_fin_forecast_data"},
            },
            {
                "class": SharePledgeWeeklyTag,
                'param': {},
                "output_name_mapping": {"hist_shares_pledge_tag_data": "hist_shares_pledge_tag_data"},
            },
            {
                "class": ShareBonusWeeklyTag,
                'param': {},
                "output_name_mapping": {"hist_bonus_tag_data": "hist_bonus_tag_data"},
            },
            {
                "class": LimitedSharesUnlockWeeklyTag,
                'param': {},
                "output_name_mapping": {"hist_share_reward_limited_shares_unlock_data": "hist_share_reward_limited_shares_unlock_data"},
            },
            {
                "class": RepurchaseWeeklyTag,
                'param': {},
                "output_name_mapping": {
                    "hist_repurchase_data": "hist_repurchase_data"},
            },
            {
                "class": SharesFrozenWeeklyTag,
                'param': {},
                "output_name_mapping": {
                    "hist_frozen_unfrozen_tag_data": "hist_frozen_unfrozen_tag_data"},
            },
            {
                "class": LargeShareholderShareChangeWeeklyTag,
                "param": {},
                "output_name_mapping": {
                    "hist_share_change_plus_tag_data": "hist_share_change_plus_tag_data"
                }
            },
            {
                "func": merge_data,
                "param": {},
                "input_data": {
                    "1": "hist_fin_forecast_data",
                    "2": "hist_shares_pledge_tag_data",
                    "3": "hist_bonus_tag_data",
                    "4": "hist_share_reward_limited_shares_unlock_data",
                    "5": "hist_repurchase_data",
                    "6": "hist_frozen_unfrozen_tag_data",
                    "7": "hist_share_change_plus_tag_data",
                },
                "output": ["event_weekly_factor"]
            },
            {
                "func": save_data_to_table,
                "param": {"engine": self.save_info['engine'], "table": self.save_info['table'],
                          "if_exists": self.save_info.get("if_exists", "replace")},
                "input_data": {"data": "event_weekly_factor"},
                "output": ["event_weekly_factor"]
            }
        ]
        self.output_vars = ["event_weekly_factor"]

        
        
#####weight data####



# class IndexWeight(FactorCompute):
#     """
#     A class that computes the weight of a CSISmallcap500.
#
#     Attributes:
#     fac_value (pandas DataFrame): DataFrame containing the computed factor values.
#     fac_index (pandas DataFrame): DataFrame containing the index for the computed factor values.
#
#     Methods:
#     compute(): Computes the weight of a benchmark by importing data from a specified table and field. The factor values are then indexed and any missing values are filled with zeros.
#     """
#
#     def __init__(self, param_info, input_name_mapping, output_name_mapping):
#         super().__init__(param_info, input_name_mapping, output_name_mapping)
#         self.index_code = param_info.get("index_code", "csi500")
#         self.weight_name = param_info.get("weight_name", "weight")
#         self.operators = [
#             {
#                 'func': get_hist_data_4_factor_compute,
#                 'param': {
#                     "read_engine": None,
#                     "save_engine": None,
#                     "start_date": None,
#                     "end_date": None,
#                     "table": "{}_avg_weight".format(self.index_code),
#                     # "table": "real_index_weight",
#                     "field": ['trade_date', 'code', self.weight_name],
#                     "hist_year": -1,
#                     "name_dict": {}},
#                 "input_data": {},
#                 "output": ['{}_weight'.format(self.index_code)]
#              },
#
#             {
#                 "class": FactorIndex,
#                 "output_name_mapping": {"factor_index": "factor_index"},
#             },
#             {
#                 "func": align_data_to_index,
#                 "param": {"fill_method": "zero"},
#                 "input_data": {"data": '{}_weight'.format(self.index_code), "index": "factor_index"},
#                 "output": ['{}_weight_weekly'.format(self.index_code)],
#             },
#             {
#                 "class": CirculatingMarketCap,
#                 "output_name_mapping": {"circulating_market_cap": "circulating_market_cap"},
#             },
#             {
#                 "func": align_data_to_index,
#                 "param": {"fill_method": "ffill"},
#                 "input_data": {"data": 'circulating_market_cap', "index": "factor_index"},
#                 "output": ['circulating_market_cap_weekly'],
#             },
#             {
#                 "class": MarketCap,
#                 "output_name_mapping": {"market_cap": "market_cap"},
#             },
#             {
#                 "func": align_data_to_index,
#                 "param": {"fill_method": "ffill"},
#                 "input_data": {"data": 'market_cap', "index": "factor_index"},
#                 "output": ['market_cap_weekly'],
#             },
#             {
#                 "func": merge_data,
#                 "param": {},
#                 "input_data": {"1": '{}_weight_weekly'.format(self.index_code), "2": "circulating_market_cap_weekly",
#                                "3": "market_cap_weekly"},
#                 "output": ['{}_weight_weekly'.format(self.index_code)]
#             },
#             {
#                 "func": gen_mv_based_weight_flag,
#                 "param": {
#                     "circ_mv_name": "CirculatingMarketCap",
#                     "total_mv_name": "MarketCap",
#                     "raw_weight_name": self.weight_name,
#                     "output_name": "{}MvWeight".format(self.index_code.upper()),
#                 },
#                 "input_data": {"data": '{}_weight_weekly'.format(self.index_code)},
#                 "output": ['{}_weight_weekly'.format(self.index_code)],
#             },
#
#         ]
#         self.output_vars = ['{}_weight_weekly'.format(self.index_code)]


class IndexWeight(FactorCompute):
    """
    A class that computes the weight of a CSISmallcap500.

    Attributes:
    fac_value (pandas DataFrame): DataFrame containing the computed factor values.
    fac_index (pandas DataFrame): DataFrame containing the index for the computed factor values.

    Methods:
    compute(): Computes the weight of a benchmark by importing data from a specified table and field. The factor values are then indexed and any missing values are filled with zeros.
    """

    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.index_code = param_info.get("index_code", "csi500")
        self.weight_name = param_info.get("weight_name", "weight")
        self.operators = [
            {
                'func': get_hist_data_4_factor_compute,
                'param': {
                    "read_engine": None,
                    "save_engine": None,
                    "start_date": None,
                    "end_date": None,
                    ### 聚宽 get_index_stocks 获得的 指数成分股， 没有权重， 更新较快###
                    "table": "{}_avg_weight".format(self.index_code),
                    # "table": "real_index_weight",
                    "field": ['trade_date', 'code', self.weight_name],
                    "hist_year": -1,
                    "name_dict": {}},
                "input_data": {},
                "output": ['{}_real_time_weight'.format(self.index_code)]
             },

            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "zero"},
                "input_data": {"data": '{}_real_time_weight'.format(self.index_code), "index": "factor_index"},
                "output": ['{}_real_time_weight_weekly'.format(self.index_code)],
            },
            {
                "class": CirculatingMarketCap,
                "output_name_mapping": {"circulating_market_cap": "circulating_market_cap"},
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": 'circulating_market_cap', "index": "factor_index"},
                "output": ['circulating_market_cap_weekly'],
            },
            {
                "class": MarketCap,
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
                "input_data": {"1": '{}_real_time_weight_weekly'.format(self.index_code), "2": "circulating_market_cap_weekly",
                               "3": "market_cap_weekly"},
                "output": ['{}_real_time_weight_weekly'.format(self.index_code)]
            },
            {
                "func": gen_mv_based_weight_flag,
                "param": {
                    "circ_mv_name": "CirculatingMarketCap",
                    "total_mv_name": "MarketCap",
                    "raw_weight_name": self.weight_name,
                    "output_name": "{}RealTimeMvWeight".format(self.index_code.upper()),
                    "output_raw_weight": False,
                },
                "input_data": {"data": '{}_real_time_weight_weekly'.format(self.index_code)},
                "output": ['{}_real_time_weight_weekly'.format(self.index_code)],
            },
            {
                'func': get_hist_data_4_factor_compute,
                'param': {
                    "read_engine": None,
                    "save_engine": None,
                    "start_date": None,
                    "end_date": None,
                    "table": "{}_weight".format(self.index_code),
                    # "table": "real_index_weight",
                    "field": ['trade_date', 'code', self.weight_name],
                    "hist_year": -1,
                    "name_dict": {self.weight_name: "{}RawWeight".format(self.index_code.upper())}},
                "input_data": {},
                "output": ['{}_monthly_weight'.format(self.index_code)]
            },

            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "zero"},
                "input_data": {"data": '{}_monthly_weight'.format(self.index_code), "index": "factor_index"},
                "output": ['{}_monthly_weight_weekly'.format(self.index_code)],
            },
            {
                "class": CirculatingMarketCap,
                "output_name_mapping": {"circulating_market_cap": "circulating_market_cap"},
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": 'circulating_market_cap', "index": "factor_index"},
                "output": ['circulating_market_cap_weekly'],
            },
            {
                "class": MarketCap,
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
                "input_data": {"1": '{}_monthly_weight_weekly'.format(self.index_code),
                               "2": "circulating_market_cap_weekly",
                               "3": "market_cap_weekly"},
                "output": ['{}_monthly_weight_mv_weekly'.format(self.index_code)]
            },
            {
                "func": gen_mv_based_weight_flag,
                "param": {
                    "circ_mv_name": "CirculatingMarketCap",
                    "total_mv_name": "MarketCap",
                    "raw_weight_name": "{}RawWeight".format(self.index_code.upper()),
                    "output_name": "{}MonthlyMvWeight".format(self.index_code.upper()),
                    "output_raw_weight": True,

                },
                "input_data": {"data": '{}_monthly_weight_mv_weekly'.format(self.index_code)},
                "output": ['{}_monthly_upt_weight_weekly'.format(self.index_code)],
            },
            {
                "func": merge_data,
                "param": {},
                "input_data": {
                               "2": '{}_real_time_weight_weekly'.format(self.index_code),
                               "3": '{}_monthly_upt_weight_weekly'.format(self.index_code)
                               },
                "output": ['{}_weight_weekly'.format(self.index_code)]
            },
        ]
        self.output_vars = ['{}_weight_weekly'.format(self.index_code)]



class IndexWeightFromCodePortfolio(FactorCompute):
    """
    A class that computes the weight of a CSISmallcap500.

    Attributes:
    fac_value (pandas DataFrame): DataFrame containing the computed factor values.
    fac_index (pandas DataFrame): DataFrame containing the index for the computed factor values.

    Methods:
    compute(): Computes the weight of a benchmark by importing data from a specified table and field. The factor values are then indexed and any missing values are filled with zeros.
    """

    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.index_name = param_info.get("index_name", "AllMarket")
        self.code_portfolio_source_info = param_info.get("code_portfolio_source_info")
        self.invalid_infos = param_info.get("invalid_infos", [])
        self.operators = [
            {
                'func': get_hist_data_4_factor_compute,
                'param': {
                    "read_engine": self.code_portfolio_source_info.get('engine', None),
                    "save_engine": self.code_portfolio_source_info.get('engine', None),
                    "start_date": self.code_portfolio_source_info.get('start_date', None),
                    "end_date": self.code_portfolio_source_info.get('end_date', None),
                    "table": self.code_portfolio_source_info.get('table', ""),
                    # "table": "real_index_weight",
                    "field": self.code_portfolio_source_info.get("field", ""),
                    "index": self.code_portfolio_source_info.get('index', ['code', 'trade_date']),
                    "hist_year": self.code_portfolio_source_info.get("hist_year", 0),
                    "name_dict": self.code_portfolio_source_info.get("name_dict", {})},
                "input_data": {},
                "output": ['{}_weight'.format(self.index_name)]
            },
            {
                "func": transfer_data_to_valid_and_not_valid,
                "param": {
                    "invalid_infos": self.invalid_infos

                },
                "input_data": {"data": '{}_weight'.format(self.index_name)},
                "output": ['valid_{}_weight'.format(self.index_name), 'not_valid_{}_weight'.format(self.index_name)]
            },
            {
                "func": add_index_tag,
                "param": {"index_tag_name": "index_tag", },
                "input_data": {"index_code_info": 'valid_{}_weight'.format(self.index_name)},
                "output": ['valid_{}_weight'.format(self.index_name)],
            },
            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "zero"},
                "input_data": {"data": 'valid_{}_weight'.format(self.index_name), "index": "factor_index"},
                "output": ['{}_weight_weekly'.format(self.index_name)],
            },
            {
                "class": CirculatingMarketCap,
                "output_name_mapping": {"circulating_market_cap": "circulating_market_cap"},
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": 'circulating_market_cap', "index": "factor_index"},
                "output": ['circulating_market_cap_weekly'],
            },
            {
                "class": MarketCap,
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
                "input_data": {"1": '{}_weight_weekly'.format(self.index_name), "2": "circulating_market_cap_weekly",
                               "3": "market_cap_weekly"},
                "output": ['{}_weight_weekly'.format(self.index_name)]
            },
            {
                "func": gen_mv_based_weight_flag,
                "param": {
                    "circ_mv_name": "CirculatingMarketCap",
                    "total_mv_name": "MarketCap",
                    "raw_weight_name": "index_tag",
                    "output_name": "{}MvWeight".format(self.index_name),
                    "output_raw_weight": False
                },
                "input_data": {"data": '{}_weight_weekly'.format(self.index_name)},
                "output": ['{}_weight_weekly'.format(self.index_name)],
            },

        ]
        self.output_vars = ['{}_weight_weekly'.format(self.index_name)]


class ValidFlagFactor(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.if_exists=param_info.get("insert_way", "append")

        self.operators = [
            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},
            },
            # {
            #     "class": EndFlag,
            #     "param": {},
            #     "output_name_mapping": {"end_flag": "end_flag"},
            # },
            # {
            #     "class": PauseFlag,
            #     "param": {},
            #     "output_name_mapping": {"pause_flag": "pause_flag"},
            # },
            {
                "class": STFlag,
                "param": {},
                "output_name_mapping": {"st_flag": "st_flag"},
            },
            {
                "class": STFlagV2,
                "param": {},
                "output_name_mapping": {"st_flag_v2": "st_flag_v2"},
            },
            {
                "func": standard_and_merge_data,
                "param": {},
                "input_data":
                    {
                        "factor_index": "factor_index",
                        # "16": "end_flag",
                        # "17": "pause_flag",
                        "18": "st_flag",
                        "19": "st_flag_v2"
                    },
                "output": ["valid_flag_factor"],
            },
            {
                "func": save_data_to_table,
                "param": {"engine": param_info['common']['save_engine'], "table": "valid_flag",
                          "if_exists": self.if_exists},
                "input_data": {"data": "valid_flag_factor"},
                "output": ["valid_flag_factor"]
            },
        ]
        self.output_vars = ["valid_flag_factor"]


# class AllSTFactor(FactorCompute)
#     def __init__(self, param_info, input_name_mapping, output_name_mapping):
#         super().__init__(param_info, input_name_mapping, output_name_mapping)
#         self.operators = [
#             {
#                 "class": FactorIndex,
#                 "output_name_mapping": {"factor_index": "factor_index"},
#             },
#             {
#                 "class": STFlagNameHistory,
#                 "param": {},
#                 "output_name_mapping": {"end_flag": "end_flag"},
#             },
#             {
#                 "class": PauseFlag,
#                 "param": {},
#                 "output_name_mapping": {"pause_flag": "pause_flag"},
#             },
#             {
#                 "class": STFlag,
#                 "param": {},
#                 "output_name_mapping": {"st_flag": "st_flag"},
#             },
#             {
#                 "func": standard_and_merge_data,
#                 "param": {},
#                 "input_data":
#                     {
#                         "factor_index": "factor_index",
#                         "16": "end_flag",
#                         "17": "pause_flag",
#                         "18": "st_flag",
#                     },
#                 "output": ["valid_flag_factor"],
#             },
#
#
#         ]
#         self.output_vars = ["valid_flag_factor"]


class DailyFactor(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.if_exists=param_info.get("insert_way", "append")
        self.operators = [
            {
                "class": DailyIndex,
                "output_name_mapping": {"daily_index": "daily_index"},
            },
            {
                "class": ListedFlag,
                "param": {},
                "output_name_mapping": {"listed_flag": "listed_flag"},
            },
            {
                "class": MarketCap,
                "param": {},
                "output_name_mapping": {"market_cap": "market_cap"},
            },
            {
                "class": STFlag,
                "param": {},
                "output_name_mapping": {"st_flag": "st_flag"},
            },
            {
                "class": STFlagV2,
                "param": {},
                "output_name_mapping": {"st_flag_v2": "st_flag_v2"},
            },
            {
                "class": EndFlag,
                "param": {},
                "output_name_mapping": {"end_flag": "end_flag"},
            },
            {
                "func": standard_and_merge_data_daily,
                "param": {},
                "input_data":
                    {
                        "daily_index": "daily_index",
                        "1": "listed_flag",
                        "2": "market_cap",
                        "3": "st_flag_v2",
                        "4": "st_flag",
                        "5": "end_flag",
                    },
                "output": ["daily_factor"],
            },
            {
                "func": save_data_to_table,
                "param": {"engine": param_info['common']['save_engine'], "table": "daily_factor",
                          "if_exists": self.if_exists},
                "input_data": {"data": "daily_factor"},
                "output": ["daily_factor"]
            },

        ]
        self.output_vars = ["daily_factor"]

        
class OneTermReturn10am(FactorCompute):
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
                    "table": "daily_10am_trade_price",
                    "field": ['code', 'trade_date', 'close'],
                    "hist_year": 0,
                    "name_dict": {'close': "_10am_price_nfq"}},
                "input_data": {},
                "output": ["_10am_price"]
             },
            {
                'func': get_hist_data_4_factor_compute,
                'param': {
                    "read_engine": None,
                    "save_engine": None,
                    "start_date": None,
                    "end_date": None,
                    "table": "daily_trading_data_unadjusted",
                    "field": ['code', 'trade_date', 'close', 'open'],
                    "hist_year": 0,
                    "name_dict": {'close': "close_price_nfq", "open": "open_price_nfq"}},
                "input_data": {},
                "output": ["nfq_price"]
            },
            {
                'func': get_hist_data_4_factor_compute,
                'param': {
                    "read_engine": None,
                    "save_engine": None,
                    "start_date": None,
                    "end_date": None,
                    "table": "daily_trading_data",
                    "field": ['code', 'trade_date', 'factor'],
                    "hist_year": 0,
                    "name_dict": {}},
                "input_data": {},
                "output": ["factor"]
            },
            {
                "func": merge_data,
                "param": {},
                "input_data": {"1": '_10am_price', "2": "nfq_price",
                               "3": "factor"},
                "output": ["daily_price_factor"]
            },
            {
                "func": gen_adj_price,
                "param": {},
                "input_data": {"data": 'daily_price_factor'},
                "output": ["daily_price_factor"]
            },            
            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},

            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "daily_price_factor", "index": "factor_index"},
                "output": ["weekly_price_factor"],
            },
            {
                "func": gen_weekly_one_term_return,
                "param": {},
                "input_data": {"weekly_data": "weekly_price_factor"},
                "output": ["_10am_one_term_return"],
            },
            {
                "func": save_data_to_table,
                "param": {"engine": param_info['common']['save_engine'], "table": "one_term_return_10am",
                          "if_exists": self.if_exists},
                "input_data": {"data": "_10am_one_term_return"},
                "output": ["_10am_one_term_return"]
            },            
        ]
        self.output_vars = ["_10am_one_term_return"]

        
class NewQuarterlyFactor(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.if_exists=param_info.get("insert_way", "append")    
        self.operators = [
            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},
            },
            {
                "class": NOCFOverDebtQuarterly,
                "param": {},
                "output_name_mapping": {"nocf_over_debt_quarterly": "nocf_over_debt_quarterly"},
            },
            {
                "class": RevenueOverMktCapQuarterly,
                "param": {},
                "output_name_mapping": {"revenue_over_market_cap_quarterly": "revenue_over_market_cap_quarterly"},
            },
            {
                "func": standard_and_merge_data,
                "param": {},
                "input_data":
                    {
                        "factor_index": "factor_index",
                        "63": "nocf_over_debt_quarterly",
                        "64": "revenue_over_market_cap_quarterly",
                        
                    },
                "output": ["new_quarterly_factor"],
            },
            {
                "func": save_data_to_table,
                "param": {"engine": param_info['common']['save_engine'], "table": "new_quarterly_indicator",
                          "if_exists": self.if_exists},
                "input_data": {"data": "new_quarterly_factor"},
                "output": ["new_quarterly_factor"]
            },

        ]
        self.output_vars = ["new_quarterly_factor"]
        
class GrowthTrendFactor(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.if_exists=param_info.get("insert_way", "append")
        self.operators = [
            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},
            },        
            {
                "class": NetIncomeLRC3HistTrend,
                "param": {},
                "output_name_mapping": {"net_income_lr_c3_trend": "net_income_lr_c3_trend"},
            },
            {
                "class": NetIncomeYoyHistTrend,
                "param": {},
                "output_name_mapping": {"net_income_yoy_trend": "net_income_yoy_trend"},
            },
            {
                "class": RevenueLRC3HistTrend,
                "param": {},
                "output_name_mapping": {"revenue_lr_c3_trend": "revenue_lr_c3_trend"},
            },
            {
                "class": RevenueYoyHistTrend,
                "param": {},
                "output_name_mapping": {"revenue_yoy_trend": "revenue_yoy_trend"},
            },
            {
                "func": merge_data,
                "param": {},
                "input_data": {
                    "1": "net_income_lr_c3_trend",
                    "2": "net_income_yoy_trend",
                    "3": "revenue_lr_c3_trend",
                    "4": "revenue_yoy_trend",
                },
                "output": ["growth_trend_factor"]
            },
            {
                "func": save_data_to_table,
                "param": {"engine": param_info['common']['save_engine'], "table": "growth_trend",
                          "if_exists": self.if_exists},
                "input_data": {"data": "growth_trend_factor"},
                "output": ["growth_trend_factor"]
            },
        ]
        self.output_vars = ["growth_trend_factor"]
        
class AllFactor(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.if_exists=param_info.get("insert_way", "append")
        self.operators = [
            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},
            },

            {
                "class": IndexWeight,
                "param": {"index_code": "csi500"},
                "output_name_mapping": {"csi500_weight_weekly": "csi500_weight_weekly"},
            },
            {
                "class": IndexWeight,
                "param": {"index_code": "csi300"},
                "output_name_mapping": {"csi300_weight_weekly": "csi300_weight_weekly"},
            },
            {
                "class": IndexWeight,
                "param": {"index_code": "gz2000"},
                "output_name_mapping": {"gz2000_weight_weekly": "gz2000_weight_weekly"},
            },
            {
                "class": IndexWeightFromCodePortfolio,
                "param": {
                    "index_name": "AllMarket",
                    "code_portfolio_source_info": {
                        "engine": None,
                        "table": "daily_trading_data",
                        "field": ['trade_date', "code"],
                        "index": [ "code", 'trade_date'],
                        "name_dict": {}
                    }
                },
                "output_name_mapping": {"AllMarket_weight_weekly": "AllMarket_weight_weekly"},
            },
            {
                "class": GicsIndustry,
                "param": {},
                "output_name_mapping": {"gics_industry": "gics_industry"},
            },
            {
                "class": SWL1Industry,
                "param": {},
                "output_name_mapping": {"sw_l1_industry": "sw_l1_industry"},
            },
            {
                "class": AdjClosePrice,
                "param": {},
                "output_name_mapping": {"adj_close_price": "adj_close_price"},
            },
            {
                "class": MomentumWeeks,
                "param": {"window_size": 5},
                "output_name_mapping": {"momentum_weeks_5": "momentum_weeks_5"},
            },
            {
                "class": MomentumWeeks,
                "param": {"window_size": 29},
                "output_name_mapping": {"momentum_weeks_29": "momentum_weeks_29"},
            },
            {
                "class": MomentumWeeks,
                "param": {"window_size": 1},
                "output_name_mapping": {"momentum_weeks_1": "momentum_weeks_1"},
            },
            {
                "class": LongMinusShort,
                "param": {"long_window_size": 29, "short_window_size": 5},
                "output_name_mapping": {"long_minus_short": "long_minus_short"},
            },
            {
                "class": Volatility,
                "param": {},
                "output_name_mapping": {"volatility_60_days": "volatility_60_days"},
            },
            {
                "class": RSI,
                "param": {},
                "output_name_mapping": {"rsi_10_days": "rsi_10_days"},
            },

            {
                "class": TurnoverRatio,
                "param": {},
                "output_name_mapping": {"turnover_ratio": "turnover_ratio"},
            },

            {
                "class": STOM,
                "param": {},
                "output_name_mapping": {"stom": "stom"},
            },

            {
                "class": STOQ,
                "param": {},
                "output_name_mapping": {"stoq": "stoq"},
            },

            {
                "class": STOA,
                "param": {},
                "output_name_mapping": {"stoa": "stoa"},
            },

            {
                "class": MarketCap,
                "param": {},
                "output_name_mapping": {"market_cap": "market_cap"},
            },
            {
                "class": CirculatingMarketCap,
                "param": {},
                "output_name_mapping": {"circulating_market_cap": "circulating_market_cap"},
            },
            {
                "class": LogMktCap,
                "param": {},
                "output_name_mapping": {"log_mkt_cap": "log_mkt_cap"},
            },

            {
                "class": NonLinearSize,
                "param": {},
                "output_name_mapping": {"nonlinear_size": "nonlinear_size"},
            },

            {
                "class": EndFlag,
                "param": {},
                "output_name_mapping": {"end_flag": "end_flag"},
            },

            {
                "class": NanFlag,
                "param": {},
                "output_name_mapping": {"nan_flag": "nan_flag"},
            },

            {
                "class": PauseFlag,
                "param": {},
                "output_name_mapping": {"pause_flag": "pause_flag"},
            },

            {
                "class": STFlag,
                "param": {},
                "output_name_mapping": {"st_flag": "st_flag"},
            },

            {
                "class": ListedFlag,
                "param": {},
                "output_name_mapping": {"listed_flag": "listed_flag"},
            },

            {
                "class": MarketBeta,
                "param": {},
                "output_name_mapping": {"market_beta_000905XSHG_252": "market_beta_000905XSHG_252"},
            },

            {
                "class": PriceToBook,
                "param": {},
                "output_name_mapping": {"price_to_book": "price_to_book"},
            },
            {
                "class": BookToPrice,
                "param": {},
                "output_name_mapping": {"book_to_price": "book_to_price"},
            },
            {
                "class": PriceToEarnings,
                "param": {},
                "output_name_mapping": {"price_to_earnings": "price_to_earnings"},
            },

            {
                "class": BookValue,
                "param": {},
                "output_name_mapping": {"book_value": "book_value"},
            },
            {
                "class": TotalAssets,
                "param": {},
                "output_name_mapping": {"total_assets": "total_assets"},
            },

            {
                "class": TotalNonCurrentLiability,
                "param": {},
                "output_name_mapping": {"total_non_current_liability": "total_non_current_liability"},
            },

            {
                "class": PreferredSharesEquity,
                "param": {},
                "output_name_mapping": {"preferred_shares_equity": "preferred_shares_equity"},
            },

            {
                "class": TotalOperatingRevenue,
                "param": {},
                "output_name_mapping": {"total_operating_revenue": "total_operating_revenue"},
            },
            {
                "class": TotalCompositeIncomeQuarterly,
                "param": {},
                "output_name_mapping": {"total_composite_income_quarterly": "total_composite_income_quarterly"},
            },
            {
                "class": NetOperateCashFlow,
                "param": {},
                "output_name_mapping": {"net_operate_cash_flow": "net_operate_cash_flow"},
            },
            {
                "class": FCFFTopDown,
                "param": {},
                "output_name_mapping": {"fcff_top_down": "fcff_top_down"},
            },
            {
                "class": CashOverMktCap,
                "param": {},
                "output_name_mapping": {"cash_over_market_cap": "cash_over_market_cap"},
            },

            {
                "class": DividendYield,
                "param": {},
                "output_name_mapping": {"dividend_yield": "dividend_yield"},

            },
            {
                "class": NOCFOverDebt,
                "param": {},
                "output_name_mapping": {"nocf_over_debt": "nocf_over_debt"},
            },
            {
                "class": NOCFOverDebtQuarterly,
                "param": {},
                "output_name_mapping": {"nocf_over_debt_quarterly": "nocf_over_debt_quarterly"},
            },
            {
                "class": MarketLeverage,
                "param": {},
                "output_name_mapping": {"market_leverage": "market_leverage"},
            },
            {
                "class": BookLeverage,
                "param": {},
                "output_name_mapping": {"book_leverage": "book_leverage"},
            },
            {
                "class": DebtOverAssets,
                "param": {},
                "output_name_mapping": {"debt_over_assets": "debt_over_assets"},
            },
            {
                "class": NetIncomeLRC3,
                "param": {},
                "output_name_mapping": {"net_income_lr_c3": "net_income_lr_c3"},
            },
            {
                "class": NetIncomeYoy,
                "param": {},
                "output_name_mapping": {"net_income_yoy": "net_income_yoy"},
            },
            {
                "class": RevenueLRC3,
                "param": {},
                "output_name_mapping": {"revenue_lr_c3": "revenue_lr_c3"},
            },
            {
                "class": RevenueYoy,
                "param": {},
                "output_name_mapping": {"revenue_yoy": "revenue_yoy"},
            },
            {
                "class": RevenueOverMktCap,
                "param": {},
                "output_name_mapping": {"revenue_over_market_cap": "revenue_over_market_cap"},
            },
            {
                "class": RevenueOverMktCapQuarterly,
                "param": {},
                "output_name_mapping": {"revenue_over_market_cap_quarterly": "revenue_over_market_cap_quarterly"},
            },
            
            {
                "class": ROA,
                "param": {},
                "output_name_mapping": {"roa": "roa"},
            },
            {
                "class": ROE,
                "param": {},
                "output_name_mapping": {"roe": "roe"},
            },
            {
                "class": ROELRC3,
                "param": {},
                "output_name_mapping": {"roe_lr_c3": "roe_lr_c3"},
            },
            {
                "class": ROEYoy,
                "param": {},
                "output_name_mapping": {"roe_yoy": "roe_yoy"},
            }, 
            {
                "class": OperatingProfitYoy,
                "param": {},
                "output_name_mapping": {"operating_profit_yoy": "operating_profit_yoy", 'operating_profit_yoy_quarterly': "operating_profit_yoy_quarterly"},
            }, 
            {
                "class": OperatingProfitLRC3,
                "param": {},
                "output_name_mapping": {"operating_profit_lr_c3": "operating_profit_lr_c3"},
            },
            {
                "class": TotalOperatingRevenueYoy,
                "param": {},
                "output_name_mapping": {"total_operating_revenue_yoy": "total_operating_revenue_yoy",
                                        "total_operating_revenue_yoy_quarterly": "total_operating_revenue_yoy_quarterly"},
            },
            {
                "class": OperatingRevenueLRC3,
                "param": {},
                "output_name_mapping": {"total_operating_revenue_lr_c3": "total_operating_revenue_lr_c3"},
            },            
            {
                "class": OperatingProfitQuarterly,
                'param': {},
                "output_name_mapping": {"operating_profit_quarterly": "operating_profit_quarterly"},
            },
            {
                "class": TotalLiability,
                'param': {},
                "output_name_mapping": {"total_liability": "total_liability"},
            },
            {
                "func": standard_and_merge_data,
                "param": {},
                "input_data":
                    {
                        "factor_index": "factor_index",
                        "1": "csi500_weight_weekly",
                        "2": "gics_industry",
                        "3": "adj_close_price",
                        "4": "momentum_weeks_5",
                        "5": "momentum_weeks_29",
                        "6": "long_minus_short",
                        "7": "volatility_60_days",
                        "8": "rsi_10_days",
                        "9": "turnover_ratio",
                        "10": "stom",
                        "11": "stoq",
                        "12": "stoa",
                        "13": "market_cap",
                        "14": "log_mkt_cap",
                        "15": "nonlinear_size",
                        "16": "end_flag",
                        "17": "pause_flag",
                        "18": "st_flag",
                        "20": "listed_flag",
                        "21": "market_beta_000905XSHG_252",
                        "22": "price_to_book",
                        "23": "price_to_earnings",
                        "24": "book_value",
                        "25": "total_assets",
                        "26": "total_non_current_liability",
                        "27": "preferred_shares_equity",
                        "28": "total_operating_revenue",
                        "29": "total_composite_income_quarterly",
                        "30": "net_operate_cash_flow",
                        "31": "fcff_top_down",
                        "32": "cash_over_market_cap",
                        "33": "nocf_over_debt",
                        "34": "market_leverage",
                        "35": "book_leverage",
                        "36": "debt_over_assets",
                        "37": "net_income_lr_c3",
                        "38": "net_income_yoy",
                        "39": "revenue_yoy",
                        "40": "revenue_lr_c3",
                        "41": "roa",
                        "42": "roe",
                        "43": "circulating_market_cap",
                        "44": "book_to_price",
                        "45": "nan_flag",
                        "46": "revenue_over_market_cap",
                        "47": "total_liability",
                        "48": "operating_profit_quarterly",
                        "49": "sw_l1_industry",
                        "50": "momentum_weeks_1",
                        "51": "AllMarket_weight_weekly",
                        "52": "dividend_yield",
                        "53": "csi300_weight_weekly",
                        
                        "54": "roe_lr_c3",
                        "55": "roe_yoy",
                        "56": "operating_profit_yoy",
                        "57": "operating_profit_yoy_quarterly",
                        "58": "operating_profit_lr_c3",
                        "59": "total_operating_revenue_yoy",
                        "60": "total_operating_revenue_yoy_quarterly",
                        "61": "total_operating_revenue_lr_c3",
                        "62": "gz2000_weight_weekly",
#                         "63": "nocf_over_debt_quarterly",
#                         "64": "revenue_over_market_cap_quarterly",
                        
                    },
                "output": ["all_factor"],
            },
            {
                "func": save_data_to_table,
                "param": {"engine": param_info['common']['save_engine'], "table": "all_data_test_all_mkt_indicator",
                          "if_exists": self.if_exists},
                "input_data": {"data": "all_factor"},
                "output": ["all_factor"]
            },

        ]
        self.output_vars = ["all_factor"]


        

