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
import talib
from numba import jit
from sortedcontainers import SortedList
tqdm.pandas()
import statsmodels.api as sm
import numpy as np
from statsmodels.regression.rolling import RollingOLS
import hashlib
import json
from model import SQL_api
import time
import datetime
import talib
from operator import  neg

def default_dump(obj):
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def encryption_dict(input_dict):
    """
    对输入的 字典 进行 md5 加密
    :param input_dict:
    :return:

    """
    input_tuple = sorted(input_dict.items(), key=lambda x: x[0])
    input_dict = dict(input_tuple)
    input_json = json.dumps(input_dict, ensure_ascii=False, default=default_dump)
    return hashlib.md5(input_json.encode()).hexdigest()


def timer(func):
    """A decorator that prints how long a function took to run."""

    # Define the wrapper function to return.
    @wraps(func)
    def wrapper(*args, **kwargs):
        # When wrapper() is called, get the current time.
        t_start = time.time()
        # Call the decorated function and store the result.
        result = func(*args, **kwargs)
        class_name = func.__name__
        # Get the total time it took to run, and print it.
        t_total = time.time() - t_start
        print('{}, {} took {}s'.format(class_name, func.__name__, t_total))
        return result

    return wrapper


def memorize(func):
    """Store the factor values of the decorated factor for fast lookup
    """
    # Store factor values in a dict that maps from factor name  to facto_values
    global cache
    if 'cache' not in globals():
        cache = {}

    # Define the wrapper function to return.
    @wraps(func)
    def wrapper(*args, **kwargs):
        # If these factor haven't been seen before,
        # Call func() and store the factor value.
        # factor = args[0]
        # if factor.name not in cache:
        #     factor = func(*args, **kwargs)
        #     cache[factor.name] = factor.fac_value
        # else:
        #     factor.fac_value = cache[factor.name]
        # print(func.__name__)
        # for key, value in kwargs.items():
        #     print("key {}, value {}".format(key, value))
        new_kwargs = kwargs.copy()
        new_kwargs.update({'name': func.__name__})
        md5_id = encryption_dict(new_kwargs)
        # print("md5_id {}".format(md5_id))
        if md5_id in cache:
            return cache[md5_id]
        else:
            result = func(*args, **kwargs)
            cache.update({md5_id: result})
            return result

    return wrapper


@memorize
def create_sql_api(read_engine, save_engine):
    sql_api_clf = SQL_api.SQL_API(save_engine=save_engine,
                                  read_engine=read_engine)
    return sql_api_clf


@memorize
def get_hist_data_4_factor_compute(read_engine, save_engine, table, field=['trade_date', 'code'], name_dict={},
                                   index=['trade_date', 'code'], hist_year=1, start_date=None, end_date=None,
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
    print(other_filter_info)
    if hist_year < 0:
        trade_date_condition = [{'field': 'trade_date',
                                 'type': 'less_equal',
                                 'param': end_date}]
    else:
        trade_date_condition = [{'field': 'trade_date',
                                 'type': 'between',
                                 'param': [start_date - hist_year * 10000, end_date]}]
    if other_filter_info:
        trade_date_condition.append(other_filter_info)
    query_info = {'method': 'select',
                  'sheet_name': table,
                  'tgt_field': {'way': 'show', 'field': field},
                  'conditions': trade_date_condition}
    sql_api_clf = create_sql_api(read_engine=read_engine, save_engine=save_engine)
    raw_fac = sql_api_clf.read_data_from(query_info)
    if "start_date" in field:
        raw_fac['trade_date'] = raw_fac['start_date'].map(lambda x: int(x.strftime("%Y%m%d")))
    raw_fac = raw_fac.rename(name_dict, axis=1)
    raw_fac = raw_fac.set_index(index)

    return raw_fac


def merge_data(**kwargs):
    """
    将输入数据拼接在一起
    :param kwargs:
    :return:
    """
    try:
        df = pd.concat(list(kwargs.values()), axis=1)
    except Exception as e:
        print(e)
        import pdb
        pdb.set_trace()
        df = pd.DataFrame()
    return df


def Date2intDate(df):
    """
    convert datetime type date to int type

    Parameters
    ----------
    df : pd.DataFrame
        must have a column called trade_date.

    Returns
    -------
    df : pd.DataFrame
        with trade_date column as int.

    """

    df.trade_date = df.trade_date.astype(str).apply(lambda x: x.replace('-', '')).astype(int)

    return df


def intDate2Date(df, column_name='trade_date'):
    """
    convert int type date to datetime type

    Parameters
    ----------
    df : pd.DataFrame
        must have a column called trade_date.

    Returns
    -------
    df : pd.DataFrame
        with trade_date column as datetime type.

    """
    df[column_name] = pd.to_datetime(df[column_name].astype(str))
    return df


def trend_regress(sr):
    """
    Given a pd.Series or np.array, return trend regression coefficient

    Parameters
    ----------
    sr : pd.Series or 1-d np.array

    Returns
    -------
    params: 1-d np.array of regression params, e.g. (beta0,beta1,beta2...)

    """

    def ols(y, x):
        y_isnan = np.isnan(y)
        x_isnan = np.isnan(x)
        y = y[(~y_isnan) & (~x_isnan)]
        x = x[(~y_isnan) & (~x_isnan)]
        if len(x) < 2:
            beta = np.array([[0, np.NaN]])
            alpha = np.NaN
        else:
            beta = np.cov(y, x) / np.var(x) * (len(x) - 1) / len(x)
            alpha = np.mean(y) - np.mean(x) * beta[0, 1]
        return alpha, beta[0, 1]

    y = sr
    x = np.arange(1, len(y) + 1, 1)
    y_mean = np.abs(y).mean()
    _, beta = ols(y, x)
    if beta == np.NaN or y_mean == 0.0:
        slope = np.NaN
    else:
        slope = beta / y_mean
    return slope


def drop_extra_level_index(data):
    """
    drop index and duplicates other than trade_date ,code. duplicated indexed entries only keep the last entry

    Parameters
    ----------
    data : Multiindexed Pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    data: pd.Dataframe.

    """
    # if additional level of index in factor value drop it and keep the last data entry for the same trade_date-code index
    # if data.index.names != ['trade_date', 'code']:
    data = data.sort_index()
    removed_index = []
    for index_name in data.index.names:
        if index_name not in ['trade_date', 'code']:
            removed_index.append(index_name)
    data = data.droplevel(level=removed_index)
    # if reindex to stock universe index, only keep the last entry for the same trade_date code index
    data = data.reset_index().sort_values(['trade_date', 'code']).drop_duplicates(subset=['trade_date', 'code'],
                                                                                  keep='last').set_index(
        ['trade_date', 'code'])
    return data


def resample(fac_index, freq=None, start_date=None, end_date=None):
    """
    Resample the factor index from daily to specified frequency, currently only support weekly

    Returns
    -------
    None.

    """

    if freq == "daily":
        return fac_index
    opt_2_trade = {}
    if freq == "default":
        index_df = pd.DataFrame(index=fac_index)

        index_df['value'] = 0

        index_df = index_df.unstack()
        # convert index format d
        index_df = intDate2Date(index_df.reset_index()).set_index('trade_date')

        # resample time freq

        wednesday_time_index = pd.bdate_range(str(start_date), str(end_date), freq="W-Wed")
        tuesday_time_index = pd.bdate_range(str(start_date), str(end_date), freq="W-Tue")
        trade_dates = sorted(list(set(index_df.index.values)))
        resampled_time_index = []
        for date_, next_date_ in zip(trade_dates[:-1], trade_dates[1:]):
            if date_ in tuesday_time_index:
                resampled_time_index.append(date_)
                opt_2_trade.update({date_: next_date_})
            elif next_date_ in wednesday_time_index:
                resampled_time_index.append(date_)
                opt_2_trade.update({date_: next_date_})
            else:
                pass

        # resample the index

        index_df = index_df.reindex(resampled_time_index).stack().reset_index().rename(
            columns={'level_0': 'trade_date'})
        # get back to int date format
        index_df = Date2intDate(index_df).set_index(['trade_date', 'code'])

        fac_index = index_df.index

    else:
        index_df = pd.DataFrame(index=fac_index)

        index_df['value'] = 0

        index_df = index_df.unstack()
        # convert index format d
        index_df = intDate2Date(index_df.reset_index()).set_index('trade_date')

        # resample time freq
        if freq.startswith("daybefore-"):
            freq = freq.replace("daybefore-", "")
            _resampled_time_index = pd.bdate_range(str(start_date), str(end_date), freq=freq)
            trade_dates = sorted(list(set(index_df.index.values)))
            resampled_time_index = []
            for date_, next_date_ in zip(trade_dates[:-1], trade_dates[1:]):
                if next_date_ in _resampled_time_index:
                    resampled_time_index.append(date_)
                    opt_2_trade.update({date_: next_date_})
        else:

            resampled_time_index = pd.bdate_range(str(start_date), str(end_date), freq=freq)
            trade_dates = sorted(list(set(index_df.index.values)))
            for date_, next_date_ in zip(trade_dates[:-1], trade_dates[1:]):
                if date_ in resampled_time_index:
                    opt_2_trade.update({date_: next_date_})
        # resample the index

        index_df = index_df.reindex(resampled_time_index).stack().reset_index().rename(
            columns={'level_0': 'trade_date'})
        # get back to int date format
        index_df = Date2intDate(index_df).set_index(['trade_date', 'code'])

        fac_index = index_df.index
    return fac_index


def get_fac_idx(start_date, end_date, freq, read_engine, save_engine, data=None):
    """

    :param start_date:
    :param end_date:
    :param freq:
    :param read_engine:
    :param save_engine:
    :param hist_year:
    :param data: 如果 data is not None, start_date, end_date 分别为data 里trade_date 的最大值和最小值
    :return:
    """
    if data is not None:
        data = data.reset_index()
        start_date = data['trade_date'].min()
        end_date = data['trade_date'].max()

    raw_fac_index = get_hist_data_4_factor_compute(read_engine=read_engine,
                                                   save_engine=save_engine,
                                                   start_date=start_date,
                                                   end_date=end_date,
                                                   table="stock_universe", hist_year=0)
    start_date = raw_fac_index.reset_index()['trade_date'].min()
    end_date = raw_fac_index.reset_index()['trade_date'].max()
    fac_index = resample(fac_index=raw_fac_index.index, start_date=start_date, end_date=end_date, freq=freq)
    return fac_index


def align_data_to_index(data, index, fill_method="ffill"):
    """
    reindex financial data to new factor index, and ffill the missing data
    Parameters
    ----------
    data : pd.DataFrame or pd.Series
        Factor data indexed including trade_date and code

    index: pd.Index
        New index to reindex on


    Returns
    -------
    reindex_data : pd.DataFrame or pd.Series
        reindexed_factor_data

    """
    # import pdb
    # pdb.set_trace()

    data = drop_extra_level_index(data)
    full_fac_index = index.union(data.index)
    data = data.reindex(full_fac_index).sort_index()
    if fill_method == "ffill":
        data = data.groupby(level='code').apply(lambda x: x.fillna(method='ffill'))
    elif fill_method == "zero":
        data = data.groupby(level='code').apply(lambda x: x.fillna(value=0))
    else:
        data = data.groupby(level='code').apply(lambda x: x)
    data = data.reindex(index).sort_index()

    return data


def divide_two_variable(first_var_name, second_var_name, output_name, data):
    data[output_name] = data[first_var_name] / data[second_var_name]
    return data[[output_name]]


def multiply_two_variable(first_var_name, second_var_name, output_name, data, ):
    data[output_name] = data[first_var_name] * data[second_var_name]
    return data[[output_name]]


def minus_two_variable(first_var_name, second_var_name, output_name, data, ):
    data[output_name] = data[first_var_name] - data[second_var_name]
    return data[[output_name]]


def or_two_variable(first_var_name, second_var_name, output_name, data, ):
    data[output_name] = data[first_var_name] | data[second_var_name]
    return data[[output_name]]


def update_param(default_param, add_param):
    new_param = {}
    for key, value in default_param.items():
        if key in add_param:
            new_param.update({key: add_param[key]})
        else:
            new_param.update({key: value})
    return new_param


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
                print(class_name)
                param = operator.get('param', {})
                test_clf = operator['class'](param, {}, {})
                class_operators = test_clf.operators
                class_output_vars = test_clf.output_vars

                input_name_mapping = operator.get('input_name_mapping', dict())
                raw_output_name_mapping = operator.get('output_name_mapping', dict())

                output_name_mapping = {}
                for output_var in class_output_vars:
                    output_name_mapping.update({output_var: raw_output_name_mapping.get(output_var, output_var)})
                # if output_name_mapping is {}:
                #     output_name_mapping = {_: _ for _ in class_output_vars}

                sub_funcs = self.transfer_operators_2_func_pipline(class_operators, class_name, input_name_mapping,
                                                                   output_name_mapping)
                func_infos.extend(sub_funcs)
        update_func_infos = []
        for func_info in func_infos:
            update_func_info = func_info.copy()
            input_data = func_info['input_data']
            update_input_data = {}
            for param_name, input_name in input_data.items():
                if param_name in pre_input_data_mapping:
                    update_input_data.update({param_name: pre_input_data_mapping[param_name]})
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
        self.funcs = self.transfer_operators_2_func_pipline(self.operators, self.tag, self.input_name_mapping,
                                                            self.output_name_mapping)
        # print(self.param_info)
        #

        for func_info in self.funcs:

            func = func_info['func']
            print("="*15,"func {}".format(func.__name__),"="*15)
            print("参数默认：",func_info)
            param = func_info['param']
            print(param)
            common_param = self.param_info.get("common", {})
            param = update_param(param, common_param)
            special_param = self.param_info.get("special", {})

            if "tag" in func_info:
                input_param = special_param.get(func_info['tag'], {})
                param = update_param(param, input_param)
            input_data = func_info['input_data']
            print("参数更新：",param,'')
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
        return {output_var: self.data_cash[self.output_name_mapping.get(output_var, output_var)] for output_var in
                self.output_vars}


class UnadjClosePrice(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": "",
                 "save_engine": "",
                 "start_date": 0,
                 "end_date": 0,
                 "table": "daily_trading_data_unadjusted",
                 "field": ["trade_date", 'code', 'close'],
                 "other_filter_info": None,
                 "hist_year": 2,
                 "name_dict": {"close": self.__class__.__name__}},
             "input_data": {},
             "output": ['unadj_close_price']},

        ]
        self.output_vars = ["unadj_close_price"]

class AdjPrice(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": "",
                 "save_engine": "",
                 "start_date": 0,
                 "end_date": 0,
                 "table": "daily_trading_data",
                 "field": ['trade_date', 'code','close','high','low'],
                 "other_filter_info": None,
                 "hist_year": 2,
                 "name_dict": {"price": self.__class__.__name__}},
             "input_data": {},
             "output": ['adj_price']},
        ]
        self.output_vars = ["adj_price"]

class AdjFactor(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": "",
                 "save_engine": "",
                 "start_date": 0,
                 "end_date": 0,
                 "table": "daily_trading_data",
                 "field": ['trade_date', 'code', 'factor'],
                 "other_filter_info": None,
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
            # {
            #     "func": multiply_two_variable,
            #     "param": {"first_var_name": "UnadjClosePrice", "second_var_name": "AdjFactor",
            #               "output_name": self.__class__.__name__},
            #     "input_data": {"data": "merged_data"},
            #     "output": ["adj_close_price"],
            # }

        ]
        self.output_vars = ["merged_data"]


class FactorIndex(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)

        self.operators = [
            {
                "func": get_fac_idx,
                "param": {"start_date": 0, "end_date": 0, "freq": "default", "read_engine": "", "save_engine": ""},
                "input_data": {"data": ""},
                "output": ["factor_index"],
            }
        ]
        self.output_vars = ['factor_index']


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


def std_mkt_cp(value_name, data, output_name):
    data[output_name] = data[value_name].map(lambda x: x * 1e8)
    return data[[output_name]]


class MarketCap(FactorCompute):
    """
        MktCap
    """

    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": "",
                 "save_engine": "",
                 "start_date": 0,
                 "end_date": 0,
                 "table": "valuation_q",
                 "field": ['trade_date', 'code', 'market_cap'],
                 "hist_year": 2,
                 "name_dict": {"market_cap": self.__class__.__name__}},
             "input_data": {},
             "output": ['market_cap_jq']},
            {'func': std_mkt_cp,
             'param': {"value_name": "MarketCap", "output_name": "MarketCap"},
             "input_data": {"data": "market_cap_jq"},
             "output": ['market_cap']},
        ]
        self.output_vars = ["market_cap"]


def cal_market_log(value_name, data, output_name):
    data[output_name] = data[value_name].map(lambda x: np.log(x))
    return data[[output_name]]


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


def cal_nonlinear_size(value_name, window_size, data, output_name):
    data = sm.add_constant(data)
    data["{}Cube".format(value_name)] = data[value_name] ** 3
    # import pdb
    # pdb.set_trace()
    # codes = data.reset_index()['code'].unique()
    # data = data.reset_index()
    # data = data[data['code'].map(lambda x: x in codes[:100])]
    # data = data.set_index(['trade_date', 'code'])
    nonlinear_size = data.groupby(level='code').progress_apply(
        lambda x: rolling_slope_regress(x_df=x.droplevel(level='code')[["const", value_name]],
                                        y_df=x.droplevel(level='code')[["{}Cube".format(value_name)]],
                                        window=window_size)[1])
    nonlinear_size.name = output_name

    nonlinear_size = nonlinear_size.reset_index().set_index(['trade_date', 'code'])
    return nonlinear_size


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
                "param": {"value_name": "LogMktCap", "window_size": 252, "output_name": self.__class__.__name__},
                "input_data": {"data": "log_mkt_cap"},
                "output": ["nonlinear_size"],
            }
        ]
        self.output_vars = ['nonlinear_size']


def cal_pre_day_price(value_name, data, output_name):
    # data = data.reset_index()
    data[output_name] = data.sort_index(level=['code', 'trade_date']).groupby(level='code')[value_name].shift(
        1).sort_index(level=['trade_date', 'code'])
    return data[[output_name]]


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
                "param": {"value_name": "AdjClosePrice", "output_name": self.__class__.__name__},
                "input_data": {"data": "merged_data"},
                "output": ["pre_adj_close_price"]
            },
        ]
        self.output_vars = ['pre_adj_close_price']


def cal_pct_chg_hfq(adj_close_price_name, pre_adj_close_price_name, data, output_name):
    pct_chg_hfq = divide_two_variable(adj_close_price_name, pre_adj_close_price_name, output_name, data).applymap(
        lambda x: (x - 1) * 100)

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
                "param": {"adj_close_price_name": "AdjClosePrice", "pre_adj_close_price_name": "PreAdjClosePrice",
                          "output_name": self.__class__.__name__},
                "input_data": {"data": "merged_data"},
                "output": ["pct_chg_hfq_daily"],
            }

        ]
        self.output_vars = ["pct_chg_hfq_daily"]


#### 量价指标 #####

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
                "param": {"value_name": "AdjClosePrice", "window_size": self.window_size,
                          "output_name": "{}{}".format(self.__class__.__name__, self.window_size)},
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
                "output_name_mapping": {"momentum_weeks_{}".format(self.long_window_size): "momentum_weeks_{}".format(
                    self.long_window_size)},
            },
            {
                "class": MomentumWeeks,
                "param": {"window_size": self.short_window_size},
                "output_name_mapping": {"momentum_weeks_{}".format(self.long_window_size): "momentum_weeks_{}".format(
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


def cal_rsi(rolling_n, pct_chg_hfq_name, output_name, data):
    pct_chg_hfq = data[pct_chg_hfq_name]
    up_tag = pct_chg_hfq.map(lambda x: x > 0)
    down_tag = pct_chg_hfq.map(lambda x: x < 0)
    up_pct_chg_hfq = up_tag * pct_chg_hfq
    down_pct_chg_hfq = down_tag * pct_chg_hfq
    up_sum = up_pct_chg_hfq.groupby(level='code').progress_apply(lambda x: x.rolling(rolling_n).sum())
    up_sum.name = 'up_sum'
    down_sum = down_pct_chg_hfq.groupby(level='code').progress_apply(lambda x: x.rolling(rolling_n).sum())
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
                          "output_name": "{}{}Days".format(self.__class__.__name__, self.window_size)
                          },
                "input_data": {"data": "pct_chg_hfq_daily"},
                "output": ["rsi_{}_days".format(self.window_size)],
            }
        ]
        self.output_vars = ["rsi_{}_days".format(self.window_size)]


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
                "param": {"value_name": "PctChgHfqDaily", "window_size": self.window_size,
                          "output_name": "{}{}Days".format(self.__class__.__name__, self.window_size)},
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
                 "read_engine": "",
                 "save_engine": "",
                 "start_date": 0,
                 "end_date": 0,
                 "table": "valuation_q",
                 "field": ["trade_date", 'code', 'turnover_ratio'],
                 "hist_year": 2,
                 "name_dict": {"turnover_ratio": self.__class__.__name__}},
             "input_data": {},
             "output": ['turnover_ratio']},

        ]
        self.output_vars = ["turnover_ratio"]

    """Daily TurnOverRatio as a continous factor"""


def cal_share_turnover(value_name, month_count, data, output_name):
    days = month_count * 21
    share_turnover = data[value_name].groupby(level='code').progress_apply(
        lambda x: x.rolling(days).sum()) / month_count
    share_turnover.name = output_name
    share_turnover = share_turnover.reset_index().set_index(['trade_date', 'code'])
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
                "output_name_mapping": {"turnover_ratio", "turnover_ratio"}
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
                "class": TurnoverRatio,
                "output_name_mapping": {"turnover_ratio", "turnover_ratio"}
            },
            {
                "func": cal_share_turnover,
                "param": {
                    "value_name": "TurnoverRatio",
                    "output_name": self.__class__.__name__,
                    "month_count": 3,
                },
                "input_data": {"data": "turnover_ratio"},
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
                "class": TurnoverRatio,
                "output_name_mapping": {"turnover_ratio": "turnover_ratio"},
            },
            {
                "func": cal_share_turnover,
                "param": {
                    "value_name": "TurnoverRatio",
                    "output_name": self.__class__.__name__,
                    "month_count": 12,
                },
                "input_data": {"data": "turnover_ratio"},
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
                    "read_engine": "",
                    "save_engine": "",
                    "start_date": 0,
                    "end_date": 0,
                    "table": "index_level",
                    "field": ["trade_date", "code", 'close'],
                    "hist_year": 2,
                    "name_dict": {
                        "close": "{}{}".format(self.__class__.__name__, self.benchmark_code.replace(".", ""))}},
                "input_data": {},
                "output": ['benchmark_price_{}'.format(self.benchmark_code.replace(".", ""))]
            },
        ]
        self.output_vars = ['benchmark_price_{}'.format(self.benchmark_code.replace(".", ""))]


def cal_benchmark_pct_chg(value_name, data, output_name):
    data = data.droplevel('code')
    data = data.sort_index(level='trade_date')
    data[output_name] = (data[value_name] / data[value_name].shift(1) - 1).map(lambda x: x * 100)
    return data[[output_name]]


class BenchmarkPctChg(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.benchmark_code = self.param_info.get("benchmark_code", "000905.XSHG")
        self.operators = [
            {
                "class": BenchmarkPrice,
                "output_name_mapping": {
                    'benchmark_price_{}'.format(self.benchmark_code.replace(".", "")): 'benchmark_price_{}'.format(
                        self.benchmark_code.replace(".", ""))}
            },
            {
                "func": merge_data,
                "param": {},
                "input_data": {"1": 'benchmark_price_{}'.format(self.benchmark_code.replace(".", "")), },
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

def rolling_slope_regress(x_df, y_df, window):
    x_name = x_df.columns
    y_name = y_df.columns[0]
    # x_df = x_df.droplevel(level='code')
    x_df = x_df.reset_index()
    y_df = y_df.reset_index()
    df = pd.merge(x_df, y_df, how='inner', on='trade_date')
    df = df.dropna()
    df = df.set_index(['trade_date'])
    if df.shape[0] < window:
        params = pd.DataFrame(data=np.NAN, index=df.index, columns=x_name)
    else:
        model = RollingOLS(endog=df[y_name], exog=df[x_name], window=window).fit()
        params = model.params.copy()

    fitted_values = (params * df[x_name]).sum(axis=1).replace(0, np.NAN)
    # Calculate the residuals #
    residuals = df[y_name] - fitted_values

    return params.iloc[:, 1], residuals


def cal_market_beta(code_pct_chg_hfq, benchmark_pct_chg, window_size, output_name):
    code_pct_chg_hfq = sm.add_constant(code_pct_chg_hfq)

    code_pct_chg_hfq[output_name] = code_pct_chg_hfq.groupby(level='code').progress_apply(
        lambda x: rolling_slope_regress(x.droplevel(level='code'), benchmark_pct_chg, window_size)[0])
    return code_pct_chg_hfq[[output_name]]


class MarketBeta(FactorCompute):
    """CAPM market beta estimated by 252 rolling regression"""

    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.benchmark_code = self.param_info.get("benchmark_code", "000905.XSHG")
        self.window_size = self.param_info.get("window_size", 252)
        self.operators = [
            {
                "class": BenchmarkPctChg,
                "output_name_mapping": {
                    'benchmark_pct_chg_{}'.format(self.benchmark_code.replace(".", "")): 'benchmark_pct_chg_{}'.format(
                        self.benchmark_code.replace(".", ""))}
            },
            {
                "class": PctChgHfqDaily,
                # "output_name_mapping": {"pct_chg_hfq_daily": "pct_chg_hfq_daily"}
            },
            {
                "func": cal_market_beta,
                "param": {
                    "output_name": "{}{}{}".format(self.__class__.__name__, self.benchmark_code.replace(".", ""),
                                                   self.window_size),
                    "window_size": self.window_size
                },
                "input_data": {"code_pct_chg_hfq": "pct_chg_hfq_daily",
                               "benchmark_pct_chg": 'benchmark_pct_chg_{}'.format(
                                   self.benchmark_code.replace(".", ""))},
                "output": ["market_beta_{}_{}".format(self.benchmark_code.replace(".", ""), self.window_size)]
            },
        ]


##### financial factor
class TotalOperatingRevenue(FactorCompute):
    """total opertaing revenue as a continous factor"""

    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": "",
                 "save_engine": "",
                 "start_date": 0,
                 "end_date": 0,
                 "table": "income_stk",
                 "field": ["trade_date", "code", 'total_operating_revenue', "end_date"],
                 "index": ['trade_date', 'code', 'end_date'],
                 "hist_year": 2,
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
                 "read_engine": "",
                 "save_engine": "",
                 "start_date": 0,
                 "end_date": 0,
                 "table": "income_stk",
                 "field": ["trade_date", "code", 'operating_profit', "end_date"],
                 "index": ['trade_date', 'code', 'end_date'],
                 "hist_year": 2,
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
                 "read_engine": "",
                 "save_engine": "",
                 "start_date": 0,
                 "end_date": 0,
                 "table": "income_q",
                 "field": ["trade_date", "code", 'total_composite_income', "end_date"],
                 "index": ['trade_date', 'code', 'end_date'],
                 "hist_year": 2,
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
                 "read_engine": "",
                 "save_engine": "",
                 "start_date": 0,
                 "end_date": 0,
                 "table": "income_q",
                 "field": ["trade_date", "code", 'total_operating_revenue', 'end_date'],
                 "index": ['trade_date', 'code', 'end_date'],
                 "hist_year": 2,
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
                 "read_engine": "",
                 "save_engine": "",
                 "start_date": 0,
                 "end_date": 0,
                 "table": "indicator_q",
                 "field": ["trade_date", "code", 'operating_profit', 'end_date'],
                 "index": ['trade_date', 'code', 'end_date'],
                 "hist_year": 2,
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
                 "read_engine": "",
                 "save_engine": "",
                 "start_date": 0,
                 "end_date": 0,
                 "table": "income_stk",
                 "field": ["trade_date", "code", 'Operating_Tax_Surcharges', 'end_date'],
                 "index": ['trade_date', 'code', 'end_date'],

                 "hist_year": 2,
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
                 "read_engine": "",
                 "save_engine": "",
                 "start_date": 0,
                 "end_date": 0,
                 "table": "income_stk",
                 "field": ["trade_date", "code", 'operating_cost', 'end_date'],
                 "index": ['trade_date', 'code', 'end_date'],

                 "hist_year": 2,
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
                 "read_engine": "",
                 "save_engine": "",
                 "start_date": 0,
                 "end_date": 0,
                 "table": "income_stk",
                 "field": ["trade_date", "code", 'sale_expense', 'end_date'],
                 "index": ['trade_date', 'code', 'end_date'],

                 "hist_year": 2,
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
                 "read_engine": "",
                 "save_engine": "",
                 "start_date": 0,
                 "end_date": 0,
                 "table": "income_stk",
                 "field": ["trade_date", "code", 'administration_expense', 'end_date'],
                 "index": ['trade_date', 'code', 'end_date'],

                 "hist_year": 2,
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
                 "read_engine": "",
                 "save_engine": "",
                 "start_date": 0,
                 "end_date": 0,
                 "table": "income_stk",
                 "field": ["trade_date", "code", 'interest_expense', 'end_date'],
                 "index": ['trade_date', 'code', 'end_date'],

                 "hist_year": 2,
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
                 "read_engine": "",
                 "save_engine": "",
                 "start_date": 0,
                 "end_date": 0,
                 "table": "income_q",
                 "field": ["trade_date", "code", 'interest_expense', 'end_date'],
                 "index": ['trade_date', 'code', 'end_date'],

                 "hist_year": 2,
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
                 "read_engine": "",
                 "save_engine": "",
                 "start_date": 0,
                 "end_date": 0,
                 "table": "income_stk",
                 "field": ["trade_date", "code", 'commission_expense', 'end_date'],
                 "index": ['trade_date', 'code', 'end_date'],

                 "hist_year": 2,
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
                 "read_engine": "",
                 "save_engine": "",
                 "start_date": 0,
                 "end_date": 0,
                 "table": "income_stk",
                 "field": ["trade_date", "code", 'rd_expenses', 'end_date'],
                 "index": ['trade_date', 'code', 'end_date'],

                 "hist_year": 2,
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
                 "read_engine": "",
                 "save_engine": "",
                 "start_date": 0,
                 "end_date": 0,
                 "table": "income_stk",
                 "field": ["trade_date", "code", 'asset_impairment_loss', 'end_date'],
                 "index": ['trade_date', 'code', 'end_date'],

                 "hist_year": 2,
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
                 "read_engine": "",
                 "save_engine": "",
                 "start_date": 0,
                 "end_date": 0,
                 "table": "income_stk",
                 "field": ["trade_date", "code", 'other_earnings', 'end_date'],
                 "index": ['trade_date', 'code', 'end_date'],

                 "hist_year": 2,
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
                 "read_engine": "",
                 "save_engine": "",
                 "start_date": 0,
                 "end_date": 0,
                 "table": "income_stk",
                 "field": ["trade_date", "code", 'income_tax', 'end_date'],
                 "index": ['trade_date', 'code', 'end_date'],

                 "hist_year": 2,
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
                 "read_engine": "",
                 "save_engine": "",
                 "start_date": 0,
                 "end_date": 0,
                 "table": "income_stk",
                 "field": ["trade_date", "code", 'total_profit', 'end_date'],
                 "index": ['trade_date', 'code', 'end_date'],

                 "hist_year": 2,
                 "name_dict": {"total_profit": self.__class__.__name__}},
             "input_data": {},
             "output": ['total_profit']},
        ]
        self.output_vars = ["total_profit"]


def cal_ebit(total_operating_revenue_name, operating_tax_surcharges_name, operating_cost_name, sale_expense_name,
             administration_expense_name, interest_expense_name, commission_expense_name, rd_expenses_name,
             asset_impairment_loss_name,
             other_earnings_name, data, output_name):
    data = data.fillna(0)
    data = data.astype(float)
    data[output_name] = (data[total_operating_revenue_name] -
                         data[operating_tax_surcharges_name] -
                         (data[operating_cost_name] + data[sale_expense_name] +
                          data[administration_expense_name] + data[interest_expense_name] + data[
                              commission_expense_name] +
                          data[rd_expenses_name] + data[asset_impairment_loss_name]) +
                         data[other_earnings_name])
    return data[[output_name]]


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


def cal_tax_rate(income_tax_name, total_profit_name, output_name, data):
    data[output_name] = data[income_tax_name] / data[total_profit_name]
    # make sure the tax rate is greater than 0
    data[output_name] = data[output_name] * (data[output_name] > 0)
    return data[[output_name]]


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
                 "read_engine": "",
                 "save_engine": "",
                 "start_date": 0,
                 "end_date": 0,
                 "table": "balance_stk",
                 "field": ["trade_date", "code", 'cash_equivalents', 'end_date'],
                 "index": ['trade_date', 'code', 'end_date'],
                 "hist_year": 2,
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
                 "read_engine": "",
                 "save_engine": "",
                 "start_date": 0,
                 "end_date": 0,
                 "table": "balance_stk",
                 "field": ["trade_date", "code", 'total_assets', 'end_date'],
                 "index": ['trade_date', 'code', 'end_date', ],
                 "hist_year": 2,
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
                 "read_engine": "",
                 "save_engine": "",
                 "start_date": 0,
                 "end_date": 0,
                 "table": "balance_stk",
                 "field": ["trade_date", "code", 'total_liability', 'end_date'],
                 "index": ['trade_date', 'code', 'end_date', ],
                 "hist_year": 2,
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
                 "read_engine": "",
                 "save_engine": "",
                 "start_date": 0,
                 "end_date": 0,
                 "table": "balance_stk",
                 "field": ["trade_date", "code", 'total_owner_equities', 'end_date'],
                 "index": ['trade_date', 'code', 'end_date', ],
                 "hist_year": 2,
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
                 "read_engine": "",
                 "save_engine": "",
                 "start_date": 0,
                 "end_date": 0,
                 "table": "balance_stk",
                 "field": ["trade_date", "code", 'equities_parent_company_owners', 'end_date'],
                 "index": ['trade_date', 'code', 'end_date', ],

                 "hist_year": 2,
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
                 "read_engine": "",
                 "save_engine": "",
                 "start_date": 0,
                 "end_date": 0,
                 "table": "balance_stk",
                 "field": ["trade_date", "code", ''],
                 "hist_year": 2,
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
                 "read_engine": "",
                 "save_engine": "",
                 "start_date": 0,
                 "end_date": 0,
                 "table": "balance_stk",
                 "field": ["trade_date", "code", 'preferred_shares_equity', 'end_date'],
                 "index": ['trade_date', 'code', 'end_date', ],
                 "hist_year": 2,
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
                 "read_engine": "",
                 "save_engine": "",
                 "start_date": 0,
                 "end_date": 0,
                 "table": "balance_stk",
                 "field": ["trade_date", "code", 'total_current_assets', 'end_date'],
                 "index": ['trade_date', 'code', 'end_date', ],

                 "hist_year": 2,
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
                 "read_engine": "",
                 "save_engine": "",
                 "start_date": 0,
                 "end_date": 0,
                 "table": "balance_stk",
                 "field": ["trade_date", "code", 'total_current_liability', 'end_date'],
                 "index": ['trade_date', 'code', 'end_date', ],
                 "hist_year": 2,
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
                 "read_engine": "",
                 "save_engine": "",
                 "start_date": 0,
                 "end_date": 0,
                 "table": "balance_stk",
                 "field": ["trade_date", "code", 'shortterm_loan', 'end_date'],
                 "index": ['trade_date', 'code', 'end_date', ],

                 "hist_year": 2,
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
                 "read_engine": "",
                 "save_engine": "",
                 "start_date": 0,
                 "end_date": 0,
                 "table": "balance_stk",
                 "field": ["trade_date", "code", 'non_current_liability_in_one_year', 'end_date'],
                 "index": ['trade_date', 'code', 'end_date', ],

                 "hist_year": 2,
                 "name_dict": {"non_current_liability_in_one_year": self.__class__.__name__}},
             "input_data": {},
             "output": ['non_current_liability_in_one_year']},
        ]
        self.output_vars = ["non_current_liability_in_one_year"]


class ToatalNonCurrentLiability(FactorCompute):
    """Total NonCurrentLiability as a continous factor"""

    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": "",
                 "save_engine": "",
                 "start_date": 0,
                 "end_date": 0,
                 "table": "balance_stk",
                 "field": ["trade_date", "code", 'total_non_current_liability', 'end_date'],
                 "index": ['trade_date', 'code', 'end_date', ],

                 "hist_year": 2,
                 "name_dict": {"total_non_current_liability": self.__class__.__name__}},
             "input_data": {},
             "output": ['total_non_current_liability']},
        ]
        self.output_vars = ["total_non_current_liability"]


def cal_operating_cash(total_current_assets_name, cash_equivalents_name, total_current_liability_name,
                       shortterm_loan_name, non_current_liability_in_one_year_name, data, output_name):
    data = data.fillna(0)
    # compute NOCF_Over_TORev factor

    data[output_name] = (data[total_current_assets_name] -
                         data[cash_equivalents_name]) - \
                        (data[total_current_liability_name] -
                         data[shortterm_loan_name] - data[
                             non_current_liability_in_one_year_name])
    return data[[output_name]]


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
                 "read_engine": "",
                 "save_engine": "",
                 "start_date": 0,
                 "end_date": 0,
                 "table": "cash_flow_stk",
                 "field": ["trade_date", "code", 'net_operate_cash_flow', 'end_date'],
                 "index": ['trade_date', 'code', 'end_date', ],

                 "hist_year": 2,
                 "name_dict": {"net_operate_cash_flow": self.__class__.__name__}},
             "input_data": {},
             "output": ['net_operate_cash_flow']},
        ]
        self.output_vars = ["net_operate_cash_flow"]


class NetInvestCashFlow(FactorCompute):
    """Net Investing CashFlow as a continous factor"""

    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {'func': get_hist_data_4_factor_compute,
             'param': {
                 "read_engine": "",
                 "save_engine": "",
                 "start_date": 0,
                 "end_date": 0,
                 "table": "cash_flow_stk",
                 "field": ["trade_date", "code", 'net_invest_cash_flow', 'end_date'],
                 "index": ['trade_date', 'code', 'end_date', ],

                 "hist_year": 2,
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
                 "read_engine": "",
                 "save_engine": "",
                 "start_date": 0,
                 "end_date": 0,
                 "table": "cash_flow_stk",
                 "field": ["trade_date", "code", 'intangible_assets_amortization', 'end_date'],
                 "index": ['trade_date', 'code', 'end_date', ],
                 "hist_year": 2,
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
                 "read_engine": "",
                 "save_engine": "",
                 "start_date": 0,
                 "end_date": 0,
                 "table": "cash_flow_stk",
                 "field": ["trade_date", "code", 'fixed_assets_depreciation', 'end_date'],
                 "index": ['trade_date', 'code', 'end_date', ],

                 "hist_year": 2,
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
                 "read_engine": "",
                 "save_engine": "",
                 "start_date": 0,
                 "end_date": 0,
                 "table": "cash_flow_stk",
                 "field": ["trade_date", "code", 'defferred_expense_amortization', 'end_date'],
                 "index": ['trade_date', 'code', 'end_date', ],

                 "hist_year": 2,
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
                 "read_engine": "",
                 "save_engine": "",
                 "start_date": 0,
                 "end_date": 0,
                 "table": "cash_flow_stk",
                 "field": ["trade_date", "code", 'fix_intan_other_asset_acqui_cash', 'end_date'],
                 "index": ['trade_date', 'code', 'end_date', ],

                 "hist_year": 2,
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
                 "read_engine": "",
                 "save_engine": "",
                 "start_date": 0,
                 "end_date": 0,
                 "table": "cash_flow_q",
                 "field": ["trade_date", "code", 'net_invest_cash_flow', 'end_date'],
                 "index": ['trade_date', 'code', 'end_date', ],

                 "hist_year": 2,
                 "name_dict": {"net_invest_cash_flow": self.__class__.__name__}},
             "input_data": {},
             "output": ['net_invest_cash_flow_quarterly']},
        ]
        self.output_vars = ["net_invest_cash_flow_quarterly"]


def cal_fcff_top_down(earning_before_interest_and_taxes_name, tax_rate_name, intangible_assets_amortization_name,
                      fixed_assets_depreciation_name, defferred_expense_amortization_name,
                      fix_intan_other_asset_acqui_cash_name,
                      operating_cash_name, data, output_name):
    data_ = data.reset_index()
    data_['year'] = data_.end_date.apply(lambda x: x.year)
    year_end_operating_cash = data_.loc[
        data_.end_date.apply(lambda x: x.month == 12), ['code', 'end_date', operating_cash_name]]
    year_end_operating_cash['year'] = year_end_operating_cash.end_date.apply(lambda x: x.year + 1)
    data = data_.merge(year_end_operating_cash[['code', 'year', operating_cash_name]],
                       on=['code', 'year'],
                       how='left',
                       suffixes=('', '_last_yr_end')).drop(
        columns='year').set_index(['trade_date', 'code', 'end_date'])

    data = data.fillna(0)
    # import pdb
    # pdb.set_trace()
    data[output_name] = (data[earning_before_interest_and_taxes_name] * (1 - data[tax_rate_name]) +
                         data[intangible_assets_amortization_name] + data[
                             fixed_assets_depreciation_name] + data[defferred_expense_amortization_name] -
                         data[fix_intan_other_asset_acqui_cash_name] - (
                                 data[operating_cash_name] - data[
                             '{}_last_yr_end'.format(operating_cash_name)]))
    return data[[output_name]]


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

def cal_st_based_on_hist_name(value_name, data, output_name):
    data[output_name] = (data[value_name].map(lambda x: "st" in x.lower() if type(x) is str else False)).map(int)
    return data[[output_name]]


def transfer_timestamp_to_int(value_name, data, output_name):
    data[output_name] = data[value_name].map(lambda x: int(x.strftime("%Y%m%d")))


class STFlagNameHistory(FactorCompute):
    """st_flag from history name"""

    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                'func': get_hist_data_4_factor_compute,
                'param': {
                    "read_engine": "",
                    "save_engine": "",
                    "start_date": 0,
                    "end_date": 0,
                    "table": "name_history_stk",
                    "field": ["start_date", "code", 'new_name'],
                    "index": ['trade_date', 'code'],

                    "hist_year": -1,
                    "name_dict": {}},
                "input_data": {},
                "output": ['hist_name']
            },
            {
                "class": FactorIndex,
                "output_name_mapping": {"factor_index": "factor_index"},

            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "ffill"},
                "input_data": {"data": "hist_name", "index": "factor_index"},
                "output": ["hist_name_weekly"],
            },
            {
                "func": cal_st_based_on_hist_name,
                "param": {"value_name": "new_name", "output_name": self.__class__.__name__},
                "input_data": {"data": "hist_name_weekly"},
                "output": ["st_flag_name_history"],
            },
        ]
        self.output_vars = ["st_flag_name_history"]


def cal_st_flag_based_on_net_profit(value_name, data, output_name):
    def cal_st_flag(code_df):
        code_df = code_df.reset_index()

        code_df['end_date'] = code_df['end_date'].map(lambda x: int(x.strftime("%Y%m%d")))

        code_df = code_df.sort_values('trade_date')
        st_flags = []
        for trade_date in code_df['trade_date'].values:
            hist_code_df = code_df[code_df.trade_date <= trade_date]
            end_date = hist_code_df.end_date.values % 10000
            hist_code_df = hist_code_df[(end_date == 1231) | (end_date == 930)]
            if len(hist_code_df) == 0:
                st_flags.append(0)
                continue
            end_date1 = list(hist_code_df.end_date)[-1]
            end_date2 = int(end_date1 / 10000 - 1) * 10000 + 1231
            net_income1 = hist_code_df[value_name][hist_code_df.end_date == end_date1].min()
            net_income2 = hist_code_df[value_name][hist_code_df.end_date == end_date2].min()
            if (net_income2 < 0) & (net_income1 < 0):
                st_flags.append(1)
            else:
                st_flags.append(0)
        code_df['STFlagNetProfit'] = st_flags
        return code_df.set_index(['trade_date'])[['STFlagNetProfit']]

    st_flag = data.groupby(level='code').apply(lambda s: cal_st_flag(s))
    return st_flag


class STFlagNetProfit(FactorCompute):
    """st_flag based on net income data,  """

    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                'func': get_hist_data_4_factor_compute,
                'param': {
                    "read_engine": "",
                    "save_engine": "",
                    "start_date": 0,
                    "end_date": 0,
                    "table": "income_stk",
                    "field": ["trade_date", "end_date", "code", 'net_profit', ],
                    "index": ['trade_date', 'code', 'end_date'],

                    "hist_year": -1,
                    "name_dict": {"net_profit": "NetProfit"}},
                "input_data": {},
                "output": ['net_profit']
            },
            {
                "func": cal_st_flag_based_on_net_profit,
                "param": {"value_name": "NetProfit", "output_name": self.__class__.__name__},
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


class PauseFlag(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                'func': get_hist_data_4_factor_compute,
                'param': {
                    "read_engine": "",
                    "save_engine": "",
                    "start_date": 0,
                    "end_date": 0,
                    "table": "daily_trading_data",
                    "field": ["trade_date", "code", 'paused', ],
                    "index": ['trade_date', 'code'],

                    "hist_year": -1,
                    "name_dict": {"paused": self.__class__.__name__}},
                "input_data": {},
                "output": ['paused_flag']
            },
        ]
        self.output_vars = ['paused_flag']


def cal_end_flag(data, output_name):
    data = data.reset_index()
    data['end_date'] = data['end_date'].map(lambda x: int(x.strftime("%Y%m%d")))
    data[output_name] = (data['trade_date'] - data['end_date']).map(lambda x: 1 if x > 0 else 0)
    return data.set_index(['trade_date', 'code'])[[output_name]]


class EndFlag(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                'func': get_hist_data_4_factor_compute,
                'param': {
                    "read_engine": "",
                    "save_engine": "",
                    "start_date": 0,
                    "end_date": 0,
                    "table": "stock_universe",
                    "field": ["trade_date", "code", 'end_date', ],
                    "index": ['trade_date', 'code'],
                    "hist_year": 2,
                    "name_dict": {}},
                "input_data": {},
                "output": ['stock_info']
            },
            {
                'func': cal_end_flag,
                'param': {
                    "output_name": self.__class__.__name__
                },
                "input_data": {"data": "stock_info"},
                "output": ['end_flag']
            },
        ]
        self.output_vars = ['end_flag']


def cal_listed_flag(mini_list_days, output_name, data):
    data = data.reset_index()
    data['trade_date_'] = pd.to_datetime(data['trade_date'], format='%Y%m%d')
    data['list_days'] = (data['trade_date_'] - data['start_date']).map(lambda x: x.days)
    data[output_name] = data['list_days'].map(lambda x: x < mini_list_days)
    return data[[output_name]]


class ListedFlag(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                'func': get_hist_data_4_factor_compute,
                'param': {
                    "read_engine": "",
                    "save_engine": "",
                    "start_date": 0,
                    "end_date": 0,
                    "table": "stock_universe",
                    "field": ["trade_date", "code", 'start_date', ],
                    "index": ['trade_date', 'code'],
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


def cal_nan_flag(data, output_name):
    data[output_name] = data.isna().any(axis=1)
    return data[[output_name]]


class NanFlag(FactorCompute):
    """missing key financial value as a categorical factor"""

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
                "class": NetOperateCashFlow,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"net_operate_cash_flow": "net_operate_cash_flow"},
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
                "class": OperatingProfitQuarterly,
                'param': {},
                "input_name_mapping": {},
                "output_name_mapping": {"operating_profit_quarterly": "operating_profit_quarterly"},
                "output": ["operating_profit_quarterly"],
            },
            {
                "func": merge_data,
                "param": {},
                "input_data": {"1": "fcff_top_down", "2": "total_liability", "3": "total_assets",
                               "4": "net_operate_cash_flow", "5": "total_operating_revenue",
                               "6": "operating_profit_quarterly"},
                "output": ["merged_data"]
            },
            {
                "func": cal_nan_flag,
                "param": {},
                "input_data": {"data": "merged_data"},
                "output": ["nan_flag"]
            },
        ]
        self.output_vars = ['nan_flag']


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
                "input_data": {"data": "fcff_top_down", "index": "factor_index"},
                "output": ["fcff_top_down"],
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


def cal_all_code_quarterly_trend(value_name, data, output_name):
    def cal_quarterly_regress(code_df, factor_name):

        code_df = code_df.reset_index()
        code_ = code_df['code'].values[0]

        trade_dates = code_df['trade_date'].unique()
        trade_date_2_trend = {}
        for trade_date in trade_dates:
            tmp_df = code_df[code_df.trade_date <= trade_date].copy()
            tmp_df.sort_values('end_date', inplace=True)
            tmp_df = tmp_df.fillna(method='ffill')
            last_quarter_values = tmp_df[factor_name].values[-3:]
            if len(last_quarter_values) == 3:
                trend_value = trend_regress(last_quarter_values)
                trade_date_2_trend.update({trade_date: trend_value})
        code_df['trend'] = code_df['trade_date'].map(trade_date_2_trend)

        return code_df.set_index(['trade_date', 'end_date'])['trend']

    trend = data.groupby(level='code').progress_apply(lambda x: cal_quarterly_regress(x, value_name))
    trend.name = output_name
    trend = trend.reset_index().set_index(['trade_date', 'code', 'end_date']).sort_index()
    return trend


def cal_all_code_quarter_2_yoy(value_name, data, output_name):
    def cal_quarter_2_yoy(code_df, factor_name):
        """
        计算特定财报 yoy指标
        :param code_df: dataframe
        :param factor_name:  因子名字
        :return:
        """
        code_df = code_df.reset_index()
        code_df.sort_values(['trade_date', 'end_date'], inplace=True)
        code_df.fillna(method='ffill', inplace=True)
        # if "000553" in code_df['code'].values[0]:
        #     import pdb
        #     pdb.set_trace()
        code_df['end_date_'] = code_df['end_date'].map(lambda x: str(x)[:10])
        code_df['last_year_end_date_'] = code_df['end_date_'].map(
            lambda x: datetime.datetime.strptime(x, "%Y-%m-%d")).map(
            lambda x: datetime.datetime(year=x.year - 1, month=x.month, day=x.day).strftime("%Y-%m-%d"))

        end_date_2_factor = dict(zip(code_df['end_date_'].values, code_df[factor_name].values))
        code_df['last_year_factor'] = code_df['last_year_end_date_'].map(end_date_2_factor)
        code_df['yoy'] = (code_df[factor_name] - code_df['last_year_factor']) / code_df['last_year_factor'].map(
            lambda x: abs(x))

        code_df.drop_duplicates('trade_date', keep='last', inplace=True)
        code_df['last_max_end_date'] = code_df['end_date'].cummax()
        code_df['is_max'] = code_df['end_date'] == code_df['last_max_end_date']
        code_df = code_df[code_df['is_max']]
        return code_df.set_index(['trade_date', 'end_date'])['yoy']

    yoy = data.groupby(level='code').progress_apply(lambda x: cal_quarter_2_yoy(x, value_name))
    yoy.name = output_name
    yoy = yoy.reset_index().set_index(['trade_date', 'code', 'end_date']).sort_index()
    yoy = yoy.replace(np.inf, -999)

    return yoy


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
                "func": cal_all_code_quarter_2_yoy,
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
                 "read_engine": "",
                 "save_engine": "",
                 "start_date": 0,
                 "end_date": 0,
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
                 "read_engine": "",
                 "save_engine": "",
                 "start_date": 0,
                 "end_date": 0,
                 "table": "valuation_q",
                 "field": ['trade_date', 'code', 'pb_ratio'],
                 "hist_year": 2,
                 "name_dict": {"pb_ratio": self.__class__.__name__}},
             "input_data": {},
             "output": ['price_to_book']},
        ]
        self.output_vars = ["price_to_book"]


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

                "param": {"first_var_name": "MarketCap", "second_var_name": "PriceToBook",
                          "output_name": self.__class__.__name__},
                "input_data": {"data": "merged_data"},
                "output": ["book_value"],
            },
        ]
        self.output_vars = ['book_value']


def cal_blev(book_value_name, preferred_shares_equity_name, total_non_current_liability_name, data, output_name):
    data[output_name] = (data[book_value_name] + data[
        preferred_shares_equity_name] + data[total_non_current_liability_name]) / (data[book_value_name])
    return data[[output_name]]


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
                "class": ToatalNonCurrentLiability,
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
                "func": merge_data,
                "param": {},
                "input_data": {"1": "book_value", "2": "preferred_shares_equity", "3": "total_non_current_liability"},
                "output": ["merged_data"]
            },
            {
                "func": cal_blev,
                "param": {
                    "book_value_name": "BookValue",
                    "preferred_shares_equity_name": "PreferredSharesEquity",
                    "total_non_current_liability_name": "ToatalNonCurrentLiability",
                    "output_name": self.__class__.__name__
                },
                "input_data": {"data": "merged_data"},
                "output": ["book_leverage"],
            },

        ]
        self.output_vars = ['book_leverage']


def cal_mlev(market_cap_name, preferred_shares_equity_name, total_non_current_liability_name, data, output_name):
    data[output_name] = (data[market_cap_name] + data[
        preferred_shares_equity_name] + data[total_non_current_liability_name]) / (data[market_cap_name])
    return data[[output_name]]


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
                "class": ToatalNonCurrentLiability,
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
                "func": merge_data,
                "param": {},
                "input_data": {"1": "market_cap", "2": "preferred_shares_equity", "3": "total_non_current_liability"},
                "output": ["merged_data"]
            },
            {
                "func": cal_mlev,
                "param": {
                    "market_cap_name": "MarketCap",
                    "preferred_shares_equity_name": "PreferredSharesEquity",
                    "total_non_current_liability_name": "ToatalNonCurrentLiability",
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
                 "read_engine": "",
                 "save_engine": "",
                 "start_date": 0,
                 "end_date": 0,
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
                 "read_engine": "",
                 "save_engine": "",
                 "start_date": 0,
                 "end_date": 0,
                 "table": "indicator_q",
                 "field": ['trade_date', 'code', 'roe'],
                 "hist_year": 2,
                 "name_dict": {"roe": self.__class__.__name__}},
             "input_data": {},
             "output": ['roe']},
        ]
        self.output_vars = ["roe"]


class NOCF_Over_Debt(FactorCompute):
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


# author : yangrenshuo 2023/03/22
def boll(x, rolling_n, std_cof, output_name):
    a, b, c = talib.BBANDS(x, timeperiod=rolling_n, nbdevup=std_cof, nbdevdn=std_cof)
    return pd.DataFrame({output_name[0]: a, output_name[1]: b, output_name[2]: c})


def cal_boll(rolling_n, std_cof, adjcloseprice_name, output_name, data):
    AdjClosePrice = data[adjcloseprice_name]
    df = AdjClosePrice.groupby(level='code').apply(lambda x: boll(x, rolling_n, std_cof, output_name))
    return df


class BollBands(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.window_size = self.param_info.get("common", {}).get("rolling_n", 10)
        self.cof = self.param_info.get("cof", 2)
        self.operators = [
            {
                "class": AdjClosePrice,
                "output_name_mapping": {"adj_close_price": "adj_close_price"},
            },
            {
                "func": cal_boll,
                "param": {"rolling_n": self.window_size,
                          "std_cof": self.cof,
                          "adjcloseprice_name": "AdjClosePrice",
                          "output_name": ["bollup", "bollmid", "bolldown"]
                          },
                "input_data": {"data": "adj_close_price"},
                "output": ["bollbands"],
            }
        ]
        self.output_vars = ["bollbands"]


#rsi 指标
def rsi(x, rolling_n, output_name):
    a= talib.RSI(x['close'],timeperiod=rolling_n)
    return pd.DataFrame({output_name[0]: a})


def cal_rsi(rolling_n, output_name, data):
    # print(data)
    # AdjClosePrice = data[adjcloseprice_name]
    df = data.groupby(level='code').apply(lambda x: rsi(x, rolling_n, output_name))
    if len(output_name)>1:
        df[output_name[1:]]=data[output_name[1:]]
    return df


class RSI(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)

        self.operators = [
            {
                "class": AdjPrice,
                "output_name_mapping": {"adj_price": "adj_price"},
            },
            {
                "func": cal_rsi,
                "param": {"rolling_n":10,
                          "output_name": ['rsi']
                          },
                "input_data": {"data": "adj_price"},
                "output": ["rsi"],
            }
        ]
        self.output_vars = ["rsi"]
######################################################动量因子##########################
#aroon 指标
def aroon(x, rolling_n, output_name):
    a, b = talib.AROON(x['high'],x['low'], timeperiod=rolling_n)
    return pd.DataFrame({output_name[0]: a, output_name[1]: b})


def cal_aroon(rolling_n, output_name, data):
    # print(data)
    # AdjClosePrice = data[adjcloseprice_name]
    df = data.groupby(level='code').apply(lambda x: aroon(x, rolling_n, output_name))
    if len(output_name)>2:
        df[output_name[2]]=data[output_name[2]]
    return df


class AROON(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)

        self.operators = [
            {
                "class": AdjPrice,
                "output_name_mapping": {"adj_price": "adj_price"},
            },
            {
                "func": cal_aroon,
                "param": {"rolling_n":10,
                          "output_name": ["aroondown", "aroonup"]
                          },
                "input_data": {"data": "adj_price"},
                "output": ["aroon"],
            }
        ]
        self.output_vars = ["aroon"]

#BBIC
def bbic(x, rolling_n,output_name):
    close = x['close']
    ma1 = talib.MA(close, timeperiod=rolling_n[0])
    ma2 = talib.MA(close, timeperiod=rolling_n[1])
    ma3 = talib.MA(close, timeperiod=rolling_n[2])
    ma4 = talib.MA(close, timeperiod=rolling_n[3])
    bbi = (ma1+ ma2+ma3+ma4)/4
    bbic = bbi/close
    return pd.DataFrame({output_name[0]: bbic, output_name[1]: bbi})


def cal_bbic(rolling_n, output_name, data):
    # print(data)
    # AdjClosePrice = data[adjcloseprice_name]
    df = data.groupby(level='code').apply(lambda x: bbic(x, rolling_n, output_name))
    if len(output_name)>2:
        try:
            df[output_name[2]]=data[output_name[2]]
        except:
            pass
    return df


class BBIC(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)

        self.operators = [
            {
                "class": AdjPrice,
                "output_name_mapping": {"adj_price": "adj_price"},
            },
            {
                "func": cal_bbic,
                "param": {"rolling_n":(3,6,12,24),
                          "output_name": ["bbic", "bbi"]
                          },
                "input_data": {"data": "adj_price"},
                "output": ["bbic"],
            }
        ]
        self.output_vars = ["bbic"]

#bearpower
def bearpower(x, rolling_n,output_name):
    close = x['close']
    ema = talib.EMA(close, timeperiod=rolling_n)
    bearpower = (x['low']-ema)/close
    return pd.DataFrame({output_name[0]: bearpower})


def cal_bearpower(rolling_n, output_name, data):

    df = data.groupby(level='code').apply(lambda x: bearpower(x, rolling_n, output_name))
    if len(output_name)>1:
        try:
            df[output_name[1:]]=data[output_name[1:]]
        except:
            pass
    return df


class BearPower(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)

        self.operators = [
            {
                "class": AdjPrice,
                "output_name_mapping": {"adj_price": "adj_price"},
            },
            {
                "func": cal_bearpower,
                "param": {"rolling_n":13,
                          "output_name": ["bearpower"]
                          },
                "input_data": {"data": "adj_price"},
                "output": ["bearpower"],
            }
        ]
        self.output_vars = ["bearpower"]

#n日乖离率
def deviate(x, rolling_n,output_name):
    close = x['close']
    ma = talib.MA(close, timeperiod=rolling_n)
    deviate = (close-ma)/ma*100
    return pd.DataFrame({output_name[0]: deviate})


def cal_deviate(rolling_n, output_name, data):

    df = data.groupby(level='code').apply(lambda x: deviate(x, rolling_n, output_name))
    if len(output_name)>1:
        try:
            df[output_name[1:]]=data[output_name[1:]]
        except:
            pass
    return df


class Deviate(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)

        self.operators = [
            {
                "class": AdjPrice,
                "output_name_mapping": {"adj_price": "adj_price"},
            },
            {
                "func": cal_deviate,
                "param": {"rolling_n":13,
                          "output_name": ["deviate"]
                          },
                "input_data": {"data": "adj_price"},
                "output": ["deviate"],
            }
        ]
        self.output_vars = ["deviate"]
#bull_power
def bullpower(x, rolling_n,output_name):
    close = x['close']
    ema = talib.EMA(close, timeperiod=rolling_n)
    bullpower = (x['high']-ema)/close
    return pd.DataFrame({output_name[0]: bullpower})


def cal_bullpower(rolling_n, output_name, data):

    df = data.groupby(level='code').apply(lambda x: bullpower(x, rolling_n, output_name))
    if len(output_name)>1:
        try:
            df[output_name[1:]]=data[output_name[1:]]
        except:
            pass
    return df


class BullPower(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)

        self.operators = [
            {
                "class": AdjPrice,
                "output_name_mapping": {"adj_price": "adj_price"},
            },
            {
                "func": cal_bullpower,
                "param": {"rolling_n":13,
                          "output_name": ["bullpower"]
                          },
                "input_data": {"data": "adj_price"},
                "output": ["bullpower"],
            }
        ]
        self.output_vars = ["bullpower"]

#CCI
def cci(x, rolling_n,output_name):
    cci = talib.CCI(x['high'],x['low'],x['close'], timeperiod=rolling_n)
    return pd.DataFrame({output_name[0]: cci})


def cal_cci(rolling_n, output_name, data):

    df = data.groupby(level='code').apply(lambda x: cci(x, rolling_n, output_name))
    if len(output_name)>1:
        try:
            df[output_name[1:]]=data[output_name[1:]]
        except:
            pass
    return df


class CCI(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)

        self.operators = [
            {
                "class": AdjPrice,
                "output_name_mapping": {"adj_price": "adj_price"},
            },
            {
                "func": cal_cci,
                "param": {"rolling_n":13,
                          "output_name": ["cci"]
                          },
                "input_data": {"data": "adj_price"},
                "output": ["cci"],
            }
        ]
        self.output_vars = ["cci"]

#CR
def cr(x, rolling_n,output_name):
    close = x['close']
    low = x['low']
    high = x['high']
    midvalue = (high+low).shift(1)/2
    upvalue = (high-midvalue)
    upvalue[upvalue<0]=0
    downvalue = midvalue-low
    downvalue[downvalue < 0] = 0
    cr = upvalue.rolling(rolling_n).sum()/downvalue.rolling(rolling_n).sum()*100
    return pd.DataFrame({output_name[0]: cr})


def cal_cr(rolling_n, output_name, data):

    df = data.groupby(level='code').apply(lambda x: cr(x, rolling_n, output_name))
    if len(output_name)>1:
        try:
            df[output_name[1:]]=data[output_name[1:]]
        except:
            pass
    return df


class CR(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)

        self.operators = [
            {
                "class": AdjPrice,
                "output_name_mapping": {"adj_price": "adj_price"},
            },
            {
                "func": cal_cr,
                "param": {"rolling_n":13,
                          "output_name": ["cr"]
                          },
                "input_data": {"data": "adj_price"},
                "output": ["cr"],
            }
        ]
        self.output_vars = ["cr"]

#year_rank
def TS_RANK(row, rolling_n,output_name):
    res=[]
    x = row['close'].values
    sl = SortedList(x[:rolling_n],key= neg)
    for i in range(rolling_n,len(x)):
        sl.add(x[i])
        res.append(sl.bisect_left(x[i]))
        sl.remove(x[i-rolling_n])

    res = pd.Series([np.NaN] * min(len(row),rolling_n) + res, index=row.index)
    return pd.DataFrame({output_name[0]: res})


def cal_rank(rolling_n, output_name, data):
    df = data.groupby(level='code').apply(lambda x: TS_RANK(x, rolling_n, output_name))
    if len(output_name)>1:
        try:
            df[output_name[1:]]=data[output_name[1:]]
        except:
            pass
    return df


class YearRank(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)

        self.operators = [
            {
                "class": AdjPrice,
                "output_name_mapping": {"adj_price": "adj_price"},
            },
            {
                "func": cal_rank,
                "param": {"rolling_n":250,
                          "output_name": ["yearrank"]
                          },
                "input_data": {"data": "adj_price"},
                "output": ["yearrank"],
            }
        ]
        self.output_vars = ["yearrank"]
#mass:MASS:SUM(MA(HIGH-LOW,N1)/MA(MA(HIGH-LOW,N1),N1),N2);
def mass(x, rolling_n, output_name):
    dif = x['high'] - x['low']
    difma = talib.MA(dif, rolling_n[0])
    mass = (difma / difma.rolling(rolling_n[0]).mean()).rolling(rolling_n[1]).sum()
    return pd.DataFrame({output_name[0]: mass})


def cal_mass(rolling_n, output_name, data):
    df = data.groupby(level='code').apply(lambda x: mass(x, rolling_n, output_name))
    if len(output_name) > 1:
        try:
            df[output_name[1:]] = data[output_name[1:]]
        except:
            pass
    return df


class MASS(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": AdjPrice,
                "output_name_mapping": {"adj_price": "adj_price"},
            },
            {
                "func": cal_mass,
                "param": {"rolling_n": 13,
                          "output_name": ["mass"]
                          },
                "input_data": {"data": "adj_price"},
                "output": ["mass"],
            }
        ]
        self.output_vars = ["mass"]

#PLRC
def plrc(x, rolling_n,output_name):
    x['time'] = [i for i in range(len(x))]
    y = x['close'] / (x['close'].rolling(rolling_n).mean())
    xy = y * x['time']
    dep = x['time']
    b = (xy.rolling(rolling_n).mean() - (y.rolling(rolling_n).mean()) * (dep.rolling(rolling_n).mean())) / dep.rolling(
        rolling_n).var()
    return pd.DataFrame({output_name[0]: b * (rolling_n / (rolling_n - 1))})
def cal_plrc(rolling_n, output_name, data):
    df = data.groupby(level='code').apply(lambda x: plrc(x, rolling_n, output_name))
    if len(output_name) > 1:
        try:
            df[output_name[1:]] = data[output_name[1:]]
        except:
            pass
    return df


class PLRC(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": AdjPrice,
                "output_name_mapping": {"adj_price": "adj_price"},
            },
            {
                "func": cal_plrc,
                "param": {"rolling_n": 13,
                          "output_name": ["plrc"]
                          },
                "input_data": {"data": "adj_price"},
                "output": ["plrc"],
            }
        ]
        self.output_vars = ["plrc"]

#pricemonth
def pricemonth(x, rolling_n,output_name):
    x_shift= x['close'].shift(rolling_n)
    return pd.DataFrame({output_name[0]: x['close']/x_shift-1})

def cal_pricemonth(rolling_n, output_name, data):
    df = data.groupby(level='code').apply(lambda x: pricemonth(x, rolling_n, output_name))
    if len(output_name) > 1:
        try:
            df[output_name[1:]] = data[output_name[1:]]
        except:
            pass
    return df


class PrcM(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": AdjPrice,
                "output_name_mapping": {"adj_price": "adj_price"},
            },
            {
                "func": cal_pricemonth,
                "param": {"rolling_n": 30,
                          "output_name": ["PrcM"]
                          },
                "input_data": {"data": "adj_price"},
                "output": ["prcm"],
            }
        ]
        self.output_vars = ["prcm"]
#ROC
def roc(x, rolling_n,output_name):
    roc = talib.ROC(x['close'],rolling_n)
    return pd.DataFrame({output_name[0]: roc})

def cal_roc(rolling_n, output_name, data):
    df = data.groupby(level='code').apply(lambda x: roc(x, rolling_n, output_name))
    if len(output_name) > 1:
        try:
            df[output_name[1:]] = data[output_name[1:]]
        except:
            pass
    return df


class ROC(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": AdjPrice,
                "output_name_mapping": {"adj_price": "adj_price"},
            },
            {
                "func": cal_roc,
                "param": {"rolling_n": 12,
                          "output_name": ["roc"]
                          },
                "input_data": {"data": "adj_price"},
                "output": ["roc"],
            }
        ]
        self.output_vars = ["roc"]

#trix
def trix(x, rolling_n,output_name):
    trix = talib.TRIX(x['close'],rolling_n[0])
    trma = trix.rolling(rolling_n[1]).mean()
    return pd.DataFrame({output_name[0]: trix, output_name[1]: trma})
def cal_trix(rolling_n, output_name, data):
    df = data.groupby(level='code').apply(lambda x: trix(x, rolling_n, output_name))
    if len(output_name) > 2:
        try:
            df[output_name[2:]] = data[output_name[2:]]
        except:
            pass
    return df


class TRIX(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": AdjPrice,
                "output_name_mapping": {"adj_price": "adj_price"},
            },
            {
                "func": cal_trix,
                "param": {"rolling_n": (12,5),
                          "output_name": ["trix",'trma']
                          },
                "input_data": {"data": "adj_price"},
                "output": ["trix"],
            }
        ]
        self.output_vars = ["trix"]

#volume_amplitude
def va(x, rolling_n,output_name):
    v_average = x['volume'].rolling(rolling_n).mean()
    amplitude = x['close'].diff(1)/x['close'].shift(1)
    amp_average = amplitude.rolling(rolling_n).mean()
    res = x['volume']/v_average*amp_average
    return pd.DataFrame({output_name[0]: res})

def cal_va(rolling_n, output_name, data):
    df = data.groupby(level='code').apply(lambda x: va(x, rolling_n, output_name))
    if len(output_name) > 1:
        try:
            df[output_name[1:]] = data[output_name[1:]]
        except:
            pass
    return df


class VA(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": AdjPrice,
                "output_name_mapping": {"adj_price": "adj_price"},

            },
            {
                "func": cal_va,
                "param": {"rolling_n": 20,
                          "output_name": ["va"]
                          },
                "input_data": {"data": "adj_price"},
                "output": ["va"],
            }
        ]
        self.output_vars = ["va"]
#hatch
def hatch(x, rolling_n,output_name):
    up_hatch = ((x['high']-x['close'])/np.where(x['close']>x['open'],x['close'],x['open'])).rolling(rolling_n).mean()
    down_hatch = ((x['close']-x['low'])/np.where(x['close']<x['open'],x['close'],x['open'])).rolling(rolling_n).mean()
    return pd.DataFrame({output_name[0]: up_hatch,output_name[1]:down_hatch})

def cal_hatch(rolling_n, output_name, data):
    df = data.groupby(level='code').apply(lambda x: hatch(x, rolling_n, output_name))
    if len(output_name) > 2:
        try:
            df[output_name[2:]] = data[output_name[2:]]
        except:
            pass
    return df


class Hatch(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": AdjPrice,
                "output_name_mapping": {"adj_price": "adj_price"},

            },
            {
                "func": cal_hatch,
                "param": {"rolling_n": 20,
                          "output_name": ["uphatch",'downhatch']
                          },
                "input_data": {"data": "adj_price"},
                "output": ["hatch"],
            }
        ]
        self.output_vars = ["hatch"]
######################风险类因子###########################
#Variance
def variance(x, rolling_n,output_name):
    amp = x['close'].diff(1)/x['close'].shift(1)+1
    amp = pow(amp,250/rolling_n)
    var = amp.rolling(rolling_n).std()
    return pd.DataFrame({output_name[0]: var})

def cal_variance(rolling_n, output_name, data):
    df = data.groupby(level='code').apply(lambda x: variance(x, rolling_n, output_name))
    if len(output_name) > 1:
        try:
            df[output_name[1:]] = data[output_name[1:]]
        except:
            pass
    return df


class Variance(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": AdjPrice,
                "output_name_mapping": {"adj_price": "adj_price"},

            },
            {
                "func": cal_variance,
                "param": {"rolling_n": 20,
                          "output_name": ["variance"]
                          },
                "input_data": {"data": "adj_price"},
                "output": ["variance"],
            }
        ]
        self.output_vars = ["variance"]

def variance2(x, rolling_n,output_name):
    amp = x['close'].diff(1)/x['close'].shift(1)+1
    amp = pow(amp,250/rolling_n)
    var = amp.rolling(rolling_n).std()
    var = var*np.where(x['close'].diff(1)>0,1,-1)
    return pd.DataFrame({output_name[0]: var})

def cal_variance2(rolling_n, output_name, data):
    df = data.groupby(level='code').apply(lambda x: variance2(x, rolling_n, output_name))
    if len(output_name) > 1:
        try:
            df[output_name[1:]] = data[output_name[1:]]
        except:
            pass
    return df


class Variance2(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.operators = [
            {
                "class": AdjPrice,
                "output_name_mapping": {"adj_price": "adj_price"},

            },
            {
                "func": cal_variance2,
                "param": {"rolling_n": 20,
                          "output_name": ["variance"]
                          },
                "input_data": {"data": "adj_price"},
                "output": ["variance2"],
            }
        ]
        self.output_vars = ["variance2"]