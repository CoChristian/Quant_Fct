#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 17:14:24 2022
data process pipeline no class, not recursive
@author: lianrui
"""
import time
import os
import pandas as pd
from functools import wraps
from tqdm import tqdm

tqdm.pandas()
import statsmodels.api as sm
import time
from statsmodels.distributions.empirical_distribution import ECDF
import datetime
# from model.tools import stand, std_winsor, load_obj, show_process
import multiprocessing
import pathlib
from sqlalchemy import create_engine
import re
import numpy as np
import pmdarima as pm
from scipy.optimize import lsq_linear
import scipy.stats as stats
import calendar
import tushare as ts
from scipy.stats import norm
from sklearn.metrics import confusion_matrix
import matplotlib
import pdb
import time
# from model import SQL_api
from statsmodels.regression.rolling import RollingOLS
import hashlib
import json

from model import SQL_api


def encryption_dict(input_dict):
    """
    对输入的 字典 进行 md5 加密
    :param input_dict:
    :return:

    """
    input_tuple = sorted(input_dict.items(), key=lambda x: x[0])
    input_dict = dict(input_tuple)
    input_json = json.dumps(input_dict)
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


def update_arguments(func):
    """
    根据函数的入参， 去除传入的函数字典内多余的参数
    :param func:
    :return:
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        arg_varnames = func.__code__.co_varnames
        param = {varname: kwargs[varname] for varname in arg_varnames if varname in kwargs}
        fac_value = func(*args, **param)
        return fac_value
    return wrapper



def fac_value_to_sr(func):
    """A decorator that adjusts for colunames and data type for fac_value
    general rule is to return a pd.Series, indexed first by trade_date, then by code."""

    # Define the wrapper function to return.
    @wraps(func)
    def wrapper(*args, **kwargs):
        fac_value = func(*args, **kwargs)
        # convert factor values to a pd Series

        if isinstance(fac_value, pd.DataFrame):
            # import pdb
            # pdb.set_trace()
            assert fac_value.shape[1] == 1
            fac_value.columns = [func.__name__]
        else:
            fac_value.name = func.__name__
        return fac_value

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
        print(func.__name__)
        for key, value in kwargs.items():
            print("key {}, value {}".format(key, value))
        new_kwargs = kwargs.copy()
        new_kwargs.update({'name': func.__name__})
        md5_id = encryption_dict(new_kwargs)
        print("md5_id {}".format(md5_id))
        if md5_id in cache:
            return cache[md5_id]
        else:
            result = func(*args, **kwargs)
            cache.update({md5_id: result})
            return result
    return wrapper


@timer
@memorize
def test(c=10):
    time.sleep(2)



@memorize
def create_sql_api(read_engine, save_engine):
    sql_api_clf = SQL_api.SQL_API(save_engine=save_engine,
                                  read_engine=create_engine(read_engine))
    return sql_api_clf

@update_arguments
def get_hist_data_4_factor_compute(read_engine, save_engine, table, field, index=['trade_date', 'code'],  hist_year=0, start_date=None, end_date=None, other_filter_info=None):
    """
    读取特定数据
    :param sql_api_clf:
    :param table:
    :param field:
    :param index, 输出的 index
    :param hist_year: 需要 历史数据，如果为0表示不需要start_date以前的历史数据， 如果为正表示需要 hist_year 年份的历史数据，如果为-1表示需要所有的历史数据
    :param start_date:
    :param end_date:
    :return: 读取的数据
    """
    if hist_year < 0:
        trade_date_condition = [{'field': 'trade_date',
                                'type': 'less_equal',
                                'param': end_date}]
    else:
        trade_date_condition = [{'field': 'trade_date',
                                'type': 'between',
                                'param': [start_date-hist_year*10000, end_date]}]
    if other_filter_info:
        trade_date_condition.append(other_filter_info)
    query_info = {'method': 'select',
                  'sheet_name': table,
                  'tgt_field': {'way': 'show', 'field': ['trade_date', 'code'] + field},
                  'conditions': trade_date_condition}
    sql_api_clf = create_sql_api(read_engine=read_engine, save_engine=save_engine)
    raw_fac = sql_api_clf.read_data_from(query_info)
    raw_fac = raw_fac.set_index(index)
    return raw_fac



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


@update_arguments
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


@memorize
def get_fac_idx(**kwargs):

    """
    generate the trade_date, code specific factor index from dynamic index constituent stocks


    Returns
    -------
    None.

    """
    # engine = kwargs['read_engine']
    # save_engine = kwargs['save_engine']
    # end_date = kwargs['end_date']
    # start_date = kwargs['start_date']
    # freq = kwargs['freq']
    # sql_api_clf = create_sql_api(engine, save_engine)
    # param = kwargs.copy()
    # param.update({'table': "stock_universe", 'field': [], 'hist_year': 0})
    raw_fac_index = get_hist_data_4_factor_compute(table="stock_universe", field=[], hist_year=0, **kwargs)

    fac_index = resample(raw_fac_index.index, **kwargs)
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
    # drop index and duplicates other than trade_date ,code
    # index = get_fac_idx(sql_aip_clf, start_date, end_date, freq=None)
    data = drop_extra_level_index(data)
    # reindex to our full fac index  for forward fill
    # if data is pd Series, convert to dataframe
    # get the union of data index and factor index
    # def get_date_from_index(index):
    #     index_info = dict(zip(list(index.names), zip(*index.values)))
    #     return index_info['trade_date']
    # data_trade_dates = get_date_from_index(data.index)
    # std_trade_dates = get_date_from_index(index)

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


def merge_result(subset_funs, **kwargs):
    """
    将 子函数序列的结果返回
    :param subset_funs: 子函数的名字
    :return:
    """
    results = []
    for func in subset_funs:
        if type(func) == str:
            func = eval(func)
        print("func name {}".format(func.__name__))
        results.append(func(**kwargs))
    result_df = pd.concat(results, axis=1)
    return result_df


class RollingSlopeRegression:
    """
    A class that performs rolling slope regression on a pandas DataFrame.
    """

    def __init__(self, df, window=252, x_name=['const', 'BenchmarkReturn'], y_name='PctChgHfqNone2Zero'):
        """
        Initialize the class with the input data and parameters.

        Parameters:
        df (pandas DataFrame): DataFrame containing the data to be used in the regression.
        window (int): Window size for rolling regression. Defaults to 252.
        x_name (list): List of column names to be used as independent variables. Defaults to ['const', 'BenchmarkReturn'].
        y_name (str): Column name to be used as dependent variable. Defaults to 'Return'.
        """
        self.df = df
        self.window = window
        self.x_name = x_name
        self.y_name = y_name
        self.params = None
        self.fitted_values = None

    def fit(self):
        """
        Perform the rolling slope regression and return the regression coefficients.

        Returns:
        pandas DataFrame: DataFrame of regression coefficients for each step of the rolling window.
        """
        if self.df.shape[0] < self.window:
            self.params = pd.DataFrame(data=np.NAN, index=self.df.index, columns=self.x_name)
        else:
            model = RollingOLS(endog=self.df[self.y_name], exog=self.df[self.x_name], window=self.window).fit()
            self.params = model.params.copy()

        # Calculate the fitted values
        self.fitted_values = (self.params * self.df.drop(columns=self.y_name)).sum(axis=1).replace(0, np.NAN)

        # Calculate the residuals
        self.residuals = self.df[self.y_name] - self.fitted_values

        return self


@memorize
@fac_value_to_sr
def unadj_close_price(**kwargs):
    fac_value = get_hist_data_4_factor_compute(table='daily_trading_data_unadjusted', field=['close'], index=['trade_date', 'code'], hist_year=3, **kwargs)
    return fac_value

@memorize
@fac_value_to_sr
def adj_factor(**kwargs):
    # engine = kwargs['read_engine']
    # save_engine = kwargs['save_engine']
    # end_date = kwargs['end_date']
    # start_date = kwargs['start_date']
    # sql_api_clf = create_sql_api(engine, save_engine)
    # fac_value = get_hist_data_4_factor_compute(sql_api_clf, table='daily_trading_data', field=['factor'], index=['trade_date', 'code'], hist_year=-3, start_date=start_date, end_date=end_date)
    raw_factor = get_hist_data_4_factor_compute(table='daily_trading_data', field=['factor'], index=['trade_date', 'code'], hist_year=3, **kwargs)
    fac_value = raw_factor
    return fac_value

@memorize
@fac_value_to_sr
def adj_close_price(**kwargs):
    subfunc_factor_value = merge_result([unadj_close_price, adj_factor], **kwargs)
    factor_value = subfunc_factor_value['unadj_close_price'] * subfunc_factor_value['adj_factor']
    return factor_value

@memorize
@fac_value_to_sr
def unadj_pre_close_price(**kwargs):
    """前一天的未复权价格"""
    raw_factor = get_hist_data_4_factor_compute(table='daily_trading_data_unadjusted', field=['pre_close'], index=['trade_date', 'code'], hist_year=3, **kwargs)
    fac_value = raw_factor
    return fac_value

@memorize
@fac_value_to_sr
def adj_pre_close_price(**kwargs):
    """
    前一天复原后价格
    :param kwargs:
    :return:
    """
    subfunc_factor_value = merge_result([adj_close_price], **kwargs)
    fac_value = subfunc_factor_value.sort_index(level=[1, 0]).groupby(level=1)['adj_close_price'].shift(
        1).sort_index(level=[0, 1])
    return fac_value

@memorize
@fac_value_to_sr
def mkt_cap(**kwargs):
    """
    获取市值数据
    :param kwargs:
    :return:
    """
    raw_factor = get_hist_data_4_factor_compute(table='valuation_q', field=['market_cap'], index=['trade_date', 'code'], hist_year=3, **kwargs)
    fac_value = raw_factor * 10 ** 8
    index = get_fac_idx(**kwargs)
    fac_value = align_data_to_index(fac_value, index)
    return fac_value



@memorize
@fac_value_to_sr
def adj_close_price_weekly(**kwargs):
    subfunc_factor_value = merge_result([adj_close_price], **kwargs)
    index = get_fac_idx(**kwargs)
    fac_value = align_data_to_index(subfunc_factor_value, index)
    return fac_value
# class AdjClosePriceWeekly(GroupFactor):
#     """weekly sampled AdjClosePrice as a continous factor"""
#     def compute(self):
#         self.instantiate_child_factors(AdjClosePrice = AdjClosePrice,
#                                     )
#
#         # np.argmax +1 because python is 0 indexed
#         self.get_fac_idx()
# #         import pdb
# #         pdb.set_trace()
#         self.fac_value = self.align_data_to_index(self.children_factor_value, self.fac_index)
#         return self

@memorize
@fac_value_to_sr
def pct_chg_hfq(**kwargs):
    """
    股票的日收益
    :param kwargs:
    :return:
    """
    subfunc_factor_value = merge_result([adj_pre_close_price, adj_close_price], **kwargs)
    fac_value = (subfunc_factor_value['adj_close_price'] / subfunc_factor_value['adj_pre_close_price']).map(
    lambda x: (x - 1) * 100)
    fac_value = fac_value.fillna(value=0)

    return fac_value

@memorize
@fac_value_to_sr
def volatility_60_days(**kwargs):
    subfunc_factor_value = merge_result([pct_chg_hfq], **kwargs)
    fac_value = subfunc_factor_value['pct_chg_hfq'].groupby(level='code').progress_apply(
        lambda x: x.rolling(60).std(ddof=0))
    index = get_fac_idx(**kwargs)
    fac_value = align_data_to_index(fac_value, index)
    return fac_value

@memorize
@fac_value_to_sr
def long_mom_29_weeks(**kwargs):
    subfunc_factor_value = merge_result([adj_close_price_weekly], **kwargs)
    fac_value = subfunc_factor_value.groupby(level = 'code').progress_apply(lambda x: x/x.shift(29)-1)
    return fac_value


@memorize
@fac_value_to_sr
def short_mom_5_weeks(**kwargs):
    subfunc_factor_value = merge_result([adj_close_price_weekly], **kwargs)
    fac_value = subfunc_factor_value.groupby(level = 'code').progress_apply(lambda x: x/x.shift(5)-1)
    return fac_value


@memorize
@fac_value_to_sr
def long_29_weeks_minus_short_5_weeks_mom(**kwargs):
    subfunc_factor_value = merge_result([long_mom_29_weeks, short_mom_5_weeks], **kwargs)
    fac_value = subfunc_factor_value['long_mom_29_weeks'] - subfunc_factor_value['short_mom_5_weeks']
    return fac_value

# class Long29weeksMinusShort5weekdsMoM(GroupFactor):
#     def compute(self):
#         self.instantiate_child_factors(LongMoM29weeks = LongMoM29weeks,
#                                        ShortMoM5weeks = ShortMoM5weeks)
#         # np.argmax +1 because python is 0 indexed
#         self.fac_value = self.children_factor_value['LongMoM29weeks'] - self.children_factor_value['ShortMoM5weeks']

# class LongMoM29weeks(GroupFactor):
#     """LongMoM29weeks = Close/Close.shift(29) -1 """
#     def compute(self):
#         self.instantiate_child_factors(AdjClosePriceWeekly = AdjClosePriceWeekly)
#         # np.argmax +1 because python is 0 indexed
#         self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x/x.shift(29)-1)
#         return self
#
# class ShortMoM5weeks(GroupFactor):
#     """ShortMoM5weeks = Close/Close.shift(5) -1 """
#     def compute(self):
#         self.instantiate_child_factors(AdjClosePriceWeekly = AdjClosePriceWeekly)
#         # np.argmax +1 because python is 0 indexed
#         self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x/x.shift(5)-1)
#         return self



@memorize
@fac_value_to_sr
def bench_mark_000905_daily_price(**kwargs):

    raw_fac = get_hist_data_4_factor_compute(table='index_level', field=['close'], index=['trade_date', 'code'], hist_year=3, other_filter_info={"field": 'code', "type": "equal", "param": "000905.XSHG"}, **kwargs)
    fac_value = raw_fac
    return fac_value

@memorize
@fac_value_to_sr
def bench_mark_000905_daily_return(**kwargs):
    subfunc_factor_value = merge_result([bench_mark_000905_daily_price], **kwargs)

    fac_value = (subfunc_factor_value['bench_mark_000905_daily_price'] - subfunc_factor_value['bench_mark_000905_daily_price'].shift(
        1)) / subfunc_factor_value['bench_mark_000905_daily_price'].shift(1)
    fac_value = fac_value.droplevel(1)
    # index = get_fac_idx(**kwargs)
    # code_ls = index.get_level_values(1).unique()
    # date_ls = fac_value.index
    # multiindex = pd.MultiIndex.from_product([code_ls, date_ls])
    # tmp_df = pd.DataFrame(index=multiindex).reset_index().set_index('trade_date')
    # tmp_df['BenchmarkReturn'] = fac_value
    # tmp_df = tmp_df.set_index('code', append=True)
    # fac_value = tmp_df['BenchmarkReturn']
    return fac_value

@memorize
@fac_value_to_sr
def market_beta_252(**kwargs):
    pct_chg_hfq_df = pct_chg_hfq(**kwargs)
    benchmark_daily_return = bench_mark_000905_daily_return(**kwargs)
    pct_chg_hfq_df = sm.add_constant(pct_chg_hfq_df)


    def rolling_slope_regress(pct_chg_hfq, benchmark, window):

        x_name = pct_chg_hfq.columns
        y_name = benchmark.name
        pct_chg_hfq = pct_chg_hfq.droplevel(level='code')
        pct_chg_hfq = pct_chg_hfq.reset_index()
        benchmark = benchmark.reset_index()
        df = pd.merge(pct_chg_hfq, benchmark, how='inner', on='trade_date')
        df = df.dropna()
        df = df.set_index(['trade_date'])
        if df.shape[0] < window:
            params = pd.DataFrame(data=np.NAN, index=df.index, columns=df.columns)
        else:
            model = RollingOLS(endog=df[y_name], exog=df[x_name], window=window).fit()
            params = model.params.copy()
        return params.iloc[:,1]
    fac_value = pct_chg_hfq_df.groupby(level='code').progress_apply(lambda x: rolling_slope_regress(x, benchmark_daily_return, 252))

    # subfunc_factor_value = merge_result(['pct_chg_hfq', 'bench_mark_000905_daily_return'], **kwargs)
    #
    #
    # # perform rolling regression
    # # self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: RollingSlopeRegression(x).fit().params.iloc[:,1].droplevel(1))
    # fac_value = subfunc_factor_value.groupby(level='code').progress_apply(
    #     lambda x: RollingSlopeRegression(x).fit().params.iloc[:, 1]).droplevel(2).swaplevel('trade_date', 'code')
    # index = get_fac_idx(**kwargs)
    # fac_value = align_data_to_index(fac_value, index, fill_method='zero')
    return fac_value

@fac_value_to_sr
def roll_up_sum_10(**kwargs):
    subfunc_factor_value = merge_result([pct_chg_hfq], **kwargs)
    fac_value = subfunc_factor_value['pct_chg_hfq'] * (
                subfunc_factor_value['pct_chg_hfq'] > 0)
    n_roll = 10
    fac_value = fac_value.groupby(level='code').progress_apply(lambda x: x.rolling(n_roll).sum())
    return fac_value

@fac_value_to_sr
def roll_down_sum_10(**kwargs):
    subfunc_factor_value = merge_result([pct_chg_hfq], **kwargs)
    fac_value = subfunc_factor_value['pct_chg_hfq'] * (
                subfunc_factor_value['pct_chg_hfq'] < 0)
    n_roll = 10
    fac_value = fac_value.groupby(level='code').progress_apply(lambda x: x.rolling(n_roll).sum())
    return fac_value


def rsi(**kwargs):
    """
    计算rsi值
    :param kwargs:
    :return:
    """
    subfunc_factor_value = merge_result([roll_up_sum_10, roll_down_sum_10], **kwargs)
    fac_value = subfunc_factor_value['roll_up_sum_10'] / (
                subfunc_factor_value['roll_up_sum_10'] + subfunc_factor_value['roll_down_sum_10'])
    fac_value = fac_value * 100
    fac_value = fac_value.replace(np.inf, 0)
    index = get_fac_idx(**kwargs)
    fac_value = align_data_to_index(fac_value, index, fill_method='zero')
    return fac_value


@memorize
@fac_value_to_sr
def st_flag_name_history(**kwargs):
    # engine = kwargs['read_engine']
    # save_engine = kwargs['save_engine']
    # end_date = kwargs['end_date']
    # start_date = kwargs['start_date']
    # freq = kwargs['freq']
    # sql_api_clf = create_sql_api(engine, save_engine)
    #
    # raw_fac = get_hist_data_4_factor_compute(sql_api_clf, table='name_history_stk', field=['new_name'], index=['trade_date', 'code'], hist_year=-1, end_date=end_date)

    # param = kwargs.copy()
    # param.update({'table': 'name_history_stk', 'field': ['new_name'], 'index': ['trade_date', 'code'], 'hist_year': -1})
    raw_fac = get_hist_data_4_factor_compute(table='name_history_stk', field=['new_name'], index=['trade_date', 'code'], hist_year=-1, **kwargs)
    fac_value = raw_fac['new_name'].map(lambda x: "st" in x.lower() if type(x) is str else False)
    index = get_fac_idx(**kwargs)
    fac_value = align_data_to_index(fac_value, index)
    return fac_value

@memorize
@fac_value_to_sr
def st_flag_netprofit(**kwargs):
    # engine = kwargs['read_engine']
    # save_engine = kwargs['save_engine']
    # end_date = kwargs['end_date']
    # start_date = kwargs['start_date']
    # freq = kwargs['freq']
    # sql_api_clf = create_sql_api(engine, save_engine)
    # raw_fac = get_hist_data_4_factor_compute(sql_api_clf, table='income_stk', field=['net_profit', 'end_date'], index=['trade_date', 'end_date', 'code'], hist_year=-1, end_date=end_date)
    # param = kwargs.copy()
    # param.update({'table': 'income_stk', 'field': ['net_profit', 'end_date'], 'index': ['trade_date', 'end_date', 'code'], 'hist_year': -1})
    raw_fac = get_hist_data_4_factor_compute(table='income_stk', field=['net_profit', 'end_date'], index=['trade_date', 'end_date', 'code'], hist_year=-1, **kwargs)
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
            net_income1 = hist_code_df.net_profit[hist_code_df.end_date == end_date1].min()
            net_income2 = hist_code_df.net_profit[hist_code_df.end_date == end_date2].min()
            if (net_income2 < 0) & (net_income1 < 0):
                st_flags.append(1)
            else:
                st_flags.append(0)
        code_df['st_flag'] = st_flags
        return code_df.set_index(['trade_date'])['st_flag']
    fac_value = raw_fac.groupby(level='code').apply(lambda s: cal_st_flag(s))
    index = get_fac_idx(**kwargs)
    fac_value = align_data_to_index(fac_value, index)
    return fac_value


@memorize
@fac_value_to_sr
def st_flag(**kwargs):
    subfunc_factor_value = merge_result([st_flag_name_history, st_flag_netprofit], **kwargs)
    factor_value = (
                subfunc_factor_value['st_flag_name_history'] + subfunc_factor_value['st_flag_netprofit']).map(
        lambda x: 1 if x > 0 else 0)
    return factor_value


@memorize
@fac_value_to_sr
def total_asset(**kwargs):
    raw_fac = get_hist_data_4_factor_compute(table='balance_stk', field=['total_assets', 'end_date'], index=['trade_date', 'end_date', 'code'], hist_year=-1, **kwargs)
    fac_value = raw_fac
    return fac_value


@memorize
@fac_value_to_sr
def fcff_over_mktcap(*kwargs):
    pass


# class CashOverMketCap(ContinousFactor):
#     """CashOverMketCap = FCFF_top_down/MktCap"""
#
#     def compute(self):
#         self.instantiate_child_factors(FCFF_top_down=FCFF_top_down,
#                                        MktCap=MktCap)
#         self.children_factor_value = self.children_factor_value.reset_index(level='end_date') \
#             .groupby(level='code').apply(lambda x: x.fillna(method='ffill')).dropna(subset=['end_date']) \
#             .set_index(['end_date'], append=True)
#         # import pdb
#         # pdb.set_trace()
#         self.fac_value = self.children_factor_value['FCFF_top_down'] / self.children_factor_value['MktCap']
#
#         return self



@memorize
@fac_value_to_sr
def total_operating_revenue(**kwargs):
    """total opertaing revenue as a continous factor"""
    raw_fac = get_hist_data_4_factor_compute(table='income_stk', field=['total_operating_revenue', 'end_date'], index=['trade_date', 'end_date', 'code'], hist_year=2, **kwargs)
    # index = get_fac_idx(**kwargs)
    # fac_value = align_data_to_index(raw_fac, index, fill_method='no')
    fac_value = raw_fac

    return fac_value


@memorize
@fac_value_to_sr
def operating_tax_surcharges(**kwargs):
    """total opertaing revenue as a continous factor"""
    raw_fac = get_hist_data_4_factor_compute(table='income_stk', field=['Operating_Tax_Surcharges', 'end_date'], index=['trade_date', 'end_date', 'code'], hist_year=2, **kwargs)
    # index = get_fac_idx(**kwargs)
    # fac_value = align_data_to_index(raw_fac, index, fill_method='no')
    fac_value = raw_fac

    return fac_value


@memorize
@fac_value_to_sr
def operating_cost(**kwargs):
    """total opertaing revenue as a continous factor"""
    raw_fac = get_hist_data_4_factor_compute(table='income_stk', field=['operating_cost', 'end_date'], index=['trade_date', 'end_date', 'code'], hist_year=2, **kwargs)
    # index = get_fac_idx(**kwargs)
    # fac_value = align_data_to_index(raw_fac, index, fill_method='no')
    fac_value = raw_fac
    return fac_value


@memorize
@fac_value_to_sr
def sale_expense(**kwargs):
    """total opertaing revenue as a continous factor"""
    raw_fac = get_hist_data_4_factor_compute(table='income_stk', field=['sale_expense', 'end_date'], index=['trade_date', 'end_date', 'code'], hist_year=2, **kwargs)
    # index = get_fac_idx(**kwargs)
    # fac_value = align_data_to_index(raw_fac, index, fill_method='no')
    fac_value = raw_fac

    return fac_value


@memorize
@fac_value_to_sr
def administration_expense(**kwargs):
    """total opertaing revenue as a continous factor"""
    raw_fac = get_hist_data_4_factor_compute(table='income_stk', field=['administration_expense', 'end_date'], index=['trade_date', 'end_date', 'code'], hist_year=2, **kwargs)
    # index = get_fac_idx(**kwargs)
    # fac_value = align_data_to_index(raw_fac, index, fill_method='no')
    fac_value = raw_fac

    return fac_value


@memorize
@fac_value_to_sr
def interest_expense(**kwargs):
    """total opertaing revenue as a continous factor"""
    raw_fac = get_hist_data_4_factor_compute(table='income_stk', field=['interest_expense', 'end_date'], index=['trade_date', 'end_date', 'code'], hist_year=2, **kwargs)
    # index = get_fac_idx(**kwargs)
    # fac_value = align_data_to_index(raw_fac, index, fill_method='no')
    fac_value = raw_fac
    return fac_value

@memorize
@fac_value_to_sr
def commission_expense(**kwargs):
    """total opertaing revenue as a continous factor"""
    raw_fac = get_hist_data_4_factor_compute(table='income_stk', field=['commission_expense', 'end_date'], index=['trade_date', 'end_date', 'code'], hist_year=2, **kwargs)
    # index = get_fac_idx(**kwargs)
    # fac_value = align_data_to_index(raw_fac, index, fill_method='no')
    fac_value = raw_fac

    return fac_value


@memorize
@fac_value_to_sr
def rd_expenses(**kwargs):
    """total opertaing revenue as a continous factor"""
    raw_fac = get_hist_data_4_factor_compute(table='income_stk', field=['rd_expenses', 'end_date'], index=['trade_date', 'end_date', 'code'], hist_year=2, **kwargs)
    # index = get_fac_idx(**kwargs)
    # fac_value = align_data_to_index(raw_fac, index, fill_method='no')
    fac_value = raw_fac

    return fac_value


@memorize
@fac_value_to_sr
def asset_impairment_loss(**kwargs):
    """total opertaing revenue as a continous factor"""
    raw_fac = get_hist_data_4_factor_compute(table='income_stk', field=['asset_impairment_loss', 'end_date'], index=['trade_date', 'end_date', 'code'], hist_year=2, **kwargs)
    # index = get_fac_idx(**kwargs)
    # fac_value = align_data_to_index(raw_fac, index, fill_method='no')
    fac_value = raw_fac

    return fac_value


@memorize
@fac_value_to_sr
def other_earnings(**kwargs):
    """total opertaing revenue as a continous factor"""
    raw_fac = get_hist_data_4_factor_compute(table='income_stk', field=['other_earnings', 'end_date'], index=['trade_date', 'end_date', 'code'], hist_year=2, **kwargs)
    # index = get_fac_idx(**kwargs)
    # # fac_value = align_data_to_index(raw_fac, index, fill_method='no')
    fac_value = raw_fac.astype(float)
    return fac_value

@memorize
@fac_value_to_sr
def income_tax(**kwargs):
    """total opertaing revenue as a continous factor"""
    raw_fac = get_hist_data_4_factor_compute(table='income_stk', field=['income_tax', 'end_date'], index=['trade_date', 'end_date', 'code'], hist_year=2, **kwargs)
    # index = get_fac_idx(**kwargs)
    # # fac_value = align_data_to_index(raw_fac, index, fill_method='no')
    fac_value = raw_fac.astype(float)
    return fac_value


@memorize
@fac_value_to_sr
def total_profit(**kwargs):
    """total opertaing revenue as a continous factor"""
    raw_fac = get_hist_data_4_factor_compute(table='income_stk', field=['total_profit', 'end_date'], index=['trade_date', 'end_date', 'code'], hist_year=2, **kwargs)
    # index = get_fac_idx(**kwargs)
    # # fac_value = align_data_to_index(raw_fac, index, fill_method='no')
    fac_value = raw_fac.astype(float)
    return fac_value

@memorize
@fac_value_to_sr
def intangible_assets_amortization(**kwargs):
    """Amortization as a continous factor"""
    raw_fac = get_hist_data_4_factor_compute(table='cash_flow_stk', field=['intangible_assets_amortization', 'end_date'], index=['trade_date', 'end_date', 'code'], hist_year=2, **kwargs)
    # index = get_fac_idx(**kwargs)
    # # fac_value = align_data_to_index(raw_fac, index, fill_method='no')
    fac_value = raw_fac.astype(float)
    return fac_value

@memorize
@fac_value_to_sr
def fixed_assets_depreciation(**kwargs):
    """Amortization as a continous factor"""
    raw_fac = get_hist_data_4_factor_compute(table='cash_flow_stk', field=['fixed_assets_depreciation', 'end_date'], index=['trade_date', 'end_date', 'code'], hist_year=2, **kwargs)
    fac_value = raw_fac.astype(float)
    return fac_value

@memorize
@fac_value_to_sr
def defferred_expense_amortization(**kwargs):
    """Amortization as a continous factor"""
    raw_fac = get_hist_data_4_factor_compute(table='cash_flow_stk', field=['defferred_expense_amortization', 'end_date'], index=['trade_date', 'end_date', 'code'], hist_year=2, **kwargs)
    fac_value = raw_fac.astype(float)
    return fac_value


@memorize
@fac_value_to_sr
def fix_intan_other_asset_acqui_cash(**kwargs):
    """Amortization as a continous factor"""
    raw_fac = get_hist_data_4_factor_compute(table='cash_flow_stk', field=['fix_intan_other_asset_acqui_cash', 'end_date'], index=['trade_date', 'end_date', 'code'], hist_year=2, **kwargs)
    fac_value = raw_fac.astype(float)
    return fac_value

@memorize
@fac_value_to_sr
def total_current_assets(**kwargs):
    """Amortization as a continous factor"""
    raw_fac = get_hist_data_4_factor_compute(table='balance_stk', field=['total_current_assets', 'end_date'], index=['trade_date', 'end_date', 'code'], hist_year=2, **kwargs)
    fac_value = raw_fac.astype(float)
    return fac_value

@memorize
@fac_value_to_sr
def cash_equivalents(**kwargs):
    """Amortization as a continous factor"""
    raw_fac = get_hist_data_4_factor_compute(table='balance_stk', field=['cash_equivalents', 'end_date'], index=['trade_date', 'end_date', 'code'], hist_year=2, **kwargs)
    fac_value = raw_fac.astype(float)
    return fac_value


@memorize
@fac_value_to_sr
def total_current_liability(**kwargs):
    """Amortization as a continous factor"""
    raw_fac = get_hist_data_4_factor_compute(table='balance_stk', field=['total_current_liability', 'end_date'], index=['trade_date', 'end_date', 'code'], hist_year=2, **kwargs)
    fac_value = raw_fac.astype(float)
    return fac_value


@memorize
@fac_value_to_sr
def shortterm_loan(**kwargs):
    """Amortization as a continous factor"""
    raw_fac = get_hist_data_4_factor_compute(table='balance_stk', field=['shortterm_loan', 'end_date'], index=['trade_date', 'end_date', 'code'], hist_year=2, **kwargs)
    fac_value = raw_fac.astype(float)
    return fac_value


@memorize
@fac_value_to_sr
def non_current_liability_in_one_year(**kwargs):
    """Amortization as a continous factor"""
    raw_fac = get_hist_data_4_factor_compute(table='balance_stk', field=['non_current_liability_in_one_year', 'end_date'], index=['trade_date', 'end_date', 'code'], hist_year=2, **kwargs)
    fac_value = raw_fac.astype(float)
    return fac_value


@memorize
@fac_value_to_sr
def operating_cash(**kwargs):
    """total opertaing revenue as a continous factor"""
    subfunc_factor_value = merge_result([total_current_assets, cash_equivalents, total_current_liability,
                                         shortterm_loan, non_current_liability_in_one_year], **kwargs)
    subfunc_factor_value = subfunc_factor_value.fillna(0)
    # compute NOCF_Over_TORev factor
    fac_value = (subfunc_factor_value['total_current_assets'] -
                      subfunc_factor_value['cash_equivalents']) - \
                     (subfunc_factor_value['total_current_liability'] -
                      subfunc_factor_value['shortterm_loan'] - subfunc_factor_value[
                          'non_current_liability_in_one_year'])
    return fac_value


@memorize
@fac_value_to_sr
def tax_rate(**kwargs):
    """total opertaing revenue as a continous factor"""
    subfunc_factor_value = merge_result([income_tax, total_profit], **kwargs)
    fac_value = subfunc_factor_value['income_tax'] / subfunc_factor_value['total_profit']
    fac_value = fac_value * (fac_value > 0)
    return fac_value

@memorize
@fac_value_to_sr
def ebit(**kwargs):
    """Compute EBIT using top down approach"""
    subfunc_factor_value = merge_result([total_operating_revenue,  operating_tax_surcharges, operating_cost,
                                         sale_expense, administration_expense, interest_expense, commission_expense,
                                         rd_expenses, asset_impairment_loss, other_earnings], **kwargs)

    subfunc_factor_value = subfunc_factor_value.fillna(0)
    subfunc_factor_value = subfunc_factor_value.astype(float)
    # compute NOCF_Over_TORev factor
    fac_value = (subfunc_factor_value['total_operating_revenue'] -
                subfunc_factor_value['operating_tax_surcharges'] -
                      (subfunc_factor_value['operating_cost'] + subfunc_factor_value['sale_expense'] +
                       subfunc_factor_value['administration_expense'] + subfunc_factor_value[
                           'interest_expense'] + subfunc_factor_value['commission_expense'] +
                       subfunc_factor_value['rd_expenses'] + subfunc_factor_value[
                           'asset_impairment_loss']) +
                      subfunc_factor_value['other_earnings'])
    return fac_value


@memorize
@fac_value_to_sr
def fcff_top_down(**kwargs):
    """
        EBIT(1 - TaxRate) + IntangibleAmortization + Depreciation
        + DeferredExpenseAmortization - CapitalExpense
        -(OperatingCash - OperatingCash.shift(1))
        """
    subfunc_factor_value = merge_result([ebit, tax_rate, intangible_assets_amortization, fixed_assets_depreciation,
                                         defferred_expense_amortization, fix_intan_other_asset_acqui_cash, operating_cash], **kwargs)
    subfunc_factor_value = subfunc_factor_value.reset_index()
    subfunc_factor_value['year'] = subfunc_factor_value.end_date.apply(lambda x: x.year)
    year_end_operating_cash = subfunc_factor_value.loc[
        subfunc_factor_value.end_date.apply(lambda x: x.month == 12), ['code', 'end_date', 'operating_cash']]
    year_end_operating_cash['year'] = year_end_operating_cash.end_date.apply(lambda x: x.year + 1)
    subfunc_factor_value = subfunc_factor_value.merge(year_end_operating_cash[['code', 'year', 'operating_cash']],
                                                       on=['code', 'year'],
                                                       how='left',
                                                       suffixes=('', '_last_yr_end')).drop(
        columns='year').set_index(['trade_date', 'end_date', 'code'])

    subfunc_factor_value = subfunc_factor_value.fillna(0)

    fac_value = (subfunc_factor_value['ebit'] * (1 - subfunc_factor_value['tax_rate']) +
                      subfunc_factor_value['intangible_assets_amortization'] + subfunc_factor_value[
                          'fixed_assets_depreciation'] + subfunc_factor_value['defferred_expense_amortization'] -
                      subfunc_factor_value['fix_intan_other_asset_acqui_cash'] - (
                                  subfunc_factor_value['operating_cash'] - subfunc_factor_value[
                              'operating_cash_last_yr_end']))
    return fac_value
#

# class FCFF_top_down(FundamentalFactor):
#     """
#     EBIT(1 - TaxRate) + IntangibleAmortization + Depreciation
#     + DeferredExpenseAmortization - CapitalExpense
#     -(OperatingCash - OperatingCash.shift(1))
#     """
#
#     def compute(self):
#         # set factor index
#
#         # creating and compute all child factors
#
#         self.instantiate_child_factors(EBIT=EBIT,
#                                        TaxRate=TaxRate,
#                                        IntangibleAmortization=IntangibleAmortization,
#                                        Depreciation=Depreciation,
#                                        DeferredExpenseAmortization=DeferredExpenseAmortization,
#                                        CapitalExpense=CapitalExpense,
#                                        OperatingCash=OperatingCash
#                                        )
#         # get last year end operating cash
#         child_factor_df = self.children_factor_value.reset_index()
#         child_factor_df['year'] = child_factor_df.end_date.apply(lambda x: x.year)
#         year_end_operating_cash = child_factor_df.loc[
#             child_factor_df.end_date.apply(lambda x: x.month == 12), ['code', 'end_date', 'OperatingCash']]
#         year_end_operating_cash['year'] = year_end_operating_cash.end_date.apply(lambda x: x.year + 1)
#         self.children_factor_value = child_factor_df.merge(year_end_operating_cash[['code', 'year', 'OperatingCash']],
#                                                            on=['code', 'year'],
#                                                            how='left',
#                                                            suffixes=('', '_last_yr_end')).drop(
#             columns='year').set_index(['trade_date', 'end_date', 'code'])
#
#         self.children_factor_value = self.children_factor_value.fillna(0)
#         # import pdb
#         # pdb.set_trace()
#         self.fac_value = (self.children_factor_value['EBIT'] * (1 - self.children_factor_value['TaxRate']) +
#                           self.children_factor_value['IntangibleAmortization'] + self.children_factor_value[
#                               'Depreciation'] + self.children_factor_value['DeferredExpenseAmortization'] -
#                           self.children_factor_value['CapitalExpense'] - (
#                                       self.children_factor_value['OperatingCash'] - self.children_factor_value[
#                                   'OperatingCash_last_yr_end']))
#
#         return self


class Factor(object):
    def __init__(self, param):
        self.sql_api_clf = create_sql_api(read_engine=param['read_engine'], save_engine=param['save_engine'])
        self.param = param

    def compute(self):
        pass


class StFlag(Factor):
    def __init__(self, param):
        super(StFlag, self).__init__(param)
        self.func = st_flag

    def compute(self):
        print(self.param)
        result = self.func(**self.param)
        return result


class Vol60Days(Factor):
    def __init__(self, param):
        super(Vol60Days, self).__init__(param)
        self.func = volatility_60_days

    def compute(self):
        print(self.param)
        result = self.func(**self.param)
        return result



class MarketBeta252(Factor):
    """CAPM market beta estimated by 252 rolling regression"""

    def __init__(self, param):
        super(MarketBeta252, self).__init__(param)
        self.func = market_beta_252

    def compute(self):
        print(self.param)
        result = self.func(**self.param)
        return result


