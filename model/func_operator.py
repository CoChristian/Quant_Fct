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
import time
import datetime

from statsmodels.distributions.empirical_distribution import ECDF
import SQL_api
import math
from jqdatasdk import *

auth("13764432461", "Nfhq12345")


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
            result = cache[md5_id]

        else:
            result = func(*args, **kwargs)
            cache.update({md5_id: result})
        if type(result) == pd.DataFrame:
            return result.copy()
        else:
            return result

    return wrapper


@memorize
def create_sql_api(read_engine, save_engine):
    sql_api_clf = SQL_api.SQL_API(save_engine=save_engine,
                                  read_engine=read_engine)
    return sql_api_clf


@timer
def save_data_to_table(engine, table, data, if_exists='append'):
    # if if_reset_index:
    #     data = data.reset_index()
    sql_api_clf = create_sql_api(read_engine=engine, save_engine=engine)

    if if_exists == 'append':

        if 'trade_date' not in data.index.names:

            data = data.reset_index()
            if 'index' in data.columns:
                data = data.drop('index', axis=1)
            if 'code' in data.columns and 'trade_date' in data.columns:
                data = data.set_index(['trade_date', 'code'])
            if 'trade_date' in data.columns and "code" not in data.columns:
                data = data.set_index('trade_date')
            if 'trade_date' not in data.columns and "code" in data.columns:
                data = data.set_index('code')

        sql_api_clf.insert_new_data_to(data, table)
    elif if_exists == "delete_and_append":

        sql_api_clf.delete_old_data_and_insert_new_data(data, table)
    else:
        sql_api_clf.save_data(data, table)
    return data


@timer
@memorize
def get_hist_data_4_factor_compute(read_engine, save_engine, table, field=['trade_date', 'code'], name_dict={},
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
    # if "start_date" in field:
    #     raw_fac['trade_date'] =raw_fac['start_date'].map(lambda x: int(x.strftime("%Y%m%d")))
    raw_fac = raw_fac.rename(name_dict, axis=1)

    raw_fac = raw_fac.set_index(index)

    return raw_fac


@timer
def merge_data(**kwargs):
    """
    将输入数据拼接在一起
    :param kwargs:
    :return:
    """
    try:
        df = pd.concat(list(kwargs.values()), axis=1)
        # features = df.columns
        # df.to_pickle(r"D:\PycharmProjects\test_data\{}.pkl".format("_".join(features)))
    except Exception as e:
        print(e)
        import pdb
        pdb.set_trace()
        df = pd.DataFrame()

    return df


@timer
def standard_and_merge_data(**kwargs):
    """
    将输入数据拼接在一起
    :param kwargs:
    :return:
    """
    try:
        factor_index = kwargs.pop('factor_index')

        datas = [align_data_to_index(data=_, index=factor_index) for _ in list(kwargs.values())]
        df = pd.concat(datas, axis=1)
        df = clear_data(df)
        # features = df.columns
        # df.to_pickle(r"D:\PycharmProjects\test_data\{}.pkl".format("_".join(features)))
    except Exception as e:
        print(e)
        import pdb
        pdb.set_trace()
        df = pd.DataFrame()
    return df


@timer
def standard_and_merge_data_daily(**kwargs):
    """
    将输入数据拼接在一起
    :param kwargs:
    :return:
    """
    try:
        factor_index = kwargs.pop('daily_index')

        datas = [align_data_to_index(data=_, index=factor_index) for _ in list(kwargs.values())]
        df = pd.concat(datas, axis=1)
        df = clear_data(df)
        # features = df.columns
        # df.to_pickle(r"D:\PycharmProjects\test_data\{}.pkl".format("_".join(features)))
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

    y = sr
    x = np.arange(1, len(y) + 1, 1)
    y_mean = np.abs(y).mean()
    _, beta = ols(y, x)
    if beta == np.NaN or y_mean == 0.0:
        slope = np.NaN
    else:
        slope = beta / y_mean
    return slope


def trend_regress_4_roe(sr):
    """
    Given a pd.Series or np.array, return trend regression coefficient

    Parameters
    ----------
    sr : pd.Series or 1-d np.array

    Returns
    -------
    params: 1-d np.array of regression params, e.g. (beta0,beta1,beta2...)

    """
    y = sr
    x = np.arange(1, len(y) + 1, 1)
    _, beta = ols(y, x)
    slope = beta
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
        return fac_index, {}
    opt_2_trade = {}
    if freq == "default":
        index_df = pd.DataFrame(index=fac_index)
        add_opt_dates = [20230428, 2023110, 20240913, 20250127]
        # add_opt_dates = [pd.to_datetime(str(_)) for _ in add_opt_dates]
        index_df['value'] = 0

        index_df = index_df.unstack()
        # convert index format d
        # index_df = intDate2Date(index_df.reset_index()).set_index('trade_date')
        index_df = index_df.reset_index().set_index('trade_date')
        # resample time freq
#         end_date = datetime.datetime.strptime(str(end_date), "%Y%m%d") + datetime.timedelta(days=1)
#         end_date = int(end_date.strftime("%Y%m%d"))
        trade_dates = sorted(list(set(index_df.index.values)))
#         trade_dates.append(end_date)

        wednesday_time_index = pd.bdate_range(str(start_date), str(end_date), freq="W-Wed")
        tuesday_time_index = pd.bdate_range(str(start_date), str(end_date), freq="W-Tue")

        wednesday_time_index = [int(_.strftime("%Y%m%d")) for _ in wednesday_time_index]
        tuesday_time_index = [int(_.strftime("%Y%m%d")) for _ in tuesday_time_index]

        resampled_time_index = []
        for date_, next_date_ in zip(trade_dates[:-1], trade_dates[1:]):
            if date_ in tuesday_time_index or date_ in add_opt_dates:
                resampled_time_index.append(date_)
                opt_2_trade.update({date_: next_date_})
            elif next_date_ in wednesday_time_index:
                resampled_time_index.append(date_)
                opt_2_trade.update({date_: next_date_})
            else:
                pass
        if trade_dates[-1] in tuesday_time_index:
            resampled_time_index.append(trade_dates[-1])
            opt_2_trade.update({trade_dates[-1]: None})
        opt_2_trade.update({20240913: 20240918, 20250127: 20250205})
        # resample the index

        index_df = index_df.reindex(resampled_time_index).stack().reset_index().rename(
            columns={'level_0': 'trade_date'})
        # get back to int date format
        # index_df = Date2intDate(index_df).set_index(['trade_date', 'code'])
        index_df = index_df.set_index(['trade_date', 'code'])

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

    return fac_index, opt_2_trade


@timer
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
    fac_index, opt_to_trade = resample(fac_index=raw_fac_index.index, start_date=start_date, end_date=end_date,
                                       freq=freq)
    # print(opt_to_trade)

    return fac_index

@timer
def get_fac_idx_online(start_date, end_date, freq, read_engine, save_engine, data=None):
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
        if end_date is not None:
            end_date = max(data['trade_date'].max(), end_date)
        else:
            end_date = data['trade_date'].max()

    raw_fac_index = get_hist_data_4_factor_compute(read_engine=read_engine,
                                                   save_engine=save_engine,
                                                   start_date=start_date,
                                                   end_date=end_date,
                                                   table="stock_universe", hist_year=0)
    start_date = raw_fac_index.reset_index()['trade_date'].min()
    end_date = raw_fac_index.reset_index()['trade_date'].max()
    fac_index, opt_to_trade = resample(fac_index=raw_fac_index.index, start_date=start_date, end_date=end_date,
                                       freq=freq)
    # print(opt_to_trade)

    return fac_index

def gen_opt_2_trade(start_date, end_date, freq, read_engine, save_engine):
    raw_fac_index = get_hist_data_4_factor_compute(read_engine=read_engine,
                                                   save_engine=save_engine,
                                                   start_date=start_date,
                                                   end_date=end_date,
                                                   table="stock_universe", hist_year=0)
    start_date = raw_fac_index.reset_index()['trade_date'].min()
    end_date = raw_fac_index.reset_index()['trade_date'].max()
    fac_index, opt_to_trade = resample(fac_index=raw_fac_index.index, start_date=start_date, end_date=end_date,
                                       freq=freq)

    opt_2_trade_infos = []
    for opt_date, trade_date in opt_to_trade.items():
        opt_2_trade_infos.append({'opt_date': opt_date, 'trade_date': trade_date})
    opt_to_trade_df = pd.DataFrame(opt_2_trade_infos)

    # opt_to_trade_df = opt_to_trade_df.applymap(lambda x: int(pd.to_datetime(str(x)).strftime("%Y%m%d")))
    return opt_to_trade_df


@timer
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

    if ('end_date' in data.index.names):
        data['end_date'] = data.index.get_level_values('end_date')
        data['end_date_max'] = data.groupby(level='code')['end_date'].cummax()
        data = data[data['end_date'] == data['end_date_max']]
        data = data.drop(['end_date_max', 'end_date'], axis=1)
    try:
        data = drop_extra_level_index(data)
    except Exception as e:
        import pdb
        pdb.set_trace()
    full_fac_index = index.union(data.index)
    data = data.reindex(full_fac_index).sort_index()
    if (fill_method == "ffill"):
        data = data.groupby(level='code',group_keys=False).apply(lambda x: x.fillna(method='ffill'))
    elif fill_method == "zero":
        data = data.groupby(level='code',group_keys=False).apply(lambda x: x.fillna(value=0))
    else:
        data = data.groupby(level='code',group_keys=False).apply(lambda x: x)
    data = data.reindex(index).sort_index()

    return data


def resample_data_to_index(data, index, drop_max=False, fill_method='pad'):
    """
    将数据影视到 index 里的日期
    可以实现功能如下
        index 为daily index ，将非交易日发布的数据映射到交易日
        index 为 weekly index，将 daily的数据映射到 weekly 的日期
    :param data: 输入数据
    :param index: 交易日的index
    :param drop_max: 是否去掉 pub_dates 里大于最大交易日的数据
    :param fill_method: 映射方法，如果为pad，则 pub_date映射到 index里前一个交易日
                        如果为 backfill 则pub_date映射到 index里后一个交易日
    :return:
    """
    # import pdb
    # pdb.set_trace()
    # data = drop_extra_level_index(data)

    index_df = pd.DataFrame([1 for _ in index], index=index)
    valid_trade_dates = index_df.reset_index()['trade_date'].unique()
    data = data.reset_index()
    pub_dates = data['trade_date'].unique()

    max_trade_date = max(valid_trade_dates)
    min_trade_date = min(valid_trade_dates)

    if drop_max:
        pub_dates = [_ for _ in pub_dates if _ <= max_trade_date]
    pub_dates = [_ for _ in pub_dates if _ >= min_trade_date]
    all_dates = list(valid_trade_dates) + list(pub_dates)
    all_dates = list(set(all_dates))
    all_date_df = pd.DataFrame({'date': all_dates})
    all_date_df['new_date'] = all_date_df['date'].map(lambda x: x if x in valid_trade_dates else None)
    all_date_df.sort_values('date', inplace=True)
    all_date_df.fillna(method=fill_method, inplace=True)

    date2newdate = dict(zip(all_date_df.dropna()['date'].values, all_date_df.dropna()['new_date'].values))
    data['trade_date'] = data['trade_date'].map(date2newdate)
    data = data[data['trade_date'].notnull()]
    return data.set_index(['code', 'trade_date'])


def trade_date_to_weight_date(weight_dates, trade_dates):
    weight_dates = sorted(weight_dates)
    trade_dates = sorted(trade_dates)
    trade_date_to_weight_date_info = {}
    for trade_date in trade_dates:
        print("trade_date {}".format(trade_date))
        for first_weight_date, next_weight_date in zip(weight_dates[:-1], weight_dates[1:]):
            print("first weight, {}, second weight ".format(first_weight_date, next_weight_date))
            if trade_date < next_weight_date and trade_date > first_weight_date:
                trade_date_to_weight_date_info.update({trade_date: first_weight_date})
        if (trade_date > weight_dates[-1]):
            trade_date_to_weight_date_info.update({trade_date: weight_dates[-1]})

    return trade_date_to_weight_date_info


def process_index_weight(weight_data, index, weight_name):
    new_weight_df = pd.DataFrame([0 for j in range(len(index))], index=index, columns=['test'])
    new_weight_df = new_weight_df.reset_index()
    all_trade_dates = new_weight_df['trade_date'].unique()
    weight_data = weight_data.reset_index()
    weight_data['weight_date'] = weight_data['trade_date']
    weight_data = weight_data.drop('trade_date', axis=1)
    weight_dates = weight_data['weight_date'].unique()
    trade_date_2_weight_date = trade_date_to_weight_date(weight_dates, all_trade_dates)
    new_weight_df['weight_date'] = new_weight_df['trade_date'].map(trade_date_2_weight_date)
    new_weight_df = pd.merge(new_weight_df, weight_data, how='left', on=['weight_date', 'code'])

    new_weight_df.fillna(0, inplace=True)

    aligned_weight = new_weight_df.set_index(['trade_date', 'code'])[[weight_name]].sort_index()
    return aligned_weight


@timer
def divide_two_variable(first_var_name, second_var_name, output_name, data):
    data[output_name] = data[first_var_name] / data[second_var_name]

    #     data[output_name] = data.apply(lambda x: x[first_var_name]/x[second_var_name] if x[second_var_name] != 0 else 0, axis=1)
    return data[[output_name]]


@timer
def divide_two_variable_4_zero(first_var_name, second_var_name, output_name, data):
    data[[first_var_name, second_var_name]] = data[[first_var_name, second_var_name]].fillna(0)
    data[output_name] = data.apply(lambda x: x[first_var_name] / x[second_var_name] if x[second_var_name] != 0 else 0,
                                   axis=1)
    return data[[output_name]]


@timer
def multiply_two_variable(first_var_name, second_var_name, output_name, data, ):
    data[output_name] = data[first_var_name] * data[second_var_name]
    return data[[output_name]]


@timer
def minus_two_variable(first_var_name, second_var_name, output_name, data, ):
    data[output_name] = data[first_var_name] - data[second_var_name]
    return data[[output_name]]


@timer
def or_two_variable(first_var_name, second_var_name, output_name, data, ):
    print(data.columns)
    data[output_name] = (data[first_var_name] + data[second_var_name]).map(lambda x: 1 if x > 0 else 0)
    return data[[output_name]]


@timer
def cal_reciprocal(value_name, data, output_name):
    data[output_name] = data[value_name].map(lambda x: 1 / x)
    return data[[output_name]]


def update_param(default_param, common_param):
    new_param = {}
    for key, value in default_param.items():
        if key in common_param and value is None:
            new_param.update({key: common_param[key]})
        else:
            new_param.update({key: value})
    return new_param


@timer
def std_mkt_cp(value_name, data, output_name):
    data[output_name] = data[value_name].map(lambda x: x * 1e8)
    return data[[output_name]]


@timer
def cal_market_log(value_name, data, output_name):
    data[output_name] = data[value_name].map(lambda x: np.log(x))
    return data[[output_name]]


@timer
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


@timer
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


@timer
def cal_pre_day_price(value_name, data, output_name):
    # data = data.reset_index()
    data[output_name] = data.sort_index(level=['code', 'trade_date']).groupby(level='code')[value_name].shift(
        1).sort_index(level=['trade_date', 'code'])
    return data[[output_name]]


# @timer
def cal_beta(x_df, y_df, lag):
    x_name = x_df.columns[0]
    y_name = y_df.columns[0]
    # x_df = x_df.droplevel(level='code')
    x_df = x_df.reset_index()
    y_df = y_df.reset_index()

    df = pd.merge(x_df, y_df, how='inner', on='trade_date')
    df = df.dropna()
    df = df.set_index(['trade_date']).sort_index()

    # daily_data = daily_data.sort_values(by=['ts_code', 'trade_date'])
    pct_chg_hfq = df[x_name].to_numpy()  # 个股收益率
    pct_chg_mkt = df[y_name].to_numpy()  # 市场收益率
    beta_ = []
    if len(df) < lag:
        df['beta'] = np.nan
    else:
        #         for i in tqdm(range(lag, len(df) + 1)):
        for i in range(lag, len(df) + 1):
            y = pct_chg_hfq[i - lag: i]
            x = pct_chg_mkt[i - lag: i]
            _, beta_tmp = ols(y, x)
            beta_.append(beta_tmp)
        beta_ = [np.nan] * (lag - 1) + beta_
        df['beta'] = beta_

    return df['beta']


@timer
def cal_stoq(value_name, data, output_name):
    def stoq(stom):
        stom = np.array([stom[0], stom[21], stom[42]])
        tmp = sum(np.exp(stom)) / 3
        result = np.log(tmp)
        return result

    stoq_data = data[value_name].groupby(level='code').progress_apply(
        lambda x: x.sort_index(level='trade_date').rolling(43).apply(stoq))
    stoq_data.name = output_name
    stoq_data = stoq_data.reset_index().set_index(['trade_date', 'code'])
    stoq_data = stoq_data.replace(np.inf, np.nan).replace(-np.inf, np.nan)

    return stoq_data


@timer
def cal_stoa(value_name, data, output_name):
    def stoa(stom):
        index_list = [i * 21 for i in range(12)]
        stom = stom[index_list]
        tmp = sum(np.exp(stom)) / 12
        result = np.log(tmp)
        return result

    # data[output_name] = data[value_name].rolling(232).apply(stoa)
    stoa_data = data[value_name].groupby(level='code').progress_apply(
        lambda x: x.sort_index(level='trade_date').rolling(232).apply(stoa))
    stoa_data.name = output_name
    stoa_data = stoa_data.reset_index().set_index(['trade_date', 'code'])
    stoa_data = stoa_data.replace(np.inf, np.nan).replace(-np.inf, np.nan)

    return stoa_data


@timer
def cal_benchmark_pct_chg(value_name, data, output_name):
    data = data.droplevel('code')
    data = data.sort_index(level='trade_date')
    data[output_name] = (data[value_name] / data[value_name].shift(1) - 1).map(lambda x: x * 100)
    return data[[output_name]]


def cal_benchmark_pct_chg_from_index_weight(code_pct_chg_hfq, index_weight, weight_name):
    #     opt_2_dict = dict(zip(opt2trade_data['opt_date'].values, opt2trade_data['trade_date'].values))
    index_weight = index_weight.reset_index()
    code_pct_chg_hfq = code_pct_chg_hfq.reset_index()
    #     index_weight['trade_date'] = index_weight['trade_date'].map(opt_2_dict)
    all_trade_dates = sorted(code_pct_chg_hfq['trade_date'].unique())
    benchmark_pct_chg_dict = {}
    for trade_date in all_trade_dates:
        hist_index_weight = index_weight[index_weight.trade_date < trade_date]
        if len(hist_index_weight):
            tgt_index_weight = hist_index_weight[hist_index_weight.trade_date == hist_index_weight['trade_date'].max()]
            this_day_code_pct_chg = code_pct_chg_hfq[code_pct_chg_hfq['trade_date'] == trade_date]
            code_pct_chg_weight_data = pd.merge(this_day_code_pct_chg, tgt_index_weight, how='left', on=['code'])
            benchmark_pct_chg = (
                        code_pct_chg_weight_data[weight_name] * code_pct_chg_weight_data['PctChgHfqDaily']).sum()
            benchmark_pct_chg_dict.update({trade_date: benchmark_pct_chg})
    benchmark_pct_chg_s = pd.Series(benchmark_pct_chg_dict)
    benchmark_pct_chg_s.name = "IndexPctChg"
    benchmark_pct_chg_s.index.name = "trade_date"
    return benchmark_pct_chg_s.to_frame()


@timer
def cal_market_beta(code_pct_chg_hfq, benchmark_pct_chg, window_size, output_name):
    # code_pct_chg_hfq = sm.add_constant(code_pct_chg_hfq)
    # codes = code_pct_chg_hfq.reset_index()['code'].unique()
    # code_pct_chg_hfq = code_pct_chg_hfq.reset_index()
    # code_pct_chg_hfq = code_pct_chg_hfq[code_pct_chg_hfq['code'].map(lambda x: x in codes[:100])]
    # code_pct_chg_hfq = code_pct_chg_hfq.set_index(['trade_date', 'code'])

    # beta = code_pct_chg_hfq.groupby(level='code').progress_apply(lambda x: rolling_slope_regress(x.droplevel(level='code'), benchmark_pct_chg, window_size)[0])
    #     import pdb
    #     pdb.set_trace()
    code_pct_chg_hfq = code_pct_chg_hfq[code_pct_chg_hfq.PctChgHfqDaily.notnull()]

    beta = code_pct_chg_hfq.groupby(level='code').progress_apply(
        lambda x: cal_beta(x.droplevel(level='code'), benchmark_pct_chg.applymap(lambda x: x / 100), window_size))

    beta.name = output_name
    beta = beta.reset_index().set_index('trade_date')
    return beta


@timer
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


@timer
def cal_tax_rate(income_tax_name, total_profit_name, output_name, data):
    data[output_name] = data[income_tax_name] / data[total_profit_name]
    # make sure the tax rate is greater than 0
    data[output_name] = data[output_name] * (data[output_name] > 0)
    return data[[output_name]]


@timer
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


@timer
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
    # data.to_pickle(r"D:\PycharmProjects\test_data\new_fcff_detail.pkl")
    return data[[output_name]]


@timer
def cal_st_based_on_hist_name(value_name, data, output_name):
    data = data.reset_index()
    data['trade_date'] = data['start_date'].map(lambda x: int(x.strftime("%Y%m%d")))
    data = data.set_index(['trade_date', 'code'])
    data[output_name] = (data[value_name].map(lambda x: "st" in x.lower() if type(x) is str else False)).map(int)
    return data[[output_name]]


@timer
def transfer_timestamp_to_int(value_name, data, output_name):
    data[output_name] = data[value_name].map(lambda x: int(x.strftime("%Y%m%d")))


@timer
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
        code_df[output_name] = st_flags
        return code_df.set_index(['trade_date'])[[output_name]]

    st_flag = data.groupby(level='code').apply(lambda s: cal_st_flag(s))
    return st_flag


# @timer
# def cal_st_flag_based_on_net_profit_revenue(data, output_name):
#     def cal_st_flag(code_df):
#         code_df = code_df.reset_index()

#         code_df['end_date'] = code_df['end_date'].map(lambda x: int(x.strftime("%Y%m%d")))

#         code_df = code_df.sort_values('trade_date')
#         st_flags = []
#         for trade_date in code_df['trade_date'].values:
#             hist_code_df = code_df[code_df.trade_date <= trade_date]
#             end_date = hist_code_df.end_date.values % 10000
#             hist_code_df = hist_code_df[(end_date == 1231) | (end_date == 930)]
#             if len(hist_code_df) == 0:
#                 st_flags.append(0)
#                 continue
#             end_date1 = list(hist_code_df.end_date)[-1]
#             end_date2 = int(end_date1 / 10000 - 1) * 10000 + 1231
#             net_income1 = hist_code_df["NetProfit"][hist_code_df.end_date == end_date1].min()
#             net_income2 = hist_code_df["NetProfit"][hist_code_df.end_date == end_date2].min()
#             operating_revenue = hist_code_df["TotalOperatingRevenue"][hist_code_df.end_date == end_date1].min()
#             if trade_date < 20200101:
#                 if (net_income2 < 0) & (net_income1 < 0):
#                     st_flags.append(1)
#                 else:
#                     st_flags.append(0)
#             else:
#                 if  (net_income1 < 0) & (operating_revenue < 1e8):
#                     st_flags.append(1)
#                 else:
#                     st_flags.append(0)
#         code_df[output_name] = st_flags
#         return code_df.set_index(['trade_date'])[[output_name]]

#     st_flag = data.groupby(level='code').apply(lambda s: cal_st_flag(s))
#     return st_flag

@timer
def cal_st_flag_based_on_profit_revenue(net_profit, adjusted_profit, net_asset, output_name):
    data = pd.concat([net_profit, adjusted_profit, net_asset], axis=1)

    def cal_st_flag(code_df):
        code_df = code_df.reset_index()
        code_df['year'] = code_df['end_date'].map(lambda x: int(x.strftime("%Y")))

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
            year = hist_code_df['year'].values[-1]
            this_year_data = hist_code_df[hist_code_df.year == year]
            this_year_data = this_year_data.drop_duplicates('end_date', keep='last')
            adjusted_profit = this_year_data['AdjustedProfit'].sum()
            end_date2 = int(end_date1 / 10000 - 1) * 10000 + 1231
            net_income1 = hist_code_df["NetProfit"][hist_code_df.end_date == end_date1].min()
            net_income2 = hist_code_df["NetProfit"][hist_code_df.end_date == end_date2].min()
            total_income1 = hist_code_df["TotalProfit"][hist_code_df.end_date == end_date1].min()
            net_asset = hist_code_df["NetAsset"][hist_code_df.end_date == end_date1].min()
            operating_revenue = hist_code_df["TotalOperatingRevenue"][hist_code_df.end_date == end_date1].min()
            if trade_date < 20200101:
                if (net_income2 < 0) & (net_income1 < 0):
                    st_flags.append(1)
                else:
                    st_flags.append(0)
            elif trade_date < 20240430:
                if (net_income1 < 0) & (operating_revenue < 1e8):
                    st_flags.append(1)
                else:
                    st_flags.append(0)
            else:
                income = min(net_income1, total_income1, adjusted_profit)
                if ((income < 0) & (operating_revenue < 1e8)) | (net_asset < 0):
                    st_flags.append(1)
                else:
                    st_flags.append(0)
        code_df[output_name] = st_flags
        return code_df.set_index(['trade_date'])[[output_name]]

    st_flag = data.groupby(level='code').apply(lambda s: cal_st_flag(s))
    return st_flag


@timer
def cal_end_flag_based_on_delisting_time(data, output_name):
    data = data.reset_index()
    data['end_date'] = data['end_date'].map(lambda x: int(x.strftime("%Y%m%d")))
    data[output_name] = (data['trade_date'] - data['end_date']).map(lambda x: 1 if x > 0 else 0)
    return data.set_index(['trade_date', 'code'])[[output_name]]


@timer
def cal_end_flag_based_on_hist_name(value_name, data, output_name):
    data = data.reset_index()
    data['trade_date'] = data['start_date'].map(lambda x: int(x.strftime("%Y%m%d")))
    data = data.set_index(['trade_date', 'code'])
    data[output_name] = (data[value_name].map(lambda x: x.startswith("退市") or x.endswith("退") if type(x) is str else False)).map(int)
    return data[[output_name]]


@timer
def cal_nan_flag(data, fcff_top_down_name, total_liability_name, total_assets_name, net_operate_cash_flow_name,
                total_operating_revenue_name,  operating_profit_quarterly_name,  total_non_current_liability_name,
                preferred_shares_equity_name, gics_industry_name, output_name):
    finance_data = data[data[gics_industry_name] == "金融"]
    no_finance_data = data[data[gics_industry_name] != "金融"]
    no_finance_data[output_name] = no_finance_data[[fcff_top_down_name, total_liability_name, total_assets_name, net_operate_cash_flow_name,
                total_operating_revenue_name,  operating_profit_quarterly_name,  total_non_current_liability_name,
                preferred_shares_equity_name]].isna().any(axis=1)
    finance_data[output_name] = finance_data[[fcff_top_down_name, total_liability_name, total_assets_name, net_operate_cash_flow_name,
                total_operating_revenue_name,  operating_profit_quarterly_name]].isna().any(axis=1)
    data = pd.concat([finance_data, no_finance_data])
    return data[[output_name]]


@timer
def cal_all_code_quarterly_trend(value_name, data, output_name, hist_quarter_count=3):
    def cal_quarterly_regress(code_df, factor_name):

        code_df = code_df.reset_index()
        code_ = code_df['code'].values[0]

        trade_dates = code_df['trade_date'].unique()
        trade_date_2_trend = {}
        for trade_date in trade_dates:
            tmp_df = code_df[code_df.trade_date <= trade_date].copy()
            tmp_df.sort_values('end_date', inplace=True)
            tmp_df = tmp_df.fillna(method='ffill')
            last_quarter_values = tmp_df[factor_name].values[-hist_quarter_count:]
            if len(last_quarter_values) == hist_quarter_count:
                trend_value =  trend_regress(last_quarter_values)
                trade_date_2_trend.update({trade_date: trend_value})
        code_df['trend'] = code_df['trade_date'].map(trade_date_2_trend)

        return code_df.set_index(['trade_date',  'end_date'])['trend']

    trend = data.groupby(level = 'code').progress_apply(lambda x: cal_quarterly_regress(x, value_name))
    trend.name = output_name
    trend = trend.reset_index().set_index(['trade_date', 'code', 'end_date']).sort_index()
    trend = trend.replace(np.inf, np.nan).replace(-np.inf, np.nan)
    return trend

@timer
def cal_all_code_hist_corr(value_name, data, output_name, hist_week_count=50):
    def cal_corr(sr):
        y = sr.values
        x = np.arange(1, len(y) + 1, 1)
        corr_ = np.corrcoef(x, y)[0,1]
        return corr_
#     hist_corr = data.groupby(level = 'code').progress_apply(lambda x: cal_corr(x, value_name))
    data = data.reset_index()
    hist_corr_infos = []
    for code, code_factor in data.groupby('code'):
        tmp_factor_corr = code_factor.set_index(['trade_date', 'code'])[value_name].rolling(hist_week_count).apply(lambda x: cal_corr(x))
        
        hist_corr_infos.append(tmp_factor_corr)
#     hist_corr = data.groupby('code')[value_name].rolling(hist_week_count).apply(lambda x: cal_corr(x))

    hist_corr = pd.concat(hist_corr_infos)

    hist_corr.name = output_name

    hist_corr = hist_corr.reset_index().set_index(['trade_date', 'code']).sort_index()
    hist_corr = hist_corr.replace(np.inf, np.nan).replace(-np.inf, np.nan)
    return hist_corr
    

# @timer
# def cal_all_code_quarterly_trend_4_roe(value_name, data, output_name):
#     def cal_quarterly_regress_4_roe(code_df, factor_name):
#         code_df = code_df.reset_index()
#         code_ = code_df['code'].values[0]
#         trade_dates = code_df['trade_date'].unique()
#         trade_date_2_trend = {}
# #         trade_date_2_trend_vol_adj = {}
#         for trade_date in trade_dates:
#             tmp_df = code_df[code_df.trade_date <= trade_date].copy()
#             tmp_df.sort_values('end_date', inplace=True)
#             tmp_df = tmp_df.fillna(method='ffill')
#             last_quarter_values = tmp_df[factor_name].values[-3:]
            
#             if len(last_quarter_values) == 3:
#                 trend_value =  trend_regress_4_roe(last_quarter_values)
#                 trade_date_2_trend.update({trade_date: trend_value})
                
#         code_df['trend'] = code_df['trade_date'].map(trade_date_2_trend)
#         return code_df.set_index(['trade_date',  'end_date'])['trend']

#     trend = data.groupby(level = 'code').progress_apply(lambda x: cal_quarterly_regress_4_roe(x, value_name))
#     trend.name = output_name
#     trend = trend.reset_index().set_index(['trade_date', 'code', 'end_date']).sort_index()
#     trend = trend.replace(np.inf, np.nan).replace(-np.inf, np.nan)
#     return trend

@timer
def cal_all_code_quarterly_trend_4_roe(value_name, data, output_name):
    def cal_quarterly_regress_4_roe(code_df, factor_name):
        code_df = code_df.reset_index()
        code_ = code_df['code'].values[0]
        trade_dates = code_df['trade_date'].unique()
        trade_date_2_trend = {}
        trade_date_2_trend_vol_adj = {}
        for trade_date in trade_dates:
            tmp_df = code_df[code_df.trade_date <= trade_date].copy()
            tmp_df.sort_values('end_date', inplace=True)
            tmp_df = tmp_df.fillna(method='ffill')
            last_quarter_values = tmp_df[factor_name].values[-3:]
            
            if len(last_quarter_values) == 3:
                trend_value =  trend_regress_4_roe(last_quarter_values)
                trade_date_2_trend.update({trade_date: trend_value})
                factor_std = np.std(last_quarter_values)/np.abs(last_quarter_values).mean()
                trend_value_vol_adj = trend_value/factor_std
                trade_date_2_trend_vol_adj.update({trade_date: trend_value_vol_adj})
        code_df['trend'] = code_df['trade_date'].map(trade_date_2_trend)
        code_df['trend_vol_adj'] = code_df['trade_date'].map(trade_date_2_trend_vol_adj)
        
        return code_df.set_index(['trade_date',  'end_date'])[['trend', 'trend_vol_adj']]

    trend = data.groupby(level = 'code').progress_apply(lambda x: cal_quarterly_regress_4_roe(x, value_name))
#     trend.name = output_name
    trend.columns = [output_name, "{}VolAdj".format(output_name)]
    
    trend = trend.reset_index().set_index(['trade_date', 'code', 'end_date']).sort_index()
    trend = trend.replace(np.inf, np.nan).replace(-np.inf, np.nan)
    return trend

# @timer
# def cal_all_code_quarter_2_diff(value_name, data, output_name):
#     def cal_quarter_2_diff(code_df, factor_name):
#         """
#         计算特定财报 yoy指标
#         :param code_df: dataframe
#         :param factor_name:  因子名字
#         :return:
#         """
#         code_df = code_df.reset_index()
#         code_df.sort_values(['trade_date', 'end_date'], inplace=True)
#         code_df.fillna(method='ffill', inplace=True)
#         # if "000553" in code_df['code'].values[0]:
#         #     import pdb
#         #     pdb.set_trace()
#         code_df['end_date_'] = code_df['end_date'].map(lambda x: str(x)[:10])
#         code_df['last_year_end_date_'] = code_df['end_date_'].map(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d")).map(
#             lambda x: datetime.datetime(year=x.year-1, month=x.month, day=x.day).strftime("%Y-%m-%d"))

#         end_date_2_factor = dict(zip(code_df['end_date_'].values, code_df[factor_name].values))
#         code_df['last_year_factor'] = code_df['last_year_end_date_'].map(end_date_2_factor)
#         code_df['yoy'] = (code_df[factor_name] - code_df['last_year_factor'])
#         code_df.drop_duplicates('trade_date', keep='last', inplace=True)
#         code_df['last_max_end_date'] = code_df['end_date'].cummax()
#         code_df['is_max'] = code_df['end_date'] == code_df['last_max_end_date']
#         code_df = code_df[code_df['is_max']]
#         return code_df.set_index(['trade_date', 'end_date'])['yoy']
 
#     yoy = data.groupby(level = 'code').progress_apply(lambda x: cal_quarter_2_diff(x, value_name))
#     yoy.name = output_name
#     yoy = yoy.reset_index().set_index(['trade_date', 'code', 'end_date']).sort_index()
#     # yoy = yoy.replace(np.inf, -999)
#     yoy = yoy.replace(np.inf, np.nan).replace(-np.inf, np.nan)
#     yoy = yoy.fillna(value=-999)
#     return yoy

@timer
def cal_all_code_quarter_2_diff(value_name, data, output_name):
    def cal_quarter_2_diff(code_df, factor_name):
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
        code_df['last_year_end_date_'] = code_df['end_date_'].map(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d")).map(
            lambda x: datetime.datetime(year=x.year-1, month=x.month, day=x.day).strftime("%Y-%m-%d"))
        code_df['last_last_year_end_date_'] = code_df['end_date_'].map(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d")).map(
            lambda x: datetime.datetime(year=x.year-2, month=x.month, day=x.day).strftime("%Y-%m-%d"))
        end_date_2_factor = dict(zip(code_df['end_date_'].values, code_df[factor_name].values))
        code_df['last_year_factor'] = code_df['last_year_end_date_'].map(end_date_2_factor)
        code_df['last_last_year_factor'] = code_df['last_last_year_end_date_'].map(end_date_2_factor)
        
        code_df['yoy'] = (code_df[factor_name] - code_df['last_year_factor'])
        code_df['factor_std'] = code_df[[factor_name, 'last_year_factor', 'last_last_year_factor']].apply(lambda x: x.std(), axis=1)

        code_df['yoy_vol_adj'] = code_df['yoy']/code_df['factor_std']
        code_df.drop_duplicates('trade_date', keep='last', inplace=True)
        code_df['end_date_max'] = code_df['end_date'].cummax()
        code_df['is_max'] = code_df['end_date'] == code_df['end_date_max']
        code_df = code_df[code_df['is_max']]
        return code_df.set_index(['trade_date', 'end_date'])[['yoy', 'yoy_vol_adj']]

    yoy = data.groupby(level='code').progress_apply(lambda x: cal_quarter_2_diff(x, value_name))
    #     yoy.name = output_name
    yoy.columns = [output_name, "{}VolAdj".format(output_name)]
    yoy = yoy.reset_index().set_index(['trade_date', 'code', 'end_date']).sort_index()
    # yoy = yoy.replace(np.inf, -999)
    yoy = yoy.replace(np.inf, np.nan).replace(-np.inf, np.nan)
    yoy = yoy.fillna(value=-999)
    return yoy


@timer
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
    # yoy = yoy.replace(np.inf, -999)
    yoy = yoy.replace(np.inf, np.nan).replace(-np.inf, np.nan)
    yoy = yoy.fillna(value=-999)
    return yoy


@timer
def cal_all_code_trend_with_performance_letters(factor_data_quarterly, factor_data_performance_letters, factor_name,
                                                output_name):
    def cal_regress_with_performance_letters(code_df, factor_name):
        code_df = code_df.reset_index()
        code_ = code_df['code'].values[0]

        trade_dates = code_df['trade_date'].unique()
        trade_date_2_trend = {}
        for trade_date in trade_dates:
            tmp_df = code_df[code_df.trade_date <= trade_date].copy()
            tmp_df.sort_values(['end_date', 'trade_date'], inplace=True)

            tmp_df = tmp_df.fillna(method='ffill')
            tmp_df = tmp_df.drop_duplicates('end_date', keep='last')
            last_quarter_values = tmp_df[factor_name].values[-3:]
            if len(last_quarter_values) == 3:
                trend_value = trend_regress(last_quarter_values)
                trade_date_2_trend.update({trade_date: trend_value})
        code_df['trend'] = code_df['trade_date'].map(trade_date_2_trend)
        return code_df.set_index(['trade_date', 'end_date'])['trend']

    factor_data_quarterly['PerformanceLettersTag'] = 0
    factor_data_performance_letters['PerformanceLettersTag'] = 1
    factor_data_performance_letters = factor_data_performance_letters. \
        rename({"{}FromPerformanceLetters".format(factor_name): factor_name,
                "{}FromPerformanceLettersQuarterly".format(factor_name): "{}Quarterly".format(factor_name)}, axis=1)

    factor_quarterly_data = pd.concat([factor_data_quarterly.reset_index(),
                                       factor_data_performance_letters.reset_index()[
                                           ['code', 'end_date', 'trade_date', "{}Quarterly".format(factor_name),
                                            'PerformanceLettersTag']]], axis=0)
    trend_quarterly = factor_quarterly_data.groupby('code').progress_apply(
        lambda x: cal_regress_with_performance_letters(x, "{}Quarterly".format(factor_name)))
    trend_quarterly.name = output_name
    trend_quarterly = trend_quarterly.reset_index().set_index(['trade_date', 'code', 'end_date']).sort_index()
    trend_quarterly = trend_quarterly.replace(np.inf, np.nan).replace(-np.inf, np.nan)
    return trend_quarterly


# @timer
# def cal_all_code_yoy_with_performance_letters(factor_data, factor_data_quarterly, factor_data_performance_letters, factor_name, output_name):
#     def cal_yoy_with_performance_letters(code_df, factor_name):
#         """
#         计算特定财报 yoy指标, 结合业绩快报
#         :param code_df: dataframe
#         :param factor_name:  因子名字
#         :return:
#         """
#         code_df = code_df.reset_index()
#         code_df.sort_values(['trade_date', 'end_date'], inplace=True)
#         code_df.fillna(method='ffill', inplace=True)
#         # if "000553" in code_df['code'].values[0]:
#         #     import pdb
#         #     pdb.set_trace()
#         code_df['end_date_'] = code_df['end_date'].map(lambda x: str(x)[:10])
#         code_df['last_year_end_date_'] = code_df['end_date_'].map(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d")).map(
#             lambda x: datetime.datetime(year=x.year-1, month=x.month, day=x.day).strftime("%Y-%m-%d"))
#         code_df['last_last_year_end_date_'] = code_df['end_date_'].map(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d")).map(
#             lambda x: datetime.datetime(year=x.year-2, month=x.month, day=x.day).strftime("%Y-%m-%d"))
#         valid_code_df = code_df[code_df.PerformanceLettersTag == 0]

#         end_date_2_factor = dict(zip(valid_code_df['end_date_'].values, valid_code_df[factor_name].values))
#         code_df['last_year_factor'] = code_df['last_year_end_date_'].map(end_date_2_factor)
#         code_df['last_last_year_factor'] = code_df['last_last_year_end_date_'].map(end_date_2_factor)

#         code_df['yoy_1'] = (code_df[factor_name] - code_df['last_year_factor']) / code_df['last_year_factor'].map(
#             lambda x: abs(x))
#         code_df['yoy_2'] = (code_df['last_year_factor'] - code_df['last_last_year_factor']) / code_df['last_last_year_factor'].map(
#             lambda x: abs(x))
# #         code_df['yoy'] = code_df['yoy_2'].map(lambda x: x*0.3) + code_df['yoy_1'].map(lambda x: x*0.7)
#         code_df['yoy'] = code_df['yoy_1']
#         code_df.drop_duplicates('trade_date', keep='last', inplace=True)
#         code_df['last_max_end_date'] = code_df['end_date'].cummax()
#         code_df['is_max'] = code_df['end_date'] == code_df['last_max_end_date']
#         code_df = code_df[code_df['is_max']]
#         return code_df.set_index(['trade_date', 'end_date'])['yoy']


#     factor_data_quarterly['PerformanceLettersTag'] = 0
#     factor_data['PerformanceLettersTag'] = 0
#     factor_data_performance_letters['PerformanceLettersTag'] = 1
#     factor_data_performance_letters = factor_data_performance_letters.\
#         rename({"{}FromPerformanceLetters".format(factor_name): factor_name,
#                 "{}FromPerformanceLettersQuarterly".format(factor_name): "{}Quarterly".format(factor_name)}, axis=1)

#     factor_data = pd.concat([factor_data.reset_index(), factor_data_performance_letters.reset_index()[['code', 'end_date', 'trade_date', factor_name, 'PerformanceLettersTag']]], axis=0)

# #     factor_data = pd.concat([factor_data.reset_index(), factor_data_performance_letters[[factor_name, 'PerformanceLettersTag']]], axis=0)

#     yoy = factor_data.groupby('code').progress_apply(lambda x: cal_yoy_with_performance_letters(x, factor_name))

#     yoy.name = output_name
#     yoy = yoy.reset_index().set_index(['trade_date', 'code', 'end_date']).sort_index()
#     # yoy = yoy.replace(np.inf, -999)
#     yoy = yoy.replace(np.inf, np.nan).replace(-np.inf, np.nan)
#     yoy = yoy.fillna(value=-999)

#     factor_quarterly_data = pd.concat([factor_data_quarterly.reset_index(), factor_data_performance_letters.reset_index()[['code', 'end_date', 'trade_date', "{}Quarterly".format(factor_name), 'PerformanceLettersTag']]], axis=0)
#     yoy_quarterly = factor_quarterly_data.groupby('code').progress_apply(lambda x: cal_yoy_with_performance_letters(x, "{}Quarterly".format(factor_name)))
#     yoy_quarterly.name = "{}Quarterly".format(output_name)
#     yoy_quarterly = yoy_quarterly.reset_index().set_index(['trade_date', 'code', 'end_date']).sort_index()
#     # yoy = yoy.replace(np.inf, -999)
#     yoy_quarterly = yoy_quarterly.replace(np.inf, np.nan).replace(-np.inf, np.nan)
#     yoy_quarterly = yoy_quarterly.fillna(value=-999)

#     return yoy, yoy_quarterly

def transfer_trade_date_2_financial_report_season(trade_date):
    trade_date = str(trade_date)
    year = int(trade_date[:4])
    month = trade_date[4:6]
    if month in ["01", '02', '03']:
        season1 = datetime.date(year=year - 1, month=9, day=30).strftime("%Y%m%d")
        season2 = datetime.date(year=year - 1, month=12, day=31).strftime("%Y%m%d")
        season3 = None
    elif month in ['04']:
        season1 = datetime.date(year=year - 1, month=9, day=30).strftime("%Y%m%d")
        season2 = datetime.date(year=year - 1, month=12, day=31).strftime("%Y%m%d")
        season3 = datetime.date(year=year, month=3, day=31).strftime("%Y%m%d")
    elif month in ["05", '06']:
        season1 = datetime.date(year=year, month=3, day=31).strftime("%Y%m%d")
        season2 = None
        season3 = None
    elif month in ['07', '08']:
        season1 = datetime.date(year=year, month=3, day=31).strftime("%Y%m%d")
        season2 = datetime.date(year=year, month=6, day=30).strftime("%Y%m%d")
        season3 = None
    elif month in ['09']:
        season1 = datetime.date(year=year, month=6, day=30).strftime("%Y%m%d")
        season2 = None
        season3 = None
    elif month in ['10']:
        season1 = datetime.date(year=year, month=6, day=30).strftime("%Y%m%d")
        season2 = datetime.date(year=year, month=9, day=30).strftime("%Y%m%d")
        season3 = None
    else:
        season1 = datetime.date(year=year, month=9, day=30).strftime("%Y%m%d")
        season2 = None
        season3 = None
    return season1, season2, season3


def transfer_end_date_2_trade_dates(trade_dates):
    trade_dates = sorted(trade_dates)
    end_date_to_season1_trade_date = {}
    end_date_to_season2_trade_date = {}
    end_date_to_season3_trade_date = {}
    for trade_date in trade_dates:
        end_date_season1, end_date_season2, end_date_season3 = transfer_trade_date_2_financial_report_season(trade_date)
        if end_date_season1:
            if end_date_season1 in end_date_to_season1_trade_date:
                end_date_to_season1_trade_date[end_date_season1].append(trade_date)
            else:
                end_date_to_season1_trade_date.update({end_date_season1: [trade_date]})
        if end_date_season2:
            if end_date_season2 in end_date_to_season2_trade_date:
                end_date_to_season2_trade_date[end_date_season2].append(trade_date)
            else:
                end_date_to_season2_trade_date.update({end_date_season2: [trade_date]})
        if end_date_season3:
            if end_date_season3 in end_date_to_season3_trade_date:
                end_date_to_season3_trade_date[end_date_season3].append(trade_date)
            else:
                end_date_to_season3_trade_date.update({end_date_season3: [trade_date]})
    return end_date_to_season1_trade_date, end_date_to_season2_trade_date, end_date_to_season3_trade_date


# @timer
# def align_data_2_index_quarterly(data, index, nonevalue_percentile, name):
#     factors = data.columns
#     unique_index_names = data.index.names
#     data = data.reset_index()
#     data = data.drop_duplicates(subset=unique_index_names)
#     data.to_pickle("{}_daily.pkl".format(name))
#     index_df = pd.DataFrame([list(_) for _ in index], columns=['trade_date', 'code'])
#     trade_dates = sorted(index_df['trade_date'].unique())
#     end_date_to_season1_trade_date, end_date_to_season2_trade_date, end_date_to_season3_trade_date = transfer_end_date_2_trade_dates(trade_dates)
#     data = data.sort_values(['code', 'end_date', 'trade_date'])
#     data['end_date'] = data['end_date'].map(lambda x: x.strftime("%Y%m%d"))
#     def transfer_pub_date_2_trade_date(pub_dates, trade_dates):
#         pub_date_count = len(pub_dates)
#         pub_date_2_trade_date = {}
#         if len(trade_dates):
#             for idx, pub_date in enumerate(pub_dates):
#                 if idx < pub_date_count-1:
#                     next_pub_date = pub_dates[idx+1]
#                     tmp_trade_dates = [_ for _ in trade_dates if _ >=pub_date and _ < next_pub_date]
#                     pub_date_2_trade_date.update({pub_date: tmp_trade_dates})
#                 else:
#                     tmp_trade_dates = [_ for _ in trade_dates if _ >=pub_date]
#                 pub_date_2_trade_date.update({pub_date: tmp_trade_dates})
#         return pub_date_2_trade_date
#     def align_data_2_index_multi_season(season_data, trade_dates, factors, code):
#         pub_dates = season_data['trade_date'].values
#         pub_date_2_factor = {}
#         season_data = season_data.set_index('trade_date')
#         for pub_date in pub_dates:
#             tmp_factor_info = season_data.loc[pub_date][factors].to_dict()
#             tmp_factor_info.update({'code': code})
#             pub_date_2_factor.update({pub_date: tmp_factor_info})
#         pub_date_2_trade_date = transfer_pub_date_2_trade_date(pub_dates, trade_dates)
#         index_factor_infos = []
#         for pub_date, trade_dates in pub_date_2_trade_date.items():
#             factor_info = pub_date_2_factor[pub_date]
#             for trade_date in trade_dates:
#                 factor_info_ = factor_info.copy()
#                 factor_info_.update({'trade_date': trade_date})
#                 index_factor_infos.append(factor_info_)
#         return index_factor_infos
#     index_factor_infos_4_season1 = []
#     index_factor_infos_4_season2 = []
#     index_factor_infos_4_season3 = []

#     for (code, end_date), tmp_season_data in data.groupby(['code', 'end_date']):
#         trade_dates_4_season1 = end_date_to_season1_trade_date.get(end_date, [])
#         trade_dates_4_season2 = end_date_to_season2_trade_date.get(end_date, [])
#         trade_dates_4_season3 = end_date_to_season3_trade_date.get(end_date, [])
#         tmp_index_factor_infos_4_season1 = align_data_2_index_multi_season(tmp_season_data, trade_dates_4_season1, factors, code)
#         tmp_index_factor_infos_4_season2 = align_data_2_index_multi_season(tmp_season_data, trade_dates_4_season2, factors, code)
#         tmp_index_factor_infos_4_season3 = align_data_2_index_multi_season(tmp_season_data, trade_dates_4_season3, factors, code)
#         index_factor_infos_4_season1.extend(tmp_index_factor_infos_4_season1)
#         index_factor_infos_4_season2.extend(tmp_index_factor_infos_4_season2)
#         index_factor_infos_4_season3.extend(tmp_index_factor_infos_4_season3)
#     index_factor_df_4_season1 = pd.DataFrame(index_factor_infos_4_season1).set_index(['trade_date', 'code'])
#     index_factor_df_4_season2 = pd.DataFrame(index_factor_infos_4_season2).set_index(['trade_date', 'code'])
#     index_factor_df_4_season3 = pd.DataFrame(index_factor_infos_4_season3).set_index(['trade_date', 'code'])
#     index_factor_df_4_season1 = index_factor_df_4_season1.reindex(index).reset_index()
#     index_factor_df_4_season2 = index_factor_df_4_season2.reindex(index).reset_index()
#     index_factor_df_4_season3 = index_factor_df_4_season3.reindex(index).reset_index()
#     index_factor_df_4_season1 = index_factor_df_4_season1.fillna(-999)
#     index_factor_df_4_season2 = index_factor_df_4_season2.fillna(-999)
#     index_factor_df_4_season3 = index_factor_df_4_season3.fillna(-999)

#     index_factor_df_4_season1.to_pickle("{}_season1.pkl".format(name))
#     index_factor_df_4_season2.to_pickle("{}_season2.pkl".format(name))
#     index_factor_df_4_season3.to_pickle("{}_season3.pkl".format(name))

#     def process_non_value(data, factor, percentile):
#         valid_data = data[data[factor].notnull() & data[factor] != -999]
#         if len(valid_data) > 10:
#             nonevalue = valid_data[factor].quantile(percentile)
#         else:
#             nonevalue = 0
#         data[factor] = data[factor].fillna(nonevalue)
#         data[factor] = data[factor].replace(-999, nonevalue)
#         return data
#     merged_factor_infos = []
#     for trade_date in tqdm(trade_dates, desc='Processing'):
#         tgt_season1_df = index_factor_df_4_season1[index_factor_df_4_season1.trade_date == trade_date]
#         tgt_season2_df = index_factor_df_4_season2[index_factor_df_4_season2.trade_date == trade_date]
#         tgt_season3_df = index_factor_df_4_season3[index_factor_df_4_season3.trade_date == trade_date]
#         season2_valid_count = tgt_season2_df[factors].applymap(lambda x: x!= -999).sum().sum()
#         season2_valid_rate = season2_valid_count/(len(tgt_season2_df)*len(factors))
#         season3_valid_count = tgt_season3_df[factors].applymap(lambda x: x!= -999).sum().sum()
#         season3_valid_rate = season3_valid_count/(len(tgt_season3_df)*len(factors))
#         for factor in factors:
#             try:
#                 tgt_season1_df = process_non_value(tgt_season1_df, factor, nonevalue_percentile)
#                 tgt_season2_df = process_non_value(tgt_season2_df, factor, nonevalue_percentile)
#                 tgt_season3_df = process_non_value(tgt_season3_df, factor, nonevalue_percentile)
#             except Exception as e:
#                 import pdb
#                 pdb.set_trace()


#         merged_factor = tgt_season1_df.set_index(['trade_date', 'code'])[factors] + tgt_season2_df.set_index(['trade_date', 'code'])[factors].applymap(lambda x: x*1.5*season2_valid_rate) + tgt_season3_df.set_index(['trade_date', 'code'])[factors].applymap(lambda x: x*2*season3_valid_rate)
#         merged_factor_infos.append(merged_factor)
#     all_merged_factor = pd.concat(merged_factor_infos)
#     return all_merged_factor


def cal_factor_ttm(data, factor_name):
    if "end_date" not in data.columns:
        data = data.reset_index()
    data = data.sort_values(['code', 'trade_date', 'end_date'])
    data['quarter'] = data['end_date'].map(lambda x: "{}Q{}".format(x.strftime("%Y"), int(int(x.strftime("%m")) / 3)))
    ttm_factor_infos = []
    for code, tmp_data in tqdm(data.groupby('code')):
        #         quarter_2_factor = dict(zip(tmp_data['quarter'].values, tmp_data[factor_name]))
        #         quarter_2_factor_ttm = {}
        #         for quarter, factor_value in quater_2_factor.items():
        unique_trade_dates = sorted(tmp_data['trade_date'].unique())
        trade_date_2_factor_ttm = {}
        for trade_date in unique_trade_dates:
            hist_factor_data = tmp_data[tmp_data.trade_date <= trade_date].drop_duplicates('quarter', keep='last')
            max_qurter = hist_factor_data['quarter'].max()
            #             print(max_qurter)
            last_year_quarter = "{}{}".format(int(max_qurter[:4]) - 1, max_qurter[4:])
            last_year_q4 = "{}Q4".format(int(max_qurter[:4]) - 1)
            if last_year_quarter in hist_factor_data['quarter'].values:
                last_year_quarter_factor = \
                hist_factor_data[hist_factor_data.quarter == last_year_quarter][factor_name].values[-1]
            else:
                last_year_quarter_factor = None
            if last_year_q4 in hist_factor_data['quarter'].values:
                last_year_q4_factor = hist_factor_data[hist_factor_data.quarter == last_year_q4][factor_name].values[-1]
            else:
                last_year_q4_factor = None

            max_quarter_factor = hist_factor_data[hist_factor_data.quarter == max_qurter][factor_name].values[-1]
            if last_year_q4_factor is not None and last_year_quarter_factor is not None and max_quarter_factor is not None:
                factor_ttm = last_year_q4_factor - last_year_quarter_factor + max_quarter_factor
            else:
                factor_ttm = None
            trade_date_2_factor_ttm.update({trade_date: factor_ttm})
        tmp_data['{}TTM'.format(factor_name)] = tmp_data['trade_date'].map(trade_date_2_factor_ttm)

        ttm_factor_infos.append(tmp_data[['code', 'trade_date', 'end_date', '{}TTM'.format(factor_name)]])
    all_data = pd.concat(ttm_factor_infos)
    return all_data.set_index(['trade_date', 'end_date', 'code'])[['{}TTM'.format(factor_name)]]


def align_data_2_index_merge(data, index, name, merge_rate):
    factors = data.columns
    unique_index_names = data.index.names
    data = data.reset_index()
    data = data.drop_duplicates(subset=unique_index_names)
    data.to_pickle("{}_daily.pkl".format(name))
    index_df = pd.DataFrame([list(_) for _ in index], columns=['trade_date', 'code'])
    trade_dates = sorted(index_df['trade_date'].unique())
    end_date_to_season1_trade_date, end_date_to_season2_trade_date, end_date_to_season3_trade_date = transfer_end_date_2_trade_dates(
        trade_dates)
    data = data.sort_values(['code', 'end_date', 'trade_date'])
    data['end_date'] = data['end_date'].map(lambda x: x.strftime("%Y%m%d"))

    def transfer_pub_date_2_trade_date(pub_dates, trade_dates):
        pub_date_count = len(pub_dates)
        pub_date_2_trade_date = {}
        if len(trade_dates):
            for idx, pub_date in enumerate(pub_dates):
                if idx < pub_date_count - 1:
                    next_pub_date = pub_dates[idx + 1]
                    tmp_trade_dates = [_ for _ in trade_dates if _ >= pub_date and _ < next_pub_date]
                    pub_date_2_trade_date.update({pub_date: tmp_trade_dates})
                else:
                    tmp_trade_dates = [_ for _ in trade_dates if _ >= pub_date]
                pub_date_2_trade_date.update({pub_date: tmp_trade_dates})
        return pub_date_2_trade_date

    def align_data_2_index_multi_season(season_data, trade_dates, factors, code):
        pub_dates = season_data['trade_date'].values
        pub_date_2_factor = {}
        season_data = season_data.set_index('trade_date')
        for pub_date in pub_dates:
            tmp_factor_info = season_data.loc[pub_date][factors].to_dict()
            tmp_factor_info.update({'code': code})
            pub_date_2_factor.update({pub_date: tmp_factor_info})
        pub_date_2_trade_date = transfer_pub_date_2_trade_date(pub_dates, trade_dates)
        index_factor_infos = []
        for pub_date, trade_dates in pub_date_2_trade_date.items():
            factor_info = pub_date_2_factor[pub_date]
            for trade_date in trade_dates:
                factor_info_ = factor_info.copy()
                factor_info_.update({'trade_date': trade_date})
                index_factor_infos.append(factor_info_)
        return index_factor_infos

    index_factor_infos_4_season1 = []
    index_factor_infos_4_season2 = []
    index_factor_infos_4_season3 = []

    for (code, end_date), tmp_season_data in data.groupby(['code', 'end_date']):
        trade_dates_4_season1 = end_date_to_season1_trade_date.get(end_date, [])
        trade_dates_4_season2 = end_date_to_season2_trade_date.get(end_date, [])
        trade_dates_4_season3 = end_date_to_season3_trade_date.get(end_date, [])
        tmp_index_factor_infos_4_season1 = align_data_2_index_multi_season(tmp_season_data, trade_dates_4_season1,
                                                                           factors, code)
        tmp_index_factor_infos_4_season2 = align_data_2_index_multi_season(tmp_season_data, trade_dates_4_season2,
                                                                           factors, code)
        tmp_index_factor_infos_4_season3 = align_data_2_index_multi_season(tmp_season_data, trade_dates_4_season3,
                                                                           factors, code)
        index_factor_infos_4_season1.extend(tmp_index_factor_infos_4_season1)
        index_factor_infos_4_season2.extend(tmp_index_factor_infos_4_season2)
        index_factor_infos_4_season3.extend(tmp_index_factor_infos_4_season3)
    index_factor_df_4_season1 = pd.DataFrame(index_factor_infos_4_season1).set_index(['trade_date', 'code'])
    index_factor_df_4_season2 = pd.DataFrame(index_factor_infos_4_season2).set_index(['trade_date', 'code'])
    index_factor_df_4_season3 = pd.DataFrame(index_factor_infos_4_season3).set_index(['trade_date', 'code'])
    index_factor_df_4_season1 = index_factor_df_4_season1.reindex(index).reset_index()
    index_factor_df_4_season2 = index_factor_df_4_season2.reindex(index).reset_index()
    index_factor_df_4_season3 = index_factor_df_4_season3.reindex(index).reset_index()
    #     index_factor_df_4_season1 = index_factor_df_4_season1.fillna(-999)
    #     index_factor_df_4_season2 = index_factor_df_4_season2.fillna(-999)
    #     index_factor_df_4_season3 = index_factor_df_4_season3.fillna(-999)

    index_factor_df_4_season1.to_pickle("{}_season1.pkl".format(name))
    index_factor_df_4_season2.to_pickle("{}_season2.pkl".format(name))
    index_factor_df_4_season3.to_pickle("{}_season3.pkl".format(name))

    merged_factor_infos = []
    for trade_date in tqdm(trade_dates, desc='Processing'):
        tgt_season1_df = \
        index_factor_df_4_season1[index_factor_df_4_season1.trade_date == trade_date].set_index(['trade_date', 'code'])[
            factors]
        tgt_season2_df = \
        index_factor_df_4_season2[index_factor_df_4_season2.trade_date == trade_date].set_index(['trade_date', 'code'])[
            factors]
        tgt_season3_df = \
        index_factor_df_4_season3[index_factor_df_4_season3.trade_date == trade_date].set_index(['trade_date', 'code'])[
            factors]
        #         season3_valid_count = tgt_season3_df[factors].applymap(lambda x: x!= -999).sum().sum()
        season3_valid_count = tgt_season3_df[factors].notnull().sum().sum()
        if season3_valid_count > 0:
            data_4_merge = [tgt_season1_df, tgt_season2_df, tgt_season3_df]
        else:
            data_4_merge = [tgt_season1_df, tgt_season2_df]
        merge_factor_datas = []
        for factor in factors:
            factor_infos = [_[factor] for _ in data_4_merge]
            factor_info_df = pd.concat(factor_infos, axis=1)
            #             factor_info_df = factor_info_df.replace(-999, np.nan)
            factor_info_df = factor_info_df.fillna(method='pad', axis=1)
            season_count = factor_info_df.shape[1]
            if season_count == 3:
                valid_factor_info_df = factor_info_df.iloc[:, 1:]
            else:
                valid_factor_info_df = factor_info_df
            valid_factor_info_df.columns = ['first_season', 'second_season']
            valid_factor_info_df[factor] = valid_factor_info_df['first_season'] * merge_rate + valid_factor_info_df[
                'second_season'] * (1 - merge_rate)
            merge_factor_datas.append(valid_factor_info_df[factor])
        merged_factor_infos.append(pd.concat(merge_factor_datas, axis=1))
    all_merged_factor = pd.concat(merged_factor_infos)
    all_merged_factor = all_merged_factor.reset_index()
    all_merged_factor = all_merged_factor.sort_values(['code', 'trade_date'])
    all_merged_factor[factors] = all_merged_factor.groupby('code')[factors].fillna(method='pad')
    all_merged_factor[factors] = all_merged_factor.groupby('code')[factors].fillna(-999)
    return all_merged_factor


@timer
def cal_all_code_yoy_with_performance_letters(factor_data, factor_data_quarterly, factor_data_performance_letters,
                                              factor_name, output_name):
    def cal_yoy_with_performance_letters(code_df, factor_name):
        """
        计算特定财报 yoy指标, 结合业绩快报
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
        code_df['last_last_year_end_date_'] = code_df['end_date_'].map(
            lambda x: datetime.datetime.strptime(x, "%Y-%m-%d")).map(
            lambda x: datetime.datetime(year=x.year - 2, month=x.month, day=x.day).strftime("%Y-%m-%d"))
        valid_code_df = code_df[code_df.PerformanceLettersTag == 0]

        end_date_2_factor = dict(zip(valid_code_df['end_date_'].values, valid_code_df[factor_name].values))
        code_df['last_year_factor'] = code_df['last_year_end_date_'].map(end_date_2_factor)
        code_df['last_last_year_factor'] = code_df['last_last_year_end_date_'].map(end_date_2_factor)

        code_df['yoy_1'] = (code_df[factor_name] - code_df['last_year_factor']) / code_df['last_year_factor'].map(
            lambda x: abs(x))
        code_df['yoy_2'] = (code_df['last_year_factor'] - code_df['last_last_year_factor']) / code_df[
            'last_last_year_factor'].map(
            lambda x: abs(x))
        #         code_df['yoy'] = code_df['yoy_2'].map(lambda x: x*0.3) + code_df['yoy_1'].map(lambda x: x*0.7)
        #         code_df['yoy'] = code_df['yoy_1']

        code_df.drop_duplicates('trade_date', keep='last', inplace=True)
        code_df['last_max_end_date'] = code_df['end_date'].cummax()
        code_df['is_max'] = code_df['end_date'] == code_df['last_max_end_date']
        code_df = code_df[code_df['is_max']]
        return code_df.set_index(['trade_date', 'end_date'])[['yoy_1', 'yoy_2']]

    factor_data_quarterly['PerformanceLettersTag'] = 0
    factor_data['PerformanceLettersTag'] = 0
    factor_data_performance_letters['PerformanceLettersTag'] = 1
    factor_data_performance_letters = factor_data_performance_letters. \
        rename({"{}FromPerformanceLetters".format(factor_name): factor_name,
                "{}FromPerformanceLettersQuarterly".format(factor_name): "{}Quarterly".format(factor_name)}, axis=1)

    factor_data = pd.concat([factor_data.reset_index(), factor_data_performance_letters.reset_index()[
        ['code', 'end_date', 'trade_date', factor_name, 'PerformanceLettersTag']]], axis=0)

    #     factor_data = pd.concat([factor_data.reset_index(), factor_data_performance_letters[[factor_name, 'PerformanceLettersTag']]], axis=0)

    yoy = factor_data.groupby('code').progress_apply(lambda x: cal_yoy_with_performance_letters(x, factor_name))

    #     yoy.name = output_name
    yoy.columns = ["{}1".format(output_name), "{}2".format(output_name)]
    yoy = yoy.reset_index().set_index(['trade_date', 'code', 'end_date']).sort_index()
    # yoy = yoy.replace(np.inf, -999)
    yoy = yoy.replace(np.inf, np.nan).replace(-np.inf, np.nan)
    yoy = yoy.fillna(value=-999)

    factor_quarterly_data = pd.concat([factor_data_quarterly.reset_index(),
                                       factor_data_performance_letters.reset_index()[
                                           ['code', 'end_date', 'trade_date', "{}Quarterly".format(factor_name),
                                            'PerformanceLettersTag']]], axis=0)
    yoy_quarterly = factor_quarterly_data.groupby('code').progress_apply(
        lambda x: cal_yoy_with_performance_letters(x, "{}Quarterly".format(factor_name)))
    #     yoy_quarterly.name = "{}Quarterly".format(output_name)
    yoy_quarterly.columns = ["{}Quarterly1".format(output_name), "{}Quarterly2".format(output_name)]
    yoy_quarterly = yoy_quarterly.reset_index().set_index(['trade_date', 'code', 'end_date']).sort_index()
    # yoy = yoy.replace(np.inf, -999)
    yoy_quarterly = yoy_quarterly.replace(np.inf, np.nan).replace(-np.inf, np.nan)

    yoy_quarterly = yoy_quarterly.fillna(value=-999)
    #     import pdb
    #     pdb.set_trace()
    return yoy, yoy_quarterly


@timer
def cal_factor_from_performance_letter_quarterly(factor_data, factor_data_from_performance_letter, factor_name):
    factor_data = factor_data.reset_index()
    factor_data_from_performance_letter = factor_data_from_performance_letter.reset_index()
    factor_data = factor_data.sort_values(['code', 'end_date', 'trade_date'])
    factor_data = factor_data.drop_duplicates(['code', 'end_date'])
    factor_data['end_date'] = factor_data['end_date'].map(lambda x: x.strftime("%Y%m%d"))
    factor_info = dict(zip(factor_data['end_date'].values, factor_data[factor_name].values))

    def get_last_quarter(end_date):
        end_date = end_date.strftime("%Y%m%d")
        quarter = end_date[4:]
        if quarter == "0331":
            return None
        elif quarter == "0630":
            return "{}0331".format(end_date[:4])
        elif quarter == "0930":
            return "{}0630".format(end_date[:4])
        else:
            return "{}0930".format(end_date[:4])

    factor_data_from_performance_letter['pre_quarter'] = factor_data_from_performance_letter['end_date'].map(lambda x: get_last_quarter(x))
    factor_data_from_performance_letter = pd.merge(factor_data_from_performance_letter, factor_data, how='left', left_on=['code', 'pre_quarter'], right_on=['code', 'end_date'])
#     factor_data_from_performance_letter['pre_quarter_factor'] = factor_data_from_performance_letter['pre_quarter'].map(factor_info)

#     factor_data_from_performance_letter['pre_quarter_factor'] = factor_data_from_performance_letter['pre_quarter_factor'].fillna(0)
#     factor_data_from_performance_letter[factor_name] = factor_data_from_performance_letter[factor_name].fillna(0)
    factor_data_from_performance_letter['{}FromPerformanceLetters'.format(factor_name)] = factor_data_from_performance_letter['{}FromPerformanceLetters'.format(factor_name)].replace({0: np.NaN})
    factor_data_from_performance_letter['{}FromPerformanceLettersQuarterly'.format(factor_name)] = factor_data_from_performance_letter['{}FromPerformanceLetters'.format(factor_name)] - factor_data_from_performance_letter[factor_name]
    factor_data_from_performance_letter = factor_data_from_performance_letter.rename({"end_date_x": "end_date", 'trade_date_x': "trade_date"}, axis=1)
 
    return factor_data_from_performance_letter.set_index(['code', 'end_date', 'trade_date'])[['{}FromPerformanceLetters'.format(factor_name),
                                                '{}FromPerformanceLettersQuarterly'.format(factor_name)]]

def process_operationg_revenue_from_performance_letters(performance_letters_data):
    performance_letters_data[['total_operating_revenue', 'operating_revenue']] = performance_letters_data[['total_operating_revenue', 'operating_revenue']].fillna(0)
    performance_letters_data['TotalOperatingRevenueFromPerformanceLetters'] = performance_letters_data[['total_operating_revenue', 'operating_revenue']].apply(lambda x: x.max(), axis=1)
    
    return performance_letters_data[['TotalOperatingRevenueFromPerformanceLetters']]
    

@timer
def cal_blev(book_value_name, preferred_shares_equity_name, total_non_current_liability_name, industry_name, debt_over_assets_name, data, output_name):
    finance_data = data[data[industry_name] == "金融"]
    no_finance_data = data[data[industry_name] != "金融"]
    no_finance_data[output_name] = (no_finance_data[book_value_name] + no_finance_data[
        preferred_shares_equity_name] + no_finance_data[total_non_current_liability_name]) / (no_finance_data[book_value_name])
    finance_data[output_name] = finance_data[debt_over_assets_name]
    data = pd.concat([finance_data, no_finance_data])
    return data[[output_name]]


@timer
def cal_mlev(market_cap_name, preferred_shares_equity_name, total_non_current_liability_name, industry_name, debt_over_assets_name, data, output_name):
    finance_data = data[data[industry_name] == "金融"]
    no_finance_data = data[data[industry_name] != "金融"]
    no_finance_data[output_name] = (no_finance_data[market_cap_name] + no_finance_data[
        preferred_shares_equity_name] + no_finance_data[total_non_current_liability_name]) / (no_finance_data[market_cap_name])
    finance_data[output_name] = finance_data[debt_over_assets_name]
    data = pd.concat([finance_data, no_finance_data])
    return data[[output_name]]


@timer
def cal_cash_dividend_last_year(data, capital_data, output_name):
    code_cash_dividend_infos = []
    data = data.reset_index()
    capital_data = capital_data.reset_index()
    pub_dates = data['trade_date'].unique()
    trade_dates = sorted(capital_data['trade_date'].unique())
    pub_dates = [_ for _ in pub_dates if _ >= min(trade_dates)]
    pub_date_2_trade_date = {}
    for pub_date in pub_dates:
        if pub_date in trade_dates:
            pub_date_2_trade_date.update({pub_date: pub_date})
        else:
            trade_dates_after_pub_dates = [_ for _ in trade_dates if _ >= pub_date]
            if len(trade_dates_after_pub_dates):
                pub_date_2_trade_date.update({pub_date: trade_dates_after_pub_dates[0]})

    data = data[data.trade_date.map(lambda x: x in pub_date_2_trade_date)]
    data['trade_date'] = data['trade_date'].map(pub_date_2_trade_date)
    data = pd.merge(data, capital_data, how='left', on=['code', 'trade_date'])
    data['bonus_ratio_rmb'] = data['bonus_ratio_rmb'].fillna(0).map(float)
    data['capitalization_4_bonus'] = data['distributed_share_base_board'].isnull()*data['capitalization'] + data['distributed_share_base_board'].fillna(0)
    data['cash_dividend'] = data['capitalization_4_bonus'].map(lambda x: x*1e4)*data['bonus_ratio_rmb'].map(lambda x: x/10)

    data['year'] = data['end_date'].map(lambda x: x.year)
    cash_dividend_last_year_infos = []
    for (code, year), code_year_data in data.groupby(['code', 'year']):
        if '年度分红' in code_year_data['bonus_type'].values:
            all_cash_dividend = code_year_data['cash_dividend'].sum()
            pub_date = code_year_data[code_year_data.bonus_type == "年度分红"]['trade_date'].values[-1]
            cash_dividend_last_year_infos.append({'code': code, 'trade_date': pub_date, output_name: all_cash_dividend, 'year': year})
    cash_dividend_last_year_df = pd.DataFrame(cash_dividend_last_year_infos)
    cash_dividend_last_year_df = cash_dividend_last_year_df.fillna(0)
    cash_dividend_last_year_df = cash_dividend_last_year_df.set_index(['trade_date', 'code'])
    return cash_dividend_last_year_df

def cal_cash_dividend_history(cash_dividend_last_year_data, history_year_count):
    cash_dividend_last_year_data = cash_dividend_last_year_data.reset_index()
    
    cash_dividend_last_year_data = cash_dividend_last_year_data.sort_values(['code', 'year'])
    
    cash_dividend_last_year_data["CashDividend{}Years".format(history_year_count)] = cash_dividend_last_year_data.groupby('code').rolling(history_year_count, min_periods=1)['CashDividendLastYear'].mean().fillna(0).values

    return cash_dividend_last_year_data[['trade_date', 'code', "CashDividend{}Years".format(history_year_count)]].set_index(['trade_date', 'code'])

    

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
    funcs = [lambda y: np.ceil(y * 100) / 100,
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


def gen_mv_based_weight_flag(data, circ_mv_name, total_mv_name, raw_weight_name, output_name, output_raw_weight):
    """
    基于 mv 数据 产生新的weight 和 flag
    """
    circ_mv = data[circ_mv_name].to_numpy()
    total_mv = data[total_mv_name].to_numpy()
    circ_total_weight = circ_mv/total_mv
    circ_total_weight = myround(circ_total_weight)
    correct_mv = total_mv * circ_total_weight
    data['correct_mv'] = correct_mv
    data['correct_mv'] = data['correct_mv']*(data[raw_weight_name].map(lambda x: 1 if x>0 else 0))
    def mv_based_weight(mv_value):
        return mv_value/mv_value.sum()
    data[output_name] = data.groupby(level='trade_date')['correct_mv'].apply(lambda x: mv_based_weight(x))
    data[raw_weight_name] = data.groupby(level='trade_date')[raw_weight_name].apply(lambda x: mv_based_weight(x))
    
    data["{}Flag".format(output_name)] = data[output_name].map(lambda x: 0 if x > 0 else 1)
    if output_raw_weight:
        return data[[raw_weight_name, output_name, "{}Flag".format(output_name)]]
    else:
        return data[[output_name, "{}Flag".format(output_name)]]

def add_index_tag(index_code_info, index_tag_name):

    index_code_info[index_tag_name]=1

    return index_code_info


def clear_data(data):
    data = data.replace(np.inf, np.nan).replace(-np.inf, np.nan)
    return data


def win(x, trim=0.2, limit='both'):
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
#     import pdb
#     pdb.set_trace()
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
    min_std = -3 * z_std
    max_std = 3 * z_std
    z[z < min_std] = min_std
    z[z > max_std] = max_std
    z[z == None] = min_std
    return z


#
# df = pd.DataFrame({"x": ['a', 'a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b', 'b'], 'y': np.random.random(12)})
#
# res = df.set_index('x').groupby(level='x')['y'].apply(lambda x: sd_win_sort(x))


def concat_data_process(valid_data, invalid_data, tgt_factors):

    data = pd.concat([invalid_data, valid_data])

    data = data.fillna(-999)

    return data[tgt_factors]


def get_tgt_factor_from_data(data, tgt_factors):
    if len(tgt_factors):
        return data[tgt_factors]
    else:
        return data


# def cal_mv_weight(data, circ_mv_name, total_mv_name, raw_weight_name, output_name):
#     circ_mv = data[circ_mv_name].to_numpy()
#     total_mv = data[total_mv_name].to_numpy()
#     circ_total_weight = circ_mv/total_mv
#     circ_total_weight = myround(circ_total_weight)
#     correct_mv = total_mv * circ_total_weight
#     data['correct_mv'] = correct_mv
#     data['correct_mv'] = data['correct_mv']*(data[raw_weight_name].map(lambda x: 1 if x>0 else 0))
#     def mv_based_weight(mv_value):
#         return mv_value/mv_value.sum()
#     data[output_name] = data.groupby(level='trade_date')['correct_mv'].apply(lambda x: mv_based_weight(x))
#     return data


def cal_reciprocal(value_name, data, output_name):
    data[output_name] = data[value_name].map(lambda x: 1 / x)
    return data


# def save_data_to_multi_place(save_infos):
#     for info in save_infos:
#         engine = info['engine']
#         table = info['table']


def get_data_from_multi_source(data_source_infos, start_date, end_date, join_method='outer'):
    multi_datas = []
    # print(data_source_infos)
    for info in data_source_infos:
        engine = info['engine']
        table = info["table"]
        field = info["field"]
        index = info.get("index", ['trade_date', 'code'])
        hist_year = info.get("hist_year", 0)
        name_dict = info.get("name_dict", {})
        other_filter = info.get("other_filter")
        data = get_hist_data_4_factor_compute(
            read_engine=engine,
            save_engine=engine,
            start_date=start_date,
            end_date=end_date,
            table=table,
            field=field,
            index=index,
            hist_year=hist_year,
            name_dict=name_dict,
            other_filter_info=other_filter
        )
        multi_datas.append(data)

    multi_data = pd.concat(multi_datas, axis=1, join=join_method)

    #     except Exception as e:
    #         import pdb
    #         pdb.set_trace()
    # new_data =  pd.concat(multi_datas, axis=1, join='inner')

    return multi_data


def transfer_data_to_valid_and_not_valid(data, invalid_infos):
    invalid_tag = pd.Series([False for j in range(len(data))], index=data.index)

    #     data[['code', 'trade_date', "ResearchRptFactor", "Weeks50CountLog", "GicsIndustryCode"]].to_excel("research_rpt_data.xlsx")
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
            invalid_tag |= data["rank"].map(lambda x: x > feature_value)
        elif type_ == "not_bottom":
            data['rank'] = data.groupby('trade_date')[feature_name].rank(ascending=True)
            invalid_tag |= data["rank"].map(lambda x: x > feature_value)
        elif type_ == "not_top_v2":
            data['rank'] = data.groupby('trade_date')[feature_name].rank(method='first', ascending=False)
            invalid_tag |= data["rank"].map(lambda x: x > feature_value)
        elif type_ == "isnull":
            invalid_tag |= data[feature_name].isnull()
        else:
            pass
    # if "ResearchRptFactor" in data.columns:
    #     import pdb
    #     pdb.set_trace()
    invalid_data = data[invalid_tag]
    valid_data = data[~invalid_tag]

    return valid_data, invalid_data


def sd_win_sort(raw_fac, limit=0.05, sort_func=ECDF, reverse=False, is_3_sigma_std=True):
    """
    Perform  5% trime, zscore and 3 sigma winsorization and Ecdf sort on a group of a single factor
    """

    idx = raw_fac.index
    sd_fac = stand(raw_fac, limit)
    if reverse:
        sd_fac = - sd_fac
    if is_3_sigma_std:
        sd_win_fac = std_winsor(sd_fac)
    else:
        sd_win_fac = sd_fac
    fac_cdf_clf = sort_func(sd_win_fac)
    fac_cdf = fac_cdf_clf(sd_win_fac)
    fac_cdf_series = pd.Series(fac_cdf, index=idx)
    return fac_cdf_series


def fast_sd_win_ecdf(raw_fac, code, feature_name, limit=0.05, reverse=False):
    """
    快速版本，直接计算特定值的ecdf,
    :param raw_fac:
    :param limit:
    :param sort_func:
    :param reverse:
    :return:
    """
    feature_s = raw_fac.set_index("code")[feature_name]
    if reverse:
        feature_s = feature_s.map(lambda x: -x)
    code_feature = feature_s[code]
    code_ecdf = 1 - (feature_s - code_feature).map(lambda x: 1 if x > 0 else 0).sum() / len(feature_s)
    code_ecdf = limit if code_ecdf < limit else code_ecdf
    code_ecdf = 1 if code_ecdf > (1 - limit) else code_ecdf
    return code_ecdf


def sum_data_with_weight(data, output_name, features=[], weights=[]):
    assert len(features) == len(weights), print(features)
    try:
        data[output_name] = (data[features] * weights).sum(axis=1)
    except Exception as e:
        import pdb
        pdb.set_trace()
    return data


def pipline_sum_data_weight_weight(data, sum_infos):
    """
    根据权重对输入数据加权求和
    :param data:
    :param sum_infos:
    :return:
    """
    for info in sum_infos:
        if "condition_feature" not in info:
            features = info['features']
            weights = info['weights']
            output_name = info['output_name']
            data = sum_data_with_weight(data, output_name, features, weights)
        else:
            condition_feature = info['condition_feature']
            output_name = info['output_name']
            feature_combined_infos = info['feature_combined_infos']
            hist_values = []
            special_datas = []
            for feature_info in feature_combined_infos:
                condition_value = feature_info['condition_value']
                features = feature_info['features']
                weights = feature_info['weights']
                if condition_value != 'other':
                    special_data = data[data[condition_feature] == condition_value]
                else:
                    special_data = data[data[condition_feature].map(lambda x: x not in hist_values)]
                special_data = sum_data_with_weight(special_data, output_name, features, weights)
                special_datas.append(special_data)
                hist_values.append(condition_value)
            data = pd.concat(special_datas)
    return data


def gen_mktcap_bin(data, mv_name):
    import math

    data["{}Bin".format(mv_name)] = data.groupby('trade_date')[mv_name].rank(pct=True).map(lambda x: math.ceil(x * 10))
    return data


def standardized_factor(value_name, industry_name, data, output_name, limit=0.05, sort_func=ECDF, reverse=False):
    data = data.reset_index()
    data[output_name] = data.groupby(['trade_date', industry_name])[value_name].apply(
        lambda x: sd_win_sort(x, limit=limit, sort_func=sort_func, reverse=reverse))
    return data.set_index(['trade_date', 'ts_code'])


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
        is_3_sigma_std = feature_info.get('is_3_sigma_std', True)
        try:
            data[output_name] = data.groupby(['trade_date', industry_name],group_keys=False)[featuren_name].apply(
                lambda x: sd_win_sort(x, limit=limit_value, sort_func=sort_func, reverse=reverse,
                                      is_3_sigma_std=is_3_sigma_std))
        except Exception as e:
            print(e)
            import pdb
            pdb.set_trace()
            pass
    data = data.set_index(index_names)
    return data



def group_feature(data, group_infos):
    for group_info in group_infos:
        group_bin_count = group_info['group_bin_count']
        feature = group_info['feature']
        null_value = group_info['null_value']
        data[feature] = data[feature].fillna(-999)
        valid_data = data[data[feature] != -999]
        novalid_data = data[data[feature] == -999]
        valid_data[feature] = valid_data[feature].rank(pct=True).map(lambda x: math.ceil(group_bin_count*x))
        max_bin = valid_data[feature].max()
        min_bin = valid_data[feature].min()
        valid_data[feature] = valid_data[feature].map(lambda x: (x-min_bin)/(max_bin/2))
        novalid_data[feature] = null_value
        data = pd.concat([valid_data, novalid_data])
    return data
        


def peer_standardized_factor(valid_data, peer_mapping, feature_neutral_infos):
    valid_data = valid_data.reset_index()
    features = list(set([_['feature_name'] for _ in feature_neutral_infos]))
    peer_mapping = peer_mapping.reset_index()
    # peer_mapping['trade_date'] = peer_mapping['trade_date'].map(lambda x: int(x.strftime("%Y%m%d")))
    valid_data['peer'] = valid_data['code']
    peer_mapping_factor = pd.merge(peer_mapping, valid_data[['peer', 'trade_date'] + features], how='left',
                                   left_on=['trade_date', 'peer'], right_on=['trade_date', 'peer'])
    print(peer_mapping_factor.columns)
    factor_peer_neutral_infos = []
    for feature_info in feature_neutral_infos:
        limit_value = feature_info['limit_value']
        reverse = feature_info['reverse']
        output_name = feature_info['output_name']
        feature_name = feature_info['feature_name']
        print("feature {}".format(feature_name))
        sort_func = feature_info['sort_func']
        peer_mapping_factor['neutral_value'] = peer_mapping_factor.groupby(['trade_date', 'code'])[feature_name].apply(lambda x: sd_win_sort(x, limit=limit_value, sort_func=sort_func, reverse=reverse))
        factor_peer_neutral = peer_mapping_factor[peer_mapping_factor['code'] == peer_mapping_factor['peer']]
        factor_peer_neutral = factor_peer_neutral.rename({"neutral_value": output_name}, axis=1)
        factor_peer_neutral_infos.append(factor_peer_neutral.set_index(['trade_date', 'code'])[[output_name]])
    factor_peer_neutral_df = pd.concat(factor_peer_neutral_infos, axis=1)
    return factor_peer_neutral_df


def fast_peer_standardized_factor(valid_data, peer_mapping, feature_neutral_infos):
    valid_data = valid_data.reset_index()
    features = list(set([_['feature_name'] for _ in feature_neutral_infos]))
    peer_mapping = peer_mapping.reset_index()
    # peer_mapping['trade_date'] = peer_mapping['trade_date'].map(lambda x: int(x.strftime("%Y%m%d")))
    valid_data['peer'] = valid_data['code']
    # peer_mapping_factor = pd.merge(peer_mapping, valid_data[['peer', 'trade_date']+features], how='left', left_on=['trade_date', 'peer'], right_on=['trade_date', 'peer'])
    peer_mapping_factor = pd.merge(peer_mapping, valid_data[['peer', 'trade_date']+features], how='left', left_on=['trade_date', 'peer'], right_on=['trade_date', 'peer'])
    # print(peer_mapping_factor.columns)
    factor_peer_neutral_infos = []
    for feature_info in feature_neutral_infos:
        limit_value = feature_info['limit_value']
        reverse = feature_info['reverse']
        output_name = feature_info['output_name']
        feature_name = feature_info['feature_name']
        print("feature {}".format(feature_name))
        sort_func = feature_info['sort_func']
        for (trade_date, code), tmp_df in peer_mapping_factor.groupby(['trade_date', 'code']):
            code_ecdf = fast_sd_win_ecdf(tmp_df, code, feature_name, limit=limit_value, reverse=reverse)
            factor_peer_neutral_infos.append({'trade_date': trade_date, 'code': code, output_name: code_ecdf})
    factor_peer_neutral_df = pd.DataFrame(factor_peer_neutral_infos)
    return factor_peer_neutral_df


def fast_peer_standardized_factor_v2(valid_data, peer_mapping, feature_neutral_infos):
    valid_data = valid_data.reset_index()
    features = list(set([_['feature_name'] for _ in feature_neutral_infos]))
    peer_mapping = peer_mapping.reset_index()
    # peer_mapping['trade_date'] = peer_mapping['trade_date'].map(lambda x: int(x.strftime("%Y%m%d")))
    valid_data['peer'] = valid_data['code']
    peer_mapping_factor = pd.merge(peer_mapping, valid_data[['code', 'trade_date']+features], how='left', left_on=['trade_date', 'code'], right_on=['trade_date', 'code'])
    peer_mapping_factor = pd.merge(peer_mapping_factor, valid_data[['peer', 'trade_date']+features], how='left', left_on=['trade_date', 'peer'], right_on=['trade_date', 'peer'])
    # print(peer_mapping_factor.columns)

    factor_peer_neutral_infos = []
    for feature_info in feature_neutral_infos:
        limit_value = feature_info['limit_value']
        reverse = feature_info['reverse']
        output_name = feature_info['output_name']
        feature_name = feature_info['feature_name']
        print("feature {}".format(feature_name))
        sort_func = feature_info['sort_func']
        peer_mapping_factor['less_than_peer_tag'] = (peer_mapping_factor['{}_x'.format(feature_name)] - peer_mapping_factor['{}_y'.format(feature_name)]).map(lambda x: -x if reverse else x).map(lambda x: 1 if x < 0 else 0)
        peer_ecdf = 1-peer_mapping_factor.groupby(['trade_date', 'code'])['less_than_peer_tag'].sum()/peer_mapping_factor.groupby(['trade_date', 'code'])['less_than_peer_tag'].count()
        peer_ecdf = peer_ecdf.map(lambda x: 1 if x > (1-limit_value) else x).map(lambda x: limit_value if x < limit_value else x)
        peer_ecdf.name = output_name
        factor_peer_neutral_infos.append(peer_ecdf)
    factor_peer_neutral_df = pd.concat(factor_peer_neutral_infos, axis=1)

    return factor_peer_neutral_df


def gen_research_report_weekly_count(data):

    data['WeekCount'] = 1
    research_report_weekly_count = data.groupby(level=['code', 'trade_date'])['WeekCount'].sum()

    return research_report_weekly_count

def gen_research_report_hist_count(data):
    import math
    data.sort_index(level=['code', 'trade_date'], inplace=True)
    for j in range(50, 52, 2):
        data['Weeks{}Count'.format(j)] = data.groupby(level='code',group_keys=False)['WeekCount'].apply(lambda x: x.rolling(j).sum().shift(1))
        data['Weeks{}CountLog'.format(j)] = data['Weeks{}Count'.format(j)].map(lambda x: math.log(x) if x > 0 else -1)
        data['Weeks{}CountTag'.format(j)] = data['Weeks{}Count'.format(j)].map(lambda x: 1 if x > 0 else 0)
#     is_leading_stock = data['OneYearCount'] > 0
#     leading_stock_data = data[is_leading_stock]
#     noleading_stock_data = data[~is_leading_stock]
#     leading_stock_data['OneYearCountPct'] = leading_stock_data.groupby('trade_date')['OneYearCount'].rank(pct=True)
#     leading_stock_data['FourWeeksCountPct'] = leading_stock_data.groupby('trade_date')['FourWeeksCount'].rank(pct=True)
#     leading_stock_data['EightWeeksCountPct'] = leading_stock_data.groupby('trade_date')['EightWeeksCount'].rank(pct=True)
#     data = pd.concat([leading_stock_data, noleading_stock_data])
#     data[['OneYearCountPct', 'FourWeeksCountPct', 'EightWeeksCountPct']] = data[['OneYearCountPct', 'FourWeeksCountPct', 'EightWeeksCountPct']].fillna(-999)
#     data.query('trade_date == 20250916').to_excel("rpt_count.xlsx")
    return data

def gen_weekly_fin_forecast_tag(data):

    data['FinanceGoodPredTag'] = data['type'].map(lambda x: x in ['业绩大幅上升', "业绩预增", "预计扭亏", "预计减亏", "大幅减亏"])
    data['FinancePoorPredTag'] = data['type'].map(lambda x: x in ['业绩预亏', "业绩大幅下降", "业绩预降"])
    # data.sort_index(level=['code', 'trade_date'], inplace=True)
    # data['HistGoodPredTag'] = data['finance_good_pred_tag'].groupby(level='code').apply(lambda x: x.rolling(window_size, min_periods=1).sum().map(lambda x: 1 if x > 0 else 0))
    # data['HistPoorPredTag'] = data['finance_poor_pred_tag'].groupby(level='code').apply(lambda x: x.rolling(window_size, min_periods=1).sum().map(lambda x: 1 if x > 0 else 0))
    finance_good_pred_tag = data.groupby(level=['code', 'trade_date'])['FinanceGoodPredTag'].sum().map(lambda x: 1 if x>0 else 0).to_frame()
    finance_poor_pred_tag = data.groupby(level=['code', 'trade_date'])['FinancePoorPredTag'].sum().map(lambda x: 1 if x>0 else 0).to_frame()
    return finance_good_pred_tag, finance_poor_pred_tag

def gen_hist_event_tag(data, tag_name, window_size, hist_tag, shift_window_size=0):

    data.sort_index(level=['code', 'trade_date'], inplace=True)
    data[hist_tag] = data[tag_name].groupby(level='code').apply(lambda x: x.rolling(window_size, min_periods=1).sum().shift(shift_window_size).map(lambda x: 1 if x > 0 else 0))
    return data[[hist_tag]]


def std_gics_industry(data):
    data['GicsIndustryCode'] = data['GicsIndustryCode'].fillna(20)
    data['GicsIndustryCode'] = data['GicsIndustryCode'].map(int)
    data['GicsIndustryCode'] = data['GicsIndustryCode'].map(lambda x: 45 if x == 50 else x)
    
    data['GicsIndustryName'] = data['GicsIndustryName'].map(lambda x: "信息技术" if x == "通讯服务" else x)
    data['GicsIndustryName'] = data['GicsIndustryName'].fillna("工业")
    return data
    
def generate_index_vol_tag(index_name, bin_count=3):
#     from jqdatasdk import * 
#     auth("13764432461", "Nfhq12345")
    
    today = datetime.datetime.today().date().strftime("%Y%m%d")
    index_price = get_price(index_name, start_date="2009-01-01", end_date=today)
    index_price = index_price.sort_index()
    index_price['r'] = index_price['close'].pct_change()
    index_price['vol_20'] = index_price['r'].rolling(20).std()
    index_price['trade_date'] = index_price.index.map(lambda x: int(pd.to_datetime(str(x)).strftime('%Y%m%d')))
    index_price['vol_20_pct'] = index_price['vol_20'].rolling(1000).apply(lambda x: x.rank(pct=True).iloc[-1])
    index_price = index_price[index_price.vol_20_pct.notnull()]
    index_price['vol_20_pct_bin'] = index_price['vol_20_pct'].map(lambda x: math.ceil(x*bin_count))
    return index_price[['trade_date', 'vol_20_pct_bin', 'vol_20', 'vol_20_pct']]



def generate_index_vol_tag_real_bin(index_name):
#     from jqdatasdk import * 
#     auth("13764432461", "Nfhq12345")
    
    today = datetime.datetime.today().date().strftime("%Y%m%d")
    index_price = get_price(index_name, start_date="2009-01-01", end_date=today)
    index_price = index_price.sort_index()
    index_price['r'] = index_price['close'].pct_change()
    index_price['vol_20'] = index_price['r'].rolling(20).std()
    index_price['trade_date'] = index_price.index.map(lambda x: int(pd.to_datetime(str(x)).strftime('%Y%m%d')))
#     index_price['vol_20_pct'] = index_price['vol_20'].rolling(1000).apply(lambda x: x.rank(pct=True).iloc[-1])
#     index_price = index_price[index_price.vol_20_pct.notnull()]
#     index_price['vol_20_pct_bin'] = index_price['vol_20_pct'].map(lambda x: math.ceil(x*3))
    index_price = index_price[index_price.vol_20.notnull()]
    def cal_vol_bin(x):
        if x > 0.015655:
            return 3
        elif x > 0.01093:
            return 2
        else:
            return 1
    index_price['vol_20_pct_bin'] = index_price['vol_20'].map(lambda x: cal_vol_bin(x))
    return index_price[['trade_date', 'vol_20_pct_bin']]

def generate_fixed_weight_growth(all_factor_data, start_date, end_date):
    # factor_names = [info['name'] for info in factor_bin_infos]
    factor_names = ["ValueFactor", "LiquidityFactor", "GrowthFactor", "QualityFactor", "LongMomentumFactorReverse",
                     "ShortMomentumFactorReverse", "VolatilityFactor"]
    # import pdb
    # pdb.set_trace()

    for factor_name in factor_names:
        if factor_name == 'OverallMomentumFactor':
            all_factor_data[factor_name] = all_factor_data[factor_name].map(lambda x: x / 2)
        else:
            all_factor_data[factor_name] = all_factor_data[factor_name].map(lambda x: x - 0.5)
    all_factor_data = all_factor_data.reset_index()
    trade_dates = sorted(all_factor_data['trade_date'].unique())
    fixed_score_infos = []
    # bin_count = 10
    for i, date in enumerate(tqdm(trade_dates)):
        if date > start_date and date <= end_date:
            tmp_data = all_factor_data[all_factor_data.trade_date == date]
            for growth_weight in range(2, 12, 2):
                bin_count = 3
                tmp_data.sort_values("ValueFactor", inplace=True)
                valid_count = len(tmp_data)

                tmp_data.sort_values("ValueFactor", inplace=True)
                tmp_data['ValueFactorBin'] = [int(bin_count * j/valid_count) for j in range(valid_count)]
                bin_2_value = tmp_data.groupby('ValueFactorBin')['ValueFactor'].mean().to_dict()
                tmp_data['ValueFactorBinValue'] = tmp_data['ValueFactorBin'].map(bin_2_value)

                tmp_data.sort_values("GrowthFactor", inplace=True)
                
                tmp_data['GrowthFactorBin'] = [int(50 * j / valid_count) for j in range(valid_count)]
                #     valid_tmp_df['growth_factor_bin'] = valid_tmp_df['growth_factor_bin'].map(lambda x: min(x, 1))
                bin_2_value = tmp_data.groupby('GrowthFactorBin')['GrowthFactor'].mean().to_dict()
                tmp_data['GrowthFactorBinValue'] = tmp_data['GrowthFactorBin'].map(bin_2_value)

                tmp_data.sort_values("QualityFactor", inplace=True)
                tmp_data['QualityFactorBin'] = [int(bin_count * j / valid_count) for j in range(valid_count)]
                bin_2_value = tmp_data.groupby('QualityFactorBin')['QualityFactor'].mean().to_dict()
                tmp_data['QualityFactorBinValue'] = tmp_data['QualityFactorBin'].map(bin_2_value)

                tmp_data.sort_values("LiquidityFactor", inplace=True)
                tmp_data['LiquidityFactorBin'] = [int(5 * j / valid_count) for j in range(valid_count)]
                tmp_data['LiquidityFactorBin'] = tmp_data['LiquidityFactorBin'].map(lambda x: min(x, 1))
                bin_2_value = tmp_data.groupby('LiquidityFactorBin')['LiquidityFactor'].mean().to_dict()
                tmp_data['LiquidityFactorBinValue'] = tmp_data['LiquidityFactorBin'].map(bin_2_value)

                tmp_data.sort_values("VolatilityFactor", inplace=True)
                tmp_data['VolatilityFactorBin'] = [int(5 * j / valid_count) for j in range(valid_count)]
                tmp_data['VolatilityFactorBin'] = tmp_data['VolatilityFactorBin'].map(lambda x: min(x, 1))
                bin_2_value = tmp_data.groupby('VolatilityFactorBin')['VolatilityFactor'].mean().to_dict()
                tmp_data['VolatilityFactorBinValue'] = tmp_data['VolatilityFactorBin'].map(bin_2_value)

                tmp_data.sort_values("ShortMomentumFactorReverse", inplace=True)
                tmp_data['ShortMomentumFactorReverseBin'] = [int(5 * j / valid_count) for j in range(valid_count)]
                tmp_data['ShortMomentumFactorReverseBin'] = tmp_data['ShortMomentumFactorReverseBin'].map(lambda x: min(x, 1))
                bin_2_value = tmp_data.groupby('ShortMomentumFactorReverseBin')['ShortMomentumFactorReverse'].mean().to_dict()
                tmp_data['ShortMomentumFactorReverseBinValue'] = tmp_data['ShortMomentumFactorReverseBin'].map(bin_2_value)

                tmp_data.sort_values("LongMomentumFactorReverse", inplace=True)
                tmp_data['LongMomentumFactorReverseBin'] = [int(5 * j / valid_count) for j in range(valid_count)]
                bin_2_value = tmp_data.groupby('LongMomentumFactorReverseBin')['LongMomentumFactorReverse'].mean().to_dict()

                bin_2_value[0] = bin_2_value[4]
                tmp_data['LongMomentumFactorReverseBinValue'] = tmp_data['LongMomentumFactorReverseBin'].map(
                    bin_2_value)

                tmp_data["CSI500FixedWeightScoreBin{}GrowthWeight{}".format(bin_count, growth_weight)] = tmp_data['ValueFactorBinValue'] + tmp_data[
                    'LiquidityFactorBinValue'] + tmp_data['GrowthFactorBinValue'] * growth_weight + \
                                              + tmp_data[
                                                         'QualityFactorBinValue'] - tmp_data[
                                                         'LongMomentumFactorReverseBinValue'] + tmp_data[
                                                         'ShortMomentumFactorReverseBinValue'] + 0*tmp_data[
                                                         'VolatilityFactorBinValue']
                # tmp_data["CSI500FixedWeightScoreWithFinanceForecast"] = tmp_data['ValueFactorBinValue'] + tmp_data[
                #     'LiquidityFactorBinValue'] + tmp_data['GrowthFactorBinValue'] * 2 + 0.6*tmp_data['HistFinanceGoodPredTag']\
                #                               - 0.6*tmp_data['HistFinancePoorPredTag'] + tmp_data[
                #                                          'QualityFactorBinValue'] - tmp_data[
                #                                          'LongMomentumFactorReverseBinValue'] + tmp_data[
                #                                          'ShortMomentumFactorReverseBinValue'] + 0*tmp_data[
                #                                          'VolatilityFactorBinValue']
                score_std = tmp_data["CSI500FixedWeightScoreBin{}GrowthWeight{}".format(bin_count, growth_weight)].std()
                tmp_data["CSI500FixedWeightScoreBin{}GrowthWeight{}".format(bin_count, growth_weight)] = tmp_data["CSI500FixedWeightScoreBin{}GrowthWeight{}".format(bin_count, growth_weight)].map(lambda x: x/score_std * 0.002)
#                 print(tmp_data["CSI500FixedWeightScoreBin{}GrowthWeight{}".format(bin_count, growth_weight)].std())
                # tmp_data["CSI500FixedWeightScoreWithFinanceForecast"] = tmp_data["CSI500FixedWeightScoreWithFinanceForecast"].map(lambda x: x * 0.0016)


            fixed_score_infos.append(tmp_data)
    fixed_score_df = pd.concat(fixed_score_infos)
    fixed_score_df = fixed_score_df.set_index(['code', 'trade_date'])[["GrowthFactor", "ValueFactor"]+["CSI500FixedWeightScoreBin3GrowthWeight{}".format(growth_weight) for growth_weight in range(2, 12, 2)]]
    return fixed_score_df


def generate_pure_fixed_weight_growth(all_factor_data, start_date, end_date):
    # factor_names = [info['name'] for info in factor_bin_infos]
    factor_names = ["ValueFactor", "LiquidityFactor", "GrowthFactor", "QualityFactor", "LongMomentumFactorReverse",
                     "ShortMomentumFactorReverse", "VolatilityFactor"]
    # import pdb
    # pdb.set_trace()

    for factor_name in factor_names:
        if factor_name == 'OverallMomentumFactor':
            all_factor_data[factor_name] = all_factor_data[factor_name].map(lambda x: x / 2)
        else:
            all_factor_data[factor_name] = all_factor_data[factor_name].map(lambda x: x - 0.5)
    all_factor_data = all_factor_data.reset_index()
    trade_dates = sorted(all_factor_data['trade_date'].unique())
    fixed_score_infos = []
    # bin_count = 10
    for i, date in enumerate(tqdm(trade_dates)):
        if date > start_date and date <= end_date:
            tmp_data = all_factor_data[all_factor_data.trade_date == date]
            for growth_weight in range(2, 12, 2):
                bin_count = 3
                tmp_data.sort_values("ValueFactor", inplace=True)
                valid_count = len(tmp_data)

                tmp_data.sort_values("ValueFactor", inplace=True)
                tmp_data['ValueFactorBin'] = [int(bin_count * j/valid_count) for j in range(valid_count)]
                bin_2_value = tmp_data.groupby('ValueFactorBin')['ValueFactor'].mean().to_dict()
                tmp_data['ValueFactorBinValue'] = tmp_data['ValueFactorBin'].map(bin_2_value)

                tmp_data.sort_values("GrowthFactor", inplace=True)
                
                tmp_data['GrowthFactorBin'] = [int(50 * j / valid_count) for j in range(valid_count)]
                #     valid_tmp_df['growth_factor_bin'] = valid_tmp_df['growth_factor_bin'].map(lambda x: min(x, 1))
                bin_2_value = tmp_data.groupby('GrowthFactorBin')['GrowthFactor'].mean().to_dict()
                tmp_data['GrowthFactorBinValue'] = tmp_data['GrowthFactorBin'].map(bin_2_value)

                tmp_data.sort_values("QualityFactor", inplace=True)
                tmp_data['QualityFactorBin'] = [int(bin_count * j / valid_count) for j in range(valid_count)]
                bin_2_value = tmp_data.groupby('QualityFactorBin')['QualityFactor'].mean().to_dict()
                tmp_data['QualityFactorBinValue'] = tmp_data['QualityFactorBin'].map(bin_2_value)

                tmp_data.sort_values("LiquidityFactor", inplace=True)
                tmp_data['LiquidityFactorBin'] = [int(5 * j / valid_count) for j in range(valid_count)]
                tmp_data['LiquidityFactorBin'] = tmp_data['LiquidityFactorBin'].map(lambda x: min(x, 1))
                bin_2_value = tmp_data.groupby('LiquidityFactorBin')['LiquidityFactor'].mean().to_dict()
                tmp_data['LiquidityFactorBinValue'] = tmp_data['LiquidityFactorBin'].map(bin_2_value)

                tmp_data.sort_values("VolatilityFactor", inplace=True)
                tmp_data['VolatilityFactorBin'] = [int(5 * j / valid_count) for j in range(valid_count)]
                tmp_data['VolatilityFactorBin'] = tmp_data['VolatilityFactorBin'].map(lambda x: min(x, 1))
                bin_2_value = tmp_data.groupby('VolatilityFactorBin')['VolatilityFactor'].mean().to_dict()
                tmp_data['VolatilityFactorBinValue'] = tmp_data['VolatilityFactorBin'].map(bin_2_value)

                tmp_data.sort_values("ShortMomentumFactorReverse", inplace=True)
                tmp_data['ShortMomentumFactorReverseBin'] = [int(5 * j / valid_count) for j in range(valid_count)]
                tmp_data['ShortMomentumFactorReverseBin'] = tmp_data['ShortMomentumFactorReverseBin'].map(lambda x: min(x, 1))
                bin_2_value = tmp_data.groupby('ShortMomentumFactorReverseBin')['ShortMomentumFactorReverse'].mean().to_dict()
                tmp_data['ShortMomentumFactorReverseBinValue'] = tmp_data['ShortMomentumFactorReverseBin'].map(bin_2_value)

                tmp_data.sort_values("LongMomentumFactorReverse", inplace=True)
                tmp_data['LongMomentumFactorReverseBin'] = [int(5 * j / valid_count) for j in range(valid_count)]
                bin_2_value = tmp_data.groupby('LongMomentumFactorReverseBin')['LongMomentumFactorReverse'].mean().to_dict()

                bin_2_value[0] = bin_2_value[4]
                tmp_data['LongMomentumFactorReverseBinValue'] = tmp_data['LongMomentumFactorReverseBin'].map(
                    bin_2_value)

                tmp_data["CSI500FixedWeightScoreBin{}GrowthWeight{}NonValue".format(bin_count, growth_weight)] = tmp_data['ValueFactorBinValue']*0 + tmp_data[
                    'LiquidityFactorBinValue'] + tmp_data['GrowthFactorBinValue'] * growth_weight + \
                                              + tmp_data[
                                                         'QualityFactorBinValue'] - tmp_data[
                                                         'LongMomentumFactorReverseBinValue'] + tmp_data[
                                                         'ShortMomentumFactorReverseBinValue'] + 0*tmp_data[
                                                         'VolatilityFactorBinValue']
                tmp_data["CSI500FixedWeightScoreBin{}GrowthWeight{}NonValueQuality".format(bin_count, growth_weight)] = tmp_data['ValueFactorBinValue']*0 + tmp_data[
                    'LiquidityFactorBinValue'] + tmp_data['GrowthFactorBinValue'] * growth_weight + \
                                              + tmp_data[
                                                         'QualityFactorBinValue']*0 - tmp_data[
                                                         'LongMomentumFactorReverseBinValue'] + tmp_data[
                                                         'ShortMomentumFactorReverseBinValue'] + 0*tmp_data[
                                                         'VolatilityFactorBinValue']
                tmp_data["CSI500FixedWeightScoreBin{}GrowthWeight{}NonValueQualityLiquidity".format(bin_count, growth_weight)] = tmp_data['ValueFactorBinValue']*0 + tmp_data[
                    'LiquidityFactorBinValue']*0 + tmp_data['GrowthFactorBinValue'] * growth_weight + \
                                              + tmp_data[
                                                         'QualityFactorBinValue']*0 - tmp_data[
                                                         'LongMomentumFactorReverseBinValue'] + tmp_data[
                                                         'ShortMomentumFactorReverseBinValue'] + 0*tmp_data[
                                                         'VolatilityFactorBinValue']                
                # tmp_data["CSI500FixedWeightScoreWithFinanceForecast"] = tmp_data['ValueFactorBinValue'] + tmp_data[
                #     'LiquidityFactorBinValue'] + tmp_data['GrowthFactorBinValue'] * 2 + 0.6*tmp_data['HistFinanceGoodPredTag']\
                #                               - 0.6*tmp_data['HistFinancePoorPredTag'] + tmp_data[
                #                                          'QualityFactorBinValue'] - tmp_data[
                #                                          'LongMomentumFactorReverseBinValue'] + tmp_data[
                #                                          'ShortMomentumFactorReverseBinValue'] + 0*tmp_data[
                #                                          'VolatilityFactorBinValue']
                score_std = tmp_data["CSI500FixedWeightScoreBin{}GrowthWeight{}NonValue".format(bin_count, growth_weight)].std()
                tmp_data["CSI500FixedWeightScoreBin{}GrowthWeight{}NonValue".format(bin_count, growth_weight)] = tmp_data["CSI500FixedWeightScoreBin{}GrowthWeight{}NonValue".format(bin_count, growth_weight)].map(lambda x: x/score_std * 0.002)
                score_std = tmp_data["CSI500FixedWeightScoreBin{}GrowthWeight{}NonValueQuality".format(bin_count, growth_weight)].std()
                tmp_data["CSI500FixedWeightScoreBin{}GrowthWeight{}NonValueQuality".format(bin_count, growth_weight)] = tmp_data["CSI500FixedWeightScoreBin{}GrowthWeight{}NonValueQuality".format(bin_count, growth_weight)].map(lambda x: x/score_std * 0.002)
                score_std = tmp_data["CSI500FixedWeightScoreBin{}GrowthWeight{}NonValueQualityLiquidity".format(bin_count, growth_weight)].std()
                tmp_data["CSI500FixedWeightScoreBin{}GrowthWeight{}NonValueQualityLiquidity".format(bin_count, growth_weight)] = tmp_data["CSI500FixedWeightScoreBin{}GrowthWeight{}NonValueQualityLiquidity".format(bin_count, growth_weight)].map(lambda x: x/score_std * 0.002)
                
                
#                 print(tmp_data["CSI500FixedWeightScoreBin{}GrowthWeight{}".format(bin_count, growth_weight)].std())
                # tmp_data["CSI500FixedWeightScoreWithFinanceForecast"] = tmp_data["CSI500FixedWeightScoreWithFinanceForecast"].map(lambda x: x * 0.0016)


            fixed_score_infos.append(tmp_data)
    fixed_score_df = pd.concat(fixed_score_infos)
    fixed_score_df = fixed_score_df.set_index(['code', 'trade_date'])
    return fixed_score_df


def gen_adj_price(data):
    data = data.reset_index()
    data = data.sort_values(['code', 'trade_date'])
    data['_10am_price_fq'] = data['_10am_price_nfq'] * data['factor']
    data['close_price_fq'] = data['close_price_nfq'] * data['factor']
    data['open_price_fq'] = data['open_price_nfq'] * data['factor']
    data['nextday_10am_price_fq'] = data.groupby('code')['_10am_price_fq'].shift(-1)
    data['nextday_open_price_fq'] = data.groupby('code')['open_price_fq'].shift(-1)
    return data.set_index(['code', 'trade_date'])


def gen_weekly_one_term_return(weekly_data):
    weekly_data = weekly_data.reset_index()
    weekly_data = weekly_data.sort_values(['code', 'trade_date'])
    weekly_data['next_week_10am_price_fq'] = weekly_data.groupby('code')['nextday_10am_price_fq'].shift(-1)
    weekly_data['_10amOneTermReturn'] = weekly_data['next_week_10am_price_fq']/weekly_data['nextday_10am_price_fq'] -1
    weekly_data['next_week_close_price_fq'] = weekly_data.groupby('code')['close_price_fq'].shift(-1)
    weekly_data['_10amOneTermReturn4LastWeek'] = weekly_data['next_week_close_price_fq']/weekly_data['nextday_10am_price_fq'] - 1

    return weekly_data.set_index(['code', 'trade_date'])[['_10amOneTermReturn', '_10amOneTermReturn4LastWeek']]

def discretize(sr,q):
    """
    discretize a continous varable into bins according to its ranking

    Parameters
    ----------
    sr : pd.Series
        continous variable values.
    q : int or list of float
        if int, indicating number of bins
        if list of float indicating percentile intervals of bins e.g. [0,0.25,0.5,0.75,1.0].

    Returns
    -------
    discretize_sr : pd.Series
        discretized variable.

    """
    bins = pd.qcut(sr,q,retbins = True)[1]
    # discretize the varable
    if isinstance(q,int):
        labels = [i for i in range(q)]
    elif isinstance(q,list):
        labels = [i for i in range(len(q)-1)]
    bins = sorted(bins)
    bins[0] = bins[0] - 0.0001
    discretize_sr = pd.cut(sr,bins,labels = [i for i in labels])
    return discretize_sr


