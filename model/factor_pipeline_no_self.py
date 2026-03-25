#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 17:14:24 2022

@author: yitao hu
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
from model.tools import stand, std_winsor, load_obj, show_process
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
from model import SQL_api
from statsmodels.regression.rolling import RollingOLS


#### pipline decorators
def clean_memory(func):
    """
    empty the factor values stored in kernel memory

    Returns
    -------
    None.

    """
    global cache
    cache = {}
    print('Factor memeory cleared')
    return None


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
        factor = args[0]
        if factor.name not in cache:
            factor = func(*args, **kwargs)
            cache[factor.name] = factor.fac_value
        else:
            factor.fac_value = cache[factor.name]
        return factor

    return wrapper


def timer(func):
    """A decorator that prints how long a function took to run."""

    # Define the wrapper function to return.
    @wraps(func)
    def wrapper(*args, **kwargs):
        # When wrapper() is called, get the current time.
        t_start = time.time()
        # Call the decorated function and store the result.
        result = func(*args, **kwargs)
        class_name = result.__class__.__name__
        # Get the total time it took to run, and print it.
        t_total = time.time() - t_start
        print('{}, {} took {}s'.format(class_name, func.__name__, t_total))
        return result

    return wrapper


def reindex_to_fac_index(func):
    """A decorator that reindex factor value to factor index (dynamic benchmark stock-date)."""

    # Define the wrapper function to return.
    @wraps(func)
    def wrapper(*args, **kwargs):
        factor = func(*args, **kwargs)
        print(factor)
        # reindex the factor value to the factor index
        if factor.reindex:
            factor.fac_value = factor.align_data_to_index(factor.fac_value, factor.fac_index, factor.fill_method)
            factor.index = factor.fac_value.index
            # change the column name as
            factor.fac_value.columns = [factor.name]
        return factor

    return wrapper


def fac_value_to_sr(func):
    """A decorator that adjusts for colunames and data type for fac_value
    general rule is to return a pd.Series, indexed first by trade_date, then by code."""

    # Define the wrapper function to return.
    @wraps(func)
    def wrapper(*args, **kwargs):
        factor = func(*args, **kwargs)
        # convert factor values to a pd Series
        if isinstance(factor.fac_value, pd.DataFrame):
            # import pdb
            # pdb.set_trace()
            factor.fac_value.columns = [factor.name]
            factor.fac_value = factor.fac_value[factor.name]
        else:
            factor.fac_value.name = factor.name
        return factor

    return wrapper


def get_fac_index(func):
    """A decorator that implement get_fac_indx method."""

    # Define the wrapper function to return.
    @wraps(func)
    def wrapper(*args, **kwargs):
        factor = args[0]
        factor.get_fac_idx()
        factor = func(*args, **kwargs)
        return factor

    return wrapper


#### utility functions
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
    if data.index.names != ['trade_date', 'code']:
        data = data.sort_index()
        for index_name in data.index.names:
            if index_name not in ['trade_date', 'code']:
                data = data.droplevel(level=index_name)
        # if reindex to stock universe index, only keep the last entry for the same trade_date code index
    data = data.reset_index().sort_values(['trade_date', 'code']).drop_duplicates(subset=['trade_date', 'code'],
                                                                                  keep='last').set_index(
        ['trade_date', 'code'])
    return data


def _sd_win_sort(raw_fac, sort_func=ECDF, reverse=False):
    """
    Perform  5% trime, zscore and 3 sigma winsorization and Ecdf sort on a group of a single factor

    Parameters
    ----------
    raw_fac : np.array or pd.Series
        unpreprocessed raw_fac.
    sort_func : function, optional
        preprocessing function. The default is ECDF.

    Returns
    -------
    preprocessed_factor
        pd.Series.

    """

    sd_fac = stand(raw_fac, 0.05)
    if reverse:
        sd_fac = -sd_fac
    sd_win_fac = std_winsor(sd_fac)
    fac_cdf = sort_func(sd_win_fac)

    return fac_cdf(sd_win_fac)


def groupby_fillna(df: pd.DataFrame, by: list = ['code'], method: str = 'ffill') -> pd.DataFrame:
    """
    fill na group by a particular index level

    Parameters
    ----------
    df : pd.DataFrame
        Multiindex df .
    by : list, optional
        the index level name to group by . The default is ['code'].
    method : str, optional
        na fill method. The default is 'ffill'.

    Returns
    -------
    df : Multiindex pd.DataFrame
        DESCRIPTION.

    """

    df = df.groupby(level=by).progress_apply(lambda x: x.fillna(method=method))

    return df


def percentile_inv(sr, value=None):
    """
    Given a pd.Series or np.array, return the  percentile

    Parameters
    ----------
    sr : pd.Series

    value: float or int,
        if None, default value would be the last value
        if 'all', value would be all the values in sr
    Returns
    -------
    float
        last time step's value percentile

    """
    idx = None
    if isinstance(sr, pd.DataFrame):
        sr = sr.iloc[:, 0]
    if isinstance(sr, pd.Series):
        idx = sr.index
    if value is None:
        value = sr[-1]
    elif value == 'all':
        value = sr
    ecdf = ECDF(sr)
    percentile = ecdf(value)
    if idx is not None:
        percentile = pd.Series(percentile, index=idx)
    return percentile


def discretize(sr, q):
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
    bins = pd.qcut(sr, q, retbins=True)[1]
    # discretize the varable
    if isinstance(q, int):
        labels = [i for i in range(q)]
    elif isinstance(q, list):
        labels = [i for i in range(len(q) - 1)]
    discretize_sr = pd.cut(sr, bins, labels=[i for i in labels])
    return discretize_sr


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

    # time_steps = np.array([i+1 for i in range(sr.shape[0])])
    # time_steps = sm.add_constant(time_steps)
    # model = sm.OLS(sr,exog=time_steps,hasconst=True).fit()
    # params = model.params
    # slope = params[1]
    # slope = slope/(np.abs(sr).mean())
    # if mean value of the y is 0, return NaN
    # if slope == np.NaN or sr.mean()==0:
    #     slope = np.NaN
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



def cal_quarterly_regress(code_df, factor_name):
    code_df = code_df.reset_index()
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


def top_minus_bottom(sr, quntiles=[0.9, 0.1], reverse=False):
    """
    compute cross-sectionally factor dispersion as top quentile minus bottom quantile

    Parameters
    ----------
    sr : pd.Series
        cross-sectional factor values.
    quntiles : list of float, optional
        top and bottom quantile . The default is [0.9,0.1].
    reverse : bool, optional
        indicator whether to reverse ranking of the factors . The default is FALSE.

    Returns
    -------
    sr： pd.Series
        replicated spread pd.Seres as factor value filter.

    """
    if reverse:
        sr = -sr

    top = sr.quantile(quntiles[0])
    bottom = sr.quantile(quntiles[1])

    return pd.Series(top - bottom, index=sr.index)


def rolling_oos_fitting(train_data, model,
                        window_size,
                        feature_names,
                        label_name,
                        sample_weighted=False):
    """ML-based rolling training and out_of_sample predicting

    Params:

    train_data: pd.DataFrame, where label is the last columns
    model: sklearn instanticated classifier or regressor
    window_size: int, rolling window size
    feature_names: list of strings, feature name to fit the model
    label_name: string, label name to fit the model
    Returns:
    pred_label: pd.Series, predicted labels forward in time
    """
    n_sample = train_data.shape[0]
    pred_label = pd.Series(index=train_data.index)
    for i in tqdm(range(n_sample - window_size), desc='Fitting ' + str(type(model))):

        feature_set = train_data.iloc[i:i + window_size][feature_names]
        # label must be shifted because performing forward prediction
        label_set = train_data.iloc[i:i + window_size][label_name]
        if sample_weighted:
            sample_weight = np.arange(1, window_size + 1)
            model = model.fit(feature_set, label_set.values.ravel(), sample_weight=sample_weight)
        else:
            model = model.fit(feature_set, label_set.values.ravel())

        # out of sample features for forward prediction
        oos_features = train_data.iloc[i + window_size][feature_names]
        next_pred = model.predict(oos_features.to_frame().T)

        pred_label.iloc[i + window_size] = next_pred

    return pred_label


def dataframe_fillna(data, factor_name, fill_method='ffill'):
    unstack_df = data[factor_name].unstack()
    if fill_method == "ffill":
        unstack_df = unstack_df.fillna(method='ffill')
    elif fill_method == "zero":
        unstack_df = unstack_df.fillna(value=0)
    else:
        unstack_df = unstack_df
    # import pdb
    # pdb.set_trace()
    data = unstack_df.stack()
    data = pd.DataFrame(data, columns=[factor_name])
    return data


def series_fillna(data, fill_method='ffill'):
    unstack_df = data.unstack()
    if fill_method == "ffill":
        unstack_df = unstack_df.fillna(method='ffill')
    elif fill_method == "zero":
        unstack_df = unstack_df.fillna(value=0)
    else:
        unstack_df = unstack_df
    data = unstack_df.stack()
    return data


def transfer_quarter_2_yoy(code_df, factor_name):
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


#### Framework
class Factor(object):
    """Basic factor class for further inheritance

       Compute method is modified by overwriting .compute method

       Factor values is reached by .fac_value attribute
    """

    def __init__(self, start_date=None, end_date=None, **kwargs):
        """
        Factor initialization
            pass in kwargs can overwrite default params
            can use getattr function to inspect additional attributes of the class

        The Factor class provides a framework for accessing and processing factor data stored in SQL databases. It contains default values for the SQL connection strings and the frequency of the data. The get_fac_idx method generates a factor index based on dynamic stock universe and the resample method allows the user to resample the index to a different frequency. The get_data method reads in one feature of raw source data, forward fills it to daily frequency, reindexes it to the time-evolved universe, and assigns the values to the fac_values attribute.

        Parameters
        ----------
        read_engine :str
            str name of the mysql engine.
            default= 'mysql+pymysql://develop:haikuan_2025@localhost:3306/factor_values_db'
        save_engine :  str
            str name of the mysql engine.
            default = 'mysql+pymysql://develop:haikuan_2025@localhost:3306/preprocessed_factor_data'
        start_date : int
            start of fac compute date
        end_date : int
            end_of fac compute date
        freq : str, optional
            frequency. The default is daily.
            if 'W-Tue', factor index is set to every Tuesday
        universe_table: str
            str name of the dynamic stock universe to trade
        reindex: bool
            bool indicating whether reindex the final factor value back to dynamic stock univser index. The default is True.
        rolling_win: int terms of days


        Returns
        -------
        None.

        """
        # initialize all default params
        self.save_engine = 'mysql+pymysql://develop:haikuan_2025@localhost:3306/factor_values_db'
        # self.read_engine = 'mysql+pymysql://develop:haikuan_2025@localhost:3306/preprocessed_factor_data'
        self.read_engine = 'mysql+pymysql://develop:haikuan_2025@localhost:3306/factor_research_full_a_share'
        self.freq = 'daily'
        self.sql_api = SQL_api.SQL_API(save_engine=create_engine(self.save_engine),
                                       read_engine=create_engine(self.read_engine))
        self.start_date = start_date
        self.end_date = end_date

        self.reindex = True
        # initialize all further computing attributes
        # save factor names
        self.name = str(type(self)).split('.', -1)[-1].split("'")[0]
        self.universe_table = 'stock_universe'

        # initialize other factor attributes
        self.rolling_win = None
        self.fac_index = None
        self.opt_2_trade = {}
        self.children_factor_value_name = None
        self.children_factor_value = None
        self.children_filter_factor_name = None
        self.children_filter_factor = None
        self.mask_factor_name = None
        self.group_by_factor_name = None
        self.n_shift = None
        self.n_roll = None  # rolling window size, specific for rolling factors
        self.fac_value = None
        self.fill_method = "ffill"
        ## overwritte default attributes
        for attr, val in kwargs.items():
            if attr not in dir(self):
                raise AttributeError(attr + ' attribute does not exist')
            else:
                setattr(self, attr, val)

    def resample(self, freq=None):
        """
        Resample the factor index from daily to specified frequency, currently only support weekly

        Returns
        -------
        None.

        """

        if freq is not None:
            self.freq = freq
        if self.freq == 'daily':
            return None
        if self.freq == "default":
            index_df = pd.DataFrame(index=self.fac_index)

            index_df['value'] = 0

            index_df = index_df.unstack()
            # convert index format d
            index_df = intDate2Date(index_df.reset_index()).set_index('trade_date')

            # resample time freq

            wednesday_time_index = pd.bdate_range(str(self.start_date), str(self.end_date), freq="W-Wed")
            tuesday_time_index = pd.bdate_range(str(self.start_date), str(self.end_date), freq="W-Tue")
            trade_dates = sorted(list(set(index_df.index.values)))
            resampled_time_index = []
            for date_, next_date_ in zip(trade_dates[:-1], trade_dates[1:]):
                if date_ in tuesday_time_index:
                    resampled_time_index.append(date_)
                    self.opt_2_trade.update({date_: next_date_})
                elif next_date_ in wednesday_time_index:
                    resampled_time_index.append(date_)
                    self.opt_2_trade.update({date_: next_date_})
                else:
                    pass

            # resample the index

            index_df = index_df.reindex(resampled_time_index).stack().reset_index().rename(
                columns={'level_0': 'trade_date'})
            # get back to int date format
            index_df = Date2intDate(index_df).set_index(['trade_date', 'code'])

            self.fac_index = index_df.index

        else:
            index_df = pd.DataFrame(index=self.fac_index)

            index_df['value'] = 0

            index_df = index_df.unstack()
            # convert index format d
            index_df = intDate2Date(index_df.reset_index()).set_index('trade_date')

            # resample time freq
            if self.freq.startswith("daybefore-"):
                freq = self.freq.replace("daybefore-", "")
                _resampled_time_index = pd.bdate_range(str(self.start_date), str(self.end_date), freq=freq)
                trade_dates = sorted(list(set(index_df.index.values)))
                resampled_time_index = []
                for date_, next_date_ in zip(trade_dates[:-1], trade_dates[1:]):
                    if next_date_ in _resampled_time_index:
                        resampled_time_index.append(date_)
                        self.opt_2_trade.update({date_: next_date_})
            else:

                resampled_time_index = pd.bdate_range(str(self.start_date), str(self.end_date), freq=self.freq)
                trade_dates = sorted(list(set(index_df.index.values)))
                for date_, next_date_ in zip(trade_dates[:-1], trade_dates[1:]):
                    if date_ in resampled_time_index:
                        self.opt_2_trade.update({date_: next_date_})
            # resample the index

            index_df = index_df.reindex(resampled_time_index).stack().reset_index().rename(
                columns={'level_0': 'trade_date'})
            # get back to int date format
            index_df = Date2intDate(index_df).set_index(['trade_date', 'code'])

            self.fac_index = index_df.index

    def get_fac_idx(self):

        """
        generate the trade_date, code specific factor index from dynamic index constituent stocks


        Returns
        -------
        None.

        """
        # avoid read in index multiple times
        if self.fac_index is not None:
            pass
        else:
            query_stmt = """select trade_date,code from {universe_table}
                            where trade_date >= {start_date} 
                            and trade_date <= {end_date};""".format(universe_table=self.universe_table,
                                                                    start_date=str(self.start_date),
                                                                    end_date=str(self.end_date))
            self.fac_index = self.sql_api.read_data_from(query_stmt).set_index(['trade_date', 'code']).index
            # resample to desired frequency
            self.resample()
            pass

    def get_data(self, tablename, field):

        """
            read in one feature of raw source data, forward fill the feature to daily frequency,
            reindex it to our time evoluted universe,assign values first to attribute .fac_values


        Parameters
        ----------
        tablename : str
            source data table name in sql .
        field : list of one str
            list of columnname of sql table e.g. ['net_operate_cash_flow'].

        Returns
        -------
        None

        """
        ## generate sql query
        trade_date_ls = self.fac_index.get_level_values(0).to_list()

        ## make sure we can forward fill
        trade_date_condition = {'field': 'trade_date',
                                'type': 'between',
                                'param': [min(trade_date_ls) - 30000, max(trade_date_ls)]}
        # set the query info dict
        query_info = {'method': 'select',
                      'sheet_name': tablename,
                      'tgt_field': {'way': 'show', 'field': ['trade_date', 'code'] + field},
                      'conditions': [trade_date_condition]}

        # read in necessary raw data
        raw_fac = self.sql_api.read_data_from(query_info)
        # set factor index
        raw_fac = raw_fac.set_index(['trade_date', 'code'])
        self.fac_value = raw_fac


    @staticmethod
    def align_data_to_index(data, index, fill_method='ffill'):
        """
        reindex data to new factor index, and ffill the missing data
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
        data = drop_extra_level_index(data)
        # reindex to our full fac index  for forward fill
        # if data is pd Series, convert to dataframe
        # get the union of data index and factor index
        full_fac_index = index.union(data.index)
        data = data.reindex(full_fac_index).sort_index()
        # import pdb
        # pdb.set_trace()
        # ffill by code
        if fill_method == "ffill":
            data = data.groupby(level='code').apply(lambda x: x.fillna(method='ffill'))
        elif fill_method == "zero":
            data = data.groupby(level='code').apply(lambda x: x.fillna(value=0))
        else:
            data = data.groupby(level='code').apply(lambda x: x)
        # data = data_fillna(data, fill_method)
        # data = data.groupby(level='code').apply(lambda x: x.fillna(method='ffill'))
        # print("class %s, fill method %s" % (self.__class__.__name__, self.fill_method))
        print("align index")
        # reindex back to fac tor index
        # data = data.reindex(index).sort_index().dropna(how = 'all')
        data = data.reindex(index).sort_index()
        return data

    def pass_in_basic_param(self, other_factor, **kwargs):
        """


        Pass in basic params to another factor including:
            self.save_engine
            self.read_engine
            self.start_date
            self.end_date
            self.sql_api
            self.fac_index
            self.full_fac_index

        if self.rolling_win is not None:
            other_factor.start_date = self.start_date - self.rolling_win

        Parameters
        ----------
        other_factor : Factor
            Factor pass params to .
        kwargs: attributes of factor, if passed, overwrites the attributes of other factor

        Returns
        -------
        other_factor : Factor
            Factor after passing params to.

        """

        other_factor.save_engine = self.save_engine
        other_factor.read_engine = self.read_engine
        other_factor.sql_api = self.sql_api
        other_factor.start_date = self.start_date
        other_factor.end_date = self.end_date
        other_factor.universe_table = self.universe_table
        other_factor.fac_index = self.fac_index
        # other_factor.reindex  = self.reindex
        other_factor.freq = self.freq

        ## overwritte attributes if pass in specific args
        for attr, val in kwargs.items():
            if attr not in dir(other_factor):
                raise AttributeError(attr + ' attribute does not exist')
            else:
                setattr(other_factor, attr, val)
        return other_factor

    def compute(self):
        """
        main function how to compute each factor signal, Customized for each specific factor

        Need to be overwritten in each Factor class

        general rule is to return a pd.Series, indexed first by trade_date, then by code"""
        pass

        # def reindex_to_fac_index(self):
        """
        reindex factor value to factor index (dynamic benchmark stock-date)
        """
        # if self.reindex:
        #     self.fac_value = self.align_data_to_index(self.fac_value, self.fac_index)
        #     self.index = self.fac_value.index
        #     # change the column name as
        #     self.fac_value.columns = [self.name]

    # def fac_value_to_sr(self):
    #     if isinstance(self.fac_value,pd.DataFrame):
    #         self.fac_value.columns = [self.name]
    #         self.fac_value = self.fac_value[self.name]
    #     else:
    #         self.fac_value.name = self.name
    # @timer
    # @get_fac_index
    # @fac_value_to_sr
    # @memorize
    # @reindex_to_fac_index
    def run_pipline(self):
        """
        container for compute method"""
        self.get_fac_idx()
        self.compute()
        if self.reindex:
            self.fac_value = self.align_data_to_index(self.fac_value, self.fac_index, self.fill_method)
            self.index = self.fac_value.index
            # change the column name as
            self.fac_value.columns = [self.name]

        if isinstance(self.fac_value, pd.DataFrame):
            # import pdb
            # pdb.set_trace()
            self.fac_value.columns = [self.name]
            self.fac_value = self.fac_value[self.name]
        else:
            self.fac_value.name = self.name



    def save_to_sql(self, tablename):
        """
        Save new factor value to sql database

        Parameters
        ----------
        tablename : str
            tablename in mysql server.

        Returns
        -------
        None.

        """
        self.sql_api.insert_new_data_to(self.fac_value, tablename)

    def instantiate_child_factors(self, **kwargs):
        """
        Creating a pd.DataFrame of instanticated chidren factors to compute sythetic factors

        passing in fac_index and full_fac_index to avoid reimporting

        Parameters
        ----------
        **kwargs : Class of Continous or Sythetic Factor


        Returns
        -------
        Self


        """
        if self.children_factor_value is None:
            self.children_factor_value = pd.DataFrame()

        for factor_name, Factor in kwargs.items():
            print('Creating Children Factor: ', factor_name)
            factor = Factor()

            factor = self.pass_in_basic_param(factor, reindex=False)
            # compute child factor
            factor.run_pipline()
            fac_df = factor.fac_value.to_frame()
            if self.children_factor_value.empty:
                print('Creating Children Factor level2: , empty', factor_name)

                self.children_factor_value = fac_df

            else:
                # drop data entry with all nas to avoid the df become too large
                print('before')
                print('Creating Children Factor level2: ', factor_name)

                self.children_factor_value = self.children_factor_value.merge(fac_df,
                                                                              left_index=True,
                                                                              right_index=True,
                                                                              how='outer')

                # pdb.set_trace()
        # if end_date in the index,foward fill it, in case for further computation group by end date
        # if 'end_date' in self.children_factor_value.index.names:
        #     self.children_factor_value = self.children_factor_value.reset_index(level = 'end_date')\
        #         .groupby(level = 'code').apply(lambda x: x.fillna(method = 'ffill')).dropna(subset = ['end_date'])\
        #         .set_index(['end_date'],append = True)
        ## drop data entries with all nas
        # print("before drop %s" % len(self.children_factor_value))
        # self.children_factor_value = self.children_factor_value.dropna(how = 'all')
        # print("after drop %s" % len(self.children_factor_value))
        # store the chlild factor names
        self.children_factor_value_name = self.children_factor_value.columns.to_list()


    def instantiate_child_filter_factor(self, **kwargs):
        """
        Creating a pd.DataFrame of instanticated chidren filter or groupby factors

        passing in fac_index and full_fac_index to avoid reimporting

        Parameters
        ----------
        **kwargs : Class of Continous or Factor


        Returns
        -------
        Self.

        """
        # if we do not create the filter factor from parent factor, create from scratch
        if self.children_filter_factor is None:
            self.children_filter_factor = pd.DataFrame()
            for factor_name, Factor in kwargs.items():
                print('Creating Children Factor: ', factor_name)
                factor = Factor()
                factor = self.pass_in_basic_param(factor)
                # compute child factor
                factor.run_pipline()
                self.children_filter_factor[factor_name] = factor.fac_value
            # store the chlild factor names
            self.children_filter_factor_name = self.children_filter_factor.columns.to_list()
        # else do nothing
        else:
            pass


    def mask(self, mask_Factor, operator=None, other=None):
        """
        filter out undesireable stocks

        used by:
            numeric_fac.mask(filter_fac.fac_value (some_conditions))

        Parameters
        ----------
        mask_fac : class of Categorical Factors

        operator: str or None, one of ['<','>','<=','>=','==','!=','not']
                if None, mask_fac would be a bool factor

        other: float or Factor, default is None

        Returns
        -------
        None.

        """
        # instantiate filter factor
        mask_factor = mask_Factor()
        mask_factor = self.pass_in_basic_param(mask_factor, fac_index=self.fac_value.index)
        mask_factor.run_pipline()
        self.mask_factor_name = mask_factor.name
        self.children_filter_factor = mask_factor.fac_value
        if operator is None:
            conditional = (mask_factor.fac_value)
        elif operator == '<':
            conditional = (mask_factor < other)
        elif operator == '>':
            conditional = (mask_factor > other)
        elif operator == '<=':
            conditional = (mask_factor <= other)
        elif operator == '>=':
            conditional = (mask_factor >= other)
        elif operator == '==':
            conditional = (mask_factor == other)
        elif operator == '!=':
            conditional = (mask_factor != other)
        elif operator == 'not':
            conditional = ~(mask_factor.fac_value)
        self.fac_value = self.fac_value[conditional]
        # update factor index
        self.fac_index = self.fac_value.index



class CategoricalFactor(Factor):
    def __eq__(self, other):
        if isinstance(other, Factor):
            return self.fac_value == other.fac_value
        else:
            return self.fac_value == other

    def __ne__(self, other):
        if isinstance(other, Factor):
            return self.fac_value != other.fac_value
        else:
            return self.fac_value != other

            ## overload logic operator

    def __and__(self, other):
        return self.fac_value & other.fac_value

    def __or__(self, other):
        return self.fac_value | other.fac_value


class ContinousFactor(Factor):
    """
    Basic type of continous factor, used for inheritance
    """

    ## overload math operator
    # add two factors
    def __add__(self, other):
        if isinstance(other, Factor):
            return self.fac_value + other.fac_value

        else:
            return self.fac_value + other

    # substract two factors
    def __sub__(self, other):
        if isinstance(other, Factor):
            return self.fac_value - other.fac_value
        else:
            return self.fac_value - other

    # multiple two factors
    def __mul__(self, other):
        if isinstance(other, Factor):
            return self.fac_value * other.fac_value
        else:
            return self.fac_value * other

    # divide two factors
    def __truediv__(self, other):
        if isinstance(other, Factor):
            return self.fac_value / other.fac_value
        else:
            return self.fac_value / other

    # power
    def __pow__(self, other):
        return self.fac_value ** other

    ## overload comparision operator
    def __lt__(self, other):
        if isinstance(other, Factor):
            return self.fac_value < other.fac_value
        else:
            return self.fac_value < other

    def __gt__(self, other):
        if isinstance(other, Factor):
            return self.fac_value > other.fac_value
        else:
            return self.fac_value > other

    def __le__(self, other):
        if isinstance(other, Factor):
            return self.fac_value <= other.fac_value
        else:
            return self.fac_value <= other

    def __ge__(self, other):
        if isinstance(other, Factor):
            return self.fac_value >= other.fac_value
        else:
            return self.fac_value >= other

    def __eq__(self, other):
        if isinstance(other, Factor):
            return self.fac_value == other.fac_value
        else:
            return self.fac_value == other

    def __ne__(self, other):
        if isinstance(other, Factor):
            return self.fac_value != other.fac_value
        else:
            return self.fac_value != other

    def standrdize(self, func, groupby_Fac=None, reverse=False):

        """
        specify how to standardize each raw_factor values

        Parameters
        ----------
        groupby_Fac : Class of Categorical Factor
                         if none, standardized cross-sectionally

        reverse: optional bool

            Flag indicating whether revese the direction of raw factor values
        func: Function
            standardize function
        Returns
        -------
        Continous Factor
            continous factor after .

        """

        # convert back to DataFrame
        fac_value_df = self.fac_value.reset_index()
        if groupby_Fac:
            groupby_fac = groupby_Fac()
            groupby_fac = self.pass_in_basic_param(groupby_fac)
            groupby_fac.run_pipline()
            self.group_by_factor_name = groupby_fac.name
            # get categorical variable name
            cate_name = groupby_fac.fac_value.name
            # convert two series to dataframe
            groupby_cate_df = groupby_fac.fac_value.reset_index()
            # merge two df
            fac_value_df = fac_value_df.merge(groupby_cate_df, on=['trade_date', 'code'], how='left')
            # set index
            fac_value_df.set_index(['trade_date', cate_name, 'code'], inplace=True)
            fac_value_df = fac_value_df.groupby(level=['trade_date', cate_name]).progress_apply(
                lambda x: x.apply(func, axis=0, reverse=reverse))
            # convert back to Series
            self.fac_value = fac_value_df.droplevel(1).iloc[:, 0]

        else:
            fac_value_df.set_index(['trade_date', 'code'], inplace=True)
            fac_value_df = fac_value_df.groupby(level='trade_date').progress_apply(
                lambda x: x.apply(func, axis=0, reverse=reverse))
            # convert back to Series
            self.fac_value = fac_value_df.iloc[:, 0]


    def rolling_apply(self, window_size, apply_func):
        """
        rolling apply a function on the factor by specified window_size, beware of the frequency when using this function

        Parameters
        ----------
        window_size : int
            window size .
        func : function
            applied function .

        Returns
        -------
        self
            factor after rolling function .

        """

        rolling_fac_value = self.fac_value.unstack().rolling(window_size, min_periods=int(window_size / 2)).apply(
            apply_func).stack()
        # reindex to fac index
        rolling_fac_value = rolling_fac_value.reindex(self.fac_index)
        # forwardfill values
        rolling_fac_value = rolling_fac_value.groupby(level=1).progress_apply(
            lambda x: x.fillna(method='ffill')).sort_index()
        # fillna

        self.fac_value = rolling_fac_value.copy()



class GroupFactor(ContinousFactor):
    """
    Factor Greated by linear combined multiple Continous Factor, mask and standardization is performed at children factor level
    """

    def standrdize_child_fac(self, func, groupby_Fac=None, reverse=False):

        """
        specify how to standardize each raw_factor values

        Parameters
        ----------
        groupby_Fac : Class of Categorical Factor

        reverse: optional bool

            Flag indicating whether revese the direction of raw factor values
        func: Function
            standardize function
        Returns
        -------
        Continous Factor
            continous factor after .

        """
        # convert back to DataFrame
        fac_value_df = self.align_data_to_index(self.children_factor_value, self.fac_index).reset_index().dropna()
        if groupby_Fac is not None:
            # # instantiate group by factor
            groupby_fac = groupby_Fac()
            groupby_fac = self.pass_in_basic_param(groupby_fac)
            groupby_fac.run_pipline()
            # get categorical variable name
            cate_name = groupby_fac.fac_value.name
            # convert two series to dataframe
            groupby_cate_df = groupby_fac.fac_value.reset_index()
            # merge two df
            fac_value_df = fac_value_df.merge(groupby_cate_df, on=['trade_date', 'code'], how='left')
            # set index
            fac_value_df.set_index(['trade_date', cate_name, 'code'], inplace=True)
            fac_value_df = fac_value_df.groupby(level=['trade_date', cate_name]).progress_apply(
                lambda x: x.apply(func, axis=0, reverse=reverse))
            # convert back to Series
            self.children_factor_value = fac_value_df.droplevel(1)

        else:
            fac_value_df.set_index(['trade_date', 'code'], inplace=True)
            fac_value_df = fac_value_df.groupby(level='trade_date').progress_apply(
                lambda x: x.apply(func, axis=0, reverse=reverse))
            self.children_factor_value = fac_value_df


    def rolling_apply_child_fac(self, window_size, apply_func):
        """
        rolling apply a function on the factor by specified window_size, beware of the frequency when using this function

        Parameters
        ----------
        window_size : int
            window size .
        func : function
            applied function .

        Returns
        -------
        self
            factor after rolling function .

        """

        rolling_fac_value = self.children_factor_value.copy().stack()
        rolling_fac_value = rolling_fac_value.rename_axis(['trade_date', 'code', 'factor'])
        rolling_fac_value = rolling_fac_value.groupby(level=['code', 'factor']).progress_apply(
            lambda x: x.rolling(window_size, min_periods=int(window_size / 2)).apply(apply_func))
        rolling_fac_value = rolling_fac_value.dropna().unstack()
        # realign index
        rolling_fac_value = rolling_fac_value.reindex(self.fac_index)
        self.children_factor_value = rolling_fac_value


    def mask_child_fac(self, mask_Factor, operator=None, other=None):
        """
        filter out undesireable stocks for children facrors
        note: the children factor df must be of a pd.Dataframe first indexed
        by trade_date, then by code

        used by:
            numeric_fac.mask(filter_fac.fac_value (some_conditions))

        Parameters
        ----------
        mask_fac : class of Factor
            pd.Series of binary variable.
        operator: str or None, one of ['<','>','<=','>=','=='.'!=']
                if None, mask_fac would be a bool factor

        other: float or Factor, default is None

        Returns
        -------
        None.

        """
        # # instantiate filter factor
        # self.children_factor_value = self.align_data_to_index(self.children_factor_value, self.fac_index)
        mask_factor = mask_Factor()
        mask_factor = self.pass_in_basic_param(mask_factor, fac_index=self.children_factor_value.index)
        mask_factor.run_pipline()
        self.mask_factor_name = mask_factor.name
        if operator is None:
            conditional = (mask_factor.fac_value)
        elif operator == '<':
            conditional = (mask_factor < other)
        elif operator == '>':
            conditional = (mask_factor > other)
        elif operator == '<=':
            conditional = (mask_factor <= other)
        elif operator == '>=':
            conditional = (mask_factor >= other)
        elif operator == '==':
            conditional = (mask_factor == other)
        elif operator == '!=':
            conditional = (mask_factor != other)
        elif operator == 'not':
            conditional = ~(mask_factor.fac_value.astype(bool))
        self.children_factor_value = self.children_factor_value[conditional]



class FundamentalFactor(ContinousFactor):
    """For factors get from financial report, add report end_date as additional index for some adjustment"""

    def get_data(self, tablename, field):
        """
            read in one feature of raw source data, forward fill the feature to daily frequency,
            reindex it to our time evoluted universe,assign values first to attribute .fac_values


        Parameters
        ----------
        tablename : str
            source data table name in sql .
        field : list of one str
            list of columnname of sql table e.g. ['net_operate_cash_flow'].

        Returns
        -------
        None

        """
        ## generate sql query
        trade_date_ls = self.fac_index.get_level_values(0).to_list()
        # min_date = min(trade_date_ls)
        max_date = max(trade_date_ls)
        # fetech additional data for forward fill
        # query_stmt = """select trade_date,`code`,end_date,{field} from {tablename}
        #                 where trade_date between {min_date} and {max_date}""".format(field = field[0],
        #                                                                         tablename = tablename,
        #                                                                         min_date = min_date-10500,
        #                                                                         max_date = max_date)
        query_stmt = """select trade_date,`code`,end_date,{field} from {tablename} 
                        where trade_date < {max_date}""".format(field=field[0],
                                                                tablename=tablename,
                                                                max_date=max_date)

        # read in necessary raw data
        raw_fac = self.sql_api.read_data_from(query_stmt)
        # set factor index
        raw_fac = raw_fac.set_index(['trade_date', 'code', 'end_date'])
        self.fac_value = raw_fac


class StatusFactorStartDate(CategoricalFactor):
    def get_data(self,tablename,field):

        ## generate sql query
        trade_date_ls = self.fac_index.get_level_values(0).to_list()

        # set the query
        query_stmt = """select start_date,`code`,{field} from {tablename}
                        where trade_date <= {max_trade_date}""".format(field = field[0],
                        tablename = tablename,max_trade_date = max(trade_date_ls))
        # read in necessary raw data
        raw_fac  = self.sql_api.read_data_from(query_stmt)
        # set factor index
        raw_fac['trade_date'] =raw_fac['start_date'].map(lambda x: int(x.strftime("%Y%m%d")))
        raw_fac = raw_fac.set_index(['trade_date','code'])
        self.fac_value = raw_fac


class STFlagNameHistory(StatusFactorStartDate):
    """st_flag from history name"""

    def compute(self):
        # set factor index
        tablename = 'name_history_stk'
        field = ['new_name']
        self.get_data(tablename=tablename,
                      field=field)
        self.fac_value = (self.fac_value['new_name'].map(lambda x: "st" in x.lower() if type(x) is str else False))


class STFlagNetProfit(FundamentalFactor):
    """st_flag based on net income data,  """

    def compute(self):
        tablename = "income_stk"
        field = ['net_profit']
        self.get_data(tablename=tablename,
                      field=field)
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
        self.fac_value = self.fac_value.groupby(level='code').apply(lambda s: cal_st_flag(s))

        # import pdb
        # pdb.set_trace()


class STFlag(GroupFactor):
    """
    compute st flag, combine STFlagNameHistory and STFlagNetProfit
    """
    def compute(self):
        self.instantiate_child_factors(STFlagNameHistory=STFlagNameHistory,
                                       STFlagNetProfit=STFlagNetProfit,
                                       )
        self.fac_value = (self.children_factor_value['STFlagNameHistory'] + self.children_factor_value['STFlagNetProfit']).map(lambda x: 1 if x > 0 else 0)
