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
def clean_memory():
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
        factor.fac_value = factor.align_data_to_index(factor.fac_value,factor.fac_index, factor.fill_method)
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
    if isinstance(factor.fac_value,pd.DataFrame):
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
def intDate2Date(df,column_name = 'trade_date'):
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
    
    df.trade_date = df.trade_date.astype(str).apply(lambda x: x.replace('-','')).astype(int)
    
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
    if data.index.names!= ['trade_date','code']:
        data = data.sort_index()
        for index_name in data.index.names:
            if index_name not in ['trade_date','code']:
                data = data.droplevel(level = index_name)
        # if reindex to stock universe index, only keep the last entry for the same trade_date code index 
    data = data.reset_index().sort_values(['trade_date','code']).drop_duplicates(subset = ['trade_date','code'],keep = 'last').set_index(['trade_date','code'])
    return data
    
def _sd_win_sort(raw_fac,sort_func = ECDF,reverse = False):
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

def groupby_fillna(df: pd.DataFrame ,by: list = ['code'], method: str = 'ffill') -> pd.DataFrame: 
    
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
    
    df =  df.groupby(level=by).progress_apply(lambda x:x.fillna(method = method))
    
    return df

def percentile_inv(sr,value = None):
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
    if isinstance(sr,pd.DataFrame):
        sr = sr.iloc[:,0]
    if isinstance(sr,pd.Series):
        idx = sr.index
    if value is None:
        value = sr[-1]
    elif value =='all':
        value = sr
    ecdf = ECDF(sr)
    percentile = ecdf(value)
    if idx is not None:
        percentile = pd.Series(percentile,index = idx)
    return percentile

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
    discretize_sr = pd.cut(sr,bins,labels = [i for i in labels])
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
    x = np.arange(1, len(y)+1, 1)
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
        self.fitted_values = (self.params * self.df.drop(columns= self.y_name)).sum(axis =1).replace(0,np.NAN)
        
        # Calculate the residuals
        self.residuals = self.df[self.y_name] - self.fitted_values
        
        return self


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
            trend_value =  trend_regress(last_quarter_values)
            trade_date_2_trend.update({trade_date: trend_value})
    code_df['trend'] = code_df['trade_date'].map(trade_date_2_trend)
    return code_df.set_index(['trade_date',  'end_date'])['trend']

def top_minus_bottom(sr,quntiles = [0.9,0.1],reverse = False):
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
    
    
    return pd.Series(top - bottom,index = sr.index)


def rolling_oos_fitting(train_data,model,
                        window_size,
                        feature_names,
                        label_name,
                        sample_weighted = False):
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
    for i in tqdm(range(n_sample-window_size),desc = 'Fitting ' + str(type(model))):
        
        feature_set = train_data.iloc[i:i+window_size][feature_names]
        # label must be shifted because performing forward prediction
        label_set = train_data.iloc[i:i+window_size][label_name]
        if sample_weighted:
            sample_weight = np.arange(1,window_size+1)
            model = model.fit(feature_set,label_set.values.ravel(),sample_weight = sample_weight)
        else:
            model = model.fit(feature_set,label_set.values.ravel())
        
       
        # out of sample features for forward prediction
        oos_features = train_data.iloc[i+window_size][feature_names]
        next_pred = model.predict(oos_features.to_frame().T)
        
        pred_label.iloc[i+window_size] = next_pred
    
    
    return pred_label


def dataframe_fillna(data, factor_name, fill_method='ffill'):
    unstack_df = data[factor_name].unstack()
    if fill_method == "ffill":
        unstack_df = unstack_df.fillna(method = 'ffill')
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
        unstack_df = unstack_df.fillna(method = 'ffill')
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
        lambda x: datetime.datetime(year=x.year-1, month=x.month, day=x.day).strftime("%Y-%m-%d"))

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
    def __init__(self,start_date = None,end_date = None,**kwargs):
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
        #initialize all default params
        self.save_engine = 'mysql+pymysql://develop:haikuan_2025@192.168.110.66:3306/factor_values_db'
        # self.read_engine = 'mysql+pymysql://develop:haikuan_2025@localhost:3306/preprocessed_factor_data'
        self.read_engine = 'mysql+pymysql://develop:haikuan_2025@192.168.110.66:3306/factor_research_full_a_share'
        self.freq = kwargs.get('freq', 'daily')
        self.sql_api = SQL_api.SQL_API(save_engine = create_engine(self.save_engine), 
                                       read_engine = create_engine(self.read_engine))
        self.start_date = start_date
        self.end_date = end_date
        
        self.reindex = True
        # initialize all further computing attributes 
        # save factor names 
        self.name = str(type(self)).split('.',-1)[-1].split("'")[0]
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
        self.n_roll = None # rolling window size, specific for rolling factors 
        self.fac_value = None
        self.fill_method = "ffill"
        ## overwritte default attributes 
        for attr,val in kwargs.items():
            if attr not in dir(self):
                raise AttributeError(attr + ' attribute does not exist')
            else:
                setattr(self, attr, val)

    def resample(self,freq = None):
        """
        Resample the factor index from daily to specified frequency, currently only support weekly

        Returns
        -------
        None.

        """

        if freq is not None:
            self.freq = freq
        if self.freq  == 'daily':
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
            index_df = pd.DataFrame(index= self.fac_index)
            
            index_df['value'] = 0
            
            index_df = index_df.unstack()
            # convert index format d
            index_df = intDate2Date(index_df.reset_index()).set_index('trade_date')
            
            # resample time freq
            if self.freq.startswith("daybefore-"):
                freq = self.freq.replace("daybefore-", "")
                _resampled_time_index = pd.bdate_range(str(self.start_date),str(self.end_date),freq = freq)
                trade_dates = sorted(list(set(index_df.index.values)))
                resampled_time_index = []
                for date_, next_date_ in zip(trade_dates[:-1], trade_dates[1:]):
                    if next_date_ in _resampled_time_index:
                        resampled_time_index.append(date_)
                        self.opt_2_trade.update({date_: next_date_})
            else:

                resampled_time_index = pd.bdate_range(str(self.start_date),str(self.end_date),freq = self.freq)
                trade_dates = sorted(list(set(index_df.index.values)))
                for date_, next_date_ in zip(trade_dates[:-1], trade_dates[1:]):
                    if date_ in resampled_time_index:
                        self.opt_2_trade.update({date_: next_date_})
            # resample the index

            index_df = index_df.reindex(resampled_time_index).stack().reset_index().rename(columns = {'level_0':'trade_date'})
            # get back to int date format 
            index_df = Date2intDate(index_df).set_index(['trade_date','code'])
            
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
            return self
        else:
            query_stmt = """select trade_date,code from {universe_table}
                            where trade_date >= {start_date} 
                            and trade_date <= {end_date};""".format(universe_table = self.universe_table,
                                                                    start_date = str(self.start_date),
                                                                    end_date = str(self.end_date))
            self.fac_index = self.sql_api.read_data_from(query_stmt).set_index(['trade_date','code']).index
            # resample to desired frequency
            self.resample()
            return self
            
    def get_data(self,tablename,field):
        
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
          'param': [min(trade_date_ls)-30000, max(trade_date_ls)]}
        # set the query info dict
        query_info =   {'method': 'select',
                     'sheet_name': tablename,
                     'tgt_field': {'way': 'show', 'field': ['trade_date','code']+ field},
                     'conditions': [trade_date_condition]}
        
        # read in necessary raw data 
        raw_fac  = self.sql_api.read_data_from(query_info)
        # set factor index 
        raw_fac = raw_fac.set_index(['trade_date','code'])
        self.fac_value = raw_fac
        return self





    @staticmethod
    def align_data_to_index(data,index, fill_method='ffill'):
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
            data = data.groupby(level='code').apply(lambda x:x.fillna(method = 'ffill'))
        elif fill_method == "zero":
            data = data.groupby(level='code').apply(lambda x:x.fillna(value=0))
        else:
            data = data.groupby(level='code').apply(lambda x:x)
        # data = data_fillna(data, fill_method)
        # data = data.groupby(level='code').apply(lambda x: x.fillna(method='ffill'))
        # print("class %s, fill method %s" % (self.__class__.__name__, self.fill_method))
        print("align index")
        # reindex back to fac tor index
        # data = data.reindex(index).sort_index().dropna(how = 'all')
        data = data.reindex(index).sort_index()
        return data
    
    def pass_in_basic_param(self,other_factor,**kwargs):
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
        for attr,val in kwargs.items():
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

    @timer
    @get_fac_index
    @fac_value_to_sr
    @memorize
    @reindex_to_fac_index
    def run_pipline(self):
        """
        container for compute method"""
        self.compute()

        return self
    
    def save_to_sql(self,tablename):
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
        

    def instantiate_child_factors(self,**kwargs):
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
            print('Creating Children Factor: ',factor_name)
            factor = Factor()

            factor = self.pass_in_basic_param(factor,reindex = False)
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
                                                                              how = 'outer')


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
        return self
    
    def instantiate_child_filter_factor(self,**kwargs):
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
                print('Creating Children Factor: ',factor_name)
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
        return self
    
    def mask(self,mask_Factor,operator = None,other = None):
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
        mask_factor = self.pass_in_basic_param(mask_factor,fac_index = self.fac_value.index)
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
        return self
    
class CategoricalFactor(Factor):
    def __eq__(self,other):
        if isinstance(other,Factor): 
            return  self.fac_value == other.fac_value
        else:
            return  self.fac_value == other
    
    def __ne__(self, other):
        if isinstance(other,Factor): 
            return  self.fac_value != other.fac_value
        else:
            return  self.fac_value != other        
            
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
        if isinstance(other,Factor):
            return self.fac_value + other.fac_value

        else:
            return self.fac_value + other

    
    # substract two factors 
    def __sub__(self, other):
        if isinstance(other,Factor): 
            return self.fac_value - other.fac_value
        else:
            return self.fac_value - other
    
    # multiple two factors 
    def __mul__(self, other):
        if isinstance(other,Factor): 
            return self.fac_value * other.fac_value
        else:
            return self.fac_value * other
    
    # divide two factors 
    def __truediv__(self,other):
        if isinstance(other,Factor): 
            return  self.fac_value / other.fac_value
        else:
            return  self.fac_value / other
    # power
    def __pow__(self, other):
            return self.fac_value**other
        
    ## overload comparision operator  
    def __lt__(self, other):
        if isinstance(other,Factor): 
            return  self.fac_value < other.fac_value
        else:
            return  self.fac_value < other
    
    def __gt__(self, other):
        if isinstance(other,Factor): 
            return  self.fac_value > other.fac_value
        else:
            return  self.fac_value > other
    
    def __le__(self, other):
        if isinstance(other,Factor): 
            return  self.fac_value <= other.fac_value
        else:
            return  self.fac_value <= other
    
    def __ge__(self, other):
        if isinstance(other,Factor): 
            return  self.fac_value >= other.fac_value
        else:
            return  self.fac_value >= other
    
    def __eq__(self,other):
        if isinstance(other,Factor): 
            return  self.fac_value == other.fac_value
        else:
            return  self.fac_value == other
    
    def __ne__(self, other):
        if isinstance(other,Factor): 
            return  self.fac_value != other.fac_value
        else:
            return  self.fac_value != other        
    def standrdize(self,func,groupby_Fac = None,reverse = False):
        
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
            fac_value_df = fac_value_df.merge(groupby_cate_df, on = ['trade_date','code'], how = 'left')
            # set index 
            fac_value_df.set_index(['trade_date',cate_name,'code'],inplace = True)
            fac_value_df = fac_value_df.groupby(level=['trade_date',cate_name]).progress_apply(lambda x: x.apply(func,axis = 0,reverse = reverse))
            # convert back to Series
            self.fac_value = fac_value_df.droplevel(1).iloc[:,0]
            return self 
        else:
            fac_value_df.set_index(['trade_date','code'],inplace = True)
            fac_value_df = fac_value_df.groupby(level = 'trade_date').progress_apply(lambda x: x.apply(func,axis = 0,reverse = reverse))
            # convert back to Series
            self.fac_value = fac_value_df.iloc[:,0]
            return self
    
    def rolling_apply(self,window_size,apply_func):
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
        
        rolling_fac_value = self.fac_value.unstack().rolling(window_size,min_periods=int(window_size/2)).apply(apply_func).stack()
        # reindex to fac index 
        rolling_fac_value = rolling_fac_value.reindex(self.fac_index)
        # forwardfill values 
        rolling_fac_value = rolling_fac_value.groupby(level=1).progress_apply(lambda x:x.fillna(method = 'ffill')).sort_index()
        # fillna 
        
        self.fac_value = rolling_fac_value.copy()
        return self
    
class GroupFactor(ContinousFactor):
    
    """
    Factor Greated by linear combined multiple Continous Factor, mask and standardization is performed at children factor level 
    """
    
    def standrdize_child_fac(self,func,groupby_Fac = None,reverse = False):
        
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
            fac_value_df = fac_value_df.merge(groupby_cate_df, on = ['trade_date','code'], how = 'left')
            # set index 
            fac_value_df.set_index(['trade_date',cate_name,'code'],inplace = True)
            fac_value_df = fac_value_df.groupby(level=['trade_date',cate_name]).progress_apply(lambda x: x.apply(func,axis = 0,reverse = reverse))
            # convert back to Series
            self.children_factor_value = fac_value_df.droplevel(1)
            return self 
        else:
            fac_value_df.set_index(['trade_date','code'],inplace = True)
            fac_value_df = fac_value_df.groupby(level = 'trade_date').progress_apply(lambda x: x.apply(func,axis = 0,reverse = reverse))
            self.children_factor_value = fac_value_df
            return self
    
    
    def rolling_apply_child_fac(self,window_size,apply_func):
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
        rolling_fac_value = rolling_fac_value.rename_axis(['trade_date','code','factor'])
        rolling_fac_value = rolling_fac_value.groupby(level=['code','factor']).progress_apply(lambda x: x.rolling(window_size,min_periods=int(window_size/2)).apply(apply_func))
        rolling_fac_value = rolling_fac_value.dropna().unstack()
        # realign index 
        rolling_fac_value = rolling_fac_value.reindex(self.fac_index)
        self.children_factor_value = rolling_fac_value
        return self
    
    
    def mask_child_fac(self,mask_Factor,operator = None,other = None):
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
        mask_factor = self.pass_in_basic_param(mask_factor,fac_index = self.children_factor_value.index)
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
        return self


class FundamentalFactor(ContinousFactor):
    """For factors get from financial report, add report end_date as additional index for some adjustment"""
    def get_data(self,tablename,field):
        
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
                        where trade_date < {max_date}""".format(field = field[0],
                                                                                tablename = tablename,
                                                                                max_date = max_date)
        
        # read in necessary raw data 
        raw_fac  = self.sql_api.read_data_from(query_stmt)
        # set factor index 
        raw_fac = raw_fac.set_index(['trade_date','code','end_date'])
        self.fac_value = raw_fac
        return self


#### TradingVolume Factors
class AdjClosePrice(GroupFactor):
    """AdjClosePrice as a continous factor"""
    def compute(self):
        self.instantiate_child_factors(UnAdjClosePrice = UnAdjClosePrice,
                                       AdjFactor = AdjFactor)
        # np.argmax +1 because python is 0 indexed 
        self.fac_value = self.children_factor_value['UnAdjClosePrice'] * self.children_factor_value['AdjFactor']

        return self


class AdjClosePriceWeekly(GroupFactor):
    """weekly sampled AdjClosePrice as a continous factor"""
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice,
                                    )

        # np.argmax +1 because python is 0 indexed
        self.get_fac_idx()
#         import pdb
#         pdb.set_trace()
        self.fac_value = self.align_data_to_index(self.children_factor_value, self.fac_index)
        return self

class UnAdjClosePrice(ContinousFactor):
    """Close price unadjusted for dividends and stock split"""
    def compute(self):
        tablename = 'daily_trading_data_unadjusted'
        field = ['close']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        return self

class AdjFactor(ContinousFactor):
    """Close price unadjusted for dividends and stock split"""
    def compute(self):
        tablename = 'daily_trading_data'
        field = ['factor']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        return self


class UnAdjPreClosePrice(ContinousFactor):
    """PreClosePrice as a continous factor"""
    def compute(self):
        tablename = 'daily_trading_data_unadjusted'
        field = ['pre_close']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        return self


class AdjPreClosePrice(ContinousFactor):
    """PreClosePrice as a continous factor"""
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice,
                                    )
        # np.argmax +1 because python is 0 indexed

        self.fac_value = self.children_factor_value.sort_index(level=[1,0]).groupby(level=1)['AdjClosePrice'].shift(1).sort_index(level=[0,1])

        return self
    
class OpenPrice(ContinousFactor):
    """OpenPrice as a continous factor"""
    def compute(self):
        tablename = 'daily_trading_data'
        field = ['open']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        return self
    
class VWAPrice(ContinousFactor):
    """Volume weighted average price as a continous factor"""
    def compute(self):
        tablename = 'daily_trading_data'
        field = ['avg']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        return self

class Volume(ContinousFactor):
    """Daily Trading volume (in number of shares) as a continous factor"""

    def compute(self):
        tablename = 'daily_trading_data'
        field = ['volume']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        return self
    
class Amount(ContinousFactor):
    """Daily Trading dollar amount as a continous factor"""
    def compute(self):
        tablename = 'daily_trading_data'
        field = ['money']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        return self

class High(ContinousFactor):
    """Daily Trading  high price as a continous factor"""
    def compute(self):
        tablename = 'daily_trading_data'
        field = ['high']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        return self

class Low(ContinousFactor):
    """Daily Trading  low price as a continous factor"""
    def compute(self):
        tablename = 'daily_trading_data'
        field = ['low']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        return self
    
class HighLimit(ContinousFactor):
    """Daily Trading high limit price as a continous factor"""
    def compute(self):
        tablename = 'daily_trading_data'
        field = ['high_limit']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        return self

class LowLimit(ContinousFactor):
    """Daily Trading high limit price as a continous factor"""
    def compute(self):
        tablename = 'daily_trading_data'
        field = ['low_limit']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        return self

class TurnOverRatio(ContinousFactor):
    """Daily TurnOverRatio as a continous factor"""
    def compute(self):
        tablename = 'valuation_q'
        field = ['turnover_ratio']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        return self
    
class TypicalPrice(GroupFactor):
    """TypicalPrice = Mean(High,Low,Close)"""
    def compute(self):
        self.instantiate_child_factors(High = High,Low = Low,AdjClosePrice = AdjClosePrice).mask_child_fac(TradableStatus)
        # np.argmax +1 because python is 0 indexed 
        self.fac_value = self.children_factor_value.mean(axis =1)
        return self


class PctChgHfq(GroupFactor):
    """Return = (AdjClose - AdjPreClose)/AdjPreClose"""
    def __init__(self):
        super(PctChgHfq, self).__init__()
        self.fill_method = "no"

    def compute(self):
        print("fill method before child %s" % self.fill_method)
        self.instantiate_child_factors(AdjPreClosePrice = AdjPreClosePrice ,AdjClosePrice = AdjClosePrice).mask_child_fac(TradableStatus)
        # np.argmax +1 because python is 0 indexed
        print("fill method after child %s" % self.fill_method)
        self.fac_value = (self.children_factor_value['AdjClosePrice']/self.children_factor_value['AdjPreClosePrice']).map(lambda x: (x-1)*100)

        # self.fac_value = (self.children_factor_value['AdjClosePrice'] - self.children_factor_value['AdjPreClosePrice'])/self.children_factor_value['AdjPreClosePrice']
        return self


class PctChgHfqNone2Zero(GroupFactor):
    """Return = (AdjClose - AdjPreClose)/AdjPreClose, none value to 0"""

    def compute(self):
        print("fill method before child %s" % self.fill_method)
        self.instantiate_child_factors(AdjPreClosePrice = AdjPreClosePrice ,AdjClosePrice = AdjClosePrice).mask_child_fac(TradableStatus)
        # np.argmax +1 because python is 0 indexed
        print("fill method after child %s" % self.fill_method)
        self.fac_value = (self.children_factor_value['AdjClosePrice']/self.children_factor_value['AdjPreClosePrice']).map(lambda x: (x-1)*100)
        self.fac_value = series_fillna(self.fac_value, fill_method='zero')
        # self.fac_value = (self.children_factor_value['AdjClosePrice'] - self.children_factor_value['AdjPreClosePrice'])/self.children_factor_value['AdjPreClosePrice']
        return self
  
class CSISmallcap500Weight(ContinousFactor):
    """
    A class that computes the weight of a CSISmallcap500.
    
    Attributes:
    fac_value (pandas DataFrame): DataFrame containing the computed factor values.
    fac_index (pandas DataFrame): DataFrame containing the index for the computed factor values.
    
    Methods:
    compute(): Computes the weight of a benchmark by importing data from a specified table and field. The factor values are then indexed and any missing values are filled with zeros.
    """
    def compute(self):
        tablename = 'real_index_weight'
        field = ['weight']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        self.fac_value = self.fac_value.reindex(self.fac_index).fillna(0)
    
class CSISmallcap500WeightFlag(ContinousFactor):
    def compute(self):
        self.instantiate_child_factors(CSISmallcap500Weight = CSISmallcap500Weight)
        self.fac_value = (self.children_factor_value['CSISmallcap500Weight'] >0)

class BenchmarkLevel(ContinousFactor):
    
    """Return the benchamrk index level as a continous factor"""
    
    def get_data(self,tablename,field,benchmark_code = '000905.XSHG'):
        
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
        import pdb
        pdb.set_trace()
        query_stmt = """SELECT {fields} FROM index_level
    where `code` = '000905.XSHG' and trade_date between {min_trade_date} and {max_trade_date};""".format(fields = ','.join(['trade_date','code']+ field),
    min_trade_date = min(trade_date_ls)-30000,
    max_trade_date = max(trade_date_ls))
        # read in necessary raw data 
        raw_fac  = self.sql_api.read_data_from(query_stmt)
        # set factor index 
        raw_fac = raw_fac.set_index(['trade_date','code'])
        self.fac_value = raw_fac
        return self
    
    
    
    def compute(self):
        tablename = 'index_level'
        field = ['close']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        return self


# class Return_Raw(GroupFactor):
#     """Return = (UnAdjClosePrice - PreClose)/PreClose, ignore untradable """
#     def compute(self):
#         self.instantiate_child_factors(PreClosePrice = PreClosePrice,UnAdjClosePrice = UnAdjClosePrice)
#         # np.argmax +1 because python is 0 indexed
#         self.fac_value = (self.children_factor_value['UnAdjClosePrice'] - self.children_factor_value['PreClosePrice'])/self.children_factor_value['PreClosePrice']
#         return self

class BenchmarkReturn(GroupFactor):
    """Return = (BenchmarkLevel - BenchmarkLevel.shift(1))/BenchmarkLevel.shift(1), ignore untradable """
    def compute(self):
        self.instantiate_child_factors(BenchmarkLevel = BenchmarkLevel)
        # np.argmax +1 because python is 0 indexed 
        self.fac_value = (self.children_factor_value['BenchmarkLevel'] - self.children_factor_value['BenchmarkLevel'].shift(1))/self.children_factor_value['BenchmarkLevel'].shift(1)
    @timer
    @get_fac_index
    @fac_value_to_sr
    @memorize
    def run_pipline(self):
        """
        container for compute method"""
        self.compute()
        # drop code index level 
        self.fac_value = self.fac_value.droplevel(1)
        code_ls = self.fac_index.get_level_values(1).unique()
        date_ls = self.fac_value.index
        multiindex = pd.MultiIndex.from_product([code_ls, date_ls])
        tmp_df = pd.DataFrame(index = multiindex).reset_index().set_index('trade_date')
        tmp_df['BenchmarkReturn'] = self.fac_value
        tmp_df = tmp_df.set_index('code',append = True)
        self.fac_value = tmp_df['BenchmarkReturn']
        return self
    
class ForwardReturn(GroupFactor):
    """ForwardReturn = Return.shift(1)"""
    def compute(self):
        self.instantiate_child_factors(Return = Return)
        # np.argmax +1 because python is 0 indexed 
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.shift(-1))
        return self
    
class RollingHighPrice250(GroupFactor):
    """n-trading day highest price. where n = 250
    RollingHighPrice250 = High.rolling(250).max()"""
    def compute(self):
        self.instantiate_child_factors(High = High).mask_child_fac(TradableStatus)
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(250).max())
        return self

class CloseTo250High(GroupFactor):
    """CloseTo250High = Close/Rolling250High"""
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice,RollingHighPrice250 = RollingHighPrice250).mask_child_fac(TradableStatus)
        self.fac_value = self.children_factor_value['AdjClosePrice']/self.children_factor_value['RollingHighPrice250']
        return self

class RollUpSum10(GroupFactor):
    """Rolling n days returns sum 
    RollUpSum10 = (Returns_Raw * Bool(Returns_Raw>0)).rolling(n).mean()"""
    def compute(self):
        n_roll = 10
        self.instantiate_child_factors(PctChgHfqNone2Zero = PctChgHfqNone2Zero)
        self.fac_value = self.children_factor_value['PctChgHfqNone2Zero']* (self.children_factor_value['PctChgHfqNone2Zero']>0)
        self.fac_value = self.fac_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(n_roll).sum())
        return self

class RollDownSum10(GroupFactor):
    """Rolling n days returns sum 
    RollDownSum10 = (Returns_Raw * Bool(Returns_Raw<0)).rolling(n).mean()"""
    def compute(self):
        n_roll = 10
        self.instantiate_child_factors(PctChgHfqNone2Zero = PctChgHfqNone2Zero)
        self.fac_value = self.children_factor_value['PctChgHfqNone2Zero']* (self.children_factor_value['PctChgHfqNone2Zero']<0)
        self.fac_value = self.fac_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(n_roll).sum()).abs()
        return self

class RollUpAve10(GroupFactor):
    """Rolling n days average increase price 
    RollUpAve = ((AdjClosePrice - Pre_ClosePrice) * Bool(AdjClosePrice - Pre_ClosePrice>0)).rolling(n).mean()"""
    def compute(self):
        n_roll = 10
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice,PreClosePrice = PreClosePrice).mask_child_fac(TradableStatus)
        self.fac_value = (self.children_factor_value['AdjClosePrice'] - self.children_factor_value['PreClosePrice'])
        self.fac_value = self.fac_value * (self.fac_value>0)
        self.fac_value = self.fac_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(n_roll).mean())
        return self
    
class RollDownAve10(GroupFactor):
    """Rolling n days average decrease price 
    RollUpAve = ((AdjClosePrice - Pre_ClosePrice) * Bool(AdjClosePrice - Pre_ClosePrice<0)).rolling(n).mean()"""
    def compute(self):
        n_roll = 10
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice,PreClosePrice = PreClosePrice).mask_child_fac(TradableStatus)
        self.fac_value = (self.children_factor_value['AdjClosePrice'] - self.children_factor_value['PreClosePrice'])
        self.fac_value = self.fac_value * (self.fac_value<0)
        self.fac_value = self.fac_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(n_roll).mean())
        return self
    
class RollUpAve30(GroupFactor):
    """Rolling n days average increase price 
    RollUpAve = ((AdjClosePrice - Pre_ClosePrice) * Bool(AdjClosePrice - Pre_ClosePrice>0)).rolling(n).mean()"""
    def compute(self):
        n_roll = 30
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice,PreClosePrice = PreClosePrice).mask_child_fac(TradableStatus)
        self.fac_value = (self.children_factor_value['AdjClosePrice'] - self.children_factor_value['PreClosePrice'])
        self.fac_value = self.fac_value * (self.fac_value>0)
        self.fac_value = self.fac_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(n_roll).mean())
        return self
    
class RollDownAve30(GroupFactor):
    """Rolling n days average decrease price 
    RollUpAve = ((AdjClosePrice - Pre_ClosePrice) * Bool(AdjClosePrice - Pre_ClosePrice<0)).rolling(n).mean()"""
    def compute(self):
        n_roll = 30
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice,PreClosePrice = PreClosePrice).mask_child_fac(TradableStatus)
        self.fac_value = (self.children_factor_value['AdjClosePrice'] - self.children_factor_value['PreClosePrice'])
        self.fac_value = self.fac_value * (self.fac_value<0)
        self.fac_value = self.fac_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(n_roll).mean())
        return self

class RollUpAve50(GroupFactor):
    """Rolling n days average increase price 
    RollUpAve = ((AdjClosePrice - Pre_ClosePrice) * Bool(AdjClosePrice - Pre_ClosePrice>0)).rolling(n).mean()"""
    def compute(self):
        n_roll = 50
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice,PreClosePrice = PreClosePrice).mask_child_fac(TradableStatus)
        self.fac_value = (self.children_factor_value['AdjClosePrice'] - self.children_factor_value['PreClosePrice'])
        self.fac_value = self.fac_value * (self.fac_value>0)
        self.fac_value = self.fac_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(n_roll).mean())
        return self
    
class RollDownAve50(GroupFactor):
    """Rolling n days average decrease price 
    RollUpAve = ((AdjClosePrice - Pre_ClosePrice) * Bool(AdjClosePrice - Pre_ClosePrice<0)).rolling(n).mean()"""
    def compute(self):
        n_roll = 50
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice,PreClosePrice = PreClosePrice).mask_child_fac(TradableStatus)
        self.fac_value = (self.children_factor_value['AdjClosePrice'] - self.children_factor_value['PreClosePrice'])
        self.fac_value = self.fac_value * (self.fac_value<0)
        self.fac_value = self.fac_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(n_roll).mean())
        return self

class RollUpAve150(GroupFactor):
    """Rolling n days average increase price 
    RollUpAve = ((AdjClosePrice - Pre_ClosePrice) * Bool(AdjClosePrice - Pre_ClosePrice>0)).rolling(n).mean()"""
    def compute(self):
        n_roll = 150
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice,PreClosePrice = PreClosePrice).mask_child_fac(TradableStatus)
        self.fac_value = (self.children_factor_value['AdjClosePrice'] - self.children_factor_value['PreClosePrice'])
        self.fac_value = self.fac_value * (self.fac_value>0)
        self.fac_value = self.fac_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(n_roll).mean())
        return self
    
class RollDownAve150(GroupFactor):
    """Rolling n days average decrease price 
    RollUpAve = ((AdjClosePrice - Pre_ClosePrice) * Bool(AdjClosePrice - Pre_ClosePrice<0)).rolling(n).mean()"""
    def compute(self):
        n_roll = 150
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice,PreClosePrice = PreClosePrice).mask_child_fac(TradableStatus)
        self.fac_value = (self.children_factor_value['AdjClosePrice'] - self.children_factor_value['PreClosePrice'])
        self.fac_value = self.fac_value * (self.fac_value<0)
        self.fac_value = self.fac_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(n_roll).mean())
        return self
    
#### Momentum Factors
class ShortMoM25(GroupFactor):
    """ShortMoM25 = Close/Close.shift(25) -1 """ 
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice)
        # np.argmax +1 because python is 0 indexed 
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x/x.shift(25)-1)
        return self

class LongMoM145(GroupFactor):
    """LongMoM145 = Close/Close.shift(145) -1 """ 
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice)
        # np.argmax +1 because python is 0 indexed 
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x/x.shift(145)-1)
        return self

class LongMoM29weeks(GroupFactor):
    """LongMoM29weeks = Close/Close.shift(29) -1 """ 
    def compute(self):
        self.instantiate_child_factors(AdjClosePriceWeekly = AdjClosePriceWeekly)
        # np.argmax +1 because python is 0 indexed 
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x/x.shift(29)-1)
        return self

class ShortMoM5weeks(GroupFactor):
    """ShortMoM5weeks = Close/Close.shift(5) -1 """ 
    def compute(self):
        self.instantiate_child_factors(AdjClosePriceWeekly = AdjClosePriceWeekly)
        # np.argmax +1 because python is 0 indexed 
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x/x.shift(5)-1)
        return self

class Long145MinusShort25MoM(GroupFactor):
    def compute(self):
        self.instantiate_child_factors(LongMoM145 = LongMoM145,
                                       ShortMoM25 = ShortMoM25)
        # np.argmax +1 because python is 0 indexed 
        self.fac_value = self.children_factor_value['LongMoM145'] - self.children_factor_value['ShortMoM25']


class Long29weeksMinusShort5weekdsMoM(GroupFactor):
    def compute(self):
        self.instantiate_child_factors(LongMoM29weeks = LongMoM29weeks,
                                       ShortMoM5weeks = ShortMoM5weeks)
        # np.argmax +1 because python is 0 indexed
        self.fac_value = self.children_factor_value['LongMoM29weeks'] - self.children_factor_value['ShortMoM5weeks']


class PriceMAInc21(GroupFactor):
    """PriceMAInc = Close/MA(Close,M) -1 """ 
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice).mask_child_fac(TradableStatus)
        # np.argmax +1 because python is 0 indexed 
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x/x.rolling(21).mean()-1)
        return self


class PriceMAInc61(GroupFactor):
    """PriceMAInc = Close/MA(Close,M) -1 """
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice).mask_child_fac(TradableStatus)
        # np.argmax +1 because python is 0 indexed 
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x/x.rolling(61).mean()-1)
        return self

class PriceMAInc250(GroupFactor):
    """PriceMAInc = Close/MA(Close,M) -1 """
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice).mask_child_fac(TradableStatus)
        # np.argmax +1 because python is 0 indexed 
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x/x.rolling(250).mean()-1)
        return self

class PLRC6(GroupFactor):
    """Rolling Price linear trend regression coefficient
    PLRC = beta
    where (close / mean(close)) = beta * t + alpha"""
    def compute(self):
        M = 6
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice).mask_child_fac(TradableStatus)
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(M).apply(lambda x: trend_regress(x/x.mean())))
        return self
    
class PLRC12(GroupFactor):
    """Rolling Price linear trend regression coefficient
    PLRC = beta
    where (close / mean(close)) = beta * t + alpha"""
    def compute(self):
        M = 12
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice).mask_child_fac(TradableStatus)
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(M).apply(lambda x: trend_regress(x/x.mean())))
        return self

class PLRC24(GroupFactor):
    """Rolling Price linear trend regression coefficient
    PLRC = beta
    where (close / mean(close)) = beta * t + alpha"""
    def compute(self):
        M = 24
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice).mask_child_fac(TradableStatus)
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(M).apply(lambda x: trend_regress(x/x.mean())))
        return self

    
class ROC5(GroupFactor):
    """ROC = Close/Close.shift(M) * 100 """
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice).mask_child_fac(TradableStatus)
        # np.argmax +1 because python is 0 indexed 
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x/x.shift(5)-1)
        return self
    
class ROC6(GroupFactor):
    """ROC = Close/Close.shift(M) * 100 """
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice).mask_child_fac(TradableStatus)
        # np.argmax +1 because python is 0 indexed 
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x/x.shift(6)-1)
        return self

class ROC12(GroupFactor):
    """ROC = Close/Close.shift(M) * 100 """
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice).mask_child_fac(TradableStatus)
        # np.argmax +1 because python is 0 indexed 
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x/x.shift(12)-1)
        return self

class ROC20(GroupFactor):
    """ROC = Close/Close.shift(M) * 100 """
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice).mask_child_fac(TradableStatus)
        # np.argmax +1 because python is 0 indexed 
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x/x.shift(20)-1)
        return self
    
class ROC25(GroupFactor):
    """ROC = Close/Close.shift(M) * 100 """
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice).mask_child_fac(TradableStatus)
        # np.argmax +1 because python is 0 indexed 
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x/x.shift(25)-1)
        return self
    
class ROC60(GroupFactor):
    """ROC = Close/Close.shift(M) * 100 """
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice).mask_child_fac(TradableStatus)
        # np.argmax +1 because python is 0 indexed 
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x/x.shift(60)-1)
        return self

class ROC120(GroupFactor):
    """ROC = Close/Close.shift(M) * 100 """
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice).mask_child_fac(TradableStatus)
        # np.argmax +1 because python is 0 indexed 
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x/x.shift(120)-1)
        return self

class ROC145(GroupFactor):
    """ROC = Close/Close.shift(M) * 100 """
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice).mask_child_fac(TradableStatus)
        # np.argmax +1 because python is 0 indexed 
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x/x.shift(145)-1)
        return self

class ROC145MinusROC25(GroupFactor):
    """ROC = ROC145 - ROC25 """
    def compute(self):
        self.instantiate_child_factors(ROC145 = ROC145,ROC25 = ROC25)
        # np.argmax +1 because python is 0 indexed 
        self.fac_value = self.children_factor_value['ROC145'] - self.children_factor_value['ROC25']
        return self

class RealizedVolitality250(GroupFactor):
    """Annualized Volatility of past 250 trading days
        RealizedVolitality = Return.rolling(n).std() * 250**0.5, n = 250"""
    def compute(self):
        self.instantiate_child_factors(Return = Return)
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(250).std())*250**0.5
        return self

class RSI(GroupFactor):
    """Relative Strength 10 days, same as old computation in MyDataProcessor"""
    def compute(self):
        self.instantiate_child_factors(RollUpSum10 = RollUpSum10,
                                       RollDownSum10 = RollDownSum10)
        self.fac_value = self.children_factor_value['RollUpSum10']/(self.children_factor_value['RollUpSum10'] + self.children_factor_value['RollDownSum10'])
        self.fac_value = self.fac_value*100
        self.fac_value = self.fac_value.replace(np.inf,0)
        return self


class RSI10(GroupFactor):
    """Relative Strength 10 days"""
    def compute(self):
        self.instantiate_child_factors(RollUpAve10 = RollUpAve10,
                                       RollDownAve10 = RollDownAve10)
        self.children_factor_value['RS10'] = (self.children_factor_value['RollUpAve10'] / self.children_factor_value ['RollDownAve10'])
        self.fac_value = 100 - 100/(1+self.children_factor_value['RS10'])
        return self

class RSI30(GroupFactor):
    """Relative Strength 10 days"""
    def compute(self):
        self.instantiate_child_factors(RollUpAve30 = RollUpAve30,
                                       RollDownAve30 = RollDownAve30)
        self.children_factor_value['RS30'] = (self.children_factor_value['RollUpAve30'] / self.children_factor_value ['RollDownAve30'])
        self.fac_value = 100 - 100/(1+self.children_factor_value['RS30'])
        return self
    
class RSI50(GroupFactor):
    """Relative Strength 50 days"""
    def compute(self):
        self.instantiate_child_factors(RollUpAve50 = RollUpAve50,
                                       RollDownAve50 = RollDownAve50)
        self.children_factor_value['RS50'] = (self.children_factor_value['RollUpAve50'] / self.children_factor_value ['RollDownAve50'])
        self.fac_value = 100 - 100/(1+self.children_factor_value['RS50'])
        return self
    
class RSI150(GroupFactor):
    """Relative Strength 150 days"""
    def compute(self):
        self.instantiate_child_factors(RollUpAve150 = RollUpAve150,
                                       RollDownAve150 = RollDownAve150)
        self.children_factor_value['RS150'] = (self.children_factor_value['RollUpAve150'] / self.children_factor_value ['RollDownAve150'])
        self.fac_value = 100 - 100/(1+self.children_factor_value['RS150'])
        return self

class ArronUp(GroupFactor):
    """ArronUp Band
        = Rolling(ArgMax(Close),M)/M 
        M = 25"""
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice).mask_child_fac(TradableStatus)
        # np.argmax +1 because python is 0 indexed 
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(25).apply(lambda x: (np.argmax(x)+1)/25))
        return self
    
class ArronDown(GroupFactor):
    """ArronDown Band
        = Rolling(ArgMin(Close),M)/M 
        M = 25"""
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice).mask_child_fac(TradableStatus)
        # np.argmin +1 because python is 0 indexed 
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(25).apply(lambda x: (np.argmin(x)+1)/25))
        return self

class BBI(GroupFactor):
    """Bull And Bear lndex 
        = Mean(MA3,MA6,MA12,MA24)"""
    def compute(self):
        self.instantiate_child_factors(MA3 = MA3,MA6 = MA6,MA12 = MA12,MA24 = MA24)
        # np.argmin +1 because python is 0 indexed 
        self.fac_value = self.children_factor_value.mean(axis =1)
        return self

class BBIC(GroupFactor):
    """BBIC
        = BBI/Close"""
    def compute(self):
        self.instantiate_child_factors(BBI = BBI,AdjClosePrice = AdjClosePrice)
        # np.argmin +1 because python is 0 indexed 
        self.fac_value = self.children_factor_value['BBI']/self.children_factor_value['AdjClosePrice']
        return self
    
class BearPower(GroupFactor):
    """BearPower
        = (Low - EMA(close,13)) / close)"""
    def compute(self):
        self.instantiate_child_factors(Low = Low,EMA13 =EMA13,AdjClosePrice = AdjClosePrice)
        # np.argmin +1 because python is 0 indexed 
        self.fac_value = (self.children_factor_value['Low']-self.children_factor_value['EMA13'])/self.children_factor_value['AdjClosePrice']
        return self
    
class BullPower(GroupFactor):
    """BullPower
        = (High - EMA(close,13)) / close)"""
    def compute(self):
        self.instantiate_child_factors(High = High,EMA13 =EMA13,AdjClosePrice = AdjClosePrice)
        # np.argmin +1 because python is 0 indexed 
        self.fac_value = (self.children_factor_value['High']-self.children_factor_value['EMA13'])/self.children_factor_value['AdjClosePrice']
        return self
    
class BollDown(GroupFactor):
    """Lower BollingerBand
        = (MA(CLOSE,M)-2*STD(CLOSE,M)) / CLOSE; M=20 """
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice).mask_child_fac(TradableStatus)
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: (x.rolling(20).mean() - 2*x.rolling(20).std())/x)
        return self

class BollUp(GroupFactor):
    """Upper BollingerBand
        = (MA(CLOSE,M)+2*STD(CLOSE,M)) / CLOSE; M=20 """
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice).mask_child_fac(TradableStatus)
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: (x.rolling(20).mean() + 2*x.rolling(20).std())/x)
        return self
    
class BIAS5(GroupFactor):
    """BIAS5
        = (Close - MA(close,5)) / MA(close,5))"""
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice,MA5 = MA5)
        # np.argmin +1 because python is 0 indexed 
        self.fac_value = (self.children_factor_value['AdjClosePrice']-self.children_factor_value['MA5'])/self.children_factor_value['MA5']
        return self

class BIAS10(GroupFactor):
    """BIAS10
        = (Close - MA(close,10)) / MA(close,10))"""
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice,MA10 = MA10)
        # np.argmin +1 because python is 0 indexed 
        self.fac_value = (self.children_factor_value['AdjClosePrice']-self.children_factor_value['MA10'])/self.children_factor_value['MA10']
        return self

class BIAS20(GroupFactor):
    """BIAS20
        = (Close - MA(close,20)) / MA(close,20))"""
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice,MA20 = MA20)
        # np.argmin +1 because python is 0 indexed 
        self.fac_value = (self.children_factor_value['AdjClosePrice']-self.children_factor_value['MA20'])/self.children_factor_value['MA20']
        return self

class BIAS60(GroupFactor):
    """BIAS60
        = (Close - MA(close,60)) / MA(close,60))"""
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice,MA60 = MA60)
        # np.argmin +1 because python is 0 indexed 
        self.fac_value = (self.children_factor_value['AdjClosePrice']-self.children_factor_value['MA60'])/self.children_factor_value['MA60']
        return self

class CCI10(GroupFactor):
    """CCI = (TypicalPrice - MA(TypicalPrice,M))/STD(TypicalPrice,M) * (1/0.015), M = 10, 15 , 20, 88"""
    def compute(self):
        self.instantiate_child_factors(TypicalPrice = TypicalPrice)
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: (x - x.rolling(10).mean())/(0.015 * x.rolling(10).std()))
        return self

class CCI15(GroupFactor):
    """CCI = (TypicalPrice - MA(TypicalPrice,M))/STD(TypicalPrice,M) * (1/0.015), M = 10, 15 , 20, 88"""
    def compute(self):
        self.instantiate_child_factors(TypicalPrice = TypicalPrice)
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: (x - x.rolling(15).mean())/(0.015 * x.rolling(15).std()))
        return self

class CCI20(GroupFactor):
    """CCI = (TypicalPrice - MA(TypicalPrice,M))/STD(TypicalPrice,M) * (1/0.015), M = 10, 15 , 20, 88"""
    def compute(self):
        self.instantiate_child_factors(TypicalPrice = TypicalPrice)
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: (x - x.rolling(20).mean())/(0.015 * x.rolling(20).std()))
        return self

class CCI88(GroupFactor):
    """CCI = (TypicalPrice - MA(TypicalPrice,M))/STD(TypicalPrice,M) * (1/0.015), M = 10, 15 , 20, 88"""
    def compute(self):
        self.instantiate_child_factors(TypicalPrice = TypicalPrice)
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: (x - x.rolling(88).mean())/(0.015 * x.rolling(88).std()))
        return self

class SingleDayVPT(GroupFactor):
    """SingleDayVPT = (Close - PreClose)/PreClose * Volume"""
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice,PreClosePrice=PreClosePrice,Volume=Volume).mask_child_fac(TradableStatus).mask_child_fac(AllDayPriceLmit,operator = 'not')
        self.fac_value = (self.children_factor_value['AdjClosePrice']-self.children_factor_value['PreClosePrice'])/self.children_factor_value['PreClosePrice'] * self.children_factor_value['Volume']
        return self

class MSingleDayVPT6(GroupFactor):
    """MA(single_day_VPT, 6)"""
    def compute(self):
        M = 6
        self.instantiate_child_factors(SingleDayVPT = SingleDayVPT)
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(M).mean())
        return self
    
class MSingleDayVPT12(GroupFactor):
    """MA(single_day_VPT, 12)"""
    def compute(self):
        M = 12
        self.instantiate_child_factors(SingleDayVPT = SingleDayVPT)
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(M).mean())
        return self

class TRIX5(GroupFactor):
    """MTR = EMA(EMA(EMA(Close,M),M),M) where M = 5
       TRIX = (MTR - MTR.shift(1))/MTR.shift(1) *100"""
    def compute(self):
        M = 5
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice).mask_child_fac(TradableStatus)
        MTR =  self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.ewm(span = M).mean().ewm(span = M).mean().ewm(span = M).mean())
        self.fac_value = MTR.groupby(level = 'code').progress_apply(lambda x: (x-x.shift(1))/x.shift(1) * 100)
        return self

class TRIX10(GroupFactor):
    """MTR = EMA(EMA(EMA(Close,M),M),M) where M = 5
       TRIX = (MTR - MTR.shift(1))/MTR.shift(1) *100"""
    def compute(self):
        M = 12
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice).mask_child_fac(TradableStatus)
        MTR =  self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.ewm(span = M).mean().ewm(span = M).mean().ewm(span = M).mean())
        self.fac_value = MTR.groupby(level = 'code').progress_apply(lambda x: (x-x.shift(1))/x.shift(1) * 100)
        return self
    
class CR(GroupFactor):
    """MiddlePrice = ((High + Low)/2)
       IncValue = (High - MiddlePrice.shift(1)) if > 0 else 0 
       DecValue = (MiddlePrice.shift(1)-Low) if > 0 else 0 
       Long_Strength = IncValue.rolling(M).sum()
       Short_Strength = DecValue.rolling(M).sum()
       CR =Long_Strength/Short_Strength *100 
       where M = 20"""
    def compute(self):
        self.instantiate_child_factors(High = High,Low = Low).mask_child_fac(TradableStatus)
        self.children_factor_value['Middle_Price'] = self.children_factor_value.mean(axis = 1)
        self.children_factor_value['Middle_Price_lag1'] = self.children_factor_value['Middle_Price'].groupby(level = 'code').apply(lambda x: x.shift(1))
        IncFlag = (self.children_factor_value['High'] > self.children_factor_value['Middle_Price_lag1'])
        DecFlag = (self.children_factor_value['Middle_Price_lag1']>self.children_factor_value['Low'])
        IncValue= (self.children_factor_value['High']-self.children_factor_value['Middle_Price_lag1']) * IncFlag
        DecValue = (self.children_factor_value['Middle_Price_lag1']-self.children_factor_value['Low']) * DecFlag
        Long_Strength = IncValue.groupby(level = 'code').progress_apply(lambda x: x.rolling(20).sum())
        Short_Strength = DecValue.groupby(level = 'code').progress_apply(lambda x: x.rolling(20).sum())
        self.fac_value = Long_Strength/Short_Strength * 100
        return self
    
class TSRank250(GroupFactor):
    """Time-Series Ranking
    Percentile(Close,M) where M = 250"""
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice).mask_child_fac(TradableStatus)
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x:x.rolling(250).apply(lambda x: percentile_inv(x)))
        return self

class CrossRank20(GroupFactor):
    """Cross-sectionally Ranking on M days returns
    M= 20"""
    def compute(self):
        self.instantiate_child_factors(ROC20 = ROC20)
        self.fac_value = self.children_factor_value.dropna().groupby(level = 'trade_date').progress_apply(lambda x: percentile_inv(x,value = 'all')).droplevel(0)
        return self
    
class MAMASS(GroupFactor):
    """MASS = (High -Low.rolling(N1).mean())/(High-Low.rolling(N1).mean().rolling(N1).mean().rolling(N2).sum())
        MAMASS = MASS.rolling(M).mean()
        where N1 =9, N2 = 25, M = 6"""
    def compute(self):
        self.instantiate_child_factors(High = High,Low = Low).mask_child_fac(TradableStatus)
        N1 = 9
        N2 = 25
        M = 6
        HighPrice = self.children_factor_value['High']
        LowPrice = self.children_factor_value['Low']
        MASS = (HighPrice-LowPrice.groupby(level = 'code').progress_apply(lambda x: x.rolling(N1).mean()))/(HighPrice-LowPrice.groupby(level = 'code').progress_apply(lambda x: x.rolling(N1).mean().rolling(N1).mean().rolling(N2).sum()))
        self.fac_value = MASS.groupby(level = 'code').progress_apply(lambda x:x.rolling(M).mean())
        return self
    
class MACDC(GroupFactor):
    """ MACDC = MACD/Close"""
    def compute(self):
        self.instantiate_child_factors(MACD = MACD,AdjClosePrice = AdjClosePrice)
        self.fac_value = self.children_factor_value['MACD']/ self.children_factor_value['AdjClosePrice']
        return self
    
#### Technical Factors

class MA3(GroupFactor):
    """3days simple moving averge
        = MA(CLOSE,M); M=3 """
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice).mask_child_fac(TradableStatus)
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(3).mean())
        return self

class MA5(GroupFactor):
    """5days simple moving averge
        = MA(CLOSE,M); M=5 """
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice).mask_child_fac(TradableStatus)
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(5).mean())
        return self
    
class MA6(GroupFactor):
    """6days simple moving averge
        = MA(CLOSE,M); M=6 """
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice).mask_child_fac(TradableStatus)
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(6).mean())
        return self
    
class MA10(GroupFactor):
    """10days simple moving averge
        = MA(CLOSE,M); M=10 """
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice).mask_child_fac(TradableStatus)
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(10).mean())
        return self
    
class MA12(GroupFactor):
    """12days simple moving averge
        = MA(CLOSE,M); M=12"""
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice).mask_child_fac(TradableStatus)
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(12).mean())
        return self

class MA20(GroupFactor):
    """20days simple moving averge
        = MA(CLOSE,M); M=20"""
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice).mask_child_fac(TradableStatus)
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(20).mean())
        return self
    
class MA24(GroupFactor):
    """24days simple moving averge
        = MA(CLOSE,M); M=24"""
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice).mask_child_fac(TradableStatus)
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(12).mean())
        return self

class MA60(GroupFactor):
    """60days simple moving averge
        = MA(CLOSE,M); M=60"""
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice).mask_child_fac(TradableStatus)
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(60).mean())
        return self

class EMA12(GroupFactor):
    """12days exponential moving average ratio
        = EMA(CLOSE,M); M=12 """
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice).mask_child_fac(TradableStatus)
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.ewm(span = 12).mean())
        return self

class EMA13(GroupFactor):
    """13days exponential moving average ratio
        = EMA(CLOSE,M); M=12 """
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice).mask_child_fac(TradableStatus)
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.ewm(span = 13).mean())
        return self
    
class EMA26(GroupFactor):
    """12days exponential moving average ratio
        = EMA(CLOSE,M); M=12 """
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice).mask_child_fac(TradableStatus)
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.ewm(span = 26).mean())
        return self
    
class MAC5(GroupFactor):
    """5days simple moving average ratio
        = (MA(CLOSE,M)/CLOSE; M=5 """
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice).mask_child_fac(TradableStatus)
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(5).mean()/x)
        return self

class MAC10(GroupFactor):
    """10days simple moving average ratio
        = (MA(CLOSE,M)/CLOSE; M=20 """
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice).mask_child_fac(TradableStatus)
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(10).mean()/x)
        return self
    
class MAC20(GroupFactor):
    """20days simple moving average ratio
        = (MA(CLOSE,M)/CLOSE; M=20 """
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice).mask_child_fac(TradableStatus)
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(20).mean()/x)
        return self

class MAC60(GroupFactor):
    """60days simple moving average ratio
        = (MA(CLOSE,M)/CLOSE; M=60 """
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice).mask_child_fac(TradableStatus)
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(60).mean()/x)
        return self
  
class MAC120(GroupFactor):
    """120days simple moving average ratio
        = (MA(CLOSE,M)/CLOSE; M=60 """
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice).mask_child_fac(TradableStatus)
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(120).mean()/x)
        return self
    
class EMAC5(GroupFactor):
    """5days exponential moving average ratio
        = (EMA(CLOSE,M)/CLOSE; M=5 """
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice).mask_child_fac(TradableStatus)
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.ewm(span = 5).mean()/x)
        return self

class EMAC10(GroupFactor):
    """10days exponential moving average ratio
        = (EMA(CLOSE,M)/CLOSE; M=10 """
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice).mask_child_fac(TradableStatus)
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.ewm(span = 10).mean()/x)
        return self

class EMAC12(GroupFactor):
    """12days exponential moving average ratio
        = (EMA(CLOSE,M)/CLOSE; M=12 """
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice).mask_child_fac(TradableStatus)
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.ewm(span = 12).mean()/x)
        return self

class EMAC20(GroupFactor):
    """20days exponential moving average ratio
        = (EMA(CLOSE,M)/CLOSE; M=20 """
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice).mask_child_fac(TradableStatus)
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.ewm(span = 20).mean()/x)
        return self
    
class EMAC26(GroupFactor):
    """26days exponential moving average ratio
        = (EMA(CLOSE,M)/CLOSE; M=20 """
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice).mask_child_fac(TradableStatus)
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.ewm(span = 26).mean()/x)
        return self

class EMAC120(GroupFactor):
    """120days exponential moving average ratio
        = (EMA(CLOSE,M)/CLOSE; M=120 """
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice).mask_child_fac(TradableStatus)
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.ewm(span = 120).mean()/x)
        return self
    
class DIFF(GroupFactor):
    """EMA(CLOSE，SHORT)-EMA(CLOSE，LONG)
    where short = 12,long = 26"""
    def compute(self):
        self.instantiate_child_factors(EMA12 = EMA12,EMA26=EMA26) 
        self.fac_value = self.children_factor_value['EMA12'] - self.children_factor_value['EMA26']
        return self
    
class DEA(GroupFactor):
    """EMA(DIFF，M)
    where M = 9"""
    def compute(self):
        self.instantiate_child_factors(DIFF = DIFF) 
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.ewm(span = 9).mean())
        return self
    
class MACD(GroupFactor):
    """ MACD = DIFF- DEA"""
    def compute(self):
        self.instantiate_child_factors(DIFF = DIFF,DEA = DEA) 

        self.fac_value = self.children_factor_value['DIFF'] - self.children_factor_value['DEA']
        return self
    

#### Volume Factors
    
class TR(GroupFactor):
    """TrueRange = max(High,Low,PreClose) - (High,Low,PreClose)"""
    def compute(self):
        self.instantiate_child_factors(High = High,Low = Low,PreClosePrice=PreClosePrice).mask_child_fac(TradableStatus)
        self.fac_value = self.children_factor_value.progress_apply(lambda x: (x.max()-x.min()),axis =1)
        return self

class ATR6(GroupFactor):
    """ATR6 = TR.rolling(6).mean()"""
    def compute(self):
        self.instantiate_child_factors(TR = TR)
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(6).mean())
        return self

class ATR14(GroupFactor):
    """ATR14 = TR.rolling(14).mean()"""
    def compute(self):
        self.instantiate_child_factors(TR = TR)
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(14).mean())
        return self
    
class VROC6(GroupFactor):
    """VROC = (Volume_t - Volume_(t-n))/Volume_(t-n) *100
    default n is 6 """
    def compute(self):
        self.instantiate_child_factors(Volume = Volume).mask_child_fac(TradableStatus).mask_child_fac(AllDayPriceLmit,operator = 'not')
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: ((x-x.shift(6))/x.shift(6))*100)
        return self

class VROC12(GroupFactor):
    """VROC12 = (Volume_t - Volume_(t-n))/Volume_(t-n) *100
    default n is 12 """
    def compute(self):
        self.instantiate_child_factors(Volume = Volume).mask_child_fac(TradableStatus).mask_child_fac(AllDayPriceLmit,operator = 'not')
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: ((x-x.shift(12))/x.shift(12))*100)
        return self

class TVMA6(GroupFactor):
    """rolling moving average Amount
    default n is 6 """
    def compute(self):
        self.instantiate_child_factors(Amount = Amount).mask_child_fac(TradableStatus).mask_child_fac(AllDayPriceLmit,operator = 'not')
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(6).mean())
        return self

class TVMA20(GroupFactor):
    """rolling moving average Amount
    default n is 20 """
    def compute(self):
        self.instantiate_child_factors(Amount = Amount).mask_child_fac(TradableStatus).mask_child_fac(AllDayPriceLmit,operator = 'not')
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(20).mean())
        return self

class TVSTD6(GroupFactor):
    """rolling standard deviation Amount
    default n is 6 """
    def compute(self):
        self.instantiate_child_factors(Amount = Amount).mask_child_fac(TradableStatus).mask_child_fac(AllDayPriceLmit,operator = 'not')
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(6).std())
        return self

class TVSTD20(GroupFactor):
    """rolling standard deviation Amount
    default n is 20 """
    def compute(self):
        self.instantiate_child_factors(Amount = Amount).mask_child_fac(TradableStatus).mask_child_fac(AllDayPriceLmit,operator = 'not')
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(20).std())
        return self

class VEMA5(GroupFactor):
    """Time Exponential moving average Volume
    default span is 10 """
    def compute(self):
        self.instantiate_child_factors(Volume = Volume).mask_child_fac(TradableStatus).mask_child_fac(AllDayPriceLmit,operator = 'not')
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.ewm(span = 5).mean())
        return self

class VEMA10(GroupFactor):
    """Time Exponential moving average Volume
    default span is 10 """
    def compute(self):
        self.instantiate_child_factors(Volume = Volume).mask_child_fac(TradableStatus).mask_child_fac(AllDayPriceLmit,operator = 'not')
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.ewm(span = 10).mean())
        return self
    
class VEMA12(GroupFactor):
    """Time Exponential moving average Volume
    default span is 12 """
    def compute(self):
        self.instantiate_child_factors(Volume = Volume).mask_child_fac(TradableStatus).mask_child_fac(AllDayPriceLmit,operator = 'not')
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.ewm(span = 12).mean())
        return self

class VEMA26(GroupFactor):
    """Time Exponential moving average Volume
    default span is 26 """
    def compute(self):
        self.instantiate_child_factors(Volume = Volume).mask_child_fac(TradableStatus).mask_child_fac(AllDayPriceLmit,operator = 'not')
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.ewm(span = 26).mean())
        return self

class VSTD10(GroupFactor):
    """rolling standard deviation Volume
    default n is 10 """
    def compute(self):
        self.instantiate_child_factors(Volume = Volume).mask_child_fac(TradableStatus).mask_child_fac(AllDayPriceLmit,operator = 'not')
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(10).std())
        return self

class VSTD20(GroupFactor):
    """rolling standard deviation Volume
    default n is 20 """
    def compute(self):
        self.instantiate_child_factors(Volume = Volume).mask_child_fac(TradableStatus).mask_child_fac(AllDayPriceLmit,operator = 'not')
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(20).std())
        return self
    
class Turnover5Days(GroupFactor):
    """rolling moving average Turnover ratio
    default n is 5 """
    def compute(self):
        self.instantiate_child_factors(TurnOverRatio = TurnOverRatio).mask_child_fac(TradableStatus).mask_child_fac(AllDayPriceLmit,operator = 'not')
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(5).mean())
        return self

class Turnover10Days(GroupFactor):
    """rolling moving average Turnover ratio
    default n is 10 """
    def compute(self):
        self.instantiate_child_factors(TurnOverRatio = TurnOverRatio).mask_child_fac(TradableStatus).mask_child_fac(AllDayPriceLmit,operator = 'not')
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(10).mean())
        return self
    
class Turnover20Days(GroupFactor):
    """rolling moving average Turnover ratio
    default n is 20 """
    def compute(self):
        self.instantiate_child_factors(TurnOverRatio = TurnOverRatio).mask_child_fac(TradableStatus).mask_child_fac(AllDayPriceLmit,operator = 'not')
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(20).mean())
        return self
    
class Turnover21Days(GroupFactor):
    """rolling moving average Turnover ratio
    default n is 21 """
    def compute(self):
        self.instantiate_child_factors(TurnOverRatio = TurnOverRatio).mask_child_fac(TradableStatus).mask_child_fac(AllDayPriceLmit,operator = 'not')
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(21).mean())
        return self
    
class Turnover60Days(GroupFactor):
    """rolling moving average Turnover ratio
    default n is 60 """
    def compute(self):
        self.instantiate_child_factors(TurnOverRatio = TurnOverRatio).mask_child_fac(TradableStatus).mask_child_fac(AllDayPriceLmit,operator = 'not')
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(60).mean())
        return self

class Turnover120Days(GroupFactor):
    """rolling moving average Turnover ratio
    default n is 120 """
    def compute(self):
        self.instantiate_child_factors(TurnOverRatio = TurnOverRatio).mask_child_fac(TradableStatus).mask_child_fac(AllDayPriceLmit,operator = 'not')
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(120).mean())
        return self

class Turnover240Days(GroupFactor):
    """rolling moving average Turnover ratio
    default n is 240 """
    def compute(self):
        self.instantiate_child_factors(TurnOverRatio = TurnOverRatio).mask_child_fac(TradableStatus).mask_child_fac(AllDayPriceLmit,operator = 'not')
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(240).mean())
        return self
    
class TurnoverSTD20(GroupFactor):
    """rolling std Turnover ratio
    default n is 20 """
    def compute(self):
        self.instantiate_child_factors(TurnOverRatio = TurnOverRatio).mask_child_fac(TradableStatus).mask_child_fac(AllDayPriceLmit,operator = 'not')
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(20).std())
        return self

class VDIFF(GroupFactor):
    """EMA(VOLUME，SHORT)-EMA(VOLUME，LONG)
    where short = 12,long = 26"""
    def compute(self):
        self.instantiate_child_factors(VEMA12 = VEMA12,VEMA26=VEMA26) 
        self.fac_value = self.children_factor_value['VEMA12'] - self.children_factor_value['VEMA26']
        return self
    
class VDEA(GroupFactor):
    """EMA(VDIFF，M)
    where M = 9"""
    def compute(self):
        self.instantiate_child_factors(VDIFF = VDIFF) 
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.ewm(span = 9).mean())
        return self
    
class VMACD(GroupFactor):
    """Volume MACD = VDIFF-VDEA"""
    def compute(self):
        self.instantiate_child_factors(VDIFF = VDIFF,VDEA = VDEA) 
        self.fac_value = self.children_factor_value['VDIFF'] - self.children_factor_value['VDEA']
        return self
    
class WVAD(GroupFactor):
    """((Close - Open)/(High - Low) * Volume).rolling(6).sum()"""
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice,
                                       OpenPrice = OpenPrice,
                                       High = High,
                                       Low = Low,
                                       Volume = Volume)\
            .mask_child_fac(TradableStatus).mask_child_fac(AllDayPriceLmit,operator = 'not')
        WVAD_df = ((self.children_factor_value['AdjClosePrice'] - self.children_factor_value['OpenPrice'])/(self.children_factor_value['High'] - self.children_factor_value['Low']))*self.children_factor_value['Volume']
        self.fac_value = WVAD_df.groupby(level = 'code').progress_apply(lambda x: x.rolling(6).sum())
        return self

class MarketBeta252(GroupFactor):
    """CAPM market beta estimated by 252 rolling regression"""
    
    def compute(self):
        self.instantiate_child_factors(PctChgHfqNone2Zero = PctChgHfqNone2Zero,
                                       BenchmarkReturn = BenchmarkReturn)

        self.children_factor_value = self.children_factor_value.dropna()

        self.children_factor_value = sm.add_constant(self.children_factor_value)
        
        # perform rolling regression 
        # self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: RollingSlopeRegression(x).fit().params.iloc[:,1].droplevel(1))
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: RollingSlopeRegression(x).fit().params.iloc[:,1]).droplevel(2).swaplevel('trade_date', 'code')


                                                                                           
#### sentimental Factors
class Turnover5DaysOver120Days(GroupFactor):
    """Turnover5Day/Turnover120Days"""
    def compute(self):
        self.instantiate_child_factors(Turnover5Days = Turnover5Days,
                                       Turnover120Days = Turnover120Days)
        self.fac_value = self.children_factor_value['Turnover5Days']/self.children_factor_value['Turnover120Days']
        return self


class Volatility60DaysNone2Zero(GroupFactor):
    """n-trading day rolling volitility. where n = 60,
        pctchghfq none value changed 2 zero
    Volatility60Days = PctChgHfq.rolling(60).std()"""
    def compute(self):
        self.instantiate_child_factors(PctChgHfq = PctChgHfq)

        self.children_factor_value = dataframe_fillna(self.children_factor_value, "PctChgHfq", fill_method='zero')

        self.fac_value = self.children_factor_value['PctChgHfq'].groupby(level = 'code').progress_apply(lambda x: x.rolling(60).std(ddof=0))
        return self


class Volatility60DaysKeepNone(GroupFactor):
    """n-trading day rolling volitility. where n = 60
    keep pctchghfq none value
    Volatility60Days = PctChgHfq.rolling(60).std()"""
    def compute(self):
        self.instantiate_child_factors(PctChgHfq = PctChgHfq)
        self.children_factor_value = dataframe_fillna(self.children_factor_value, "PctChgHfq", fill_method='no')
        # def transfer_pct_chg_2_vol(pct_chg_values, window_size):
        #     vol_values = [np.NAN] * (window_size-1)
        #     for idx in range(window_size, len(pct_chg_values)+1):
        #         vol_values.append(np.nanstd(pct_chg_values[idx-window_size: idx]))
        #     return vol_values
        self.fac_value = self.children_factor_value['PctChgHfq'].groupby(level = 'code').progress_apply(lambda x: x.rolling(60).std(ddof=0))
        # self.fac_value = self.children_factor_value['PctChgHfq'].groupby(level = 'code').progress_apply(lambda x: transfer_pct_chg_2_vol(x.values, 60))

        return self


class VR(GroupFactor):
    """Volume Ratio
    VR= [(Volume_u + Volume_f / 2) / (Volume_d + Volume_f / 2)].rolling(26).sum()
    where Volume_u = Volume * PosReturn
          Volume_f = Volume * ZeroReturn
          Volume_d = Volume * NegReturn"""
    def compute(self):
        self.instantiate_child_factors(Volume = Volume,PosReturn = PosReturn,NegReturn=NegReturn,ZeroReturn = ZeroReturn).mask_child_fac(TradableStatus)
        children_factor_value_df = self.children_factor_value.fillna(0)
        children_factor_value_df['Volume_u'] = children_factor_value_df['Volume'] * children_factor_value_df['PosReturn']
        children_factor_value_df['Volume_d'] = children_factor_value_df['Volume'] * children_factor_value_df['NegReturn']
        children_factor_value_df['Volume_f'] = children_factor_value_df['Volume'] * children_factor_value_df['ZeroReturn']
        children_factor_value_df = children_factor_value_df.groupby(level = 'code').progress_apply(lambda x: x.rolling(26).sum())
        children_factor_value_df['VR'] = (children_factor_value_df['Volume_u'] + 0.5* children_factor_value_df['Volume_f'])/(children_factor_value_df['Volume_d'] + 0.5* children_factor_value_df['Volume_f'])
        self.fac_value = children_factor_value_df['VR']
        return self
    
class AR(GroupFactor):
    """AR = (High - Open).rolling(n).sum()/(Open - Low).sum() *100 
        default n is 26"""
    def compute(self):
        self.instantiate_child_factors(High = High,Low = Low,OpenPrice = OpenPrice).mask_child_fac(TradableStatus)
        rolling_sum_df = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(26).sum())
        self.fac_value = (rolling_sum_df['High'] - rolling_sum_df['OpenPrice'])/(rolling_sum_df['OpenPrice'] - rolling_sum_df['Low'])*100
        return self
    
class BR(GroupFactor):
    """BR = (High -PreClose).rolling(n).sum()/(PreClose - Low).sum() *100 
        default n is 26"""
    def compute(self):
        self.instantiate_child_factors(High = High,Low = Low,PreClosePrice = PreClosePrice).mask_child_fac(TradableStatus)
        rolling_sum_df = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(26).sum())
        self.fac_value = (rolling_sum_df['High'] - rolling_sum_df['PreClosePrice'])/(rolling_sum_df['PreClosePrice'] - rolling_sum_df['Low'])*100
        return self

class ARBR(GroupFactor):
    """ARBR = AR - BR"""
    def compute(self):
        self.instantiate_child_factors(AR = AR,BR = BR).mask_child_fac(TradableStatus)
        self.fac_value = self.children_factor_value['AR'] - self.children_factor_value['BR']
        return self
    
class VOSC(GroupFactor):
    """EMA(VOLUME，SHORT)-EMA(VOLUME，LONG)
    where short = 12,long = 26"""
    def compute(self):
        self.instantiate_child_factors(VEMA12 = VEMA12,VEMA26=VEMA26) 
        self.fac_value = ((self.children_factor_value['VEMA12'] - self.children_factor_value['VEMA26'])/self.children_factor_value['VEMA12'])*100
        return self
    

    
class VMACDC(GroupFactor):
    """ VMACDC = VMACD/Volume"""
    def compute(self):
        self.instantiate_child_factors(VMACD = VMACD,Volume = Volume) 
        self.fac_value = self.children_factor_value['VMACD']/ self.children_factor_value['Volume']
        return self
    

    
class MAWVAD(GroupFactor):
    """MAWVAD = WVAD.rolling(6).mean()"""
    def compute(self):
        self.instantiate_child_factors(WVAD = WVAD) 
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(6).mean())
        return self

class PSY(GroupFactor):
    """PSY = PosReturn.rolling(n).sum()/n , the default n is 12"""
    def compute(self):
        self.instantiate_child_factors(PosReturn = PosReturn) 
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(12).sum()/12)*100
        return self

class PSYMA(GroupFactor):
    """PSYMA = PSY.rolling(n).mean()/n , the default n is 6"""
    def compute(self):
        self.instantiate_child_factors(PSY = PSY) 
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(6).mean())
        return self


    
#### Income Statement Factor
class IncomeStatementFactor(ContinousFactor):
    """For factors get from Instatement, fill missing values as 0"""
    def get_data(self,tablename,field):
        
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
        min_date = min(trade_date_ls)
        max_date = max(trade_date_ls)
        # fetech additional data for forward fill
        query_stmt = """select trade_date,`code`,end_date,{field} from {tablename} 
                        where trade_date between {min_date} and {max_date}""".format(field = field[0],
                                                                                tablename = tablename,
                                                                                min_date = min_date-20000,
                                                                                max_date = max_date)
        
        # read in necessary raw data 
        raw_fac  = self.sql_api.read_data_from(query_stmt)
        # set factor index 
        raw_fac = raw_fac.set_index(['trade_date','code','end_date'])
        # self.fac_value = raw_fac.fillna(0)
        self.fac_value = raw_fac
        return self
    
class Rev(IncomeStatementFactor):
    """total opertaing revenue as a continous factor"""
    def compute(self):
        tablename = 'income_stk'
        field = ['total_operating_revenue']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        return self

    
class NetIncome(IncomeStatementFactor):
    """operating_profit as a continous factor"""
    def compute(self):
        tablename = 'income_stk'
        field = ['operating_profit']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        return self

class NetIncome_q(IncomeStatementFactor):
    """operating_profit as a continous factor"""
    def compute(self):
        tablename = 'income_q'
        field = ['total_composite_income']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        return self
    
class Rev_q(IncomeStatementFactor):
    """total opertaing revenue as a continous factor"""
    def compute(self):
        tablename = 'income_q'
        field = ['total_operating_revenue']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        return self

class Opincome_acc(IncomeStatementFactor):
    def compute(self):
        # set factor index 
        tablename = 'income_stk'
        field = ['operating_profit']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        return self
    
class Opincome_q(IncomeStatementFactor):
    """
    quaterly operating income 
    """
    def compute(self):
        # set factor index 
        tablename = 'indicator_q'
        field = ['operating_profit']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        
        return self

class OperatingTaxSurcharges(IncomeStatementFactor):
    """
    operating tax and surcharges 
    """
    def compute(self):
        # set factor index 
        tablename = 'income_stk'
        field = ['Operating_Tax_Surcharges']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        return self
    
class OperatingCost(IncomeStatementFactor):
    """
    operating cost
    """
    def compute(self):
        # set factor index 
        tablename = 'income_stk'
        field = ['operating_cost']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        return self

class SaleExpense(IncomeStatementFactor):
    """
    operating cost
    """
    def compute(self):
        
        # set factor index 
        
        tablename = 'income_stk'
        field = ['sale_expense']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        
        return self

class AdministrationExpense(IncomeStatementFactor):
    """
    administration expense
    """
    def compute(self):
        
        # set factor index 
        
        tablename = 'income_stk'
        field = ['administration_expense']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        
        return self
    
class InterestExpense(IncomeStatementFactor):
    """
    InterestExpense
    """

    def compute(self):
        
        # set factor index 
        
        tablename = 'income_stk'
        field = ['interest_expense']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        
        return self


class InterestExpense_q(IncomeStatementFactor):
    """
    InterestExpense
    """
    
    
    
    
    
    def compute(self):
        
        # set factor index 
        
        tablename = 'income_q'
        field = ['interest_expense']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        
        return self

class CommissionExpense(IncomeStatementFactor):
    """
    CommissionExpense
    """
    def compute(self):
       
        # set factor index 
        
        tablename = 'income_stk'
        field = ['commission_expense']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        
        return self

class RdExpenses(IncomeStatementFactor):
    """
    Research and Development Expense
    """
    def compute(self):
        
        # set factor index 
        
        tablename = 'income_stk'
        field = ['rd_expenses']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        
        return self
    
class AssetImpairmentLoss(IncomeStatementFactor):
    """
    Asset Impairment Loss
    """

    def compute(self):

        # set factor index 
        
        tablename = 'income_stk'
        field = ['asset_impairment_loss']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        return self

class OtherEarnings(IncomeStatementFactor):
    """
    Other Earnings 
    """
    def compute(self):

        # set factor index 
        
        tablename = 'income_stk'
        field = ['other_earnings']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        # convert the data type to float 
        self.fac_value = self.fac_value.astype(float)
        return self

class IncomeTax(IncomeStatementFactor):
    """
    IncomeTax
    """
    def compute(self):

        # set factor index 
        
        tablename = 'income_stk'
        field = ['income_tax']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        # convert the data type to float 
        self.fac_value = self.fac_value.astype(float)
        return self

class TotalProfit(IncomeStatementFactor):
    """
    total_profit
    """
    def compute(self):

        # set factor index 
        
        tablename = 'income_stk'
        field = ['total_profit']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        # convert the data type to float 
        self.fac_value = self.fac_value.astype(float)
        return self
    
class EBIT(FundamentalFactor):
    """Compute EBIT using top down approach"""
    
    
    def compute(self):
        
        # set factor index 
        
        # creating and compute all child factors
        self.instantiate_child_factors(Rev = Rev,
                                            OperatingTaxSurcharges = OperatingTaxSurcharges,
                                            OperatingCost = OperatingCost,
                                            SaleExpense = SaleExpense,
                                            AdministrationExpense = AdministrationExpense,
                                            InterestExpense = InterestExpense,
                                            CommissionExpense = CommissionExpense,
                                            RdExpenses = RdExpenses,
                                            AssetImpairmentLoss = AssetImpairmentLoss,
                                            OtherEarnings = OtherEarnings)
        # fill missing values with 0
        import pdb
        pdb.set_trace()
        self.children_factor_value = self.children_factor_value.fillna(0)
        self.children_factor_value = self.children_factor_value.astype(float)
        # compute NOCF_Over_TORev factor
        self.fac_value = (self.children_factor_value['Rev'] - 
                          self.children_factor_value['OperatingTaxSurcharges'] - 
                          (self.children_factor_value['OperatingCost']+self.children_factor_value['SaleExpense']+self.children_factor_value['AdministrationExpense']+ self.children_factor_value['InterestExpense'] + self.children_factor_value['CommissionExpense'] + self.children_factor_value['RdExpenses'] + self.children_factor_value['AssetImpairmentLoss']) + 
                          self.children_factor_value['OtherEarnings'])
        return self
    
class TaxRate(FundamentalFactor):
    """coporate tax rate derivated from as a continous factor"""
    def compute(self):
        # creating and compute all child factors
        self.instantiate_child_factors(IncomeTax = IncomeTax,
                                       TotalProfit = TotalProfit)
        # import raw data 
        self.fac_value = self.children_factor_value['IncomeTax']/self.children_factor_value['TotalProfit']
        # make sure the tax rate is greater than 0 
        self.fac_value = self.fac_value * (self.fac_value>0)
        return self
    
#### BalanceSheet Factors

class CashEquivalents(FundamentalFactor):
    def compute(self):
        tablename = 'balance_stk'
        field = ['cash_equivalents']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        
        return self


class TotalAsset(FundamentalFactor):
    
    def compute(self):
        tablename = 'balance_stk'
        field = ['total_assets']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        
        return self


class TotalLiability(FundamentalFactor):
    """Total Liab as a continous factor"""
    
    def compute(self):
        
        # set factor index 
        
        tablename = 'balance_stk'
        field = ['total_liability']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        
        return self

class BookEquity(FundamentalFactor):
    """Total Equity as a continous factor"""

    def compute(self):
        
        # set factor index 
        
        tablename = 'balance_stk'
        field = ['total_owner_equities']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        return self

class BookEquityParent(FundamentalFactor):
    """Total Equity for parent company as a continous factor"""
    def compute(self):
        # set factor index 
        tablename = 'balance_stk'
        field = ['equities_parent_company_owners']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        return self

class OtherEquityTools(FundamentalFactor):
    """OtherEquityTools as a continous factor"""
    def compute(self):
        # set factor index 
        tablename = 'balance_stk'
        field = ['equities_parent_company_owners']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        return self
    
class PreferredSharesEquity(FundamentalFactor):
    """Total PreferredSharesEquity as a continous factor"""
    
    def compute(self):
        
        # set factor index 
        
        tablename = 'balance_stk'
        field = ['preferred_shares_equity']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        self.fac_value = self.fac_value.fillna(0).astype(float)
        return self
    
class CurrentAssets(FundamentalFactor):
    """Total CurrentLiability as a continous factor"""
    def compute(self):
        # set factor index 
        tablename = 'balance_stk'
        field = ['total_current_assets']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        return self

class CurrentLiabilities(FundamentalFactor):
    """Total CurrentLiabilities as a continous factor"""
    def compute(self):
        # set factor index 
        tablename = 'balance_stk'
        field = ['total_current_liability']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        return self

class ShortTermLoan(FundamentalFactor):
    """Shortterm Loan as a continous factor"""
    def compute(self):
        # set factor index 
        tablename = 'balance_stk'
        field = ['shortterm_loan']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        return self

class NonCurrentLiabilityInOneYear(FundamentalFactor):
    """NonCurrentLiabilityInOneYear as a continous factor"""
    def compute(self):
        # set factor index 
        tablename = 'balance_stk'
        field = ['non_current_liability_in_one_year']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        return self
    
class NonCurrentLiability(FundamentalFactor):
    """Total NonCurrentLiability as a continous factor"""
    def compute(self):
        # set factor index 
        tablename = 'balance_stk'
        field = ['total_non_current_liability']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        
        return self

class OperatingCash(FundamentalFactor):
     """
     OperatingCash = (CurrentAssets + CashEquivalents) - (CurrentLiabilities - ShortTermLoan - NonCurrentLiabilityInOneYear)
     """
     def compute(self):
         # creating and compute all child factors
         self.instantiate_child_factors(CurrentAssets = CurrentAssets,
                                       CashEquivalents = CashEquivalents,
                                       CurrentLiabilities = CurrentLiabilities,
                                       ShortTermLoan = ShortTermLoan,
                                       NonCurrentLiabilityInOneYear = NonCurrentLiabilityInOneYear)
         import pdb
         # pdb.set_trace()
         self.children_factor_value = self.children_factor_value.fillna(0)
         # compute NOCF_Over_TORev factor
         
         self.fac_value = (self.children_factor_value['CurrentAssets'] - 
                           self.children_factor_value['CashEquivalents']) - \
             (self.children_factor_value['CurrentLiabilities'] - 
              self.children_factor_value['ShortTermLoan'] - self.children_factor_value['NonCurrentLiabilityInOneYear'])
         return self
    
#### CashFlow Factors 
class NOCF(IncomeStatementFactor):
    """Net Operating CashFlow as a continous factor"""
    def compute(self):
        
        # set factor index 
        
        tablename = 'cash_flow_stk'
        field = ['net_operate_cash_flow']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        
        return self
    
class NetInvestCashFlow(IncomeStatementFactor):
    """Net Investing CashFlow as a continous factor"""

    def compute(self):
        
        # set factor index 
        
        tablename = 'cash_flow_stk'
        field = ['net_invest_cash_flow']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        
        return self

class IntangibleAmortization(IncomeStatementFactor):
    """Amortization as a continous factor"""
    def compute(self):
        
        # set factor index 
        
        tablename = 'cash_flow_stk'
        field = ['intangible_assets_amortization']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        return self

class Depreciation(IncomeStatementFactor):
    """Depreciation as a continous factor"""
    def compute(self):
        # set factor index 
        tablename = 'cash_flow_stk'
        field = ['fixed_assets_depreciation']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        return self

class DeferredExpenseAmortization(IncomeStatementFactor):
    """DeferredExpenseAmortization as a continous factor"""
    def compute(self):
        # set factor index 
        tablename = 'cash_flow_stk'
        field = ['defferred_expense_amortization']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        return self

class CapitalExpense(IncomeStatementFactor):
    """fix_intan_other_asset_acqui_cash as a continous factor"""
    def compute(self):
        # set factor index 
        tablename = 'cash_flow_stk'
        field = ['fix_intan_other_asset_acqui_cash']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        return self
    
class NetInvestCashFlow_q(IncomeStatementFactor):
    """Net Investing CashFlow as a continous factor"""

    def compute(self):
        
        # set factor index 
        
        tablename = 'cash_flow_q'
        field = ['net_invest_cash_flow']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        
        return self
    
class FCFF_top_down(FundamentalFactor):
     """
     EBIT(1 - TaxRate) + IntangibleAmortization + Depreciation
     + DeferredExpenseAmortization - CapitalExpense
     -(OperatingCash - OperatingCash.shift(1))
     """
     def compute(self):
         
         
         # set factor index 
         
         # creating and compute all child factors
        
         self.instantiate_child_factors(EBIT = EBIT,
                                        TaxRate = TaxRate,
                                       IntangibleAmortization = IntangibleAmortization,
                                       Depreciation = Depreciation,
                                       DeferredExpenseAmortization = DeferredExpenseAmortization,
                                       CapitalExpense = CapitalExpense,
                                       OperatingCash = OperatingCash
                                       )
         # get last year end operating cash 
         child_factor_df = self.children_factor_value.reset_index()
         child_factor_df['year'] = child_factor_df.end_date.apply(lambda x: x.year)
         year_end_operating_cash = child_factor_df.loc[child_factor_df.end_date.apply(lambda x:x.month==12),['code', 'end_date', 'OperatingCash']]
         year_end_operating_cash['year'] = year_end_operating_cash.end_date.apply(lambda x: x.year+1)
         self.children_factor_value = child_factor_df.merge(year_end_operating_cash[['code','year','OperatingCash']], 
                      on = ['code','year'],
                      how = 'left',
                     suffixes=('','_last_yr_end')).drop(columns = 'year').set_index(['trade_date','end_date','code'])
             
         self.children_factor_value = self.children_factor_value.fillna(0)
         # import pdb
         # pdb.set_trace()
         self.fac_value = (self.children_factor_value['EBIT'] * (1- self.children_factor_value['TaxRate'])+ self.children_factor_value['IntangibleAmortization']+ self.children_factor_value['Depreciation'] + self.children_factor_value['DeferredExpenseAmortization'] - self.children_factor_value['CapitalExpense'] - (self.children_factor_value['OperatingCash'] - self.children_factor_value['OperatingCash_last_yr_end']))
         
         return self


class FCFF(FundamentalFactor):
     """
     FCFF = (NOCF + NetInvestCashFlow )
     """
     def compute(self):
         
         
         # set factor index 
         
         # creating and compute all child factors
        
         self.instantiate_child_factors(NOCF = NOCF,
                                       NetInvestCashFlow = NetInvestCashFlow)
         


         
         # compute NOCF_Over_TORev factor
         
         self.fac_value = (self.children_factor_value['NOCF'] + 
                           self.children_factor_value['NetInvestCashFlow'])
         
         return self

class FCFF_q(FundamentalFactor):
     """
     FCFF = (NOCF + NetInvestCashFlow )
     """
     
     
     
     
     
     def compute(self):
         
         
         # set factor index 
         
         # creating and compute all child factors
        
         self.instantiate_child_factors(NOCF = NOCF_q,
                                       NetInvestCashFlow = NetInvestCashFlow_q)
         


         
         # compute NOCF_Over_TORev factor
         
         self.fac_value = (self.children_factor_value['NOCF_q'] + 
                           self.children_factor_value['NetInvestCashFlow_q'])
         
         return self





class NOCF_q(FundamentalFactor):
    """Net Operating CashFlow as a continous factor"""
    
    
    
    
    
    def compute(self):
        
        # set factor index 
        
        tablename = 'cash_flow_q'
        field = ['net_operate_cash_flow']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        
        return self
    
class NetRepo(FundamentalFactor):
    """
    Repo Dollar value
    """
    
    
    
    
    
    def compute(self):

        # set factor index 
        
        tablename = 'cash_flow_q'
        field = ['net_buyback']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        
        return self
#### Valuation Factors

class MktCap(ContinousFactor):
    """
        MktCap 
    """
    def compute(self):
        
        # set factor index 
        
        tablename = 'valuation_q'
        field = ['market_cap']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        self.fac_value = self.fac_value* 10**8
        return self

class LogMktCap(ContinousFactor):

     def compute(self):

         # set factor index 
         
         # creating and compute all child factors
         self.instantiate_child_factors(MktCap = MktCap)
         # compute
         self.fac_value = np.log(self.children_factor_value['MktCap'])
         
         return self

class NonLinearSize(ContinousFactor):
     """
    A class that computes the Non-Linear Size Factor, a subfactor of the size factor.
     """
     def compute(self):
         """
        Compute the Non-Linear Size Factor by computing child factors and applying a rolling slope regression.
    
         """
         # set factor index 
         
         # creating and compute all child factors
         self.instantiate_child_factors(LogMktCap = LogMktCap)
         self.children_factor_value = sm.add_constant(self.children_factor_value)
         self.children_factor_value['LogMktCapCube'] = self.children_factor_value['LogMktCap']**3
         self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: RollingSlopeRegression(x,x_name = ['const','LogMktCap'],y_name = 'LogMktCapCube').fit().residuals .droplevel(1))
         
         return self


class CirMktCap(ContinousFactor):
    """
        Cir_MktCap 
    """

    def compute(self):
        
        # set factor index 
        
        tablename = 'valuation_q'
        field = ['circulating_market_cap']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        
        return self
    
class CashToPrice(ContinousFactor):
    """
        Price to Cash and cash_eq
    """
    def compute(self):
        
        # set factor index 
        
        tablename = 'valuation_q'
        field = ['pcf_ratio']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        self.fac_value = 1/self.fac_value
        return self

class PriceToBook(ContinousFactor):
    """
        Price to Book Ratio  
    """
    def compute(self):
        
        # set factor index 
        
        tablename = 'valuation_q'
        field = ['pb_ratio']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        
        return self
    
class PriceToBook_unadjust(ContinousFactor):
    """
        PB Ratio unadjusted for stock repurchase, cash dividend can etc.
    """
    def compute(self):
        
        
        # set factor index 
        
        # creating and compute all child factors
       
        self.instantiate_child_factors(MktCap = MktCap,
                                       BookEquityParent = BookEquityParent,
                                       OtherEquityTools = OtherEquityTools)
        
        self.fac_value =  self.children_factor_value['MktCap']/(self.children_factor_value['BookEquityParent'] - 
                                                                self.children_factor_value['OtherEquityTools'])
       
        
        return self



class PriceToEarnings(ContinousFactor):
    """
        PE Ratio  
    """
    def compute(self):
        
        # set factor index 
        
        tablename = 'valuation_q'
        field = ['pe_ratio']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        
        return self


class PriceToSales(ContinousFactor):
    """
        Price to Book Ratio  
    """
    def compute(self):
        
        # set factor index 
        
        tablename = 'valuation_q'
        field = ['ps_ratio']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        
        return self


#### Growth Factor


class OpIncome_yoy_q(ContinousFactor):
    """
        yoy increase in operting profit
    """
    def compute(self):
        
        # set factor index 
        
        tablename = 'indicator_q'
        field = ['inc_operation_profit_annual']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        
        return self

class OpIncome_yoy_inc(ContinousFactor):
    """
        yoy increase in operting profit
    """
    def compute(self):
        
        # set factor index 
        
        tablename = 'indicator_q'
        field = ['inc_operation_profit_year_on_year']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        
        return self


class NetIncome_yoy_inc(IncomeStatementFactor):
    """
        yoy increase in Net Income
    """
    def compute(self):
        
        self.instantiate_child_factors(NetIncome_q = NetIncome_q)
        # self.children_factor_value.to_pickle("NetIncome_yoy_inc_detail.pkl")
        self.fac_value = self.children_factor_value.groupby(level = 'code')\
            .apply(lambda x: transfer_quarter_2_yoy(x, 'NetIncome_q'))
        self.fac_value = self.fac_value.replace(np.inf,-999)
        return self

    
class Rev_yoy_inc(FundamentalFactor):
    """
        yoy increase in Revenue
    """
    def compute(self):
        # set factor index 
        self.instantiate_child_factors(Rev_q = Rev_q)
        # self.fac_value = self.children_factor_value['Rev_q'].groupby(level = 'code').apply(lambda x: x/x.shift(4).abs()-1)
        self.fac_value = self.children_factor_value.groupby(level = 'code')\
            .apply(lambda x: transfer_quarter_2_yoy(x, 'Rev_q'))
        self.fac_value = self.fac_value.replace(np.inf,-999)
        return self
    
class RevLRC3(GroupFactor):
    #### why use 3 quarters ???
    """Rolling Revenue linear trend regression coefficient
    RevLRC = beta
    where (Rev / abs(mean(Rev))) = beta * t + alpha"""
    def compute(self):
        M = 3
        self.instantiate_child_factors(Rev_q = Rev_q)
        # drop end_date
        # self.children_factor_value = drop_extra_level_index(self.children_factor_value)
        self.children_factor_value.to_pickle("Rev_q_detail.pkl")
        # self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: x.rolling(M).apply(lambda x: trend_regress(x/np.abs(x).mean())))
        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: cal_quarterly_regress(x, "Rev_q"))
        self.fac_value = drop_extra_level_index(self.fac_value)

        return self
    
class Earnings_yoy_inc(ContinousFactor):
    """
        yoy increase in Earnings
    """
    def compute(self):
        
        # set factor index 
        
        tablename = 'indicator_q'
        field = ['inc_net_profit_to_shareholders_year_on_year']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        
        return self

class EarningsLRC3(GroupFactor):
    """Rolling Earnings linear trend regression coefficient
    EarningsLRC3 = beta
    where (Earnings / abs(mean(Earnings))) = beta * t + alpha"""
    def compute(self):
        M = 3
        self.instantiate_child_factors(NetIncome_q = NetIncome_q)

        self.fac_value = self.children_factor_value.groupby(level = 'code').progress_apply(lambda x: cal_quarterly_regress(x, 'NetIncome_q'))

        self.fac_value = drop_extra_level_index(self.fac_value)

        return self
#### Value Factors    

class PE_inv(ContinousFactor):
    """
        PE_inv Ratio  
    """
    def compute(self):
        
        
        # set factor index 
        
        # creating and compute all child factors
       
        self.instantiate_child_factors(PriceToEarnings = PriceToEarnings)
        # compute NOCF_Over_TORev factor
        
        self.fac_value =  1/self.children_factor_value['PriceToEarnings']
       
        
        return self


class PB_inv(ContinousFactor):
    """
        PB_inv Ratio  
    """
    
    
    
    
    
    def compute(self):
        
        
        # set factor index 
        
        # creating and compute all child factors
       
        self.instantiate_child_factors(PriceToBook = PriceToBook)
        


        
        # compute NOCF_Over_TORev factor
        
        self.fac_value =  1/self.children_factor_value['PriceToBook']
       
        
        return self

class PS_inv(ContinousFactor):
    """
        PS_inv Ratio  
    """
    
    
    
    
    
    def compute(self):
        
        
        # set factor index 
        
        # creating and compute all child factors
       
        self.instantiate_child_factors(PriceToSales = PriceToSales)
        


        
        # compute NOCF_Over_TORev factor
        
        self.fac_value =  1/self.children_factor_value['PriceToSales']
       
        
        return self


class PEG_inv(ContinousFactor):
    """
        PEG_inv Ratio  
    """
    
    
    
    
    
    def compute(self):
        
        
        # set factor index 
        
        # creating and compute all child factors
       
        self.instantiate_child_factors(PriceToEarnings = PriceToEarnings,
                                      OpIncome_yoy_inc = OpIncome_yoy_inc)
        


        
        # compute NOCF_Over_TORev factor
        
        self.fac_value =  self.children_factor_value['OpIncome_yoy_inc']/self.children_factor_value['PriceToEarnings']
       
        
        return self

class PSG_inv(ContinousFactor):
    """
        PSG_inv Ratio  
    """
    
    
    
    
    
    def compute(self):
        
        
        # set factor index 
        
        # creating and compute all child factors
       
        self.instantiate_child_factors(PriceToSales = PriceToSales,
                                      Rev_yoy_inc = Rev_yoy_inc)
        


        
        # compute NOCF_Over_TORev factor
        
        self.fac_value = self.children_factor_value['Rev_yoy_inc']/self.children_factor_value['PriceToSales']
        
        return self

class PBG_inv(ContinousFactor):
    """
        PBG_inv Ratio  
    """
    
    
    
    
    
    def compute(self):
        
        
        # set factor index 
        
        # creating and compute all child factors
       
        self.instantiate_child_factors(PriceToBook = PriceToBook,
                                      Earnings_yoy_inc = Earnings_yoy_inc)
        


        
        # compute NOCF_Over_TORev factor
        
        self.fac_value = self.children_factor_value['Earnings_yoy_inc']/self.children_factor_value['PriceToBook']
        
        return self




class EnterpriseValue(ContinousFactor):
     """
     MktLeverage = (MktCap +TotalLiability - CashEquivalents )
     """
     
     
     
     
     
     def compute(self):
         
         
         # set factor index 
         
         # creating and compute all child factors
        
         self.instantiate_child_factors(MktCap = MktCap,
                                       TotalLiability = TotalLiability,
                                       CashEquivalents = CashEquivalents)
         


         
         # compute NOCF_Over_TORev factor
         
         self.fac_value = (self.children_factor_value['MktCap'] + 
                           self.children_factor_value['TotalLiability'] -
                           self.children_factor_value['CashEquivalents'])
         
         return self

class EBITOverEV(ContinousFactor):
     """
     EBITOverEV = EBIT/EnterpriseValue
     """
     
     
     
     
     
     def compute(self):
         
         
         # set factor index 
         
         # creating and compute all child factors
        
         self.instantiate_child_factors(EBIT = EBIT,
                                       EnterpriseValue = EnterpriseValue)
         


         
         # compute NOCF_Over_TORev factor
         
         self.fac_value = self.children_factor_value['EBIT'] /self.children_factor_value['EnterpriseValue'] 
         
         return self



class FCFFOverMktCap(ContinousFactor):
     """
     FCFFOverMktCap = FCFF/MktCap
     """
     def compute(self):
         
         
         # set factor index =
         
         # creating and compute all child factors
        
         self.instantiate_child_factors(FCFF = FCFF,
                                       MktCap = MktCap)
         self.children_factor_value = self.align_data_to_index(self.children_factor_value, self.fac_index)
         # compute NOCF_Over_TORev factor
         
         self.fac_value = self.children_factor_value['FCFF']/self.children_factor_value['MktCap']
         return self




#### Categorical Factors
class PosReturn(CategoricalFactor):
    """bool factor indicating positive return for the particular date as a categorical factor"""
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice,
                                       PreClosePrice = PreClosePrice)
        self.fac_value = (self.children_factor_value['AdjClosePrice']>self.children_factor_value['PreClosePrice'])
        return self
    
class NegReturn(CategoricalFactor):
    """bool factor indicating negative return for the particular date as a categorical factor"""
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice,
                                       PreClosePrice = PreClosePrice)
        self.fac_value = (self.children_factor_value['AdjClosePrice']<self.children_factor_value['PreClosePrice'])
        return self

class ZeroReturn(CategoricalFactor):
    """bool factor indicating zeros return for the particular date as a categorical factor"""
    def compute(self):
        self.instantiate_child_factors(AdjClosePrice = AdjClosePrice,
                                       PreClosePrice = PreClosePrice)
        self.fac_value = (self.children_factor_value['AdjClosePrice']==self.children_factor_value['PreClosePrice'])
        return self
    
# class ST_flag(CategoricalFactor):
#     """st_flag as a categorical factor"""
#     def compute(self):
#
#         # set factor index
#         tablename = 'daily_st_data'
#         field = ['st_flag']
#         self.get_data(tablename = tablename,
#  field = field)
#         return self




class ReportDate(CategoricalFactor):
    """report date as a categorical factor"""
    def compute(self):
        
        # set factor index 
        tablename = 'balance_stk'
        field = ['end_date']
        self.get_data(tablename = tablename,
 field = field)
        return self

class StatusFactor(CategoricalFactor):
    def get_data(self,tablename,field):

        ## generate sql query 
        trade_date_ls = self.fac_index.get_level_values(0).to_list()

        # set the query 
        query_stmt = """select trade_date,`code`,{field} from {tablename}
                        where trade_date <= {max_trade_date}""".format(field = field[0],
                        tablename = tablename,max_trade_date = max(trade_date_ls))
        # read in necessary raw data 
        raw_fac  = self.sql_api.read_data_from(query_stmt)
        # set factor index 
        raw_fac = raw_fac.set_index(['trade_date','code'])
        self.fac_value = raw_fac
        return self

class StatusFactorEndDate(CategoricalFactor):
    def get_data(self,tablename,field):

        ## generate sql query
        trade_date_ls = self.fac_index.get_level_values(0).to_list()

        # set the query
        query_stmt = """select end_date,`code`,{field} from {tablename}
                        where trade_date <= {max_trade_date}""".format(field = field[0],
                        tablename = tablename,max_trade_date = max(trade_date_ls))
        # read in necessary raw data
        raw_fac  = self.sql_api.read_data_from(query_stmt)
        # set factor index
        raw_fac['trade_date'] =raw_fac['end_date'].map(lambda x: int(x.strftime("%Y%m%d")))
        raw_fac = raw_fac.set_index(['trade_date','code'])
        self.fac_value = raw_fac
        return self


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
        return self


# class ST_flag(StatusFactorEndDate):
#     """delist_flag as a categorical factor"""
#
#     def compute(self):
#         # set factor index
#         tablename = 'list_status_change_stk'
#         field = ['name']
#         self.get_data(tablename=tablename,
#                       field=field)
#         self.fac_value = (self.fac_value['name'].map(lambda x: "st" in x.lower() if type(x) is str else False))
#         return self


class STFlagNameHistory(StatusFactorStartDate):
    """st_flag from history name"""

    def compute(self):
        # set factor index
        tablename = 'name_history_stk'
        field = ['new_name']
        self.get_data(tablename=tablename,
                      field=field)
        self.fac_value = (self.fac_value['new_name'].map(lambda x: "st" in x.lower() if type(x) is str else False))
        return self


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
        tmp_result = self.fac_value.groupby(level='code').apply(lambda s: cal_st_flag(s))
        self.fac_value = tmp_result
        # import pdb
        # pdb.set_trace()
        return self


class STFlag(GroupFactor):
    """
    compute st flag, combine STFlagNameHistory and STFlagNetProfit
    """
    def compute(self):
        self.instantiate_child_factors(STFlagNameHistory=STFlagNameHistory,
                                       STFlagNetProfit=STFlagNetProfit,
                                       )
        self.fac_value = (self.children_factor_value['STFlagNameHistory'] + self.children_factor_value['STFlagNetProfit']).map(lambda x: 1 if x > 0 else 0)
        return self

class DaysListed(StatusFactor):
    """
    A class that calculates the number of days a stock has been listed.
    
    This class inherits from the `StatusFactor` class and overrides its `compute` method.
    The `compute` method retrieves data from the 'list_status_change_stk' table, filters the data to 
    only include newly listed stocks, and calculates the number of days each stock has been listed 
    based on the public date and trade date. The result is stored in the 'fac_value' attribute.
    
    Attributes:
        fac_value (pandas.Series): A series representing the number of days each stock has been listed, 
            indexed by trade date and stock code.
    """
    def compute(self):
        """
        Compute the number of days each stock has been listed.
        
        This method retrieves data from the 'list_status_change_stk' table, filters the data to 
        only include newly listed stocks, and calculates the number of days each stock has been listed 
        based on the public date and trade date. The result is stored in the 'fac_value' attribute.
        """
        # set factor index 
        
        tablename = 'list_status_change_stk'
        field = ['change_type']
        self.get_data(tablename = tablename,
 field = field)
        
        self.fac_value = self.fac_value[self.fac_value=='新股上市'].dropna()
        self.fac_value = pd.Series(self.fac_value.index.get_level_values(0),index = self.fac_value.index)
        self.fac_value.name = 'public_date'
        temp_df = self.align_data_to_index(self.fac_value, self.fac_index).reset_index()
        temp_df = intDate2Date(temp_df)
        temp_df.public_date = pd.to_datetime(temp_df.public_date.astype(int).astype(str))
        temp_df['DaysListed'] = (temp_df['trade_date'] - temp_df['public_date'])
        temp_df = Date2intDate(temp_df).set_index(['trade_date','code'])
        self.fac_value = temp_df['DaysListed']
        return self


class ListFlagNOT2Years(StatusFactor):
    """
    A class that calculates the flag indicating whether a stock has been listed for more than 200 days.
    
    This class inherits from the `StatusFactor` class and overrides its `compute` method.
    The `compute` method instantiates a `DaysListed` object and compares the number of days each stock 
    has been listed to two years. The result is stored in the 'fac_value' attribute as a boolean flag, 
    indicating whether each stock has been listed for more than two years.
    
    Attributes:
        fac_value (pandas.Series): A series of boolean flags indicating whether each stock has been 
            listed for more than one year, indexed by trade date and stock code.
    """
    def compute(self):
        """
        Compute the flag indicating whether each stock has been listed for more than two years.
        
        This method instantiates a `DaysListed` object and compares the number of days each stock 
        has been listed to two years. The result is stored in the 'fac_value' attribute as a boolean flag, 
        indicating whether each stock has been listed for more than one year.
        """
        self.instantiate_child_factors(DaysListed = DaysListed)
        self.fac_value  = (self.children_factor_value<datetime.timedelta(days =600))
        return self

class EndFlag(StatusFactor):
    """delist_flag as a categorical factor"""
    
    def compute(self):
        # set factor index 
        tablename = 'list_status_change_stk'
        field = ['public_status']
        self.get_data(tablename = tablename,
 field = field)
        self.fac_value = (self.fac_value=='终止上市')
        return self
    
class StockName(StatusFactor):
    """Stockname as a categorical factor"""
    
    def compute(self):
        
        # set factor index 
        
        tablename = 'name_history_stk'
        field = ['new_name']
        self.get_data(tablename = tablename,
 field = field)
        return self
    
class Pause_Flag(CategoricalFactor):
    """pause state as a categorical factor"""
    def compute(self):
        
        # set factor index 
        
        tablename = 'daily_trading_data'
        field = ['paused']
        self.get_data(tablename = tablename,
 field = field)
        return self

class Nan_Flag(CategoricalFactor):
    """missing key financial value as a categorical factor"""
    def compute(self):
        self.instantiate_child_factors(TotalLiability = TotalLiability,
                                       TotalAsset=TotalAsset,
                                       FCFF_top_down=FCFF_top_down,
                                       NOCF = NOCF,
                                       Opincome_q = Opincome_q,
                                       Rev = Rev)

        self.fac_value  = self.children_factor_value.isna().any(axis =1)
        

        return self

class TradableStatus(StatusFactor):
    """TradableStatus as a categorical factor, 1 indicating tradable in the exchange, 0 otherwise"""
    def get_data(self,tablename,field):
        
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
          'param': [min(trade_date_ls)-20500, max(trade_date_ls)]}
        # set the query info dict
        query_info =   {'method': 'select',
                     'sheet_name': tablename,
                     'tgt_field': {'way': 'show', 'field': ['trade_date','code']+ field},
                     'conditions': [trade_date_condition]}
        query_stmt = """select trade_date,`code`,{field} from {tablename}
                        where trade_date <= {max_trade_date}""".format(field = field[0],
                        tablename = tablename,max_trade_date = max(trade_date_ls))
        # read in necessary raw data 
        raw_fac  = self.sql_api.read_data_from(query_stmt)
        # import pdb
        # pdb.set_trace()
        # set factor index 
        raw_fac = raw_fac.set_index(['trade_date','code'])
        self.fac_value = raw_fac
        return self
    
    def compute(self):
        tablename = 'daily_trading_data_unadjusted'
        field = ['paused']
        self.get_data(tablename = tablename,
 field = field)
        self.fac_value = self.fac_value['paused'] == 0
        return self

class HitHighOrLowLmit(CategoricalFactor):
    """Bool labels indicating the stock price hit high or low limit during the trading time"""
    def compute(self):
        self.instantiate_child_factors(High= High,
                                       Low = Low,
                                       HighLimit = HighLimit,
                                       LowLimit = LowLimit)
        self.fac_value = ((self.children_factor_value['High']==self.children_factor_value['HighLimit']) | (self.children_factor_value['Low']==self.children_factor_value['LowLimit']))
        return self

class AllDayPriceLmit(CategoricalFactor):
    """Bool labels indicating the stock price hit high or low limit all the time during the trading time"""
    def compute(self):
        self.instantiate_child_factors(VWAPrice= VWAPrice,
                                       HighLimit = HighLimit,
                                       LowLimit = LowLimit)
        # if vwap close to high or low limit, think it as paused all day 
        self.fac_value = ((self.children_factor_value['VWAPrice']>=self.children_factor_value['HighLimit']* 0.995) | (self.children_factor_value['VWAPrice']<=self.children_factor_value['LowLimit']*1.005))
        return self


class GicsSector(CategoricalFactor):
    """gics_code as a categorical factor"""
        
    def get_data(self,tablename,field):
        """
            read in one feature of raw source data, forward fill the feature to daily frequency,
            reindex it to our time evoluted universe,
            

        Parameters
        ----------
        tablename : str
            source data table name in sql .
        field : list of one str
            list of columnname of sql table e.g. ['net_operate_cash_flow'].

        Returns
        -------
        raw_fac : pd.Series
            feature value first indexed by trade_date, then indexed by code at daily frequency.

        """
        self.get_fac_idx()
        # set the query info dict
        query_info =   {'method': 'select',
                     'sheet_name': tablename,
                     'tgt_field': {'way': 'show', 'field': ['code']+ field},
                     'conditions': []}
        # read in necessary raw data 
        raw_fac  = self.sql_api.read_data_from(query_info)
        
        fac_index = pd.DataFrame(index=self.fac_index).reset_index()
        raw_fac = fac_index.merge(raw_fac, on = 'code', how = 'left').set_index(['trade_date','code'])
        
        self.fac_value = raw_fac[field[0]].sort_index()

    def compute(self):
        
        # set factor index 
        
        tablename = 'code2gics'
        field = ['gics_code']
        self.get_data(tablename = tablename,
 field = field)
        return self

class SWLevel1Sector(CategoricalFactor):
    """sw level 1 industry code as a categorical factor"""
    def compute(self):
        # set factor index 
        tablename = 'daily_industry_data'
        field = ['sw_l1_industry_code']
        self.get_data(tablename = tablename,
 field = field)
        return self

class SWLevel1SectorName(CategoricalFactor):
    """sw level 1 industry code as a categorical factor"""
    def compute(self):
        # set factor index 
        tablename = 'daily_industry_data'
        field = ['sw_l1_industry_name']
        self.get_data(tablename = tablename,
 field = field)
        return self

#### Multiple Filters

class CommonFilters(CategoricalFactor):
    """
    Common filters filt out st stocks, paused stocks, missing key finanical stocks and delisted stocks 
    """
    
     
    
    
    
    def compute(self):
        
        self.instantiate_child_filter_factor(ST_flag = ST_flag ,List_flag = List_flag,Pause_Flag = Pause_Flag, Nan_Flag = Nan_Flag)
        
        non_st_mask = self.children_filter_factor['ST_flag']!= 1
        non_delist_mask = self.children_filter_factor['List_flag']!= 1
        non_nan_mask = self.children_filter_factor['Nan_Flag']!= 1
        non_pause_mask = self.children_filter_factor['Pause_Flag']!= 3
        self.fac_value = non_st_mask & non_delist_mask & non_nan_mask & non_pause_mask
        
        return self 
    

#### Ratio Factors
class AssetTurnOver(ContinousFactor):
    """Asset Turnover 
        Derived from TotalAsset and Rev
    """
    def compute(self):
        
        # set factor index 
        
        # creating and compute all child factors
        self.instantiate_child_factors(TotalAsset = TotalAsset,
                                     Rev = Rev)
        # compute 
        self.fac_value = self.children_factor_value['Rev'] / self.children_factor_value['TotalAsset']
        return self
#### Profitability Factors

class GrossProfitMargin(ContinousFactor):
    """
    GrossProfitMargin
    """
    
     
    
    
    
    def compute(self):
        
        # set factor index 
        
        tablename = 'indicator_q'
        field = ['gross_profit_margin']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        
        return self

class GrossProfitToAsset(ContinousFactor):
    """
    GrossProfitToAsset
    """
    def compute(self):
        
        # set factor index 
        
        self.instantiate_child_factors(GrossProfitMargin = GrossProfitMargin,
                                       Rev_q = Rev_q,
                                       TotalAsset = TotalAsset)
        self.fac_value = self.children_factor_value['GrossProfitMargin']* self.children_factor_value['Rev_q']/self.children_factor_value['TotalAsset']
        
        return self

class NOCF_q_To_Asset(ContinousFactor):
    """
    NOCF_q_To_Asset
    """
    
     
    
    
    
    def compute(self):
        
        # set factor index 
        
        self.instantiate_child_factors(NOCF_q = NOCF_q,
                                       TotalAsset = TotalAsset)
        self.fac_value = self.children_factor_value['NOCF_q']/self.children_factor_value['TotalAsset']

        return self


class ROA(ContinousFactor):
    
     
    
    
    
    def compute(self):
        
        # set factor index 
        
        tablename = 'indicator_q'
        field = ['roa']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        
        return self

class ROE(ContinousFactor):
    
     
    
    
    
    def compute(self):
        
        # set factor index 
        
        tablename = 'indicator_q'
        field = ['roe']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        
        return self
    
class IncReturn(ContinousFactor):
    
     
    
    
    
    def compute(self):
        
        # set factor index 
        
        tablename = 'indicator_q'
        field = ['inc_return']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        
        return self



class ROEStability(ContinousFactor):
    """
    3 year rolling mean(ROE)/std(ROE)
    """
    
     
    
    
    
    def compute(self):
        
        # set factor index 
        # creating and compute all child factors
        
        stability_func = lambda x: np.mean(x)/np.std(x)
        rollingyear = 3
        # shifted start time for rolling base factor 
        shifted_start_date = self.start_date - rollingyear * 10000
        ROE_factor = self.pass_in_basic_param(ROE(),start_date = shifted_start_date,fac_index = None)
        
        # note here we want to recompute ROE instead of wrapped function 
        self.fac_value = ROE_factor\
            .compute.__wrapped__.__wrapped__.__wrapped__(ROE_factor).\
            rolling_apply(window_size = rollingyear * 50,apply_func = stability_func).fac_value
        self.fac_value = self.fac_value.reindex(self.fac_index)
        return self
    
    
class NetProfitMargin(ContinousFactor):
    
     
    
    
    
    def compute(self):
        
        # set factor index 
        
        tablename = 'indicator_q'
        field = ['net_profit_margin']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        
        return self


class FinExpenseToRev(ContinousFactor):
    
     
    
    
    
    def compute(self):
        
        # set factor index 
        
        tablename = 'indicator_q'
        field = ['financing_expense_to_total_revenue']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        
        return self

class EBITMargin(ContinousFactor):
    """EBITMargin = NetProfitMargin + FinExpenseToRev"""
    
     
    
    
    
    def compute(self):
        
        
        # set factor index 
        
        # creating and compute all child factors
       
        self.instantiate_child_factors(NetProfitMargin = NetProfitMargin,
                                      FinExpenseToRev = FinExpenseToRev)
        


        
        # compute NOCF_Over_TORev factor
        
        self.fac_value = self.children_factor_value['NetProfitMargin'] + self.children_factor_value['FinExpenseToRev']
        
        return self


class EBIT_Over_Equity(ContinousFactor):
     """
     EBIT over Total Equity 
     """
     
      
     
     
     
     def compute(self):
         
         
         # set factor index 
         
         # creating and compute all child factors
        
         self.instantiate_child_factors(EBIT = EBIT,
                                       BookEquity = BookEquity)
         


         
         # compute NOCF_Over_TORev factor
         
         self.fac_value = self.children_factor_value['EBIT'] / self.children_factor_value['BookEquity']
         
         return self
    


#### Quality Factors


class OPRev_to_Total_Rev(ContinousFactor):
    """
    Operating Revenue to Total Revenue
    """
    
     
    
    
    
    def compute(self):
        
        # set factor index 
        
        tablename = 'indicator_q'
        field = ['goods_sale_and_service_to_revenue']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        
        return self

class Opincome_Over_Rev(ContinousFactor):
    """
    operating_expense_to_total_revenue
    """
    
     
    
    
    
    def compute(self):
        
        # set factor index 
        
        tablename = 'indicator_q'
        field = ['operating_expense_to_total_revenue']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        
        return self
    
class NOCF_Over_Opincome(ContinousFactor):
    """
    ocf_to_operating_profit
    """
    
     
    
    
    
    def compute(self):
        
        # set factor index 
        
        tablename = 'indicator_q'
        field = ['ocf_to_operating_profit']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        
        return self

class Rev_Over_MktCap(ContinousFactor):
    """Total Revenue over MarketCap as a continous factor
        Derived from NOCF and Rev
    """
    def compute(self):
        
        # set factor index 
        
        # creating and compute all child factors
        self.instantiate_child_factors(MktCap = MktCap,
                                     Rev = Rev)
        # compute NOCF_Over_TORev factor
        self.fac_value = self.children_factor_value['Rev'] / self.children_factor_value['MktCap']
        
        return self

class Rev_q_Over_MktCap(ContinousFactor):
    """Total quaterly Revenue over MarketCap as a continous factor
        Derived from NOCF and Rev
    """
    
     
    
    
    
    def compute(self):
        
        # set factor index 
        
        # creating and compute all child factors
        self.instantiate_child_factors(MktCap = MktCap,
                                     Rev_q = Rev_q)


        
        # compute NOCF_Over_TORev factor
        
        self.fac_value = self.children_factor_value['Rev_q'] / self.children_factor_value['MktCap']
        
        return self


class NOCF_Over_Rev(ContinousFactor):
    """Net Operating CashFlow over Total Operating Revenue as a continous factor
        Derived from NOCF and TORev
    """
    
     
    
    
    
    def compute(self):
        
        # set factor index 
        
        # creating and compute all child factors
        
        self.instantiate_child_factors(NOCF = NOCF,
                                       Rev = Rev)
         

         
        
        
        # compute NOCF_Over_TORev factor
        
        self.fac_value = self.children_factor_value['NOCF'] / self.children_factor_value['Rev']
        
        return self


class Opincome_q_Over_MktCap(ContinousFactor):
     
      
     
     
     
     def compute(self):
         
         
         # set factor index 
         
         # creating and compute all child factors
        
         self.instantiate_child_factors(Opincome_q = Opincome_q,
                                       MktCap = MktCap)
         


         
         # compute NOCF_Over_TORev factor
         
         self.fac_value = self.children_factor_value['Opincome_q'] / self.children_factor_value['MktCap']
         
         return self

class Opincome_acc_Over_MktCap(ContinousFactor):

     def compute(self):

         # set factor index 
         
         # creating and compute all child factors
        
         self.instantiate_child_factors(Opincome_acc = Opincome_acc,
                                       MktCap = MktCap)
         # compute NOCF_Over_TORev factor
         self.fac_value = self.children_factor_value['Opincome'] / self.children_factor_value['MktCap']
         
         return self


class NOCF_Over_Debt(ContinousFactor):
     

     def compute(self):
         # set factor index 
         
         
         # creating and compute all child factors
                 
         self.instantiate_child_factors(NOCF = NOCF,
                                       TotalLiability = TotalLiability)
         # compute NOCF_Over_TORev factor
         self.fac_value = self.children_factor_value['NOCF'] / self.children_factor_value['TotalLiability']
         
         return self

class NOCF_Over_EV(ContinousFactor):
     
      
     
     
     
     def compute(self):
         # set factor index 
         
         
         # creating and compute all child factors
                 
         self.instantiate_child_factors(NOCF = NOCF,
                                       EnterpriseValue = EnterpriseValue)
         

         

         
         # compute NOCF_Over_TORev factor
         
         self.fac_value = self.children_factor_value['NOCF'] / self.children_factor_value['EnterpriseValue']
         
         return self

class OpIncomeToNetIncome(ContinousFactor):
    
     
    
    
    
    def compute(self):
        
        # set factor index 
        
        tablename = 'indicator_q'
        field = ['financing_expense_to_total_revenue']
        # import raw data 
        self.get_data(tablename = tablename,
 field = field)
        
        return self

class CashOverMketCap(ContinousFactor):
     """CashOverMketCap = FCFF_top_down/MktCap"""
     def compute(self):                 
         self.instantiate_child_factors(FCFF_top_down = FCFF_top_down,
                                       MktCap = MktCap)
         import pdb
         pdb.set_trace()
         self.children_factor_value = self.children_factor_value.reset_index(level = 'end_date')\
             .groupby(level = 'code').apply(lambda x: x.fillna(method = 'ffill')).dropna(subset = ['end_date'])\
             .set_index(['end_date'],append = True)
         # import pdb
         # pdb.set_trace()
         self.fac_value = self.children_factor_value['FCFF_top_down'] / self.children_factor_value['MktCap']
         
         return self


#### Leverage Factors
class DebtOverAssets(ContinousFactor):
     """
     Debt Over Total Assets
     """
     def compute(self):
         
         
         # set factor index 
         
         # creating and compute all child factors
        
         self.instantiate_child_factors(TotalLiability = TotalLiability,
                                       TotalAsset = TotalAsset)
         
         # compute NOCF_Over_TORev factor
         
         self.fac_value = self.children_factor_value['TotalLiability'] / self.children_factor_value['TotalAsset']
         
         return self

class MktLeverage(ContinousFactor):
     """
     MktLeverage = (MktCap +PreferredSharesEquity + NonCurrentLiability )/MktCap
     """
     
     def compute(self):
         
         
         # set factor index 
         
         # creating and compute all child factors
        
         self.instantiate_child_factors(MktCap = MktCap,
                                       PreferredSharesEquity = PreferredSharesEquity,
                                       NonCurrentLiability = NonCurrentLiability)
         
         self.children_factor_value = self.align_data_to_index(self.children_factor_value, self.fac_index)
         # compute NOCF_Over_TORev factor
         self.fac_value = self.children_factor_value.sum(axis =1)/(self.children_factor_value['MktCap'])
         
         return self

class BookValue(ContinousFactor):
     """
     Book Value of Equity:
     BookValue = MktCap/PriceToBook
     """

     def compute(self):
         
         
         # set factor index 
         
         # creating and compute all child factors
        
         self.instantiate_child_factors(MktCap = MktCap,
                                        PriceToBook = PriceToBook)
         self.children_factor_value = self.children_factor_value.groupby(level = 'code').apply(lambda x: x.fillna(method = 'ffill')).fillna(0)
         self.fac_value = self.children_factor_value['MktCap']/self.children_factor_value['PriceToBook']

         
         return self

class BookLeverage(ContinousFactor):
     """
     BookLeverage = (BookEquity +PreferredSharesEquity + NonCurrentLiability )/BookEquity
     """

     def compute(self):
         
         
         # set factor index 
         
         # creating and compute all child factors
        
         self.instantiate_child_factors(BookValue = BookValue,
                                       PreferredSharesEquity = PreferredSharesEquity,
                                       NonCurrentLiability = NonCurrentLiability)
         # self.children_factor_value = self.children_factor_value.groupby(level = 'code').apply(lambda x: x.fillna(method = 'ffill')).fillna(0)
         self.children_factor_value = self.children_factor_value.groupby(level = 'code').apply(lambda x: x.fillna(method = 'ffill'))

         self.children_factor_value.to_pickle("blev_detail.pkl")
         self.fac_value = (self.children_factor_value['BookValue'] + self.children_factor_value['PreferredSharesEquity'] +
                           self.children_factor_value['NonCurrentLiability'])/(self.children_factor_value['BookValue'])
         
         return self


#### Cross_Sectional Factors 

class QualityFactor(GroupFactor):
     """
     QualityFactor = (0.25 Rev_Over_mktCap + 0.25 Opincome_q + 0.5 NOCF_Over_Debt)
     """
     
     def compute(self):
         
         
         # set factor index 
         
         # creating and compute all child factors
        
         self.instantiate_child_factors(Rev_Over_MktCap = Rev_Over_MktCap,
                                       Opincome_q = Opincome_q,
                                       NOCF_Over_Debt = NOCF_Over_Debt)
         self.children_factor_value = self.align_data_to_index(self.children_factor_value, self.fac_index)
         # filter out some stocks
         self.mask_child_fac(TradableStatus).standrdize_child_fac(_sd_win_sort,groupby_Fac=SWLevel1Sector)
         # compute QualityFactor factor
         
         self.fac_value = (0.25 * self.children_factor_value['Rev_Over_MktCap'] + 
                           0.25* self.children_factor_value['Opincome_q'] + 
                           0.5* self.children_factor_value['NOCF_Over_Debt'])
         
         return self
     
        
class ValueFactor(GroupFactor):
    """
    ValueFactor = Mean(PE_inv,PB_inv,PS_inv,FCFFOverMktCap)
    """
    
    def compute(self):
        
        
        # set factor index 
        
        # creating and compute all child factors
       
        # self.instantiate_child_factors(PE_inv = PE_inv,
        #                                      PB_inv = PB_inv,
        #                                      PS_inv = PS_inv,
        #                                     FCFFOverMktCap  = FCFFOverMktCap)
        self.instantiate_child_factors(PE_inv = PE_inv,
                                             PB_inv = PB_inv,
                                             PS_inv = PS_inv)
        self.children_factor_value = self.align_data_to_index(self.children_factor_value, self.fac_index)
        # filter out some stocks
        self.mask_child_fac(TradableStatus).standrdize_child_fac(_sd_win_sort,groupby_Fac=SWLevel1Sector)
        # compute QualityFactor factor
        self.fac_value = self.children_factor_value.mean(axis = 1)
        
        return self


class GrowthFactor(GroupFactor):
    """
    GrowthFactor = Mean(Rev_yoy_inc,OpIncome_yoy_inc,RevLRC4,EarningsLRC4)
    """
    def compute(self):
        
        
        # creating and compute all child factors
       
        self.instantiate_child_factors(Rev_yoy_inc = Rev_yoy_inc,
                                             OpIncome_yoy_inc = OpIncome_yoy_inc,
                                             RevLRC4 = RevLRC4,
                                             EarningsLRC4 = EarningsLRC4)
        # aligned to desirable frequency  and index 
        self.children_factor_value = self.align_data_to_index(self.children_factor_value, self.fac_index)
        # filter out some stocks and standadrize 
        self.mask_child_fac(TradableStatus).standrdize_child_fac(_sd_win_sort,groupby_Fac=SWLevel1Sector)
        # compute GrowthFactor factor
        self.fac_value = self.children_factor_value.mean(axis = 1)
        
        return self


class ValueAdjGrowthFactor(GroupFactor):
    """
    This Factor computes the ValueAdjGrowthFactor = (ValueFactor + GrowthFactor)/2
    """
    
    def compute(self):
        
        
        # set factor index 
        
        # creating and compute all child factors
       
        self.instantiate_child_factors(PE_inv = PE_inv,
                                             PB_inv = PB_inv,
                                             PS_inv = PS_inv,
                                            FCFFOverMktCap  = FCFFOverMktCap,
                                           EBITOverEV = EBITOverEV,
                                           Rev_yoy_inc = Rev_yoy_inc,
                                           OpIncome_yoy_inc = OpIncome_yoy_inc)
        # filter out some stocks
        self.mask_child_fac(CommonFilters)
        self.children_factor_value = self.children_factor_value.astype(float)
        
        self.children_factor_value['RawValueFactor'] = self.children_factor_value[['PE_inv','PB_inv','PS_inv','FCFFOverMktCap','EBITOverEV']].mean(axis =1)
        self.children_factor_value['RawGrowthFactor'] = self.children_factor_value[['Rev_yoy_inc','OpIncome_yoy_inc']].mean(axis = 1)/100
        
        
        self.fac_value = self.children_factor_value['RawValueFactor'] + self.children_factor_value['RawGrowthFactor']
        
        # standadrize 
        self.standrdize(_sd_win_sort,groupby_Fac=GicsSector)
    
        
        
        return self




class LeverageFactor(GroupFactor):
     """
     LeverageFactor = (0.38 DebtOverAssets + 0.35 MktLeverage + 0.27 BookLeverage)
     """

     def compute(self):
         
         
         # set factor index 
         
         # creating and compute all child factors
        
         self.instantiate_child_factors(DebtOverAssets = DebtOverAssets,
                                       MktLeverage = MktLeverage,
                                       BookLeverage = BookLeverage)
         # aligned to desirable frequency  and index 
         self.children_factor_value = self.align_data_to_index(self.children_factor_value, self.fac_index)
         # filter out some stocks and standadrize 
         self.mask_child_fac(TradableStatus).standrdize_child_fac(_sd_win_sort,groupby_Fac=SWLevel1Sector)
         

         
         # compute QualityFactor factor
         
         self.fac_value = (0.38 * self.children_factor_value['MktLeverage'] + 
                           0.35* self.children_factor_value['DebtOverAssets'] + 
                           0.27* self.children_factor_value['BookLeverage'])
         
         return self

class ReversalFactor(GroupFactor):
    """
    ReversalFactor =  Mean(PriceMAInc250,PriceMAInc21,PriceMAInc61,TRIX10,TRIX5,PLRC12,PLRC24,BullPower,ROC20,Turnover5DaysOver120Days)
    """
    def compute(self):
        reversal_factor_names_dict = {factor_name : eval(factor_name) for factor_name in 
                              ['PriceMAInc250',
                                'PriceMAInc21',
                               'PriceMAInc61','TRIX10','TRIX5',
                               'PLRC12','PLRC24',
                               'BullPower',
                               'ROC20','Turnover5DaysOver120Days']}
        self.instantiate_child_factors(**reversal_factor_names_dict)
        self.children_factor_value = self.align_data_to_index(self.children_factor_value, self.fac_index)
        self.mask_child_fac(TradableStatus).standrdize_child_fac(_sd_win_sort,groupby_Fac = SWLevel1Sector,reverse = True).mask_child_fac(AllDayPriceLmit,operator='not')
        self.fac_value = self.children_factor_value.mean(axis =1)

class OverallMomentumFactor(GroupFactor):
    """
    OverallMomentumFactor = ECDF(ROC145 - ROC25) - ECDF(ROC25)
    """
    def compute(self):
        self.instantiate_child_factors(ROC145MinusROC25 = ROC145MinusROC25,ROC25 = ROC25)
        self.children_factor_value = self.align_data_to_index(self.children_factor_value, self.fac_index)
        self.mask_child_fac(TradableStatus).standrdize_child_fac(_sd_win_sort,
                                                                 groupby_Fac = SWLevel1Sector,
                                                                 reverse = True).mask_child_fac(AllDayPriceLmit,operator='not')
        self.fac_value = self.children_factor_value['ROC25'] - self.children_factor_value['ROC145MinusROC25'] 
#### Liqudity Factors

class STOM(GroupFactor):
    """
    STOM = rolling(21).apply(np.log(sum(x)))
    """
    def compute(self):
        # self.instantiate_child_factors(TurnOverRatio = TurnOverRatio).mask_child_fac(TradableStatus).mask_child_fac(AllDayPriceLmit,operator = 'not')
        self.instantiate_child_factors(TurnOverRatio = TurnOverRatio)
        self.fac_value = self.children_factor_value['TurnOverRatio'].groupby(level = 'code').progress_apply(lambda x: x.rolling(21).sum())
        self.fac_value = np.log(self.fac_value)

class STOQ(GroupFactor):
    """
    STOQ = rolling(63).apply(np.log(sum(x)))
    """
    def compute(self):
        self.instantiate_child_factors(TurnOverRatio = TurnOverRatio)
        self.fac_value = self.children_factor_value['TurnOverRatio'].groupby(level = 'code').progress_apply(lambda x: x.rolling(63).sum())/3
        self.fac_value = np.log(self.fac_value)

class STOA(GroupFactor):
    """
    STOA = rolling(252).apply(np.log(sum(x)))
    """
    def compute(self):
        self.instantiate_child_factors(TurnOverRatio = TurnOverRatio)
        self.fac_value = self.children_factor_value['TurnOverRatio'].groupby(level = 'code').progress_apply(lambda x: x.rolling(252).sum())/12
        self.fac_value = np.log(self.fac_value)

class LiqudityFactor(GroupFactor):
    """
    LiqudityFactor = 0.35 * Turnover20 + 0.35 * Turnover60 + 0.3 Turnover240
    """
    def compute(self):
        self.instantiate_child_factors(Turnover20Days = Turnover20Days,
                                       Turnover60Days = Turnover60Days,
                                       Turnover240Days = Turnover240Days)
        self.children_factor_value = self.align_data_to_index(self.children_factor_value, self.fac_index)
        self.mask_child_fac(TradableStatus).standrdize_child_fac(_sd_win_sort,
                                                                 groupby_Fac = SWLevel1Sector,
                                                                 reverse = True).mask_child_fac(AllDayPriceLmit,operator='not')
        self.fac_value = self.children_factor_value['Turnover20Days']*0.35 +\
            self.children_factor_value['Turnover60Days']*0.35 + \
                self.children_factor_value['Turnover240Days']*0.3
        
#### Time Series Factors

class ValueOfValueFactor(GroupFactor):
    """
    cross sectional 90 percentile - 10 percentile as an indicator of the strength of value factor 
    """

    def compute(self):
        
        
        # set factor index 
        
        # creating and compute all child factors
       
        self.instantiate_child_factors(PE_inv = PE_inv,
                                             PB_inv = PB_inv,
                                             PS_inv = PS_inv,
                                            FCFFOverMktCap  = FCFFOverMktCap)
        # filter out some stocks
        self.mask_child_fac(CommonFilters)
        
        # compute value spread of each sector  
        # self.standrdize_child_fac(top_minus_bottom,groupby_Fac=GicsSector)
        self.standrdize_child_fac(top_minus_bottom)
        # # # compute spead's historical percentile
        # self.children_factor_value = self.children_factor_value/self.children_factor_value.groupby(level = 'code').cummax()
        
        # compute value of value factor
        
        self.fac_value = self.children_factor_value.mean(axis = 1)
        
        return self
    