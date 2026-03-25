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
# from test_func_operator import *
from func_operator import *
import time
import datetime
# from factor_neutral import transfer_data_to_valid_and_not_valid




class FactorEvaluate(object):
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


class CalculateICIR(FactorEvaluate):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)

        self.save_info = param_info['save_info']
        self.operators = [
            {
                "class": PctChgHfqDaily,
                "output_name_mapping": {"pct_chg_hfq_daily": "pct_chg_hfq_daily"}
            },
            # {
            #     "func": get_factor_ic,
            #     "param":{},
            #     "input_data":{"data":"factor_data"},
            #     "output_data":["ic_score"]
            # }
        ]



class PctChgHfqDaily(FactorEvaluate):
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


class AdjClosePrice(FactorEvaluate):
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


class PreAdjClosePrice(FactorEvaluate):
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

class UnadjClosePrice(FactorEvaluate):
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


class AdjFactor(FactorEvaluate):
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



def cal_pct_chg_hfq(adj_close_price_name, pre_adj_close_price_name, data, output_name):

    pct_chg_hfq = divide_two_variable(adj_close_price_name, pre_adj_close_price_name, output_name, data).applymap(lambda x: (x-1)*100)

    return pct_chg_hfq

























'''



def del_outlier(factor_df, factor_name, method="mad", n=3):
    """
    Description
    ----------
    对每期因子进行去极值

    Parameters
    ----------
    factor_df: pandas.DataFrame. 因子数据,格式为trade_date,stock_code,factor
    factor_name: str. 因子名称
    method: str. 去极值方式,为'mad'或'sigma',默认为mad
    n: float.去极值的n值.默认取值为3

    Return
    ----------
    pandas.DataFrame.
    去极值后的因子数据, 格式为trade_date,stock_code,factor
    """
    utils._check_sub_columns(factor_df, [factor_name])
    factor_df = factor_df.copy()
    if method == "mad":
        g = factor_df.groupby("trade_date", group_keys=False)
        factor_df = g.apply(_single_mad_del, factor_name, n)
    elif method == "sigma":
        g = factor_df.groupby("trade_date", group_keys=False)
        factor_df = g.apply(_single_sigma_del, factor_name, n)
    if method not in ["mad", "sigma"]:
        raise ValueError("method must be mad or sigma")
    return factor_df


def _single_mad_del(factor_df, factor_name, n):
    """
    Description
    ----------
    单期MAD法去极值

    Parameters
    ----------
    factor_df: pandas.DataFrame. 因子值数据
    factor_name: str.因子名称
    n: float. 去极值的n

    Return
    ----------
    去极值后的因子数据
    """
    # 找出当期factor和factor_median的偏差bias_sr
    factor_median = factor_df[factor_name].median()
    bias_sr = abs(factor_df[factor_name] - factor_median)
    # 找到bias_sr的中位数new_median
    new_median = bias_sr.median()
    # 找到上下界
    dt_up = factor_median + n * new_median
    dt_down = factor_median - n * new_median

    # 超出上下界的值，赋值为上下界
    factor_df[factor_name] = factor_df[factor_name].clip(dt_down, dt_up, axis=0)
    return factor_df


def _single_sigma_del(factor_df, factor_name, n):
    """
    Description
    ----------
    单期Sigma法去极值

    Parameters
    ----------
    factor_df: pandas.DataFrame. 因子值数据
    factor_name: str. 因子名称
    n: float. 去极值的n

    Return
    ----------
    去极值后的因子数据
    """
    factor_mean = factor_df[factor_name].mean()
    factor_std = factor_df[factor_name].std()
    dt_up = factor_mean + n * factor_std
    dt_down = factor_mean - n * factor_std
    factor_df[factor_name] = factor_df[factor_name].clip(
        dt_down, dt_up, axis=0
    )  # 超出上下限的值，赋值为上下限
    return factor_df


# 标准化代码
def standardize(factor_df, factor_name, method="rank"):
    """
    Description
    ----------
    标准化

    Parameters
    ----------
    factor: pandas.DataFrame,因子值,格式为trade_date,stock_code,factor
    factor_name: str.因子名称
    method: str.中性化方式，可选为'rank'（排序标准化）或者'zscore'（Z-score标准化），默认为rank

    Return
    ----------
    pandas.DataFrame.
    标准化后的因子数据, 格式为trade_date,stock_code,factor
    """
    utils._check_sub_columns(factor_df, [factor_name])
    if method == "zscore":
        g = factor_df.groupby("trade_date", group_keys=False)
        factor_df = g.apply(_single_zscore_standardize, factor_name)
    elif method == "rank":
        g = factor_df.groupby("trade_date", group_keys=False)
        factor_df = g.apply(_single_rank_standardize, factor_name)
    else:
        raise ValueError("method must be rank or zscore")
    return factor_df


def _single_rank_standardize(factor_df, factor_name):
    """
    Description
    ----------
    单期因子数据排序标准化

    Parameters
    ----------
    factor: pandas.DataFrame,因子值,格式为trade_date,stock_code,factor
    factor_name: str.因子名称

    Return
    ----------
    pandas.DataFrame.排序标准化后的因子数据
    """
    factor_df[factor_name] = factor_df[factor_name].rank()
    return _single_zscore_standardize(factor_df, factor_name)


def _single_zscore_standardize(factor_df, factor_name):
    """
    Description
    ----------
    单期因子数据zscore标准化

    Parameters
    ----------
    factor: pandas.DataFrame,因子值,格式为trade_date,stock_code,factor
    factor_name: str.因子名称

    Return
    ----------
    pandas.DataFrame.zscore标准化后的因子数据
    """
    factor_mean = factor_df[factor_name].mean()
    factor_std = factor_df[factor_name].std()
    factor_df[factor_name] = (factor_df[factor_name] - factor_mean) / factor_std
    return factor_df


# 中性化代码
def neutralize(factor_df, factor_name, mktmv_df=None, industry_df=None):
    """
    Description
    ----------
    中性化

    Parameters
    ----------
    factor_df: pandas.DataFrame.
        因子值, 格式为trade_date,stock_code,factor
    mktmv_df: pandas.DataFrame.
        股票流通市值,格式为trade_date,stock_code,mktmv.
        默认为None即不进行市值中性化
    industry_df: pandas.DataFrame, 股票所属行业, 格式为trade_date,stock_code,ind_code.默认为None即不进行行业中性化

    Return
    ----------
    pandas.DataFrame.
    中性化后的因子数据, 格式为trade_date,stock_code,factor
    """
    neu_factor = factor_df.copy()
    if mktmv_df is not None:
        neu_factor = mktmv_neutralize(neu_factor, factor_name, mktmv_df)
    if industry_df is not None:
        neu_factor = ind_neutralize(neu_factor, factor_name, industry_df)
    return neu_factor


# 市值中性化
def mktmv_neutralize(factor_df, factor_name, mktmv_df):
    """
    Description
    ----------
    市值中性化

    Parameters
    ----------
    factor_df: pandas.DataFrame, 格式为trade_date, stock_code, factor
    factor_name: str.因子名称
    mktmv_df: pandas.DataFrame,股票流通市值,格式为trade_date,stock_code,mktmv.

    Return
    ----------
    pandas.DataFrame.中性化后的因子值
    """
    # 检查输入数据
    utils._check_sub_columns(mktmv_df, ["mktmv"])
    utils._check_sub_columns(factor_df, [factor_name])
    # 合并两个数据，groupby做回归
    df = pd.merge(factor_df, mktmv_df, on=["trade_date", "stock_code"])
    g = df.groupby("trade_date", group_keys=False)
    df = g.apply(_mktmv_reg, factor_name)
    df = df.drop(columns=["mktmv"])
    return df


def _mktmv_reg(df, factor_name):
    """
    Description
    ----------
    对单期因子进行市值中性化

    Parameters
    ----------
    df:pandas.DataFrame, 格式为trade_date, stock_code, factor, mktmv
    factor_name: str.因子名称

    Return
    ----------
    pandas.DataFrame.中性化后的因子值
    """
    x = df["mktmv"].values.reshape(-1, 1)
    y = df[factor_name]
    lr = LinearRegression()
    lr.fit(x, y)  # 拟合
    y_predict = lr.predict(x)  # 预测
    df[factor_name] = y - y_predict
    return df


# 行业中性化
def ind_neutralize(factor_df, factor_name, industry_df):
    """
    Description
    ----------
    对每期因子进行行业中性化
    方法: 先用pd.get_dummies生成行业虚拟变量, 然后用带截距项回归得到残差作为因子

    Parameters
    ----------
    factor_df: pandas.DataFrame,因子值,格式为trade_date,stock_code,factor
    factor_name: str. 因子名称
    industry_df: pandas.DataFrame, 股票所属行业, 格式为trade_date,stock_code,ind_code

    Return
    ----------
    pandas.DataFrame.行业中性化后的因子数据
    """
    # 检查输入数据
    utils._check_sub_columns(factor_df, [factor_name])
    utils._check_sub_columns(industry_df, ["ind_code"])
    # 生成虚拟变量，拼接形成新的df
    ind_dummies = pd.get_dummies(industry_df["ind_code"], drop_first=True, prefix="ind")
    # 格式为 trade_date,stock_code,dummies_ind_code
    ind_new = pd.concat([industry_df.drop(columns=["ind_code"]), ind_dummies], axis=1)
    # 拼接两个表格
    df = pd.merge(factor_df, ind_new, on=["trade_date", "stock_code"])
    g = df.groupby("trade_date", group_keys=False)
    df = g.apply(_single_ind_neutralize, factor_name)
    df = df[["trade_date", "stock_code", factor_name]].copy()
    return df


def _single_ind_neutralize(df, factor_name):
    """
    Description
    ----------
    对单期因子进行行业中性化

    Parameters
    ----------
    df: pandas.DataFrame, 因子值和行业的df, 格式为trade_date,stock_code,'factor_name',dummy_ind_code
    factor_name: str. 因子名称

    Return
    ----------
    pandas.DataFrame.行业中性化后的因子数据
    """
    x = df.iloc[:, 3:]
    y = df[factor_name]
    # 计算回归残差
    lr = LinearRegression()
    lr.fit(x, y)
    y_predict = lr.predict(x)
    df[factor_name] = y - y_predict
    return df


def get_factor_ic(factor_df, ret_df, factor_name):
    """
    Description
    ----------
    计算因子IC序列

    Parameters
    ----------
    factor_df: pandas.DataFrame. 未提前的因子数据.

    Return
    ----------
    pandas.DataFrame.
    """

    def calc_corr_func(df):
        corr_matrix = df[[factor_name, "ret"]].corr()
        correlation = corr_matrix.loc[factor_name, "ret"]
        return correlation

    # prev_factor_df = utils.get_previous_factor(factor_df)
    df = pd.merge(factor_df, ret_df, on=["trade_date", "code"])
    ic_df = df.groupby(["trade_date"], group_keys=False).apply(calc_corr_func)
    ic_df = ic_df.reset_index()
    ic_df.columns = ["trade_date", "IC"]
    return ic_df


'''
