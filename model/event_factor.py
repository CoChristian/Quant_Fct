from model.indicator_operator import merge_data,get_hist_data_4_factor_compute, resample_data_to_index, \
    align_data_to_index, save_data_to_table
from model.indicator_operator import FactorCompute, DailyIndex
from model.factor_neutral import get_data_from_multi_source, transfer_data_to_valid_and_not_valid
from datetime import datetime
import pandas as pd

###event data####

def gen_hist_tag(data, daily_tag, window_size, hist_tag, shift_window_size=0):

    data.sort_index(level=['code', 'trade_date'], inplace=True)
    data[hist_tag] = data[daily_tag].groupby(level='code').apply(lambda x: x.rolling(window_size, min_periods=1).sum().shift(shift_window_size).map(lambda x: 1 if x > 0 else 0))
    return data[[hist_tag]]

def gen_fin_forecast_tag(data):

    data['FinanceGoodPredTag'] = data['type'].map(lambda x: x in ['业绩大幅上升', "业绩预增", "预计扭亏", "预计减亏", "大幅减亏"])
    data['FinancePoorPredTag'] = data['type'].map(lambda x: x in ['业绩预亏', "业绩大幅下降", "业绩预降"])
    # data.sort_index(level=['code', 'trade_date'], inplace=True)
    # data['HistGoodPredTag'] = data['finance_good_pred_tag'].groupby(level='code').apply(lambda x: x.rolling(window_size, min_periods=1).sum().map(lambda x: 1 if x > 0 else 0))
    # data['HistPoorPredTag'] = data['finance_poor_pred_tag'].groupby(level='code').apply(lambda x: x.rolling(window_size, min_periods=1).sum().map(lambda x: 1 if x > 0 else 0))
    return data[data['FinanceGoodPredTag']>0][['FinanceGoodPredTag']], data[data['FinancePoorPredTag']>0][['FinancePoorPredTag']]


class FinForecastDailyTag(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.window_size = param_info.get("window_size", 80)
        self.operators = [
            {
                'func': get_hist_data_4_factor_compute,
                'param': {
                    "read_engine": "",
                    "save_engine": "",
                    "start_date": 0,
                    "end_date": 0,
                    "table": "fin_forecast_stk",
                    "field": [],
                    "hist_year": -1,
                    "name_dict": {}},
                "input_data": {},
                "output": ["fin_forecast_data"]
             },

            {
                "class": DailyIndex,
                "output_name_mapping": {"daily_index": "daily_index"},
            },
            {
                "func": resample_data_to_index,
                "param": {},
                "input_data": {"data": "fin_forecast_data", "index": "daily_index"},
                "output": ["daily_fin_forecast_data"],
            },
            {
                "func": gen_fin_forecast_tag,
                "param": {},
                "input_data": {"data": "daily_fin_forecast_data", },
                "output": ["good_fin_forecast_tag", 'poor_fin_forecast_tag'],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "zero"},
                "input_data": {"data": "good_fin_forecast_tag", "index": "daily_index"},
                "output": ["daily_good_fin_forecast_tag_data"],
            },
            {
                "func": gen_hist_tag,
                "param": {"daily_tag": "FinanceGoodPredTag", "hist_tag": "HistFinanceGoodPredTag", "window_size": self.window_size},
                "input_data": {"data": "daily_good_fin_forecast_tag_data"},
                "output": ["hist_daily_fin_good_forecast_data"],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "zero"},
                "input_data": {"data": "poor_fin_forecast_tag", "index": "daily_index"},
                "output": ["daily_poor_fin_forecast_tag"],
            },
            {
                "func": gen_hist_tag,
                "param": {"daily_tag": "FinancePoorPredTag", "hist_tag": "HistFinancePoorPredTag",
                          "window_size": self.window_size},
                "input_data": {"data": "daily_poor_fin_forecast_tag"},
                "output": ["hist_daily_fin_poor_forecast_data"],
            },
            {
                "func": merge_data,
                "param": {},
                "input_data": {"1": "hist_daily_fin_good_forecast_data", "2": "hist_daily_fin_poor_forecast_data"},
                "output": ["hist_daily_fin_forecast_data"]
            },
        ]
        self.output_vars = ['hist_daily_fin_forecast_data']


def gen_share_pledge_tag(data):
    data['PledgeTag'] = data['unpledged_number'].isnull()
    data = data[data['PledgeTag'] > 0]
    return data[['PledgeTag']]


class SharePledgeDailyTag(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.window_size = param_info.get("window_size", 60)
        self.operators = [
            {
                'func': get_hist_data_4_factor_compute,
                'param': {
                    "read_engine": "",
                    "save_engine": "",
                    "start_date": 0,
                    "end_date": 0,
                    "table": "shares_pledge_stk",
                    "field": [],
                    "hist_year": -1,
                    "name_dict": {}},
                "input_data": {},
                "output": ["shares_pledge_data"]
             },

            {
                "class": DailyIndex,
                "output_name_mapping": {"daily_index": "daily_index"},
            },
            {
                "func": resample_data_to_index,
                "param": {},
                "input_data": {"data": "shares_pledge_data", "index": "daily_index"},
                "output": ["daily_shares_pledge_data"],
            },
            {
                "func": gen_share_pledge_tag,
                "param": {},
                "input_data": {"data": "daily_shares_pledge_data", },
                "output": ["daily_shares_pledge_tag_data"],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "zero"},
                "input_data": {"data": "daily_shares_pledge_tag_data", "index": "daily_index"},
                "output": ["daily_shares_pledge_tag_data"],
            },
            {
                "func": gen_hist_tag,
                "param": {"daily_tag": "PledgeTag", "hist_tag": "HistPledgeTag",
                          "window_size": self.window_size},
                "input_data": {"data": "daily_shares_pledge_tag_data"},
                "output": ["hist_daily_shares_pledge_tag_data"],
            },
        ]
        self.output_vars = ['hist_daily_shares_pledge_tag_data']

def gen_bonus_tag(data):
    ##送股数据###

    data = data.reset_index()
    data = data[data['implementation_bonusnote'].notnull()]
    implementation_bonusnote_data = data.groupby(['code', 'trade_date'])['implementation_bonusnote'].sum().reset_index()
    implementation_bonusnote_data['ShareBonusTag'] = implementation_bonusnote_data['implementation_bonusnote'].map(lambda x: '送' in x and "不分配不转增" not in x)
    implementation_bonusnote_data = implementation_bonusnote_data[implementation_bonusnote_data['ShareBonusTag'] > 0]
    return implementation_bonusnote_data.set_index(['code', 'trade_date'])[['ShareBonusTag']]


class ShareBonusDailyTag(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.window_size = param_info.get("window_size", 120)
        self.operators = [
            {
                'func': get_hist_data_4_factor_compute,
                'param': {
                    "read_engine": "",
                    "save_engine": "",
                    "start_date": 0,
                    "end_date": 0,
                    "table": "xr_xd_stk",
                    "field": [],
                    "hist_year": -1,
                    "name_dict": {}},
                "input_data": {},
                "output": ["bonus_data"]
             },

            {
                "class": DailyIndex,
                "output_name_mapping": {"daily_index": "daily_index"},
            },
            {
                "func": resample_data_to_index,
                "param": {},
                "input_data": {"data": "bonus_data", "index": "daily_index"},
                "output": ["daily_bonus_data"],
            },
            {
                "func": gen_bonus_tag,
                "param": {},
                "input_data": {"data": "daily_bonus_data"},
                "output": ["daily_bonus_tag_data"],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "zero"},
                "input_data": {"data": "daily_bonus_tag_data", "index": "daily_index"},
                "output": ["daily_bonus_tag_data"],
            },
            {
                "func": gen_hist_tag,
                "param": {"daily_tag": "ShareBonusTag", "hist_tag": "HistShareBonusTag",
                          "window_size": self.window_size},
                "input_data": {"data": "daily_bonus_tag_data"},
                "output": ["hist_daily_bonus_tag_data"],
            },
        ]
        self.output_vars = ['hist_daily_bonus_tag_data']


def process_limited_share_unlock_data(data):
    data = data.reset_index()
    data['actual_unlimited_date'] = data.actual_unlimited_date.map(lambda x: int(str(x).replace('-', '')))
    data['pub_date'] = data['trade_date']
    data['trade_date'] = data['actual_unlimited_date']
    return data.set_index(['code', 'trade_date'])

def gen_limited_share_unlock_tag(data):
    ##解禁股限售数据###
    data['UnLimitTag'] = 1
    data['ShareRewardUnlimitTag'] = data['limited_reason'].map(lambda x: 1 if x == "股权激励" else 0)
    data['NoShareRewardUnlimitTag'] = data['limited_reason'].map(lambda x: 1 if x != "股权激励" else 0)
    # data.sort_index(level=['code', 'trade_date'], inplace=True)
    # data['HistShareBonusTag'] = data['share_bonus_tag'].groupby(level='code').apply(lambda x: x.rolling(window_size, min_periods=1).sum().map(lambda x: 1 if x > 0 else 0))
    # return data[['UnLimitTag', 'ShareRewardUnlimitTag', 'NoShareRewardUnlimitTag']]
    return data[data['UnLimitTag']>0][['UnLimitTag']], data[data['ShareRewardUnlimitTag']>0][['ShareRewardUnlimitTag']], data[data['NoShareRewardUnlimitTag']>0][['NoShareRewardUnlimitTag']]

class LimitedSharesUnlockDailyTag(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.short_window_size = param_info.get("show_window_size", 15)
        self.long_window_size = param_info.get("show_window_size", 85)
        self.operators = [
            {
                'func': get_hist_data_4_factor_compute,
                'param': {
                    "read_engine": "",
                    "save_engine": "",
                    "start_date": 0,
                    "end_date": 0,
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
                "class": DailyIndex,
                "output_name_mapping": {"daily_index": "daily_index"},
            },
            {
                "func": resample_data_to_index,
                "param": {"drop_max": True},
                "input_data": {"data": "limited_shares_unlock_data", "index": "daily_index"},
                "output": ["daily_limited_shares_unlock_data"],
            },
            {
                "func": gen_limited_share_unlock_tag,
                "param": {},
                "input_data": {"data": "daily_limited_shares_unlock_data",},
                "output": ["unlimit_tag", "share_reward_unlimit_tag", "no_share_reward_unlimit_tag"],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "zero"},
                "input_data": {"data": "unlimit_tag", "index": "daily_index"},
                "output": ["daily_unlimit_tag"],
            },
            {
                "func": gen_hist_tag,
                "param": {"daily_tag": "UnLimitTag", "hist_tag": "HistUnLimitTag",
                          "window_size": self.short_window_size},
                "input_data": {"data": "daily_unlimit_tag"},
                "output": ["hist_unlimit_tag_data"],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "zero"},
                "input_data": {"data": "share_reward_unlimit_tag", "index": "daily_index"},
                "output": ["daily_share_reward_unlimit_tag"],
            },
            {
                "func": gen_hist_tag,
                "param": {"daily_tag": "ShareRewardUnlimitTag", "hist_tag": "HistShareRewardUnlimitTag",
                          "window_size": self.short_window_size},
                "input_data": {"data": "daily_share_reward_unlimit_tag"},
                "output": ["hist_share_reward_unlimit_tag_data"],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "zero"},
                "input_data": {"data": "no_share_reward_unlimit_tag", "index": "daily_index"},
                "output": ["daily_no_share_reward_unlimit_tag"],
            },
            {
                "func": gen_hist_tag,
                "param": {"daily_tag": "NoShareRewardUnlimitTag", "hist_tag": "HistNoShareRewardUnlimitTag",
                          "window_size": self.short_window_size},
                "input_data": {"data": "daily_no_share_reward_unlimit_tag"},
                "output": ["hist_no_share_reward_unlimit_tag_data"],
            },
            {
                "func": gen_hist_tag,
                "param": {"daily_tag": "ShareRewardUnlimitTag", "hist_tag": "HistLongShareRewardUnlimitTag",
                          "window_size": self.long_window_size, "shift_window_size": self.short_window_size+1},
                "input_data": {"data": "daily_share_reward_unlimit_tag"},
                "output": ["hist_long_share_reward_unlimit_tag_data"],
            },
            {
                "func": merge_data,
                "param": {},
                "input_data": {"1": "hist_unlimit_tag_data", "2": "hist_share_reward_unlimit_tag_data",
                               "3": "hist_no_share_reward_unlimit_tag_data", "4": "hist_long_share_reward_unlimit_tag_data"},
                "output": ["hist_daily_limited_shares_unlock_data"]
            },
        ]
        self.output_vars = ['hist_daily_limited_shares_unlock_data']


def gen_repurchase_tag(data):
    data['RepurchasePlanTag'] = data['proc'].map(lambda x: x in ['预案', '提议'])

    return data[data['RepurchasePlanTag'] > 0][['RepurchasePlanTag']]

class RepurchaseDailyTag(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.window_size = param_info.get("window_size", 120)
        self.operators = [
            {
                'func': get_hist_data_4_factor_compute,
                'param': {
                    "read_engine": "",
                    "save_engine": "",
                    "start_date": 0,
                    "end_date": 0,
                    "table": "repurchase_data",
                    "field": [],
                    "hist_year": -1,
                    "name_dict": {}},
                "input_data": {},
                "output": ["repurchase_data"]
             },
            {
                "class": DailyIndex,
                "output_name_mapping": {"daily_index": "daily_index"},
            },
            {
                "func": resample_data_to_index,
                "param": {},
                "input_data": {"data": "repurchase_data", "index": "daily_index"},
                "output": ["daily_repurchase_data"],
            },
            {
                "func": gen_repurchase_tag,
                "param": {},
                "input_data": {"data": "daily_repurchase_data",},
                "output": ["daily_repurchase_tag_data"],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "zero"},
                "input_data": {"data": "daily_repurchase_tag_data", "index": "daily_index"},
                "output": ["daily_repurchase_tag_data"],
            },
            {
                "func": gen_hist_tag,
                "param": {"daily_tag": "RepurchasePlanTag", "hist_tag": "HistRepurchasePlanTag",
                          "window_size": self.window_size},
                "input_data": {"data": "daily_repurchase_tag_data"},
                "output": ["hist_daily_repurchase_tag_data"],
            },
        ]
        self.output_vars = ['hist_daily_repurchase_tag_data']

def gen_frozen_tag(data):
    data['UnfrozenTag'] = data['frozen_reason'].map(lambda x: 1 if "解除冻结" in x else 0)
    data['FrozenTag'] = data['frozen_reason'].map(lambda x: 1 if "解除冻结" not in x else 0)
    return data[data['UnfrozenTag'] > 0][['UnfrozenTag']], data[data['FrozenTag'] > 0][['FrozenTag']]

class SharesFrozenDailyTag(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.window_size = param_info.get("window_size", 100)
        self.operators = [
            {
                'func': get_hist_data_4_factor_compute,
                'param': {
                    "read_engine": "",
                    "save_engine": "",
                    "start_date": 0,
                    "end_date": 0,
                    "table": "shares_frozen_stk",
                    "field": [],
                    "hist_year": -1,
                    "name_dict": {}},
                "input_data": {},
                "output": ["shares_frozen_data"]
             },
            {
                "class": DailyIndex,
                "output_name_mapping": {"daily_index": "daily_index"},
            },
            {
                "func": resample_data_to_index,
                "param": {},
                "input_data": {"data": "shares_frozen_data", "index": "daily_index"},
                "output": ["daily_shares_frozen_data"],
            },
            {
                "func": gen_frozen_tag,
                "param": {},
                "input_data": {"data": "daily_shares_frozen_data",},
                "output": ["unfrozen_tag", "frozen_tag"],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "zero"},
                "input_data": {"data": "unfrozen_tag", "index": "daily_index"},
                "output": ["daily_unfrozen_tag"],
            },
            {
                "func": gen_hist_tag,
                "param": {"daily_tag": "UnfrozenTag", "hist_tag": "HistUnfrozenTag",
                          "window_size": self.window_size},
                "input_data": {"data": "daily_unfrozen_tag"},
                "output": ["hist_daily_unfrozen_tag_data"],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "zero"},
                "input_data": {"data": "frozen_tag", "index": "daily_index"},
                "output": ["daily_frozen_tag"],
            },
            {
                "func": gen_hist_tag,
                "param": {"daily_tag": "FrozenTag", "hist_tag": "HistFrozenTag",
                          "window_size": self.window_size},
                "input_data": {"data": "daily_frozen_tag"},
                "output": ["hist_daily_frozen_tag_data"],
            },
            {
                "func": merge_data,
                "param": {},
                "input_data": {"1": "hist_daily_unfrozen_tag_data", "2": "hist_daily_frozen_tag_data",
                               },
                "output": ["hist_daily_frozen_unfrozen_tag_data"]
            },
        ]
        self.output_vars = ['hist_daily_frozen_unfrozen_tag_data']

def gen_share_change_number(data):
    data['direction'] = data['type'].map(lambda x: 1 if x == 0 else -1)
    data['change_number'] = data['change_number']*data['direction']
    share_change_number = data.groupby(['code', 'trade_date']).sum()[['change_number']]
    return share_change_number[['change_number']]


class LargeShareholderShareChangeDailyTag(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.window_size = param_info.get("window_size", 120)


        self.operators = [
            {
                'func': get_hist_data_4_factor_compute,
                'param': {
                    "read_engine": "",
                    "save_engine": "",
                    "start_date": 0,
                    "end_date": 0,
                    "table": "large_shareholder_share_change_stk",
                    "field": [],
                    "hist_year": -1,
                    "name_dict": {}},
                "input_data": {},
                "output": ["large_shareholder_share_change_data"]
             },
            {
                "class": DailyIndex,
                "output_name_mapping": {"daily_index": "daily_index"},
            },
            {
                "func": resample_data_to_index,
                "param": {},
                "input_data": {"data": "large_shareholder_share_change_data", "index": "daily_index"},
                "output": ["daily_large_shareholder_share_change_data"],
            },
            {
                "func": gen_share_change_number,
                "param": {},
                "input_data": {"data": "daily_large_shareholder_share_change_data",},
                "output": ["daily_large_shareholder_share_change_number_data"],
            },
            {
                "func": align_data_to_index,
                "param": {"fill_method": "zero"},
                "input_data": {"data": "daily_large_shareholder_share_change_number_data", "index": "daily_index"},
                "output": ["daily_large_shareholder_share_change_number_data"],
            },
            {
                "func": gen_hist_tag,
                "param": {"daily_tag": "change_number", "hist_tag": "HistShareChangePlusTag",
                          "window_size": self.window_size},
                "input_data": {"data": "daily_large_shareholder_share_change_number_data"},
                "output": ["hist_daily_share_change_plus_tag_data"],
            },

        ]
        self.output_vars = ['hist_daily_share_change_plus_tag_data']


class EventDailyFactor(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.save_info = param_info['save_info']
        self.operators = [
            {
                "class": FinForecastDailyTag,
                "param": {},
                "output_name_mapping": {"hist_daily_fin_forecast_data": "hist_daily_fin_forecast_data"},
            },
            {
                "class": SharePledgeDailyTag,
                'param': {},
                "output_name_mapping": {"hist_daily_shares_pledge_tag_data": "hist_daily_shares_pledge_tag_data"},
            },
            {
                "class": ShareBonusDailyTag,
                'param': {},
                "output_name_mapping": {"hist_daily_bonus_tag_data": "hist_daily_bonus_tag_data"},
            },
            {
                "class": LimitedSharesUnlockDailyTag,
                'param': {},
                "output_name_mapping": {"hist_daily_limited_shares_unlock_data": "hist_daily_limited_shares_unlock_data"},
            },
            {
                "class": RepurchaseDailyTag,
                'param': {},
                "output_name_mapping": {
                    "hist_daily_repurchase_tag_data": "hist_daily_repurchase_tag_data"},
            },
            {
                "class": SharesFrozenDailyTag,
                'param': {},
                "output_name_mapping": {
                    "hist_daily_frozen_unfrozen_tag_data": "hist_daily_frozen_unfrozen_tag_data"},
            },
            {
                "class": LargeShareholderShareChangeDailyTag,
                "param": {},
                "output_name_mapping": {
                    "hist_daily_share_change_plus_tag_data": "hist_daily_share_change_plus_tag_data"
                }
            },
            {
                "func": merge_data,
                "param": {},
                "input_data": {
                    "1": "hist_daily_fin_forecast_data",
                    "2": "hist_daily_shares_pledge_tag_data",
                    "3": "hist_daily_bonus_tag_data",
                    "4": "hist_daily_limited_shares_unlock_data",
                    "5": "hist_daily_repurchase_tag_data",
                    "6": "hist_daily_frozen_unfrozen_tag_data",
                    "7": "hist_daily_share_change_plus_tag_data"
                },
                "output": ["event_daily_factor"]
            },
            {
                "func": save_data_to_table,
                "param": {"engine": self.save_info['engine'], "table": self.save_info['table'],
                          "if_exists": self.save_info.get("if_exists", "append"), "if_reset_index": self.save_info.get("if_reset_index", False)},
                "input_data": {"data": "event_daily_factor"},
                "output": ["event_daily_factor"]
            }
        ]
        self.output_vars = ["event_daily_factor"]


### code neutral alpha ####

def gen_opt_to_trade(daily_trading_data, real_time_opt_to_trade):
    daily_trading_data = daily_trading_data.reset_index()
    all_trade_dates = daily_trading_data.trade_date.unique()
    all_trade_dates = sorted(all_trade_dates)
    trade_dates = [trade_date for trade_date in all_trade_dates if
                   datetime.strptime(str(trade_date), "%Y%m%d").weekday() == 2]
    opt_dates = [trade_date for trade_date in all_trade_dates if
                 datetime.strptime(str(trade_date), "%Y%m%d").weekday() == 1]
    opt_2_trade = {}
    for first_date, second_date in zip(all_trade_dates[:-1], all_trade_dates[1:]):

        if first_date in opt_dates:
            opt_2_trade.update({first_date: second_date})
        if second_date in trade_dates:
            opt_2_trade.update({first_date: second_date})
    #     trade_2_opt = {trade_date: opt_date for opt_date, trade_date in opt_2_trade.items()}
    max_opt_date = max(list(opt_2_trade.keys()))
    for opt_date in opt_dates:
        if opt_date > max_opt_date:
            opt_2_trade.update({opt_date: None})
    opt_2_trade.update(real_time_opt_to_trade.to_dict())
    return opt_2_trade


def gen_week_alpha(daily_trading_data, opt_2_trade, daily_industry_data, daily_markt_mv_data):
    daily_trading_data = daily_trading_data.reset_index()
    daily_industry_data = daily_industry_data.reset_index()
    daily_markt_mv_data = daily_markt_mv_data.reset_index()
    code_opt_infos = []
    for code, code_daily_df in daily_trading_data.groupby('code'):
        #         if code in code_2_valid_start:
        #             ##### 过滤掉上市市场不超过8个月的股票 #####
        #             code_daily_df = code_daily_df[code_daily_df.trade_date.map(lambda x: x >= code_2_valid_start[code])]
        if len(code_daily_df):
            #### 获得周收益 ####
            code_daily_df.sort_values('trade_date', inplace=True)
            code_daily_df['next_open'] = code_daily_df['open'].shift(-1)
            code_opt_day_info = code_daily_df[code_daily_df.trade_date.map(lambda x: x in opt_2_trade)]

            code_opt_day_info['r'] = code_opt_day_info['next_open'].shift(-1) / code_opt_day_info['next_open']
            code_opt_infos.append(code_opt_day_info)
    weekly_trading_r_df = pd.concat(code_opt_infos)
    # ####获得股票的行业数据 ####
    # sql_query = "select * from daily_industry_data"
    # daily_industry_data = pd.read_sql_query(sql_query, engine)
    # ####获得股票的市值数据####
    # sql_query = "select trade_date, code, market_cap from valuation_q"
    # daily_markt_mv_data = pd.read_sql_query(sql_query, engine)

    weekly_close_industry_data = pd.merge(weekly_trading_r_df,
                                          daily_industry_data[['trade_date', 'code', 'sw_l1_industry_name']],
                                          how='left', on=['trade_date', 'code'])
    weekly_close_industry_mv_data = pd.merge(weekly_close_industry_data, daily_markt_mv_data, how='left',
                                             on=['trade_date', 'code'])

    code_industry_mv_infos = []
    for code, code_weekly_industry_mv_data in weekly_close_industry_mv_data.groupby('code'):
        code_weekly_industry_mv_data.sort_values('trade_date', inplace=True)
        code_weekly_industry_mv_data['sw_l1_industry_name'].fillna(method='pad', inplace=True)
        code_weekly_industry_mv_data['market_cap'].fillna(method='pad', inplace=True)
        code_industry_mv_infos.append(code_weekly_industry_mv_data)
    weekly_close_industry_mv_data = pd.concat(code_industry_mv_infos)
    ####对超额进行行业和市值中性化####
    industry_mkt_cap = weekly_close_industry_mv_data.groupby(['trade_date', 'sw_l1_industry_name']).sum()[
        'market_cap'].reset_index()

    industry_mkt_cap['industry_market_cap'] = industry_mkt_cap['market_cap']

    weekly_close_industry_mv_data = pd.merge(weekly_close_industry_mv_data, industry_mkt_cap[
        ['trade_date', 'sw_l1_industry_name', 'industry_market_cap']], how='left',
                                             on=['trade_date', 'sw_l1_industry_name'])
    weekly_close_industry_mv_data['r_rate'] = weekly_close_industry_mv_data['r'] * weekly_close_industry_mv_data[
        'market_cap'] / weekly_close_industry_mv_data['industry_market_cap']
    industry_r_df = weekly_close_industry_mv_data.groupby(['trade_date', 'sw_l1_industry_name']).sum()[
        'r_rate'].reset_index()
    industry_r_df['industry_r'] = industry_r_df['r_rate']

    weekly_close_industry_mv_data = pd.merge(weekly_close_industry_mv_data,
                                             industry_r_df[['trade_date', 'sw_l1_industry_name', 'industry_r']],
                                             how='left', on=['trade_date', 'sw_l1_industry_name'])
    weekly_close_industry_mv_data['industry_neural_alpha'] = weekly_close_industry_mv_data['r'] - \
                                                             weekly_close_industry_mv_data['industry_r']

    date_infos = []
    for trade_date, date_info in weekly_close_industry_mv_data.groupby('trade_date'):
        date_info.sort_values('market_cap', inplace=True)
        date_info['mv_alpha'] = date_info['industry_neural_alpha'].rolling(99, min_periods=49, center=True).mean()
        date_info['mv_alpha_v2'] = date_info['r'].rolling(99, min_periods=49, center=True).mean()

        date_info['alpha'] = date_info['r'] - date_info['r'].mean()
        #     date_info['mv_alpha'] = date_info['r'].rolling(99, min_periods=49, center=True).mean()
        date_info['mv_industry_neural_alpha'] = date_info['industry_neural_alpha'] - date_info['mv_alpha']
        date_info['mv_neural_alpha'] = date_info['r'] - date_info['mv_alpha_v2']

        date_infos.append(date_info)
    weekly_close_industry_mv_alpha_data = pd.concat(date_infos)
    weekly_close_industry_mv_alpha_data['ts_code'] = weekly_close_industry_mv_alpha_data['code'].map(
        lambda x: x.replace("XSHE", 'SZ').replace("XSHG", "SH"))
    return weekly_close_industry_mv_alpha_data


class WeekNeutralAlpha(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.save_info = param_info['save_info']
        self.operators = [
            {
                'func': get_hist_data_4_factor_compute,
                'param': {
                    "read_engine": "mysql+pymysql://develop:haikuan_2025@localhost:3306/factor_research_full_a_share",
                    "save_engine": "mysql+pymysql://develop:haikuan_2025@localhost:3306/factor_research_full_a_share",
                    "start_date": 0,
                    "end_date": 0,
                    "table": "daily_trading_data",
                    "field": ['trade_date', 'code', 'close', 'pre_close', 'open'],
                    "hist_year": -1,
                    "name_dict": {}},
                "input_data": {},
                "output": ["daily_trading_data"]
            },
            {
                'func': get_hist_data_4_factor_compute,
                'param': {
                    "read_engine": "mysql+pymysql://develop:haikuan_2025@localhost:3306/factor_compute_new",
                    "save_engine": "mysql+pymysql://develop:haikuan_2025@localhost:3306/factor_compute_new",
                    "start_date": 0,
                    "end_date": 0,
                    "table": "opt_to_trade",
                    "field": [],
                    "index": ['opt_date'],
                    "hist_year": -1,
                    "name_dict": {}},
                "input_data": {},
                "output": ["real_time_opt_to_trade"]
            },
            {
                'func': get_hist_data_4_factor_compute,
                'param': {
                    "read_engine": "mysql+pymysql://develop:haikuan_2025@localhost:3306/factor_research_full_a_share",
                    "save_engine": "mysql+pymysql://develop:haikuan_2025@localhost:3306/factor_research_full_a_share",
                    "start_date": 0,
                    "end_date": 0,
                    "table": "daily_industry_data",
                    "field": ['trade_date', 'code', 'sw_l1_industry_name'],
                    "hist_year": -1,
                    "name_dict": {}},
                "input_data": {},
                "output": ["daily_industry_data"]
            },
            {
                'func': get_hist_data_4_factor_compute,
                'param': {
                    "read_engine": "mysql+pymysql://develop:haikuan_2025@localhost:3306/factor_research_full_a_share",
                    "save_engine": "mysql+pymysql://develop:haikuan_2025@localhost:3306/factor_research_full_a_share",
                    "start_date": 0,
                    "end_date": 0,
                    "table": "valuation_q",
                    "field": ['trade_date', 'code',  'market_cap'],
                    "hist_year": -1,
                    "name_dict": {}},
                "input_data": {},
                "output": ["daily_markt_mv_data"]
            },
            {
                'func': gen_opt_to_trade,
                "param": {},
                "input_data":
                    {
                        "daily_trading_data": "daily_trading_data",
                        "real_time_opt_to_trade": "real_time_opt_to_trade"
                    },
                "output": ["opt_2_trade"]
            },
            {
                "func": gen_week_alpha,
                "param": {},
                "input_data":
                    {
                        "daily_trading_data": "daily_trading_data",
                        "opt_2_trade": "opt_2_trade",
                        "daily_industry_data": "daily_industry_data",
                        "daily_markt_mv_data": "daily_markt_mv_data",
                    },
                "output": ["week_alpha"]
            },
            {
                "func": save_data_to_table,
                "param": {"engine": self.save_info['engine'], "table": self.save_info['table'],
                          "if_exists": self.save_info.get("if_exists", "append")},
                "input_data": {"data": "week_alpha"},
                "output": ["week_alpha"]
            }
        ]
        self.output_vars = ["week_alpha"]


def gen_score_from_hist(week_alpha, event_daily_factor):
    """
    从历史数据生成事件策略的整体打分
    """
    week_alpha = week_alpha.reset_index()
    event_daily_factor = event_daily_factor.reset_index()
    week_event_alpha_df = pd.merge(week_alpha, event_daily_factor, how='left', on=['code','trade_date'])

    ###### 获取历史交易日期 #####
    all_dates = sorted(week_event_alpha_df['trade_date'].unique())
    ##### 获取历史因子超额数据  ######
    all_hist_infos = [week_event_alpha_df[week_event_alpha_df.trade_date == _] for _ in all_dates]
    ####利用4年的历史数据给出事件因子整体打分 #####
    pred_industry_mv_alpha_infos = []
    for j in range(200, len(all_dates)):
        hist_dates = all_dates[j-200: j-2]
        hist_industry_mv_alpha_data = pd.concat(all_hist_infos[j-200: j-2])
        param_info = {'trade_date': all_dates[j]}
        #### 获取历史超额 #####
        finance_good_pred_alpha = hist_industry_mv_alpha_data[hist_industry_mv_alpha_data.HistFinanceGoodPredTag == 1]['mv_industry_neural_alpha'].mean()
        finance_poor_pred_alpha = hist_industry_mv_alpha_data[hist_industry_mv_alpha_data.HistFinancePoorPredTag == 1]['mv_industry_neural_alpha'].mean()

        pledge_alpha = hist_industry_mv_alpha_data[hist_industry_mv_alpha_data.HistPledgeTag == 1]['mv_industry_neural_alpha'].mean()
        share_plus_alpha = hist_industry_mv_alpha_data[hist_industry_mv_alpha_data.HistShareChangePlusTag == 1]['mv_industry_neural_alpha'].mean()
        rph_alpha = hist_industry_mv_alpha_data[hist_industry_mv_alpha_data.HistRepurchasePlanTag == 1]['mv_industry_neural_alpha'].mean()
        song_alpha = hist_industry_mv_alpha_data[hist_industry_mv_alpha_data.HistShareBonusTag == 1]['mv_industry_neural_alpha'].mean()
        unlimit_alpha = hist_industry_mv_alpha_data[hist_industry_mv_alpha_data.HistUnLimitTag == 1]['mv_industry_neural_alpha'].mean()
        sh_re_unlimit_alpha = hist_industry_mv_alpha_data[hist_industry_mv_alpha_data.HistShareRewardUnlimitTag == 1]['mv_industry_neural_alpha'].mean()
        no_sh_re_unlimit_alpha = hist_industry_mv_alpha_data[hist_industry_mv_alpha_data.HistNoShareRewardUnlimitTag == 1]['mv_industry_neural_alpha'].mean()
        sh_re_unlimit_long_alpha = hist_industry_mv_alpha_data[hist_industry_mv_alpha_data.HistLongShareRewardUnlimitTag == 1]['mv_industry_neural_alpha'].mean()
        frozen_alpha = hist_industry_mv_alpha_data[hist_industry_mv_alpha_data.HistFrozenTag == 1]['mv_industry_neural_alpha'].mean()
        unfrozen_alpha = hist_industry_mv_alpha_data[hist_industry_mv_alpha_data.HistUnfrozenTag == 1]['mv_industry_neural_alpha'].mean()
        alpha_s = pd.Series({"poor_pred": finance_poor_pred_alpha, 'good_pred': finance_good_pred_alpha, "pledge": pledge_alpha,
                            "share_chg_plus": share_plus_alpha, 'rph': rph_alpha, 'sh_re_unlimit': sh_re_unlimit_alpha,
                            "no_sh_re_unlimit": no_sh_re_unlimit_alpha, 'sh_re_unlimit_long': sh_re_unlimit_long_alpha,
                            "frozen": unfrozen_alpha, 'unfrozen': unfrozen_alpha})
        #### 对超额进行排序 ####
        alpha_rank = alpha_s.rank().to_dict()
        this_industry_mv_alpha_df = all_hist_infos[j]
        #### 生成各个事件因子的超额预测值 #####
        this_industry_mv_alpha_df.fillna(0, inplace=True)
        this_industry_mv_alpha_df['poor_pred_score'] = this_industry_mv_alpha_df['HistFinancePoorPredTag']*finance_poor_pred_alpha
        this_industry_mv_alpha_df['good_pred_score'] = this_industry_mv_alpha_df['HistFinanceGoodPredTag']*finance_good_pred_alpha
        this_industry_mv_alpha_df['pledge_score'] = this_industry_mv_alpha_df['HistPledgeTag']*pledge_alpha
        this_industry_mv_alpha_df['share_chg_plus_score'] = this_industry_mv_alpha_df['HistShareChangePlusTag']*share_plus_alpha
        this_industry_mv_alpha_df['rph_score'] = this_industry_mv_alpha_df['HistRepurchasePlanTag']*rph_alpha
        this_industry_mv_alpha_df['sh_re_unlimit_score'] = this_industry_mv_alpha_df['HistShareRewardUnlimitTag']*sh_re_unlimit_alpha
        this_industry_mv_alpha_df['no_sh_re_unlimit_score'] = this_industry_mv_alpha_df['HistNoShareRewardUnlimitTag']*no_sh_re_unlimit_alpha
        this_industry_mv_alpha_df['sh_re_unlimit_long_score'] = this_industry_mv_alpha_df['HistLongShareRewardUnlimitTag']*sh_re_unlimit_long_alpha
        this_industry_mv_alpha_df['frozen_score'] = this_industry_mv_alpha_df['HistFrozenTag']*frozen_alpha
        this_industry_mv_alpha_df['unfrozen_score'] = this_industry_mv_alpha_df['HistUnfrozenTag']*unfrozen_alpha
        #### 生成各个事件因子的排序 #####

        this_industry_mv_alpha_df['poor_pred_rank'] = this_industry_mv_alpha_df['HistFinancePoorPredTag']*alpha_rank['poor_pred']
        this_industry_mv_alpha_df['good_pred_rank'] = this_industry_mv_alpha_df['HistFinanceGoodPredTag']*alpha_rank['good_pred']
        this_industry_mv_alpha_df['pledge_rank'] = this_industry_mv_alpha_df['HistPledgeTag']*alpha_rank['pledge']
        this_industry_mv_alpha_df['share_chg_plus_rank'] = this_industry_mv_alpha_df['HistShareChangePlusTag']*alpha_rank['share_chg_plus']
        this_industry_mv_alpha_df['rph_rank'] = this_industry_mv_alpha_df['HistRepurchasePlanTag']*alpha_rank['rph']
        this_industry_mv_alpha_df['sh_re_unlimit_rank'] = this_industry_mv_alpha_df['HistShareRewardUnlimitTag']*alpha_rank['sh_re_unlimit']
        this_industry_mv_alpha_df['no_sh_re_unlimit_rank'] = this_industry_mv_alpha_df['HistNoShareRewardUnlimitTag']*alpha_rank['no_sh_re_unlimit']
        this_industry_mv_alpha_df['sh_re_unlimit_long_rank'] = this_industry_mv_alpha_df['HistLongShareRewardUnlimitTag']*alpha_rank['sh_re_unlimit_long']
        this_industry_mv_alpha_df['frozen_rank'] = this_industry_mv_alpha_df['HistFrozenTag']*alpha_rank['frozen']
        this_industry_mv_alpha_df['unfrozen_rank'] = this_industry_mv_alpha_df['HistUnfrozenTag']*alpha_rank['unfrozen']
        ####获取 最高的事件因子排序 ####
        this_industry_mv_alpha_df['event_max_rank'] = this_industry_mv_alpha_df[['poor_pred_rank', 'good_pred_rank', 'pledge_rank', 'share_chg_plus_rank', 'rph_rank', 'sh_re_unlimit_rank', 'no_sh_re_unlimit_rank', 'sh_re_unlimit_long_rank', 'frozen_rank', 'unfrozen_rank']].max(axis=1)
         ####获取 事件因子的整体打分 ####
        this_industry_mv_alpha_df['event_score'] = this_industry_mv_alpha_df[['poor_pred_score', 'good_pred_score', 'pledge_score', 'share_chg_plus_score', 'rph_score', 'sh_re_unlimit_score', 'no_sh_re_unlimit_score', 'sh_re_unlimit_long_score', 'frozen_score', 'unfrozen_score']].sum(axis=1)
        #####获得事件因子数量####
        this_industry_mv_alpha_df['event_count'] = -this_industry_mv_alpha_df['HistFinancePoorPredTag'] + this_industry_mv_alpha_df['HistFinanceGoodPredTag'] + this_industry_mv_alpha_df['HistPledgeTag']*1 + this_industry_mv_alpha_df['HistShareChangePlusTag']*1 + this_industry_mv_alpha_df['HistRepurchasePlanTag']*1 + this_industry_mv_alpha_df[['HistShareRewardUnlimitTag', 'HistLongShareRewardUnlimitTag']].sum(axis=1).map(lambda x: 1 if x > 0 else 0)*1+ this_industry_mv_alpha_df['HistNoShareRewardUnlimitTag']*1 - this_industry_mv_alpha_df['HistFrozenTag'] - this_industry_mv_alpha_df['HistUnfrozenTag']
        pred_industry_mv_alpha_infos.append(this_industry_mv_alpha_df)
    pred_industry_mv_alpha_df = pd.concat(pred_industry_mv_alpha_infos)
    return pred_industry_mv_alpha_df


class ScoreGenerator(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.save_info = param_info['save_info']
        self.operators = [
            {
                'func': get_hist_data_4_factor_compute,
                'param': {
                    "read_engine": "mysql+pymysql://develop:haikuan_2025@localhost:3306/event_info",
                    "save_engine": "mysql+pymysql://develop:haikuan_2025@localhost:3306/event_info",
                    "start_date": 0,
                    "end_date": 0,
                    "table": "event_daily_factor",
                    "field": [],
                    "hist_year": -1,
                    "name_dict": {}},
                "input_data": {},
                "output": ["event_daily_factor"]
            },
            {
                'func': get_hist_data_4_factor_compute,
                'param': {
                    "read_engine": "mysql+pymysql://develop:haikuan_2025@localhost:3306/event_info",
                    "save_engine": "mysql+pymysql://develop:haikuan_2025@localhost:3306/event_info",
                    "start_date": 0,
                    "end_date": 0,
                    "table": "week_alpha",
                    "field": [],

                    "hist_year": -1,
                    "name_dict": {}},
                "input_data": {},
                "output": ["week_alpha"]
            },
            {
                "func": gen_score_from_hist,
                "param": {},
                "input_data":
                    {
                        "week_alpha": "week_alpha",
                        "event_daily_factor": "event_daily_factor",
                    },
                "output": ["event_score_info"]
            },
            {
                "func": save_data_to_table,
                "param": {"engine": self.save_info['engine'], "table": self.save_info['table'],
                          "if_exists": self.save_info.get("if_exists", "append")},
                "input_data": {"data": "event_score_info"},
                "output": ["event_score_info"]
            }
        ]
        self.output_vars = ["event_score_info"]


def merge_dataframe(left_df, right_df, how='inner', columns=['code', 'trade_date']):
    left_df = left_df.reset_index()
    right_df = right_df.reset_index()
    merged_df = pd.merge(left_df, right_df, how=how, on=columns)
    return merged_df

# def merge_portfolio_with_raw_factor():
#     pass

def portfolio_generator_from_event_count_score(event_score_flag_info, expect_count):
    last_codes = []
    chosen_infos = []
    top_neural_alphas = {}
    """
    根据 事件因子的数量和历史因子表现选择股票
    """
    import pdb
    pdb.set_trace()
    for trade_date, tmp_df in event_score_flag_info.groupby('trade_date'):
        #### 统计不同event_count的股票数量，并从高到低排序####
        event_count_cumsum = tmp_df['event_count'].value_counts().sort_index(ascending=False).cumsum()
        #### 获得 event_count的临界值 ####
        threshold_event_count = event_count_cumsum[event_count_cumsum > expect_count].index.max()
        ####优先选择 事件因子最多的股票 #####
        top_event_df = tmp_df[tmp_df.event_count > threshold_event_count]
        #### 临界的股票 ####
        middle_event_df = tmp_df[tmp_df.event_count == threshold_event_count]

        old_middle_event_df = middle_event_df[middle_event_df['ts_code'].map(lambda x: x in last_codes)]
        new_middle_event_df = middle_event_df[middle_event_df['ts_code'].map(lambda x: x not in last_codes)]
        #### 临界股票优先选择历史持仓####
        chosen_event_df = pd.concat([top_event_df, old_middle_event_df], axis=0)
        if len(chosen_event_df) < expect_count:
            need_stock_count = expect_count - len(chosen_event_df)
            ###选择历史上单因子表现最好的股票###
            middle_max_rank_count_cumsum = new_middle_event_df.event_max_rank.value_counts().sort_index(
                ascending=False).cumsum()

            max_rank_threshold = middle_max_rank_count_cumsum[
                middle_max_rank_count_cumsum >= need_stock_count].index.max()

            chosen_middel_event_df = new_middle_event_df[new_middle_event_df.event_max_rank >= max_rank_threshold]
            chosen_event_df = pd.concat([chosen_event_df, chosen_middel_event_df], axis=0)
        neural_alpha = chosen_event_df['mv_industry_neural_alpha'].mean()
        last_codes = list(chosen_event_df['ts_code'].values)
        import pdb
        pdb.set_trace()
        trade_date = chosen_event_df['trade_date'].values[0]
        top_neural_alphas.update({str(trade_date): neural_alpha})
        chosen_event_df["event_weight_{}".format(expect_count)] = 1/len(chosen_event_df)
        chosen_infos.append(chosen_event_df)
    portfolio_info = pd.concat(chosen_infos)
    portfolio_info = portfolio_info.rename({"event_score": "event_score_{}".format(expect_count)})
    return portfolio_info


class PortfolioGeneratorFromEventCountScore(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.save_info = param_info['save_info']
        self.invalid_infos = param_info['invalid_infos']
        self.operators = [
            {
                'func': get_hist_data_4_factor_compute,
                'param': {
                    "read_engine": "mysql+pymysql://develop:haikuan_2025@localhost:3306/event_info",
                    "save_engine": "mysql+pymysql://develop:haikuan_2025@localhost:3306/event_info",
                    "start_date": 0,
                    "end_date": 0,
                    "table": "event_score_info",
                    "field": [],
                    "hist_year": -1,
                    "name_dict": {}},
                "input_data": {},
                "output": ["event_score_info"]
            },
            {
                'func': get_hist_data_4_factor_compute,
                'param': {
                    "read_engine": "mysql+pymysql://develop:haikuan_2025@localhost:3306/factor_compute_new",
                    "save_engine": "mysql+pymysql://develop:haikuan_2025@localhost:3306/factor_compute_new",
                    "start_date": 0,
                    "end_date": 0,
                    "table": "all_data_test_all_mkt_indicator",
                    "field": ["code", "trade_date", "EndFlag", "PauseFlag", "STFlag", "ListedFlag", "NanFlag"],
                    "index": ["code", "trade_date"],
                    "hist_year": -1,
                    "name_dict": {}},
                "input_data": {},
                "output": ["flag_info"]
            },
            {
                "func": merge_dataframe,
                "param": {},
                "input_data":
                    {
                        "left_df": "event_score_info",
                        "right_df": "flag_info",
                    },
                "output": ["event_score_flag_info"]
            },
            {
                "func": transfer_data_to_valid_and_not_valid,
                "param": {
                    "invalid_infos": self.invalid_infos
                },
                "input_data": {"data": "event_score_flag_info"},
                "output": ['valid_event_score_flag_info', 'invalid_event_score_flag_info']
            },
            {
                "func": portfolio_generator_from_event_count_score,
                "param": {
                    "expect_count": 20,
                },
                "input_data": {"event_score_flag_info": "valid_event_score_flag_info"},
                "output": ['portfolio_info_20']
            },
            {
                "func": save_data_to_table,
                "param": {"engine": self.save_info['engine'], "table": "{}_20".format(self.save_info['table']),
                          "if_exists": self.save_info.get("if_exists", "append")},
                "input_data": {"data": "portfolio_info_20"},
                "output": ["portfolio_info_20"]
            },
            {
                "func": portfolio_generator_from_event_count_score,
                "param": {
                    "expect_count": 15,
                },
                "input_data": {"event_score_flag_info": "valid_event_score_flag_info"},
                "output": ['portfolio_info_15']
            },
            {
                "func": save_data_to_table,
                "param": {"engine": self.save_info['engine'], "table": "{}_15".format(self.save_info['table']),
                          "if_exists": self.save_info.get("if_exists", "append")},
                "input_data": {"data": "portfolio_info_15"},
                "output": ["portfolio_info_15"]
            },
            {
                "func": portfolio_generator_from_event_count_score,
                "param": {
                    "expect_count": 10,
                },
                "input_data": {"event_score_flag_info": "valid_event_score_flag_info"},
                "output": ['portfolio_info_10']
            },
            {
                "func": save_data_to_table,
                "param": {"engine": self.save_info['engine'], "table": "{}_10".format(self.save_info['table']),
                          "if_exists": self.save_info.get("if_exists", "append")},
                "input_data": {"data": "portfolio_info_10"},
                "output": ["portfolio_info_10"]
            },
        ]
        self.output_vars = ["portfolio_info_20", "portfolio_info_15", "portfolio_info_10"]