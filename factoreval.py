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
from model import factor_operator
from statsmodels.distributions.empirical_distribution import ECDF
import SQL_api
import math
import jqdatasdk as jd
import argparse
import datetime
jd.auth("13764432461", "Nfhq12345")


parser = argparse.ArgumentParser(description='add tgt trade_date')
parser.add_argument("--start_date", type=str, default="None")
parser.add_argument("--end_date", type=str, default="None")
parser.add_argument("--insert_way", type=str, default="append")
args = parser.parse_args()

if __name__ == "__main__":
    start_date = args.start_date
    end_date = args.end_date
    insert_way = args.insert_way

    if end_date == "None":
        end_date = datetime.datetime.today()
    else:
        end_date = datetime.datetime.strptime(str(end_date), "%Y%m%d")
    if start_date == "None":
        start_date = end_date - datetime.timedelta(days=7)
        start_date = int(start_date.strftime("%Y%m%d"))
    else:
        start_date = int(start_date)
    end_date = int(end_date.strftime("%Y%m%d"))


    param = {
        "start_date": start_date,
        "end_date": end_date,
        "source_data_infos":[
            {
                "engine": "mysql+pymysql://root:swisschina@192.168.110.5:3306/factor_research_full_a_share",
                "table": "daily_trading_data",
                "field": ['code', 'trade_date', 'open', 'close', 'low', 'high',
                          'volume', 'money', 'factor', 'high_limit','low_limit','pre_close','paused'],
                "index": ['trade_date', "code"],
                "name_dict":{}
            },
            {
                "engine": "mysql+pymysql://root:swisschina@192.168.110.5:3306/factor_research_full_a_share",
                "table": "daily_trading_data_unadjusted",
                "field": ['code', 'trade_date', 'close'],
                "index": ['trade_date', "code"],
                "name_dict": {
                    "close": "close_trading"
                }
            },
            {
                "engine": "mysql+pymysql://root:swisschina@192.168.110.5:3306/factor_research_full_a_share",
                "table": "daily_0935am_trade_price",
                "field": ['code', 'trade_date', 'trade_time', 'close','volume','money'],
                "index": ['trade_date', "code"],
                "name_dict": {
                    "trade_time": "trade_time0935",
                    "close": "close0935",
                    "volume": "volume0935",
                    "money": "money0935"
                }
            },
            {
                "engine": "mysql+pymysql://root:swisschina@192.168.110.5:3306/factor_research_full_a_share",
                "table": "daily_0950am_trade_price",
                "field": ['code', 'trade_date', 'trade_time', 'close', 'volume', 'money'],
                "index": ['trade_date', "code"],
                "name_dict": {
                    "trade_time": "trade_time0950",
                    "close": "close0950",
                    "volume": "volume0950",
                    "money": "money0950"
                }
            },
            {
                "engine": "mysql+pymysql://root:swisschina@192.168.110.5:3306/factor_research_full_a_share",
                "table": "daily_10am_trade_price",
                "field": ['code', 'trade_date', 'trade_time', 'close', 'volume', 'money'],
                "index": ['trade_date', "code"],
                "name_dict": {
                    "trade_time": "trade_time1000",
                    "close": "close1000",
                    "volume": "volume1000",
                    "money": "money1000"
                }
            },
            {
                "engine": "mysql+pymysql://root:swisschina@192.168.110.5:3306/factor_research_full_a_share",
                "table": "daily_245pm_trade_price",
                "field": ['code', 'trade_date', 'trade_time', 'close', 'volume', 'money'],
                "index": ['trade_date', "code"],
                "name_dict": {
                    "trade_time": "trade_time1445",
                    "close": "close1445",
                    "volume": "volume1445",
                    "money": "money1445"
                }
            },
            {
                "engine": "mysql+pymysql://root:swisschina@192.168.110.5:3306/factor_research_full_a_share",
                "table": "all_data_test_all_mkt_indicator",
                "field": ['code', 'trade_date', 'CSI500MonthlyMvWeight', 'PauseFlag', 'ListedFlag', 'NanFlag',
                          'GicsIndustryCode'],
                "index": ['trade_date', "code"]
            },
                ],
        "invalid_infos":[
            {"feature_name": "PauseFlag", "type": "not_equal", "feature_value": 0},
            {"feature_name": "STFlagV2", "type": "not_equal", "feature_value": 0},
            {"feature_name": "ListedFlag", "type": "not_equal", "feature_value": 0},
            {"feature_name": "NanFlag", "type": "not_equal", "feature_value": 0},
        ]
    }
    result = factor_operator.CalculateICIR(param, {}, {"ICIR":"ICIR"}).compute()
    print("check")




