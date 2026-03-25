import pandas as pd
import numpy as np
import preprocess
import group_calc
import factor_analysis
from model.util import get_data_from_multi_source, ts_code_to_jq_code


def get_factor_table(start_date, cur_trade_date):
    factor_table_info = {
        "engine": "mysql+pymysql://root:swisschina@192.168.110.5:3306/factor_pv",
        "table": 'ab_nr_pvfct',
        "field": [],
        "index": ['trade_date', "code"],
        "name_dict": {'AB_NR':'fac_value'}
    }
    factor_table = get_data_from_multi_source([factor_table_info], start_date, cur_trade_date)
    factor_table.reset_index(inplace=True)
    # last_date = factor_table['trade_date'].values.max()
    # last_date_valuation_table = factor_table[factor_table['trade_date'] == last_date]
    return factor_table

def get_price_table(start_date, end_date):
    price_table_info = {
        "engine": "mysql+pymysql://root:swisschina@192.168.110.5:3306/factor_research_full_a_share",
        "table": 'daily_0935am_trade_price',
        "field": ['trade_date','code','close'],
        "index": ['trade_date', "code"],
        "name_dict": {'close':'close0935'}
    }
    price_table = get_data_from_multi_source([price_table_info], start_date, end_date)
    price_table.reset_index(inplace=True)
    # last_date = factor_table['trade_date'].values.max()
    # last_date_valuation_table = factor_table[factor_table['trade_date'] == last_date]
    return price_table

def get_opt2trade_table(start_date, end_date):
    date_tale_info = {
        "engine": "mysql+pymysql://root:swisschina@192.168.110.5:3306/factor_compute_new",
        "table": 'opt2trade',
        "field": [],
        "index": ['opt_date'],
        "name_dict": {'trade_date': 'trade_date_price'}
    }
    date_table = get_data_from_multi_source([date_tale_info], start_date, end_date)
    date_table.reset_index(inplace=True)
    date_table['trade_date_price'] = date_table['trade_date_price'].astype('int')
    return date_table



