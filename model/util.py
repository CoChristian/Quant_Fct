import re
import numpy as np
import SqlApi
import pandas as pd

def create_sql_api(read_engine, save_engine):
    sql_api_clf = SqlApi.SQL_API(save_engine=save_engine,
                                  read_engine=read_engine)
    return sql_api_clf


def get_hist_data_4_factor_compute(read_engine, save_engine, table, field=['trade_date', 'code'], name_dict={},
                                   index=['trade_date', 'code'],  hist_year=0, start_date=None, end_date=None,
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
                                'param': [start_date-hist_year*10000, end_date]}]
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


def get_data_from_multi_source(data_source_infos, start_date, end_date):
    multi_datas = []
    # print(data_source_infos)
    for info in data_source_infos:
        engine = info['engine']
        table = info["table"]
        field = info["field"]
        index = info.get("index", ['trade_date', 'code'])
        hist_year = info.get("hist_year", 0)
        name_dict = info.get("name_dict", {})
        other_filter=info.get("other_filter")
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

    try:
        multi_data = pd.concat(multi_datas, axis=1)
    except Exception as e:
        import pdb
        pdb.set_trace()
        pass
    return multi_data


def ts_code_to_jq_code(ts_code):
    if ts_code == "cash":
        return 'cash'
    mkt_type_list = re.findall(r"(SZ|SH|sz|sh)", ts_code)
    if len(mkt_type_list):
        mkt_type = mkt_type_list[0].upper()
    else:
        return None
    code_num_list = re.findall("\d{6}", ts_code)
    if len(code_num_list):
        code_num = code_num_list[0]
    else:
        return None
    ts_type_to_jq_type = {"SZ": "XSHE", "SH": "XSHG"}
    return "{}.{}".format(code_num, ts_type_to_jq_type[mkt_type])

def jq_code_to_ts_code(ts_code):
    mkt_type_list = re.findall(r"(XSHE|XSHG)", ts_code)
    if len(mkt_type_list):
        mkt_type = mkt_type_list[0].upper()
    else:
        return None
    code_num_list = re.findall("\d{6}", ts_code)
    if len(code_num_list):
        code_num = code_num_list[0]
    else:
        return None
    ts_type_to_jq_type = {"XSHE": "SZ", "XSHG": "SH"}
    return "{}.{}".format(code_num, ts_type_to_jq_type[mkt_type])



def add_pre_weight_hold_share(pre_order_book, data):
    data = pd.merge(data, pre_order_book[['code', 'round_share']], how='left', on=['code'])
    data.rename({"round_share": "pre_hold_share"}, axis=1, inplace=True)
    data['pre_hold_share'] = data['pre_hold_share'].fillna(0)
    data['adj_pre_hold_share'] = data['pre_hold_share'] * data['adj_factor_compare_to_last_trade_date']
    data['adj_pre_cash_alloc'] = data['adj_pre_hold_share'] * data['trading_price']
    data['adj_pre_weight'] = data['adj_pre_cash_alloc'] / data['adj_pre_cash_alloc'].sum()
    data['adj_pre_weight'] = data['adj_pre_weight'].fillna(0)
    return data


def gj2hr(order_book):
    '''
    将符合guojun篮子导入格式的转换成符合huarong篮子导入格式
    （后续可在real_trading_sql里面更改）

    Returns
    -------
    输出符合华融买卖单的格式文件
    '''
    order_book = order_book[order_book.quantity != 0]
    order_book.index = [int(c[3:]) for c in order_book.code]
    use_order_book = pd.DataFrame(np.nan, index=order_book.index,
                                  columns=['代码', '市场', '数量', '相对权重', '方向'])
    use_order_book['代码'] = use_order_book.index
    use_order_book['数量'] = order_book.quantity

    return use_order_book.reset_index(drop=True)

