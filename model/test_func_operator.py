#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

data process pipeline no class, not recursive
@author: lianrui
"""

from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
import math
from func_operator import sd_win_sort
from sklearn.linear_model import Ridge
from statsmodels.distributions.empirical_distribution import ECDF
from factor_score_card import _OLS_estimate_fac_premium

def cal_n_day_before(trade_date, days_prior):
    target_date = datetime.strptime(str(trade_date), "%Y%m%d")
    result_date = target_date - timedelta(days=days_prior)
    return int(result_date.strftime("%Y%m%d"))

def cal_two_day_diff(first_day, second_day):
    first_day = datetime.strptime(str(first_day), "%Y%m%d")
    second_day = datetime.strptime(str(second_day), "%Y%m%d")
    diff_days = (second_day - first_day).days
    return diff_days
    

def get_valid_value(x, features):
    for feature in features:
        if not pd.isnull(x[feature]):
            return x[feature]
    return None

def transfer_date_str_2_int(date):
    date = str(date)
    try:
        date = int(date)
        return date
    except Exception as e:
        return None

def cal_data_hist_trend(data, feature, hist_window, min_hist_window):
    data_ = data.copy().reset_index()
    def get_trend(x, min_hist_window):
        x_s = pd.Series(x)
        x_s = x_s[x_s.notnull()]
        x_len = len(x_s)
        if len(x_s) >= min_hist_window:
            corr_coef =  np.corrcoef(x_s.values, np.arange(x_len))[0,1]
            if corr_coef > 0:
                return corr_coef
            elif corr_coef < 0:
                return corr_coef
            else:
                return 0
        else:
            return 0

    data_ = data_.sort_values(['code', 'trade_date'])
    trend_data = data_.groupby('code')[feature].rolling(hist_window).apply(lambda x: get_trend(x, min_hist_window))
    data_[feature+'Trend'] = trend_data.droplevel('code')
    return data_.set_index(['trade_date', 'code'])[feature+'Trend']

def cal_org_inner_forecast_trend(data, features, opt2trade):
    data = data.reset_index()
    opt2trade = opt2trade.reset_index()
    opt_dates = sorted(opt2trade['opt_date'].values)
    data = data[data['综合值计算标记'] == 1]
#     data = data[data['内部_公告日期'].notnull()]
#     data['trade_date'] = data['内部_公告日期'].map(lambda x: transfer_date_str_2_int(x))
    data = data[data['trade_date'].notnull()]
    # data = data.sort_values(['股票代码', '研究机构名称', 'trade_date'])
    data['net_profit1'] = data['预测净利润_万元']
    data['net_profit2'] = data['预测基准股本_万股'] * data['预测每股收益_基本']
    data['net_profit3'] = data['预测基准股本_万股'] * data['预测每股收益_换算']
    data['net_profit4'] = data['预测基准股本_万股'] * data['预测每股收益_摊薄']
    data['net_profit5'] = data['预测基准股本_万股'] * data['预测每股收益_稀释']
    data['NetProfit'] = data.apply(lambda x : get_valid_value(x, ['net_profit1', 'net_profit2', 'net_profit3', 'net_profit4', 'net_profit5']), axis=1)*10000
    data['NetAsset'] = data['预测基准股本_万股']*data['每股净资产']*10000
    data['MainRevenue'] = data['预测主营业务收入_万元']*10000
    org_inner_trend_infos = []
    def get_trend(x):
        x_s = pd.Series(x)
        x_s = x_s[x_s.notnull()]
        x_len = len(x_s)
        if len(x_s) >= 2:
            corr_coef =  np.corrcoef(x_s.values, np.arange(x_len))[0,1]
            if corr_coef > 0:
                return 1
            elif corr_coef < 0:
                return -1
            else:
                return 0
        else:
            return 0
    for opt_date in opt_dates:
#         print("opt date {}".format(opt_date))
        last_date = cal_n_day_before(opt_date, 180)
        hist_data =  data[data.trade_date.map(lambda x: x >= last_date and x <= opt_date)]
        if len(hist_data):
            hist_data = hist_data.sort_values(['code', '报告期', '研究机构名称', 'trade_date'])
            hist_data = hist_data.drop_duplicates(['code', '报告期', '研究机构名称', 'trade_date'], keep='last')
            org_code_hist_count = hist_data.groupby(['code', '报告期', '研究机构名称'])['trade_date'].count()
            feature_org_inner_trend_list = []
            for feature in features:
                org_inner_trend = hist_data.groupby(['code', '报告期', '研究机构名称'])[feature].apply(lambda x: get_trend(x))
                org_inner_trend.name = 'corr'
                org_inner_trend_s = org_inner_trend.fillna(0).reset_index().groupby('code')['corr'].mean()
                org_inner_trend_s.name = feature+"OrgInnerTrend"
                feature_org_inner_trend_list.append(org_inner_trend_s)
            feature_org_inner_trend_df = pd.concat(feature_org_inner_trend_list, axis=1)
            feature_org_inner_trend_df['trade_date'] = opt_date
            org_inner_trend_infos.append(feature_org_inner_trend_df.reset_index())
    all_org_inner_trend_df = pd.concat(org_inner_trend_infos)
#     all_org_inner_trend_df = all_org_inner_trend_df.rename({'股票代码': "code"}, axis=1)
    return all_org_inner_trend_df.set_index(['code', 'trade_date'])
            
            
            
            
    
    
    
def cal_consistent_earning_forecast(data, features, opt2trade):
    data = data.reset_index()
    opt2trade = opt2trade.reset_index()
    opt_dates = sorted(opt2trade['opt_date'].values)
    data = data[data['综合值计算标记'] == 1]
#     data = data[data['内部_公告日期'].notnull()]
#     data['trade_date'] = data['内部_公告日期'].map(lambda x: transfer_date_str_2_int(x))
    data = data[data['trade_date'].notnull()]
    # data = data.sort_values(['股票代码', '研究机构名称', 'trade_date'])
    data['net_profit1'] = data['预测净利润_万元']
    data['net_profit2'] = data['预测基准股本_万股'] * data['预测每股收益_基本']
    data['net_profit3'] = data['预测基准股本_万股'] * data['预测每股收益_换算']
    data['net_profit4'] = data['预测基准股本_万股'] * data['预测每股收益_摊薄']
    data['net_profit5'] = data['预测基准股本_万股'] * data['预测每股收益_稀释']
    data['NetProfit'] = data.apply(lambda x : get_valid_value(x, ['net_profit1', 'net_profit2', 'net_profit3', 'net_profit4', 'net_profit5']), axis=1)*10000
    data['NetAsset'] = data['预测基准股本_万股']*data['每股净资产']*10000
    data['MainRevenue'] = data['预测主营业务收入_万元']*10000

    consistent_earning_infos = []
    for opt_date in opt_dates:
        print("opt date {}".format(opt_date))
        last_date = cal_n_day_before(opt_date, 180)
        hist_data =  data[data.trade_date.map(lambda x: x >= last_date and x <= opt_date)]

        hist_data = hist_data.sort_values(['code', '报告期', '研究机构名称', 'trade_date'])
        hist_data = hist_data.drop_duplicates(['code', '报告期', '研究机构名称'], keep='last')
        feature_mean_value = hist_data.groupby(['code', '报告期'])[features].mean()
        feature_mean_value.columns = ["{}ConsistentMean".format(_) for _ in features]
        feature_median_value = hist_data.groupby(['code', '报告期'])[features].median()
        feature_median_value.columns = ["{}ConsistentMedian".format(_) for _ in features]
        feature_value = pd.concat([feature_mean_value, feature_median_value], axis=1)
        feature_value = feature_value.reset_index()
        feature_value['trade_date'] = opt_date
        consistent_earning_infos.append(feature_value)
    consistent_earning_data = pd.concat(consistent_earning_infos)
    consistent_earning_data = consistent_earning_data.rename({'报告期': 'end_date'}, axis=1)
    return consistent_earning_data


def roll_consistent_forecast(features, financial_date_data, consistent_data):
    financial_date_data = financial_date_data.reset_index()
    financial_date_data['end_date'] = financial_date_data['end_date'].map(lambda x: str(x)[:10].replace('-', ''))
    feature_consistent_forecast_roll_infos = []
    for (code, trade_date), code_consistent_data in consistent_data.groupby(['code', 'trade_date']):
#         code_consistent_data = consistent_data[(consistent_data.code == code) & (consistent_data.trade_date == trade_date)]
        consistent_forecast_info = {feature: {} for feature in features}
        for info in code_consistent_data.to_dict('records'):
            end_date = info['end_date']
            if str(end_date)[-4:] == '1231':
                year = end_date[:-4]
                for feature in features:
                    feature_value = info[feature]
                    feature_quarter_info = {str(year)+'Q1': feature_value/4, str(year)+'Q2': feature_value/4, str(year)+'Q3': feature_value/4, str(year)+'Q4': feature_value/4}
                    consistent_forecast_info[feature].update(feature_quarter_info)
        if len(consistent_forecast_info) == 0:
            feature_roll_info = {"Roll"+_: None for _ in features}
            feature_roll_info.update({'code': code, 'trade_date': trade_date})
            feature_consistent_forecast_roll_infos.append(feature_roll_info)

        else:
            code_hist_financial_date_data = financial_date_data[(financial_date_data['trade_date']<=trade_date) & (financial_date_data.code == code)]
            if len(code_hist_financial_date_data) == 0:
                code_hist_financial_date_data = financial_date_data[(financial_date_data['trade_date'] <= trade_date)]
            last_end_date = str(code_hist_financial_date_data['end_date'].max())
#             print("code {}, trade_date {}, last_end_date {}".format(code, trade_date, last_end_date))
            end_date_to_quarter = {'0331': 'Q1', '0630': "Q2", '0930': 'Q3', '1231': 'Q4'}
            last_quarter = last_end_date[:4] + end_date_to_quarter[last_end_date[4:]]
            feature_roll_info = {'code': code, 'trade_date': trade_date}
            for feature in features:
                feature_consistent_forecast_s = pd.Series(consistent_forecast_info[feature]).sort_index()
                future_feature_consistent_forecast_s = feature_consistent_forecast_s[feature_consistent_forecast_s.index > last_quarter]
                future_feature_consistent_forecast_s = future_feature_consistent_forecast_s[future_feature_consistent_forecast_s.notnull()]
                
                if len(future_feature_consistent_forecast_s) >= 4:
                    feature_roll_info.update({"Roll" + feature : future_feature_consistent_forecast_s.head(4).sum()})
#                     print("code, {}, trade_date, {}, last_end_date {}".format(code, trade_date,  last_quarter))
#                     print(future_feature_consistent_forecast_s.head(4))
                else:
                    feature_roll_info.update({"Roll" + feature: None})

        feature_consistent_forecast_roll_infos.append(feature_roll_info)
    feature_consistent_forecast_roll_df = pd.DataFrame(feature_consistent_forecast_roll_infos) 

    return feature_consistent_forecast_roll_df.set_index(['code', 'trade_date'])




def cal_research_report_emotion_score(research_report_time_info, research_report_score_info, opt_2_trade, start_date, end_date):

    opt_2_trade = opt_2_trade.reset_index()
    opt_2_trade = opt_2_trade[opt_2_trade['opt_date'].map(lambda x: x > start_date-10000 and x <= end_date)]
    research_report_time_info = research_report_time_info.reset_index()
    research_report_score_info = research_report_score_info.reset_index()
    def std_time_info(research_report_time_info):
        research_report_time_info['trade_date'] = research_report_time_info['entry_date'].map(lambda x: int(str(x)[:8]))
        research_report_time_info['code'] = research_report_time_info['stock_code'].map(lambda x: "%06d" % int(x))
        research_report_time_info['code'] = research_report_time_info['code'].map(lambda x: x+".XSHG" if x[0] == '6' else x+'.XSHE')
        research_report_time_info = research_report_time_info.drop_duplicates("_id")
        return research_report_time_info[['code', 'trade_date', '_id']]
    research_report_time_info = std_time_info(research_report_time_info)
    research_report_info = pd.merge(research_report_time_info, research_report_score_info, how='left', on=['_id'])
    research_report_info = research_report_info[research_report_info['score'].notnull()]
    opt_dates = sorted(opt_2_trade['opt_date'].values)

    pub_date_2_opt_date = {}
    pub_dates = research_report_time_info['trade_date'].unique()
    for first_date, second_date in zip(opt_dates[0: -1], opt_dates[1:]):
        tgt_pub_dates = [pub_date for pub_date in pub_dates if pub_date >first_date and pub_date <= second_date]
        for _ in tgt_pub_dates:
            pub_date_2_opt_date.update({_: second_date})
   
    research_report_info['trade_date'] = research_report_info['trade_date'].map(lambda x: pub_date_2_opt_date.get(x))
    research_report_info = research_report_info[research_report_info['trade_date'].notnull()]

    valid_research_report_info = research_report_info[research_report_info.score > 0]

    research_rpt_weekly_score_info = valid_research_report_info.groupby(['trade_date', 'code'])['score'].mean().reset_index()
    new_research_rpt_score_infos = []
    def cal_avg_score(x):
        avg_score = x.sum()/(x!=0).sum()
        return avg_score
    def cal_score_trend(x):
        x = x[x!=0]
        if len(x) > 1:
     
            score_trend = x.values[-1] - x.values[0]
        else:
            score_trend = 0

            
        return score_trend
    weight = pd.Series([1 for j in range(12)] + [1.5 for j in range(12)]).values
#     def cal_weight_score(x):
#         avg_score = (x*weight).sum()/((x!=0)*weight).sum()
#         return avg_score

    for code, code_score_info in research_rpt_weekly_score_info.groupby('code'):
        code_score_info = code_score_info.set_index('trade_date')
        code_score_info = code_score_info.reindex(opt_dates)
        code_score_info = code_score_info.sort_index()
        code_score_info['code'] = code
        code_score_info['TitleEmotionScoreMean'] = code_score_info['score'].fillna(0).rolling(24, min_periods=4).apply(lambda x: cal_avg_score(x))
        code_score_info['TitleEmotionScoreTrend'] = code_score_info['score'].fillna(0).rolling(24, min_periods=4).apply(lambda x: cal_score_trend(x))
#         code_score_info['hist_weight_score'] = code_score_info['score'].fillna(0).rolling(24).apply(lambda x: cal_weight_score(x))

        new_research_rpt_score_infos.append(code_score_info.reset_index())

    all_new_research_rpt_score_inf = pd.concat(new_research_rpt_score_infos)

    # all_new_research_rpt_score_inf[all_new_research_rpt_score_inf.code == "601020.XSHG"]
    return all_new_research_rpt_score_inf.set_index(['code', 'trade_date'])


def cal_report_jor_score(report_data, code_price_data, index_price_data, rolling_window, opt_2_trade, output_name):
    report_data = report_data.reset_index()
    code_price_data = code_price_data.reset_index()
    index_price_data = index_price_data.reset_index()

    all_trade_dates = sorted(code_price_data['trade_date'].unique())
    all_pub_dates = sorted(report_data['pub_date'].unique())
    all_checkin_dates = sorted(report_data['checkin_date'].unique())
    pub_date_2_next_trade_date = {}
    for pub_date in all_pub_dates:
        dates_after_pub = [date for date in all_trade_dates if date > pub_date]
        if len(dates_after_pub):
            next_trade_date = dates_after_pub[0]
            pub_date_2_next_trade_date.update({pub_date: next_trade_date})
     
    checkin_date_2_trade_date = {}
    for checkin_date in all_checkin_dates:
        dates_after_checkin = [date for date in all_trade_dates if date >= checkin_date]
        if len(dates_after_checkin):
            trade_date = dates_after_checkin[0]
            checkin_date_2_trade_date.update({checkin_date: trade_date})
               
    report_data['jor_date'] = report_data['pub_date'].map(pub_date_2_next_trade_date)
    
    report_data['checkin_date'] = report_data['checkin_date'].map(checkin_date_2_trade_date)
    report_data = report_data.sort_values(['code', 'jor_date', 'checkin_date'])
    report_data = report_data.drop_duplicates(['code', 'jor_date'], keep='first')
    code_price_data['code_jor'] = code_price_data['low']/code_price_data['pre_close'] - 1
    index_price_data = index_price_data.sort_values('trade_date')
    index_price_data['index_jor'] = index_price_data['low']/index_price_data['close'].shift(1) - 1
    date_2_index_jor = dict(zip(index_price_data['trade_date'], index_price_data['index_jor']))
    code_price_data['index_jor'] = code_price_data['trade_date'].map(date_2_index_jor)
    code_price_data['jor'] = code_price_data['code_jor'] - code_price_data['index_jor']
    report_jor_data = pd.merge(report_data[['code', 'jor_date','checkin_date']], code_price_data[['code', 'trade_date', 'jor']], how='inner', left_on=['code', 'jor_date'], right_on=['code', 'trade_date'])

    report_jor_data['trade_date'] = report_jor_data[['jor_date', 'checkin_date']].max(axis=1)
    report_jor_data = report_jor_data.sort_values(['trade_date', 'jor_date'])
    report_jor_data = report_jor_data.drop_duplicates(['code', 'trade_date'], keep='last')
    report_jor_data = report_jor_data[report_jor_data.trade_date.notnull()]
    code_price_jor_data = pd.merge(code_price_data[['code', 'trade_date']], report_jor_data[['code', 'trade_date', 'jor']], how='left', on=['code', 'trade_date'])

    code_price_jor_data = code_price_jor_data.sort_values(['code', 'trade_date'])

    code_price_jor_data[output_name+"Last"] = code_price_jor_data.groupby('code')['jor'].rolling(rolling_window, min_periods=1).apply(lambda x: x.fillna(method='pad').values[-1]).droplevel('code')

    code_price_jor_data[output_name+"Mean"] = code_price_jor_data.groupby('code')['jor'].rolling(rolling_window, min_periods=1).mean().droplevel('code')
#     code_price_jor_data[code_price_jor_data.code == "000001.XSHE"].to_excel("jor_000001_XSHE.xlsx")

    opt_to_trade_dict = dict(zip(opt_2_trade['opt_date'].values, opt_2_trade['trade_date'].values))
    weekly_code_price_jor_data =code_price_jor_data[code_price_jor_data['trade_date'].map(lambda x: x in opt_to_trade_dict)]
    weekly_code_price_jor_data[output_name+"Last"] = weekly_code_price_jor_data[output_name+"Last"].fillna(-999)
    weekly_code_price_jor_data[output_name+"Mean"] = weekly_code_price_jor_data[output_name+"Mean"].fillna(-999)

#     return weekly_code_price_jor_data[['code', 'trade_date', output_name]]
    return weekly_code_price_jor_data.set_index(['code', 'trade_date'])[[output_name+"Last", output_name+'Mean']]
    
def std_rpt_data(report_data):
    report_data['checkin_date'] = report_data['entry_date'].map(lambda x: int(str(x)[:8]))
    report_data['pub_date'] = report_data['create_date'].map(lambda x: int(str(x)[:8]))
    report_data['code'] = report_data['stock_code'].map(lambda x: "%06d" % int(x))
    report_data['code'] = report_data['code'].map(lambda x: x+".XSHG" if x[0] == '6' else x+'.XSHE')
    report_data = report_data.reset_index()
    report_data = report_data.drop_duplicates("_id")

    return report_data[['code', 'pub_date', "checkin_date"]]


def transfer_finanical_data_for_jor(report_data):
    report_data = report_data.reset_index()
    report_data['checkin_date'] = report_data['trade_date']
    report_data['pub_date'] = report_data['trade_date']
    return report_data[['code', 'pub_date', "checkin_date"]]




def time_series_std_data(data, features, window_size):
    data = data.reset_index()
    data = data.sort_values(['code', 'trade_date'])
    import numpy as np
    data[["{}TimeSeriesStd".format(feature) for feature in features]] = data.groupby('code')[features].apply(lambda x: (x-x.rolling(window_size).mean())/x.rolling(window_size).std()).replace(np.inf, 2.7).replace(-np.inf, -2.7)

    return data.set_index(['code', 'trade_date'])


def cal_consistent_target_return_indicator(tgt_price_data, nfq_price, opt_2_trade):
    nfq_price = nfq_price.reset_index()
    opt_2_trade = opt_2_trade.reset_index()
    opt_dates = opt_2_trade['opt_date'].values
    tgt_price_data = tgt_price_data.reset_index()
    def std_time_info(research_report_info):
        research_report_info['entry_date'] = research_report_info['entry_date'].map(lambda x: int(str(x)[:8]))
        research_report_info['code'] = research_report_info['stock_code'].map(lambda x: "%06d" % int(x))
        research_report_info['code'] = research_report_info['code'].map(lambda x: x+".XSHG" if x[0] == '6' else x+'.XSHE')
        research_report_info['author_name_std'] = research_report_info['author_name'].map(lambda x: "_".join(sorted(x.split(','))))
        research_report_info = research_report_info.drop_duplicates("report_id")
        return research_report_info
    tgt_price_data = std_time_info(tgt_price_data)
    nfq_price = nfq_price.rename({'close': 'nfq_close'}, axis=1)
        
    all_trade_dates = sorted(nfq_price['trade_date'].unique())
    tgt_price_data = tgt_price_data[tgt_price_data['create_date'].map(lambda x: x > min(all_trade_dates))]
    all_create_dates = sorted(tgt_price_data['create_date'].unique())
    create_date_2_price_date = {}
    for create_date in all_create_dates:
        hist_dates = [_ for _ in all_trade_dates if _ < create_date]
        if len(hist_dates):
            create_date_2_price_date.update({create_date: hist_dates[-1]})
    tgt_price_data['price_date'] = tgt_price_data['create_date'].map(create_date_2_price_date)
    tgt_price_data = pd.merge(tgt_price_data, nfq_price[['code', 'trade_date', 'nfq_close']], how='left', left_on=['code', 'price_date'], right_on=['code', 'trade_date'])
    tgt_price_data = tgt_price_data.rename({ 'nfq_close': 'forecast_day_price_nfq'}, axis=1)
    tgt_price_data['target_price'] = tgt_price_data[['target_price_ceiling', 'target_price_floor']].mean(axis=1)
    tgt_price_data = tgt_price_data[tgt_price_data['target_price'].notnull()]
    tgt_price_data = tgt_price_data[tgt_price_data['forecast_day_price_nfq'].notnull()]
    tgt_price_data['target_return'] = tgt_price_data['target_price']/tgt_price_data['forecast_day_price_nfq']    

    target_return_indicator_infos = []
    for opt_date in opt_dates:
        last_date = cal_n_day_before(opt_date, 90)
        hist_tgt_price_data = tgt_price_data[tgt_price_data.entry_date.map(lambda x: x >=last_date and x <=opt_date)].copy()
        hist_tgt_price_data['hist_day_count'] = hist_tgt_price_data['create_date'].map(lambda x: cal_two_day_diff(x, opt_date))
        hist_tgt_price_data['weight'] = hist_tgt_price_data['hist_day_count'].map(lambda x: 1.5-x/90).map(lambda x: min(max(x, 0.5), 1.5))
        
        hist_tgt_price_data = hist_tgt_price_data.sort_values(['code', 'organ_id', 'create_date'])
        hist_tgt_price_data = hist_tgt_price_data.drop_duplicates(['code', 'organ_id'], keep='last')
        hist_tgt_price_data['weight_target_return'] = hist_tgt_price_data['weight'] * hist_tgt_price_data['target_return']
        code_mean_target_return = hist_tgt_price_data.groupby('code')['target_return'].mean()
        code_weight_target_return = hist_tgt_price_data.groupby('code')['weight_target_return'].mean()

        code_date_target_return = hist_tgt_price_data.groupby(['code', 'create_date'])[['target_return', 'weight']].mean().reset_index()
        code_date_target_return = code_date_target_return.sort_values(['code', 'create_date'])
        code_date_target_return['last_target_return'] = code_date_target_return.groupby('code')['target_return'].shift(1)
        code_date_target_return['target_return_diff'] = code_date_target_return['target_return']-code_date_target_return['last_target_return']
        code_date_target_return['target_return_diff'] = code_date_target_return['target_return_diff'].fillna(0)
        code_date_target_return['weight_target_return_diff'] = code_date_target_return['target_return_diff']*code_date_target_return['weight']
        code_target_return_trend = code_date_target_return.groupby('code')['target_return_diff'].sum()
        code_weight_target_return_trend = code_date_target_return.groupby('code')['weight_target_return_diff'].sum()
        code_target_return_indicator = pd.concat([code_mean_target_return, code_weight_target_return, code_target_return_trend, code_weight_target_return_trend], axis=1)
        code_target_return_indicator.columns = ['MeanTargetReturn', 'WeightMeanTargetReturn', 'TargetReturnTrend', 'TargetReturnWeightTrend']
        
        code_target_return_indicator['trade_date'] = opt_date

        target_return_indicator_infos.append(code_target_return_indicator.reset_index())
    all_target_return_indicator = pd.concat(target_return_indicator_infos)
    
    return all_target_return_indicator.set_index(['code', 'trade_date'])
        
        
                

        

        
    


def cal_consistent_tgt_price_indicator(tgt_price_data, nfq_price, hfq_price, opt_2_trade):
    nfq_price = nfq_price.reset_index()
    hfq_price = hfq_price.reset_index()
    opt_2_trade = opt_2_trade.reset_index()
    opt_dates = opt_2_trade['opt_date'].values
    tgt_price_data = tgt_price_data.reset_index()
    def std_time_info(research_report_info):
        research_report_info['entry_date'] = research_report_info['entry_date'].map(lambda x: int(str(x)[:8]))
        research_report_info['code'] = research_report_info['stock_code'].map(lambda x: "%06d" % int(x))
        research_report_info['code'] = research_report_info['code'].map(lambda x: x+".XSHG" if x[0] == '6' else x+'.XSHE')
        research_report_info['author_name_std'] = research_report_info['author_name'].map(lambda x: "_".join(sorted(x.split(','))))
        research_report_info = research_report_info.drop_duplicates("report_id")
        return research_report_info
    tgt_price_data = std_time_info(tgt_price_data)
    nfq_price = nfq_price.rename({'close': 'nfq_close'}, axis=1)
    hfq_price = hfq_price.rename({'close': 'hfq_close'}, axis=1)
        
    all_trade_dates = sorted(hfq_price['trade_date'].unique())
    tgt_price_data = tgt_price_data[tgt_price_data['create_date'].map(lambda x: x > min(all_trade_dates))]
    all_create_dates = sorted(tgt_price_data['create_date'].unique())
    create_date_2_price_date = {}
    for create_date in all_create_dates:
        hist_dates = [_ for _ in all_trade_dates if _ < create_date]
        if len(hist_dates):
            create_date_2_price_date.update({create_date: hist_dates[-1]})
    tgt_price_data['price_date'] = tgt_price_data['create_date'].map(create_date_2_price_date)
    tgt_price_data = pd.merge(tgt_price_data, nfq_price[['code', 'trade_date', 'nfq_close']], how='left', left_on=['code', 'price_date'], right_on=['code', 'trade_date'])
    tgt_price_data = pd.merge(tgt_price_data, hfq_price[['code', 'trade_date', 'hfq_close']], how='left', left_on=['code', 'price_date'], right_on=['code', 'trade_date'])
    tgt_price_data = tgt_price_data.rename({'hfq_close': 'forecast_day_price_hfq', 'nfq_close': 'forecast_day_price_nfq'}, axis=1)
    tgt_price_data['target_price'] = tgt_price_data[['target_price_ceiling', 'target_price_floor']].mean(axis=1)
    tgt_price_data = tgt_price_data[tgt_price_data['target_price'].notnull()]
    tgt_price_data = tgt_price_data[tgt_price_data['forecast_day_price_nfq'].notnull()]

    tgt_price_data['tgt_r'] = tgt_price_data['target_price']/tgt_price_data['forecast_day_price_nfq']
    consistent_tgt_real_price_diff_infos = []
    for opt_date in opt_dates:
        last_date = cal_n_day_before(opt_date, 90)
        hist_tgt_price_data = tgt_price_data[tgt_price_data.entry_date.map(lambda x: x >=last_date and x <=opt_date)].copy()
        opt_day_price = hfq_price[hfq_price.trade_date == opt_date]
        code_2_hfq_price = dict(zip(opt_day_price['code'], opt_day_price['hfq_close']))
        hist_tgt_price_data['opt_day_hfq_price'] = hist_tgt_price_data['code'].map(code_2_hfq_price)
#         hist_tgt_price_data['tgt_r'] = hist_tgt_price_data['target_price']/hist_tgt_price_data['forecast_day_price_nfq']
        hist_tgt_price_data['real_r'] = hist_tgt_price_data['opt_day_hfq_price']/hist_tgt_price_data['forecast_day_price_hfq']
        hist_tgt_price_data['tgt_real_diff'] = hist_tgt_price_data['tgt_r']/hist_tgt_price_data['real_r']
        hist_tgt_price_data = hist_tgt_price_data.sort_values(['code', 'organ_id', 'author_name_std', 'create_date'])
        hist_tgt_price_data = hist_tgt_price_data.drop_duplicates(['code', 'organ_id', 'author_name_std'], keep='last')
        # consistent_tgt_real_price_diff = hist_tgt_price_data.groupby('code')['tgt_real_diff'].mean()
        author_center_tgt_real_price_diff = hist_tgt_price_data.groupby('author_name_std').mean()['tgt_real_diff'].to_dict()
        org_center_tgt_real_price_diff = hist_tgt_price_data.groupby('organ_id').mean()['tgt_real_diff'].to_dict()
        hist_tgt_price_data['author_center_tgt_real_price_diff'] = hist_tgt_price_data['author_name_std'].map(author_center_tgt_real_price_diff)
        hist_tgt_price_data['org_center_tgt_real_price_diff'] = hist_tgt_price_data['organ_id'].map(org_center_tgt_real_price_diff)
        
        hist_tgt_price_data['author_std_tgt_real_price_diff'] = hist_tgt_price_data['tgt_real_diff'] - hist_tgt_price_data['author_center_tgt_real_price_diff']
        hist_tgt_price_data['org_std_tgt_real_price_diff'] = hist_tgt_price_data['tgt_real_diff'] - hist_tgt_price_data['org_center_tgt_real_price_diff']
        
        # consistent_author_std_tgt_real_price_diff = hist_tgt_price_data['autor_std_tgt_real_price_diff'].mean()
        # consistent_tgt_real_price_diff_data = pd.concat([consistent_tgt_real_price_diff, consistent_author_std_tgt_real_price_diff], axis=1)
        consistent_tgt_real_price_diff = hist_tgt_price_data.groupby('code')[['tgt_real_diff', 'author_std_tgt_real_price_diff', 'org_std_tgt_real_price_diff']].mean()
        consistent_tgt_real_price_diff.columns = ['ConsistentTgtRealPriceDiff', 'ConsistentAuthorStdTgtRealPriceDiff', 'ConsistentOrgStdTgtRealPriceDiff']
        consistent_tgt_real_price_diff = consistent_tgt_real_price_diff.reset_index()
        consistent_tgt_real_price_diff['trade_date'] = opt_date
        consistent_tgt_real_price_diff_infos.append(consistent_tgt_real_price_diff)
    all_consistent_tgt_real_price_diff = pd.concat(consistent_tgt_real_price_diff_infos)
    return all_consistent_tgt_real_price_diff.set_index(['code', 'trade_date'])
    
    
def count_research_report(research_report_data, opt2trade):
    research_report_data = research_report_data.reset_index()
    opt2trade = opt2trade.reset_index()
    opt_dates = sorted(opt2trade['opt_date'].values)
    research_report_data = research_report_data[research_report_data['综合值计算标记'] == 1]
    #     data = data[data['内部_公告日期'].notnull()]
    #     data['trade_date'] = data['内部_公告日期'].map(lambda x: transfer_date_str_2_int(x))
    research_report_data = research_report_data[research_report_data['trade_date'].notnull()]
    # research_report_data = research_report_data.drop_duplicates('报告ID')
    research_report_data = research_report_data.drop_duplicates(['code', '研究机构名称', '预测日期', '分析师名称'])
    def cal_month_feature_corr(data, feature_name):
        month_feature = data.set_index('pre_month_count')[feature_name]
        month_feature = month_feature.reindex(list(range(12)))
        month_feature = month_feature.fillna(0)
        month_feature = month_feature.reset_index()
        feature_corr = month_feature.corr().loc['pre_month_count'][feature_name]

        return feature_corr
    report_count_infos = []
    for opt_date in opt_dates:
        last_date = cal_n_day_before(opt_date, 365)
        hist_data =  research_report_data[research_report_data.trade_date.map(lambda x: x >= last_date and x <= opt_date)]
        import math
        hist_data['pre_month_count'] = hist_data['trade_date'].map(lambda x: math.ceil(cal_two_day_diff(int(x), int(opt_date))/30))
        hist_data['pre_month_count'] = hist_data['pre_month_count'].map(lambda x: 12-max(min(x, 12), 1))
        code_2_month_count = hist_data.groupby(['code', 'pre_month_count'])['研究机构名称'].count()
        code_2_month_count.name = 'ReportMonthCount'
        code_2_month_count = code_2_month_count.reset_index()
        code_2_rpt_count_trend = code_2_month_count.groupby('code').apply(lambda x: cal_month_feature_corr(x, 'ReportMonthCount'))
        code_2_rpt_count_trend.name = "ReportCountTrend"
        code_2_count = hist_data.groupby('code')['研究机构名称'].count()
        code_2_count.name = "ReportLastYearCount"
        code_count_info = pd.concat([code_2_rpt_count_trend, code_2_count], axis=1)
        code_count_info = code_count_info.reset_index()
        code_count_info['trade_date'] = opt_date
        report_count_infos.append(code_count_info)
    all_report_count_data = pd.concat(report_count_infos)
    
    return all_report_count_data.set_index(['code', 'trade_date'])


def cal_research_report_financial_report_pub_date_delay(research_report_data, financial_date_data, opt2trade, min_delay_days):
    financial_date_data = financial_date_data.reset_index()
    research_report_data = research_report_data.reset_index()
    opt2trade = opt2trade.reset_index()
    opt_dates = sorted(opt2trade['opt_date'].values)
    research_report_data = research_report_data[research_report_data['综合值计算标记'] == 1]
    #     data = data[data['内部_公告日期'].notnull()]
    #     data['trade_date'] = data['内部_公告日期'].map(lambda x: transfer_date_str_2_int(x))
    research_report_data = research_report_data[research_report_data['trade_date'].notnull()]
    # research_report_data = research_report_data.drop_duplicates('报告ID')
    research_report_data = research_report_data.drop_duplicates(['code', '研究机构名称', '预测日期', '分析师名称'])

    pub_date_delay_infos = []
    two_date_diff_info = {}
    for opt_date in opt_dates:
        print(opt_date)
        last_date_4_research_report = cal_n_day_before(opt_date, 90)
        hist_research_report_data =  research_report_data[research_report_data.trade_date.map(lambda x: x >= last_date_4_research_report and x <= opt_date)]
        last_date_4_financial_report =  cal_n_day_before(opt_date, 180)
        hist_financial_report_data =  financial_date_data[financial_date_data.trade_date.map(lambda x: x >= last_date_4_financial_report and x <= opt_date)]
        for code, code_hist_research_report_data in hist_research_report_data.groupby('code'):
            all_research_report_pub_dates = code_hist_research_report_data['trade_date'].unique()
            code_hist_financial_report_data = hist_financial_report_data[hist_financial_report_data.code == code]
            all_financial_report_pub_dates = code_hist_financial_report_data['trade_date'].unique()
            date_2_min_delay_day_count = {_: {} for _ in min_delay_days}
#             print(all_research_report_pub_dates)
#             print(all_financial_report_pub_dates)
            for research_report_date in all_research_report_pub_dates:
                day_delay_infos = []
                for financial_report_pub_date in all_financial_report_pub_dates:
                    financial_report_pub_date = int(financial_report_pub_date)
                    research_report_date = int(research_report_date)
                    if (financial_report_pub_date, research_report_date) in two_date_diff_info:
                        day_delay = two_date_diff_info[(financial_report_pub_date, research_report_date)]
                    else:
                        day_delay = cal_two_day_diff(financial_report_pub_date, research_report_date)
                        two_date_diff_info.update({(financial_report_pub_date, research_report_date): day_delay})
#                     if day_delay >= min_delay_day:
                    day_delay_infos.append(day_delay)
                for min_delay_day in min_delay_days:
                    valid_day_delay_infos = [_ for _ in day_delay_infos if _ >= min_delay_day]
                    if len(valid_day_delay_infos):
                        date_2_min_delay_day_count[min_delay_day].update({research_report_date: min(valid_day_delay_infos)})
            for min_delay_day in min_delay_days:
                code_hist_research_report_data['delay_day{}'.format(abs(min_delay_day))] = code_hist_research_report_data['trade_date'].map(date_2_min_delay_day_count[min_delay_day])
                
#             code_hist_research_report_data = code_hist_research_report_data[code_hist_research_report_data['delay_day'].notnull()]
            pub_date_delay_info = {'trade_date': opt_date, 'code': code}
            for min_delay_day in min_delay_days:
                pub_date_delay_info.update({
                    'MinDelayDay{}'.format(abs(min_delay_day)): code_hist_research_report_data['delay_day{}'.format(abs(min_delay_day))].min(),
                    'MeanDelayDay{}'.format(abs(min_delay_day)): code_hist_research_report_data['delay_day{}'.format(abs(min_delay_day))].mean(),
                    'MedianDelayDay{}'.format(abs(min_delay_day)): code_hist_research_report_data['delay_day{}'.format(abs(min_delay_day))].median(),                   
                })

            pub_date_delay_infos.append(pub_date_delay_info)
    pub_date_delay_df = pd.DataFrame(pub_date_delay_infos)
    return pub_date_delay_df.set_index(['code', 'trade_date'])

    
    

def merge_data_axis0(**kwargs):
    """
    将输入数据拼接在一起
    :param kwargs:
    :return:
    """
    try:
        df = pd.concat(list(kwargs.values()), axis=0)
        
        # features = df.columns
        # df.to_pickle(r"D:\PycharmProjects\test_data\{}.pkl".format("_".join(features)))
    except Exception as e:
        print(e)
        import pdb
        pdb.set_trace()
        df = pd.DataFrame()
    return df
    

def cal_pre_quarters(end_date):
    year = end_date.year
    month = end_date.month
#     date_ = end_date.day
#     import pdb
#     pdb.set_trace()
    if month == 3:
        pre_quarters = []
    elif month == 6:
        pre_quarters = [date(year=year, month=3, day=31)]
    elif month == 9:
        pre_quarters = [date(year=year, month=3, day=31), date(year=year, month=6, day=30)]
    else:
        pre_quarters = [date(year=year, month=3, day=31), date(year=year, month=6, day=30), date(year=year, month=9, day=30)]
    return pre_quarters
        

def process_fin_forecast(data):
    data = data.reset_index()
    data['np_parent_company_owners'] = data[['profit_min', 'profit_max']].mean(axis=1)
    return data.set_index(['trade_date', 'code'])[['np_parent_company_owners', 'end_date']]

def cal_quarter_feature(data, financial_rpt, feature):
    data = data.reset_index()
    financial_rpt = financial_rpt.reset_index() 
    all_quarter_infos = []
    for info in data.to_dict('records'):
        trade_date = info['trade_date']
        code = info['code']
        hist_financial_rpt = financial_rpt[(financial_rpt.code == code) & (financial_rpt.trade_date <= trade_date)]
        end_date = info['end_date']
        hist_financial_rpt = hist_financial_rpt.sort_values(['end_date', 'trade_date']).drop_duplicates('end_date')
        quater_2_feature = dict(zip(hist_financial_rpt['end_date'], hist_financial_rpt[feature]))
        pre_quarters =  cal_pre_quarters(end_date)
        feature_cumsum_value = info[feature]
        pre_quarter_features = [quater_2_feature[_] for _ in pre_quarters if _ in quater_2_feature]
        tmp_quarter_feature = (feature_cumsum_value- sum(pre_quarter_features))/(1+len(pre_quarters)-len(pre_quarter_features))
#         info['quarter_feature'] = tmp_quarter_feature
        all_quarter_infos.append({'trade_date': trade_date, 'code': code, 'end_date': end_date, feature: tmp_quarter_feature})
    quarter_feature_df = pd.DataFrame(all_quarter_infos)
    return quarter_feature_df.set_index(['trade_date', 'code'])

    
def industry_standardized_group_factor(data, feature_neutral_infos):
    import math
    index_names = data.index.names
    data = data.reset_index()

    for feature_info in feature_neutral_infos:
        industry_name = feature_info['industry_name']
        feature_name = feature_info['feature_name']
        reverse = feature_info['reverse']
        group_count = feature_info['group_count']
        null_value = feature_info.get('null_value', 0.5)
        is_null_process_before_rank = feature_info.get('is_null_process_before_rank', False)
        data[feature_name] = data[feature_name].map(lambda x: None if x == -999 else x)
        tmp_null_rate = data[feature_name].isnull().sum()/len(data)
        if len(industry_name) > 0:
            if tmp_null_rate > 0.8:
                data["{}{}Rank".format(feature_name, industry_name)] = 0.5
            else:
                if reverse:
                    if is_null_process_before_rank:
                        data[feature_name] = data[feature_name].fillna(null_value)
                        data["{}{}Rank".format(feature_name, industry_name )] = data.groupby(['trade_date', industry_name])[feature_name].rank(pct=True, ascending=False)
                    else:
                        data["{}{}Rank".format(feature_name, industry_name )] = data.groupby(['trade_date', industry_name])[feature_name].rank(pct=True, ascending=False).fillna(null_value)
                else:
                    if is_null_process_before_rank:

                        data[feature_name] = data[feature_name].fillna(null_value)
                        data["{}{}Rank".format(feature_name, industry_name )] = data.groupby(['trade_date', industry_name])[feature_name].rank(pct=True)
                    else:
                        try:
                            data["{}{}Rank".format(feature_name, industry_name)] = data.groupby(['trade_date', industry_name])[feature_name].rank(pct=True).fillna(null_value)
                        except Exception as e:
                            import pdb
                            pdb.set_trace()
                            pass
                if group_count >= 2:
                    data["{}Bin".format(feature_name)] = data["{}{}Rank".format(feature_name, industry_name)].map(lambda x: math.ceil(group_count*x))
                    bin_2_value = data.groupby(['trade_date', "{}Bin".format(feature_name)])["{}{}Rank".format(feature_name, industry_name)].mean().reset_index()
                    data = data.drop("{}{}Rank".format(feature_name, industry_name), axis=1)
                    data = pd.merge(data, bin_2_value, how='left', on=['trade_date', "{}Bin".format(feature_name)])                
        else:
            if tmp_null_rate > 0.8:
                data["{}Rank".format(feature_name)] = 0.5
            else:
                if reverse:
                    if is_null_process_before_rank:
                        data[feature_name] = data[feature_name].fillna(null_value)
                        data["{}Rank".format(feature_name)] = data.groupby(['trade_date'])[feature_name].rank(pct=True, ascending=False)
                    else:
                        data["{}Rank".format(feature_name)] = data.groupby(['trade_date'])[feature_name].rank(pct=True, ascending=False).fillna(null_value)
                else:
                    if is_null_process_before_rank:
                        data[feature_name] = data[feature_name].fillna(null_value)
                        data["{}Rank".format(feature_name)] = data.groupby(['trade_date'])[feature_name].rank(pct=True)

                    else:
                        data["{}Rank".format(feature_name)] = data.groupby(['trade_date'])[feature_name].rank(pct=True).fillna(null_value)
                if group_count >= 2:
                    data["{}Bin".format(feature_name)] = data["{}Rank".format(feature_name)].map(lambda x: math.ceil(group_count*x))
                    bin_2_value = data.groupby(['trade_date', "{}Bin".format(feature_name)])["{}Rank".format(feature_name)].mean().reset_index()
                    data = data.drop("{}Rank".format(feature_name), axis=1)
                    data = pd.merge(data, bin_2_value, how='left', on=['trade_date', "{}Bin".format(feature_name)])
    return data.set_index(index_names)

    
def industry_group_factor(data, feature_infos):
    index_names = data.index.names
    data = data.reset_index()
    import math
    for feature_info in feature_infos:
        industry_name = feature_info['industry_name']
        feature_name = feature_info['feature_name']
        reverse = feature_info['reverse']
        data[feature_name] = data[feature_name].map(lambda x: None if x == -999 else x)
        group_count = feature_info['group_count']
        if reverse:
            data['{}Rank'.format(feature_name)] = data[feature_name].rank(pct=True, ascending=False)
        else:
            data['{}Rank'.format(feature_name)] = data[feature_name].rank(pct=True)
        industry_factor = data.groupby(['trade_date', industry_name])['{}Rank'.format(feature_name)].mean().reset_index()
#         industry_factor['{}{}RankBin'.format(feature_name, industry_name)] = industry_factor.groupby('trade_date')['{}Rank'.format(feature_name)].rank(pct=True).map(lambda x: math.ceil(group_count*x))
        industry_factor['{}{}RankBin'.format(feature_name, industry_name)] = industry_factor.groupby('trade_date')['{}Rank'.format(feature_name)].rank()      
        data = pd.merge(data, industry_factor[['trade_date', industry_name, '{}{}RankBin'.format(feature_name, industry_name)]], how='left', on=['trade_date', industry_name])
    return data.set_index(index_names)

def industry_stanadard_indicator_sum(data, chosen_indicator_info, industry_name, output_factor_name):
    index_names = data.index.names
    data = data.reset_index()
    all_output_factors = []

    for trade_date, tmp_data in data.groupby('trade_date'):
        tgt_chosen_indicator_info = chosen_indicator_info[trade_date]
        limit_value = 0.05
        sort_func = ECDF
        is_3_sigma_std = True
        neutral_factors = []
        for indicator_name, weight_direction in tgt_chosen_indicator_info.items():
            if weight_direction > 0:
                reverse = False
            else:
                reverse = True
            output_name = "{}Factor".format(indicator_name)
            try:
                tmp_data[output_name] = tmp_data.groupby([industry_name])[indicator_name].apply(
                    lambda x: sd_win_sort(x, limit=limit_value, sort_func=sort_func, reverse=reverse,
                                          is_3_sigma_std=is_3_sigma_std))
            except Exception as e:
                import pdb
                pdb.set_trace()
                pass
            neutral_factors.append(output_name)
        tmp_data[output_factor_name] = tmp_data[neutral_factors].mean(axis=1)
        all_output_factors.append(tmp_data.set_index(index_names)[[output_factor_name]])
    all_output_df = pd.concat(all_output_factors)
    return all_output_df
            
    
        
def fast_peer_standardized_group_factor(valid_data, peer_mapping, feature_neutral_infos):
    valid_data = valid_data.reset_index()
    features = list(set([_['feature_name'] for _ in feature_neutral_infos]))
    peer_mapping = peer_mapping.reset_index()
    # peer_mapping['trade_date'] = peer_mapping['trade_date'].map(lambda x: int(x.strftime("%Y%m%d")))
    valid_data['peer'] = valid_data['code']
    valid_data[features] = valid_data[features].applymap(lambda x: None if x == -999 else x)
    all_code_tags = (valid_data['code'] + valid_data['trade_date'].map(str)).values
    all_code_tags = {_ :1 for _ in all_code_tags}
    peer_mapping['tag'] = peer_mapping['code'] + peer_mapping['trade_date'].map(str)
    peer_mapping = peer_mapping[peer_mapping['tag'].map(lambda x: x in all_code_tags)]
    
                                
    peer_mapping_factor = pd.merge(peer_mapping, valid_data[['code', 'trade_date']+features], how='left', left_on=['trade_date', 'code'], right_on=['trade_date', 'code'])
    peer_mapping_factor = pd.merge(peer_mapping_factor, valid_data[['peer', 'trade_date']+features], how='left', left_on=['trade_date', 'peer'], right_on=['trade_date', 'peer'])
    # print(peer_mapping_factor.columns)

    factor_peer_neutral_infos = []
    for feature_info in feature_neutral_infos:
#         limit_value = feature_info['limit_value']
        reverse = feature_info['reverse']
        output_name = feature_info['output_name']
        feature_name = feature_info['feature_name']
        print("feature {}".format(feature_name))
#         sort_func = feature_info['sort_func']
        null_value = feature_info.get('null_value', 0.5)
        group_count = feature_info['group_count']

        null_peer_mapping_factor = peer_mapping_factor[peer_mapping_factor['{}_x'.format(feature_name)].isnull() ].drop_duplicates(['trade_date', 'code'])
        nonull_peer_mapping_factor = peer_mapping_factor[peer_mapping_factor['{}_x'.format(feature_name)].notnull()]
        nonull_peer_mapping_factor = nonull_peer_mapping_factor[nonull_peer_mapping_factor['{}_y'.format(feature_name)].notnull()]

        nonull_peer_mapping_factor['less_than_peer_tag'] = (nonull_peer_mapping_factor['{}_x'.format(feature_name)] - nonull_peer_mapping_factor['{}_y'.format(feature_name)]).map(lambda x: -x if reverse else x).map(lambda x: 1 if x < 0 else 0)
        peer_ecdf = 1-nonull_peer_mapping_factor.groupby(['trade_date', 'code'])['less_than_peer_tag'].sum()/nonull_peer_mapping_factor.groupby(['trade_date', 'code'])['less_than_peer_tag'].count()
#         peer_ecdf = peer_ecdf.map(lambda x: 1 if x > (1-limit_value) else x).map(lambda x: limit_value if x < limit_value else x)
        peer_ecdf = peer_ecdf.map(lambda x: 1e-8 if x == 0 else x)
        peer_ecdf.name = output_name
        null_peer_mapping_factor[output_name] = null_value
        null_peer_ecdf = null_peer_mapping_factor.set_index(['trade_date', 'code'])[output_name]
#         import pdb
#         pdb.set_trace()
        all_peer_ecdf = pd.concat([peer_ecdf, null_peer_ecdf])
        all_peer_ecdf = all_peer_ecdf.reset_index()
        if group_count >= 2:
            all_peer_ecdf["{}Bin".format(output_name)] = all_peer_ecdf[output_name].map(lambda x: math.ceil(group_count * x))
            bin_2_value = all_peer_ecdf.groupby(['trade_date', "{}Bin".format(output_name)])[output_name].mean().reset_index()
            all_peer_ecdf = all_peer_ecdf.drop(output_name, axis=1)
            all_peer_ecdf = pd.merge(all_peer_ecdf,bin_2_value, how='left', on=['trade_date', "{}Bin".format(output_name)] )
#         print("count {},unique count {}".format(len(all_peer_ecdf), all_peer_ecdf.drop_duplicates(['trade_date', 'code']).shape[0]))
        factor_peer_neutral_infos.append(all_peer_ecdf.set_index(['trade_date', 'code'])[output_name])

    factor_peer_neutral_df = pd.concat(factor_peer_neutral_infos, axis=1)

    return factor_peer_neutral_df

def process_research_rpt_data(research_rpt_detail):
    research_rpt_detail = research_rpt_detail.reset_index()
  
    research_rpt_detail = research_rpt_detail[research_rpt_detail['综合值计算标记'] == 1]
#     data = data[data['内部_公告日期'].notnull()]
#     data['trade_date'] = data['内部_公告日期'].map(lambda x: transfer_date_str_2_int(x))
    research_rpt_detail = research_rpt_detail[research_rpt_detail['trade_date'].notnull()]
    # data = data.sort_values(['股票代码', '研究机构名称', 'trade_date'])
    research_rpt_detail['net_profit1'] = research_rpt_detail['预测净利润_万元']
    research_rpt_detail['net_profit2'] = research_rpt_detail['预测基准股本_万股'] * research_rpt_detail['预测每股收益_基本']
    research_rpt_detail['net_profit3'] = research_rpt_detail['预测基准股本_万股'] * research_rpt_detail['预测每股收益_换算']
    research_rpt_detail['net_profit4'] = research_rpt_detail['预测基准股本_万股'] * research_rpt_detail['预测每股收益_摊薄']
    research_rpt_detail['net_profit5'] = research_rpt_detail['预测基准股本_万股'] * research_rpt_detail['预测每股收益_稀释']
    research_rpt_detail['NetProfit'] = research_rpt_detail.apply(lambda x : get_valid_value(x, ['net_profit1', 'net_profit2', 'net_profit3', 'net_profit4', 'net_profit5']), axis=1)*10000
    research_rpt_detail['NetAsset'] = research_rpt_detail['预测基准股本_万股']*research_rpt_detail['每股净资产']*10000
    research_rpt_detail['MainRevenue'] = research_rpt_detail['预测主营业务收入_万元']*10000
    return research_rpt_detail

def generate_financial_rpt_exceeding_expectation(financial_rpt, research_rpt_detail, factor_index, feature_name):
#     research_rpt_detail = research_rpt_detail.reset_index()
    financial_rpt = financial_rpt.reset_index()
    factor_index_df = pd.DataFrame(np.zeros(len(factor_index)), index=factor_index)
    factor_index_df = factor_index_df.reset_index()

#     research_rpt_detail = research_rpt_detail[research_rpt_detail['综合值计算标记'] == 1]

#     research_rpt_detail = research_rpt_detail[research_rpt_detail['trade_date'].notnull()]
#     research_rpt_detail['net_profit1'] = research_rpt_detail['预测净利润_万元']
#     research_rpt_detail['net_profit2'] = research_rpt_detail['预测基准股本_万股'] * research_rpt_detail['预测每股收益_基本']
#     research_rpt_detail['net_profit3'] = research_rpt_detail['预测基准股本_万股'] * research_rpt_detail['预测每股收益_换算']
#     research_rpt_detail['net_profit4'] = research_rpt_detail['预测基准股本_万股'] * research_rpt_detail['预测每股收益_摊薄']
#     research_rpt_detail['net_profit5'] = research_rpt_detail['预测基准股本_万股'] * research_rpt_detail['预测每股收益_稀释']
#     research_rpt_detail['NetProfit'] = research_rpt_detail.apply(lambda x : get_valid_value(x, ['net_profit1', 'net_profit2', 'net_profit3', 'net_profit4', 'net_profit5']), axis=1)*10000
#     research_rpt_detail['NetAsset'] = research_rpt_detail['预测基准股本_万股']*research_rpt_detail['每股净资产']*10000
#     research_rpt_detail['MainRevenue'] = research_rpt_detail['预测主营业务收入_万元']*10000
    
    financial_rpt = financial_rpt.sort_values(['code', 'trade_date', 'end_date'])
    financial_rpt = financial_rpt.drop_duplicates(['code', 'trade_date'], keep='last')
    total_net_profit_expectation_infos = []
    for code, code_financial_rpt in financial_rpt.groupby('code'):
        code_research_rpt_detail = research_rpt_detail[research_rpt_detail.code == code]
        for info in code_financial_rpt.to_dict('records'):
            pub_date = info['trade_date']
            end_date = str(info['end_date']).replace("-", "")
            net_profit = info['np_parent_company_owners']
            year = end_date[:4]
            quarter = end_date[4:]
            last_date = cal_n_day_before(pub_date, 180)
#             hist_data = hist_data[hist_data['报告期'] == "{}1231".format(year)]
            hist_data =  code_research_rpt_detail[(code_research_rpt_detail.trade_date.map(lambda x: x >= last_date and x <= pub_date)) & (code_research_rpt_detail['报告期'] == "{}1231".format(year))]
            if len(hist_data):
                hist_data = hist_data.sort_values(['研究机构名称', 'trade_date'])
                hist_data = hist_data.drop_duplicates(['研究机构名称'], keep='last')
                consistent_earning_forecast = hist_data['NetProfit'].mean()/4
                

            else:
                consistent_earning_forecast = None
            if consistent_earning_forecast is not None and consistent_earning_forecast != 0:
                exceeding_expection = (net_profit-consistent_earning_forecast)/abs(consistent_earning_forecast)
            else:
                exceeding_expection = None
#             if code == "000001.XSHE" and pub_date > 20190101:
#                 import pdb
#                 pdb.set_trace()
            total_net_profit_expectation_infos.append({'code': code,
                                                       'trade_date': pub_date,
                                                       'net_profit': net_profit, 
                                                       "net_profit_expectation": consistent_earning_forecast,
                                                       '{}ExceedingExpection'.format(feature_name): exceeding_expection
                                                      })
    total_net_profit_expectation_df = pd.DataFrame(total_net_profit_expectation_infos)
    total_net_profit_expectation_df.to_pickle("{}_exceedingexpectation.pkl".format(feature_name))
    code_last_pub_date_infos = []
    for code, code_index in factor_index_df.groupby('code'):
        trade_dates = code_index['trade_date'].values
        all_pub_dates = total_net_profit_expectation_df[total_net_profit_expectation_df.code == code]['trade_date'].values
        trade_date_to_last_pub_date = {}
        for trade_date in trade_dates:
            hist_pub_dates = [_ for _ in all_pub_dates if _ <= trade_date]
            if len(hist_pub_dates):
                last_pub_date = hist_pub_dates[-1]
                hist_window = cal_two_day_diff(last_pub_date, trade_date)
                if hist_window > 390:
                    last_pub_date = None
            else:
                last_pub_date = None
            trade_date_to_last_pub_date.update({trade_date: last_pub_date})
        code_index['last_pub_date'] = code_index['trade_date'].map(trade_date_to_last_pub_date)
        code_last_pub_date_infos.append(code_index)
    code_pub_date_df = pd.concat(code_last_pub_date_infos)
    net_profit_exceeding_expection = pd.merge(code_pub_date_df, total_net_profit_expectation_df[['code', 'trade_date', '{}ExceedingExpection'.format(feature_name)]], how='left', left_on=['code', 'last_pub_date'], right_on=['code', 'trade_date'])

    net_profit_exceeding_expection = net_profit_exceeding_expection.drop('trade_date_y', axis=1)
    net_profit_exceeding_expection = net_profit_exceeding_expection.rename({'trade_date_x': "trade_date"}, axis=1)
#     import pdb
#     pdb.set_trace()
    return net_profit_exceeding_expection.set_index(['trade_date', 'code'])[['{}ExceedingExpection'.format(feature_name)]]
        


def merge_growth_indicators(growth_indicator_infos, data):
    all_growth_features = []
    data = data.reset_index()
    for indicator_info in growth_indicator_infos:
        feature = indicator_info['feature']
        null_value = indicator_info['null_value']
        reverse = indicator_info['reverse']
        data["{}Bin".format(feature)] = data.groupby('trade_date')[feature].rank(pct=True,ascending=1 - reverse).fillna(
            null_value)
        all_growth_features.append("{}Bin".format(feature))
    data['GrowthIndicator'] = data[all_growth_features].mean(axis=1)
    return data.set_index(['trade_date', 'code'])

def get_zscore(score):
    score_mean = score.mean()
    score_std = score.std()
    z_score = score.map(lambda x: (x-score_mean)/score_std).map(lambda x: min(max(x, -2), 2))
    return z_score


def cal_value_indicator_residual(value_growth_factor):
    value_growth_factor = value_growth_factor.reset_index()
    value_growth_factor['SWL1IndustryCode'] = value_growth_factor['SWL1IndustryCode'].fillna("801890")
    all_bp_residual_infos = []

    value_growth_factor['LogBookToPrice'] = value_growth_factor['BookToPrice'].map(lambda x: math.log(x) if x > 0 else math.log(1/20000))
    for trade_date, tmp_factor in value_growth_factor.groupby('trade_date'):
        tmp_factor['roe_zscore'] = get_zscore(tmp_factor["ROE"])
        tmp_factor['roe_zscore'] = tmp_factor['roe_zscore'].fillna(0)
        tmp_factor['growth_indicator_zscore'] = get_zscore(tmp_factor["GrowthIndicator"])

        valid_tmp_factor = tmp_factor[tmp_factor['LogBookToPrice'].notnull()]
        valid_tmp_factor['LogBookToPriceZScore'] = get_zscore(valid_tmp_factor["LogBookToPrice"])
        novalid_tmp_factor = tmp_factor[tmp_factor['LogBookToPrice'].isnull()]
        industry_vec = pd.get_dummies(valid_tmp_factor['SWL1IndustryCode'])
        y = valid_tmp_factor['LogBookToPriceZScore']


        x = pd.concat([industry_vec, valid_tmp_factor[['growth_indicator_zscore']], valid_tmp_factor[['roe_zscore']]],
                       axis=1)

        ridge_clf = Ridge()
        ridge_clf.fit(x, y)
        bp_residual = y - ridge_clf.predict(x)
        valid_tmp_factor['BookToPriceResidual'] = bp_residual
        novalid_tmp_factor['BookToPriceResidual'] = None


        all_bp_residual_infos.append(pd.concat([valid_tmp_factor, novalid_tmp_factor])[['code', 'trade_date', 'BookToPriceResidual']])
    all_bp_residual_df = pd.concat(all_bp_residual_infos)
    return all_bp_residual_df.set_index(['trade_date', 'code'])    
        
def cal_feature_deviation(data, feature, window):
    data = data.reset_index()
    data = data.sort_values(['code', 'trade_date'])
    data["{}DeviationWindonw{}".format(feature, window)] = data[feature] - data.groupby('code')[feature].shift(window)
    return data.set_index(['trade_date', 'code'])[["{}DeviationWindonw{}".format(feature, window)]]


def divide_two_variable_4_zero_with_multi_features(first_var_names, second_var_name, data):
#     data[first_var_names + [second_var_name]] = data[first_var_names + [second_var_name]].fillna(0)
    
    output_names = []
    for first_var_name in first_var_names:
        output_name = "".join([_.capitalize() for _ in first_var_name.split("_")])
        output_names.append("{}To{}".format(output_name, second_var_name))
#         data["{}_to_{}".format(first_var_name, second_var_name)] = data.apply(lambda x: x[first_var_name] / x[second_var_name] if x[second_var_name] != 0 else 0,axis=1)
        data["{}To{}".format(output_name, second_var_name)] = data[first_var_name]/data[second_var_name]
    data[output_names] = data[output_names].fillna(0) 
    return data[output_names]

def multi_divide_two_variable(divide_infos, data):
    output_names = []
    for info in divide_infos:
        first_var_name = info['first_var_name']
        second_var_name = info['second_var_name']
        data['output'] = data[first_var_name]/data[second_var_name]
        first_var_name = "".join([_.capitalize() for _ in first_var_name.split("_")])
        second_var_name = "".join([_.capitalize()[:2] for _ in second_var_name.split("_")])
        output_name = "{}To{}".format(first_var_name, second_var_name)
        data = data.rename({'output': output_name}, axis=1)
        output_names.append(output_name)
#         data[output_name] = data[first_var_name]/data[second_var_name]
    data[output_names] = data[output_names].fillna(0)
    data = data.replace(np.inf, None)
    data = data.replace(-np.inf, None)
    data[output_names] = data[output_names].fillna(0)

    return data[output_names]
        


def ridge_regress_coef(x,y,l2):
    ridge = Ridge(alpha=l2)
    ridge.fit(x, y)
    coef_ = ridge.coef_
    return coef_

def value_factor_weekly_performance_attribution_multilayer(factor_data, factors, industry, l2, layer_tag):

    multilayer_factor_premium_infos = []
    processed_factor_infos = []
    if "trade_date" not in factor_data.columns:
        factor_data = factor_data.reset_index()

    for trade_date, tmp_df in factor_data.groupby('trade_date'):
        tmp_df["LiquidityFactor"] = tmp_df["LiquidityFactor"].rank(pct=True).map(lambda x: 0.5 if x > 0.2 else -0.5)
        tmp_df["ShortMomentumFactorReverse"] = tmp_df["ShortMomentumFactorReverse"].rank(pct=True).map(lambda x: 0.5 if x > 0.2 else -0.5)
        def std_long_momentum(x):
            if x > 0.7:
                return -0.5
            elif x > 0.3:
                return 0
            else:
                return 0.5

        tmp_df["LongMomentumFactorReverse"] = tmp_df["LongMomentumFactorReverse"].rank(pct=True).map(lambda x: std_long_momentum(x))
        mv_mean = tmp_df['LogMktCap'].mean()
        mv_std = tmp_df['LogMktCap'].std()
        tmp_df['LogMktCapZScore'] = tmp_df['LogMktCap'].map(lambda x: (x-mv_mean)/mv_std)
        for factor in factors:
            if factor in ['GrowthFactor', 'LeverageFactor', 'ValueFactor', 'QualityFactor']:
                tmp_df[factor] = tmp_df[factor].rank(pct=True).map(lambda x: math.ceil(3 * x) / 2 - 1)

        processed_factor_infos.append(tmp_df.copy())
        for _, layer_tmp_df in tmp_df.groupby(layer_tag):
            if tmp_df['OneTermReturn'].notnull().sum() != 0:
                factor_premium = ridge_regress_coef(layer_tmp_df[factors], layer_tmp_df['OneTermReturn'].fillna(0), l2)
                
                factor_premium_info = dict(zip(factors, factor_premium))
                
                factor_premium_info.update({'trade_date': trade_date, 'layer': _})
                multilayer_factor_premium_infos.append(factor_premium_info)
 

    multilayer_factor_premium_df = pd.DataFrame(multilayer_factor_premium_infos)
    processed_factor_df = pd.concat(processed_factor_infos)
    multilayer_factor_premium_df.to_excel("multilayer_factor_premium_value_factor.xlsx")

    return multilayer_factor_premium_df, processed_factor_df


def generate_score_from_perf_attribution_multilayer(start_date, end_date, factors, factor_directions, multilayer_factor_premium_df, processed_factor_df, long_window_size, short_window_size, layer_tag,  is_norm=False):
    trade_dates = sorted(processed_factor_df['trade_date'].unique())
    assert len(trade_dates) > long_window_size, print("not enough history data")
#     factor_premium_df['weight_adj_factor'] = factor_premium_df[factors].applymap(lambda x: abs(x)).sum(axis=1).map(lambda x: 0.05/x)
#     for factor in factors:
#         factor_premium_df[factor] = factor_premium_df[factor]*factor_premium_df["weight_adj_factor"]
#     if is_norm:
#         factor_premium_df[factors] = factor_premium_df[factors].applymap(lambda x: max(x, 0))
#         factor_premium_df['premium_weight_sum'] = factor_premium_df[factors].sum(axis=1)
#         factor_premium_df['adj_factor'] = factor_premium_df['premium_weight_sum'].map(lambda x: 0.012/x if x > 0 else 0)
#         for factor in factors:
#             factor_premium_df[factor] = factor_premium_df[factor]*factor_premium_df["adj_factor"]        

    score_infos = []
    premium_pred_infos = []

    for k in range(long_window_size, len(trade_dates)):
        if trade_dates[k] >= start_date and trade_dates[k] <= end_date:
            long_hist_premium_df = multilayer_factor_premium_df[multilayer_factor_premium_df.trade_date.map(lambda x: x in trade_dates[k-long_window_size: k])
            ][factors+['layer']]
#             long_premium_pred = long_hist_premium_df.mean()
            try:
                long_premium_pred = long_hist_premium_df.groupby('layer').mean()
            except Exception as e:
                import pdb
                pdb.set_trace()
                pass
            for factor, factor_direction in zip(factors, factor_directions):
                if factor_direction == 1:
#                     long_premium_pred[factor] = max(long_premium_pred[factor], 0)
                    long_premium_pred[factor] = long_premium_pred[factor].map(lambda x: max(x, 0))

                elif factor_direction == -1:
                    long_premium_pred[factor] = long_premium_pred[factor].map(lambda x: min(x, 0))
                else:
                    pass
#             long_premium_pred = long_premium_pred.map(lambda x: max(x, 0))
            short_hist_premium_df = multilayer_factor_premium_df[multilayer_factor_premium_df.trade_date.map(lambda x: x in trade_dates[k-short_window_size: k])
            ][factors+['layer']]
            short_premium_pred = short_hist_premium_df.groupby('layer').mean()
            for factor, factor_direction in zip(factors, factor_directions):
                if factor_direction == 1:
#                     short_premium_pred[factor] = max(short_premium_pred[factor], 0)
                    short_premium_pred[factor] = short_premium_pred[factor].map(lambda x: max(x, 0))

                elif factor_direction == -1:
                    short_premium_pred[factor] = short_premium_pred[factor].map(lambda x: min(x, 0))
                else:
                    pass
                
#             short_premium_pred = short_premium_pred.map(lambda x: max(x, 0))
            tmp_processed_factor_df = processed_factor_df[processed_factor_df.trade_date == trade_dates[k]]
            
            for layer, layer_tmp_processed_factor in tmp_processed_factor_df.groupby(layer_tag):
                long_score = np.matmul(layer_tmp_processed_factor[factors].values, long_premium_pred.loc[layer][factors].values.reshape(len(factors), 1))
                short_score = np.matmul(layer_tmp_processed_factor[factors].values, short_premium_pred.loc[layer][factors].values.reshape(len(factors), 1))
                layer_tmp_processed_factor['short_score'] = short_score
                layer_tmp_processed_factor['long_score'] = long_score
                layer_tmp_processed_factor['ScoreFromPerfAtt_8_2'] = layer_tmp_processed_factor['long_score']*0.8 + layer_tmp_processed_factor['short_score']*0.2
                layer_tmp_processed_factor['ScoreFromPerfAtt_5_5'] = layer_tmp_processed_factor['long_score']*0.5 + layer_tmp_processed_factor['short_score']*0.5

                score_infos.append(layer_tmp_processed_factor[['code', 'trade_date', 'ScoreFromPerfAtt_8_2', 'ScoreFromPerfAtt_5_5', 'long_score', 'short_score', layer_tag]])
    score_df = pd.concat(score_infos)

#     all_premium_pred_df = pd.concat(premium_pred_infos, axis=1)
#     all_premium_pred_df.T.to_excel("csi500_premium_pred_long_{}_short_{}.xlsx".format(long_window_size, short_window_size))
#     factor_premium_df.to_excel("csi500_premium_weekly.xlsx".format(long_window_size, short_window_size))
    return score_df


def weekly_performance_attribution_with_different_factors(factor_data, date_to_factors, long_window_size, short_window_size, start_date, end_date):
    factor_data = factor_data.reset_index()
    all_trade_dates = sorted(factor_data['trade_date'].unique())
    pred_score_infos = []
    import math
    def tansfer_data_to_bin(tmp_data, factors):
        tmp_data[factors] = tmp_data[factors].applymap(lambda x: math.ceil(3*x))
        return tmp_data
    for idx, tmp_trade_date in enumerate(all_trade_dates):
        if tmp_trade_date > start_date and tmp_trade_date < end_date:
            hist_dates = all_trade_dates[:idx][-long_window_size:]
            short_hist_dates = all_trade_dates[:idx][-short_window_size:]
#             hist_factor_data = factor_data[factor_data.trade_date.map(lambda x: x in hist_dates)]
            
            factors_4_att_info = date_to_factors[tmp_trade_date]
            factors_4_att = list(factors_4_att_info.keys())
            factor_premium_infos = []
            for trade_date in hist_dates:
                tgt_hist_data = factor_data[factor_data['trade_date'] == trade_date]
                tgt_hist_data = tansfer_data_to_bin(tgt_hist_data, factors_4_att)
#                 factor_premium = _OLS_estimate_fac_premium(tgt_hist_data[factors_4_att], tgt_hist_data['OneTermReturn'].fillna(0))
                factor_premium = ridge_regress_coef(tgt_hist_data[factors_4_att], tgt_hist_data['OneTermReturn'].fillna(0), 1000)
#                 factor_premium.index = ['bias'] + factors_4_att

#                 factor_premium_info = factor_premium.to_dict()
                factor_premium_info = dict(zip(factors_4_att, list(factor_premium)))
                factor_premium_info.update({'trade_date': trade_date})
                factor_premium_infos.append(factor_premium_info)
            factor_premium_df = pd.DataFrame(factor_premium_infos)
            long_premium_pred = factor_premium_df.mean()
            for factor, factor_direction in factors_4_att_info.items():
                if factor_direction == 1:
                    long_premium_pred[factor] = max(long_premium_pred[factor], 0)
                elif factor_direction == -1:
                    long_premium_pred[factor] = min(long_premium_pred[factor], 0)
                else:
                    pass
            short_hist_premium_df = factor_premium_df.tail(short_window_size)
            short_premium_pred = short_hist_premium_df.mean()
            for factor, factor_direction in factors_4_att_info.items():
                if factor_direction == 1:
                    short_premium_pred[factor] = max(short_premium_pred[factor], 0)
                elif factor_direction == -1:
                    short_premium_pred[factor] = min(short_premium_pred[factor], 0)
                else:
                    pass
            tmp_data = factor_data[factor_data.trade_date == tmp_trade_date]
            tmp_data = tansfer_data_to_bin(tmp_data, factors_4_att)
            long_score = np.matmul(tmp_data[factors_4_att].values, long_premium_pred.loc[factors_4_att].values.reshape(len(factors_4_att), 1))
            short_score = np.matmul(tmp_data[factors_4_att].values, short_premium_pred.loc[factors_4_att].values.reshape(len(factors_4_att), 1))
            tmp_data['short_score'] = short_score
            tmp_data['long_score'] = long_score
            tmp_data['ScoreFromPerfAtt_8_2'] = tmp_data['long_score']*0.8 + tmp_data['short_score']*0.2
            tmp_data['ScoreFromPerfAtt_5_5'] = tmp_data['long_score']*0.5 + tmp_data['short_score']*0.5
            pred_score_infos.append(tmp_data[['code', 'trade_date', 'ScoreFromPerfAtt_8_2', 'ScoreFromPerfAtt_5_5']])
     
    all_pred_score_df = pd.concat(pred_score_infos)
    return all_pred_score_df.set_index(['trade_date', 'code'])
                                           
            
