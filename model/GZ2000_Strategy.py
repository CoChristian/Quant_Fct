from model import factors_QP
import pymysql
import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
import statsmodels.api as sm


from matplotlib.pyplot import MultipleLocator
import matplotlib.pyplot as plt
from jqdatasdk import finance
from model.LinnearModel import linnear_model
import jqdatasdk as jd
from docx import Document
# docx.shared 用于设置大小（图片等）
from statsmodels.distributions.empirical_distribution import ECDF
from docx.shared import Cm, Pt
from docx.document import Document as Doc
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sqlalchemy import create_engine

from model import industry_standardized


class DataProcess():
    def __init__(self):

        jd.auth("13764432461", "Nfhq12345")
        code_df_lst = []
        for y in [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024,2025]:
            for m in range(1, 13):
                date = y * 10000 + m * 100 + 3
                code_lst = jd.get_index_stocks('399303.XSHE', date=date)  # zz1000 399852
                code_df = pd.DataFrame(code_lst, columns=['code'])
                code_df['year_month'] = date // 100
                code_df_lst.append(code_df)
        self.GZ2000_data = pd.concat(code_df_lst)
        self.data = self.dataprocess()

    def getdata(self):
        conn = pymysql.connect(host='192.168.110.66', user='develop', password='haikuan_2025',
                               database='factor_research_full_a_share')
        d = pd.read_sql('select * from daily_trading_data', conn)
        #     cursor = conn.cursor()
        #     cursor.execute("SELECT * from daily_trading_data")
        #     data = cursor.fetchall()
        #     d = pd.DataFrame(data)
        #     cursor.execute('select DISTINCT(COLUMN_NAME)  FROM information_schema.columns where table_name = "daily_trading_data" ORDER BY ORDINAL_POSITION')
        #     d.columns = [i[0] for i in cursor.fetchall()[:15]]
        # dataset = factor_compute_new  all_data_test_all_mkt_indicator
        conn = pymysql.connect(host='192.168.110.66', user='develop', password='haikuan_2025',
                               database='factor_compute_new')
        alldata = pd.read_sql('select * from all_data_test_all_mkt_indicator ', conn)
        conn = pymysql.connect(host='192.168.110.66', user='develop', password='haikuan_2025',
                               database='factor_compute_new')
        stdata = pd.read_sql('select * from valid_flag', conn)
        alldata = alldata.drop('STFlag', axis=1)
        cols = alldata.columns
        stcols = stdata.columns
        try:
            res = pd.merge(d, alldata, on=['trade_date', 'code'], how='left')
            res = pd.merge(res, stdata, on=['trade_date', 'code'], how='left')
        except:
            print(d, alldata)
            return 0, 0
        res[cols[2:]] = res.groupby('code')[cols[2:]].fillna(method='ffill')
        res[stcols[2:]] = res.groupby('code')[stcols[2:]].fillna(method='ffill')
        #     res['terminal_data'] = res['terminal_data'].fillna(0)
        conn = pymysql.connect(host='192.168.110.66', user='develop', password='haikuan_2025',
                               database='factor_research_full_a_share')
        cursor = conn.cursor()
        cursor.execute("SELECT * from stock_universe")
        data = cursor.fetchall()
        stock_universe = pd.DataFrame(data, columns=['trade_date', 'code', 'name', 'public_date', 'a', 'b'])
        stock_universe['public_time'] = (pd.to_datetime(stock_universe['trade_date'], format='%Y%m%d') - pd.to_datetime(
            stock_universe['public_date'], format='%Y-%m-%d')).apply(lambda x: x.days)
        market_cap_data = pd.read_sql('select trade_date,code,market_cap from valuation_q', conn)
        res = pd.merge(res, stock_universe[['trade_date', 'code', 'public_time']], on=['trade_date', 'code'],
                       how='left')
        res = pd.merge(res, market_cap_data[['trade_date', 'code', 'market_cap']], on=['trade_date', 'code'],
                       how='left')

        #     res.drop(['paused', 'st_flag', 'listed_flag', 'nan_flag'],axis=1,inplace=True)
        return res, alldata['trade_date'].unique()

    def dataprocess(self):
        # 获得交易日期
        conn = pymysql.connect(host='192.168.110.66', user='develop', password='haikuan_2025',
                               database='factor_compute_new')
        alldata = pd.read_sql('select * from all_data_test_all_mkt_indicator', conn)
        dates = alldata[alldata['trade_date'] > 20150101]['trade_date'].unique()

        begintime = 20150101  # 20150101
        endtime = dates[-1]  # 更新到最新日期
        raw, ll = self.getdata()

        raw = raw[(raw['trade_date'] > begintime) & (raw['trade_date'] <= endtime)]
        raw['return'] = (raw['close'] - raw['pre_close']) / raw['pre_close']
        data = raw
        raw['money_cumsum'] = raw.groupby('code')['money'].cumsum()
        data = raw[raw['trade_date'].isin(dates)]
        data['money_weekly'] = data.groupby('code')['money_cumsum'].diff(1)
        data['return'] = data.groupby('code')['close'].diff(1) / data.groupby('code')['close'].shift(1)
        data['label1'] = data.groupby('code')['return'].shift(-1)
        data = data[(data.paused != 1) & (data.STFlagV2 != 1) & (data.ListedFlag != 1) & (data.NanFlag != 1) & (
                data.EndFlag != 1) & (data.public_time > 20)]
        data['mv_rank'] = data.groupby('trade_date')['market_cap'].apply(lambda x: x.rank(ascending=True))
        data['label2'] = data.groupby('trade_date')['return'].apply(lambda x: x.rank(ascending=True))
        data['label2'] = data.groupby('code')['label2'].shift(-1)
        data['GicsIndustryCode'] = data['GicsIndustryCode'].fillna(20)
        data['year_month'] = data['trade_date'] // 100
        data['real_price'] = data['close'] / data['factor']
        data = pd.merge(self.GZ2000_data, data, how='left', on=['code', 'year_month'])
        neutral_infos = [
            {"feature_name": "LogMktCap", "limit_value": 0.05, "reverse": True, "sort_func": ECDF,
             "output_name": "SizeFactorReverse", "industry_name": "GicsIndustryCode"},
            {"feature_name": "LogMktCap", "limit_value": 0.05, "reverse": False, "sort_func": ECDF,
             "output_name": "SizeFactor", "industry_name": "GicsIndustryCode"},
            {"feature_name": "Volatility60Days", "limit_value": 0.05, "reverse": True, "sort_func": ECDF,
             "output_name": "VolatilityFactorReverse", "industry_name": "GicsIndustryCode"},
            {"feature_name": "Volatility60Days", "limit_value": 0.05, "reverse": False, "sort_func": ECDF,
             "output_name": "VolatilityFactor", "industry_name": "GicsIndustryCode"},
            {"feature_name": "RSI10Days", "limit_value": 0.05, "reverse": False, "sort_func": ECDF,
             "output_name": "RSIFactorPositive", "industry_name": "GicsIndustryCode"},
            {"feature_name": "MomentumWeeks5", "limit_value": 0.05, "reverse": True, "sort_func": ECDF,
             "output_name": "ShortMomentumFactorReverse", "industry_name": "GicsIndustryCode"},
            {"feature_name": "MomentumWeeks5", "limit_value": 0.05, "reverse": False, "sort_func": ECDF,
             "output_name": "ShortMomentumFactor", "industry_name": "GicsIndustryCode"},
            {"feature_name": "LongMinusShort", "limit_value": 0.05, "reverse": True, "sort_func": ECDF,
             "output_name": "LongMomentumFactorReverse", "industry_name": "GicsIndustryCode"},
            {"feature_name": "RevenueOverMktCap", "limit_value": 0.05, "reverse": False, "sort_func": ECDF,
             "output_name": "RevenueOverMktCapFactor", "industry_name": "GicsIndustryCode"},
            {"feature_name": "TotalCompositeIncomeQuarterly", "limit_value": 0.05, "reverse": False, "sort_func": ECDF,
             "output_name": "IncomeQuarterlyFactor", "industry_name": "GicsIndustryCode"},
            {"feature_name": "NOCFOverDebt", "limit_value": 0.05, "reverse": False,
             "sort_func": ECDF, "output_name": "NOCFOverDebtFactor", "industry_name": "GicsIndustryCode"},
            {"feature_name": "CashOverMktCap", "limit_value": 0.05, "reverse": False,
             "sort_func": ECDF, "output_name": "CashOverMktCapFactor", "industry_name": "GicsIndustryCode"},
            {"feature_name": "BookToPrice", "limit_value": 0.05, "reverse": False,
             "sort_func": ECDF, "output_name": "BookToPriceFactor", "industry_name": "GicsIndustryCode"},
            {"feature_name": "MarketBeta000905XSHG252", "limit_value": 0.05, "reverse": False,
             "sort_func": ECDF, "output_name": "BetaFactor", "industry_name": "GicsIndustryCode"},
            {"feature_name": "STOM", "limit_value": 0.05, "reverse": True,
             "sort_func": ECDF, "output_name": "STOMFactorReverse", "industry_name": "GicsIndustryCode"},
            {"feature_name": "STOQ", "limit_value": 0.05, "reverse": True,
             "sort_func": ECDF, "output_name": "STOQFactorReverse", "industry_name": "GicsIndustryCode"},
            {"feature_name": "STOA", "limit_value": 0.05, "reverse": True,
             "sort_func": ECDF, "output_name": "STOAFactorReverse", "industry_name": "GicsIndustryCode"},
            {"feature_name": "MarketLeverage", "limit_value": 0.05, "reverse": False,
             "sort_func": ECDF,
             "output_name": "MarketLeverageFactor", "industry_name": "GicsIndustryCode"},
            {"feature_name": "BookLeverage", "limit_value": 0.05, "reverse": False,
             "sort_func": ECDF,
             "output_name": "BookLeverageFactor", "industry_name": "GicsIndustryCode"},
            {"feature_name": "DebtOverAssets", "limit_value": 0.05, "reverse": False,
             "sort_func": ECDF,
             "output_name": "DebtOverAssetsFactor", "industry_name": "GicsIndustryCode"},
            {"feature_name": "NonLinearSize", "limit_value": 0.05, "reverse": False,
             "sort_func": ECDF,
             "output_name": "NonLinearSizeFactor", "industry_name": "GicsIndustryCode"},
            {"feature_name": "RevenueLRC3", "limit_value": 0.05, "reverse": False,
             "sort_func": ECDF,
             "output_name": "RevenueLRC3Factor", "industry_name": "GicsIndustryCode"},
            {"feature_name": "RevenueYoy", "limit_value": 0.05, "reverse": False,
             "sort_func": ECDF,
             "output_name": "RevenueYoyFactor", "industry_name": "GicsIndustryCode"},
            {"feature_name": "NetIncomeLRC3", "limit_value": 0.05, "reverse": False,
             "sort_func": ECDF,
             "output_name": "NetIncomeLRC3Factor", "industry_name": "GicsIndustryCode"},
            {"feature_name": "NetIncomeYoy", "limit_value": 0.05, "reverse": False,
             "sort_func": ECDF,
             "output_name": "NetIncomeYoyFactor", "industry_name": "GicsIndustryCode"},
            {"feature_name": "var20", "limit_value": 0.05, "reverse": False,
             "sort_func": ECDF,
             "output_name": "var20_stand", "industry_name": "GicsIndustryCode"},
            {"feature_name": "var20_opt", "limit_value": 0.05, "reverse": False,
             "sort_func": ECDF,
             "output_name": "var20_opt", "industry_name": "GicsIndustryCode"},
            {"feature_name": "money", "limit_value": 0.05, "reverse": False,
             "sort_func": ECDF,
             "output_name": "money_stand", "industry_name": "GicsIndustryCode"},
            {"feature_name": "money_weekly", "limit_value": 0.05, "reverse": False,
             "sort_func": ECDF,
             "output_name": "money_weekly_stand", "industry_name": "GicsIndustryCode"},
            {"feature_name": "ROEYoy", "limit_value": 0.05, "reverse": False,
             "sort_func": ECDF,
             "output_name": "ROEYoyFactor", "industry_name": "GicsIndustryCode"}
        ]
        con = "mysql+pymysql://develop:haikuan_2025@localhost/factor_research_full_a_share"
        otherfilter = {'field': 'paused', 'type': 'not_equal', 'param': 1}
        params = {'common': {'read_engine': con, 'save_engine': con,
                             'rolling_n': 20,
                             'hist_year': 0,
                             'start_date': 20150101, 'end_date': 20250122222,
                             "output_name": ["var20", 'close'],  # 参数
                             'other_filter_info': otherfilter}}
        a = factors_QP.Variance(params, {}, {}).compute()
        feature = a['variance'].swaplevel(0, 1, axis=0).sort_index(level='code').reset_index()  # 参数
        data = pd.merge(data, feature[['code', 'trade_date', "var20"]], on=['code', 'trade_date'], how='left')  # 参数
        data['var20_opt'] = data.groupby('trade_date')['var20'].apply(lambda x: abs(x - x.quantile(0.5)))
        ecdf_data = industry_standardized.industry_standardized_factor(data, neutral_infos)
        sum_infos = [
            {
                "features": ["RevenueOverMktCapFactor", "IncomeQuarterlyFactor", "NOCFOverDebtFactor"],
                "weights": [0.25, 0.25, 0.5],
                "output_name": "QualityFactor"
            },
            {
                "features": ["CashOverMktCapFactor", "RevenueOverMktCapFactor", "BookToPriceFactor"],
                "weights": [1 / 3, 1 / 3, 1 / 3],
                "output_name": "ValueFactor"
            },
            {
                "features": ["STOMFactorReverse", "STOQFactorReverse", "STOAFactorReverse"],
                "weights": [0.35, 0.35, 0.3],
                "output_name": "LiquidityFactor"
            },
            {
                "features": ["MarketLeverageFactor", "DebtOverAssetsFactor", "BookLeverageFactor"],
                "weights": [0.38, 0.35, 0.27],
                "output_name": "LeverageFactor"
            },
            {
                "features": ["RevenueLRC3Factor", "RevenueYoyFactor", "NetIncomeLRC3Factor",
                             "NetIncomeYoyFactor"],
                "weights": [0.25, 0.25, 0.25, 0.25],
                "output_name": "GrowthFactor"
            },

            {
                "features": ["LongMomentumFactorReverse", "ShortMomentumFactorReverse"],
                "weights": [0, 1],
                "output_name": "OverallMomentumFactor"
            },

        ]

        def sum_data_with_weight(data, output_name, features=[], weights=[]):
            assert len(features) == len(weights)
            data[output_name] = (data[features] * weights).sum(axis=1)
            return data

        def pipline_sum_data_weight_weight(data, sum_infos):
            """
            根据权重对输入数据加权求和
            :param data:
            :param sum_infos:
            :return:
            """
            for info in sum_infos:
                features = info['features']
                weights = info['weights']
                output_name = info['output_name']
                data = sum_data_with_weight(data, output_name, features, weights)
            return data

        sum_data = pipline_sum_data_weight_weight(ecdf_data, sum_infos)
        return sum_data
class Generate_distribute_share_flag():

    def __init__(self):

        merge_data = pd.merge(self.income(), self.distribute_share(), on=['code', 'year'])
        merge_data = pd.merge(merge_data, self.BS(), on=['code', 'year'])
        merge_data['chuangye_flag'] = merge_data['code'].apply(lambda x: 1 if (x[:2] == '30') | (x[:3] == '688') else 0)
        condition1 = (merge_data['net_profit_3year'] * 0.3 > merge_data['distributed_share_3year'] * 10000)
        condition2 = (merge_data['distributed_share_3year'] * 10000 < 50000000) & (merge_data['chuangye_flag'] == 0)
        condition3 = (merge_data['distributed_share_3year'] * 10000 < 30000000) & (merge_data['chuangye_flag'] == 1)

        condition4 = (merge_data['total_operating_revenue_3year'] * 0.15 < merge_data['rd_expenses_3year']) & (
                    merge_data['chuangye_flag'] == 1)
        condition5 = (300000000 < merge_data['rd_expenses_3year']) & (merge_data['chuangye_flag'] == 1)

        condition6 = (merge_data['retained_profit'] > 0) & (merge_data['net_profit'] > 0)

        merge_data['flag'] = np.where((condition1 & (condition2 | condition3)) & (~condition4) & (~condition5), 1, 0)
        code_data = merge_data[(merge_data['net_profit'] > 0) & (merge_data['retained_profit'] > 0)][
            ['year', 'code', 'flag']]
        code_data['year'] = code_data['year'] + 2
        code_data.to_pickle('Li_Dividend_limit_code.pkl')
    def distribute_share(self):
        conn = pymysql.connect(host='192.168.110.66', user='develop', password='haikuan_2025',
                               database='factor_research_full_a_share')
        alldata = pd.read_sql('select * from xr_xd_stk', conn)

        data = alldata[
            ['trade_date', 'end_date', 'code', 'distributed_share_base_board', 'bonus_ratio_rmb', 'plan_progress']]
        data['year'] = data['end_date'].apply(lambda x: x.year)
        data = data.sort_values(by='year', ascending=True)

        data['distributed_share_base_board'] = data.groupby('code')['distributed_share_base_board'].fillna(
            method='ffill')
        data['bonus_ratio_rmb'] = data['bonus_ratio_rmb'].fillna(0)

        data['distributed_share'] = data['distributed_share_base_board'] * data['bonus_ratio_rmb'] * 1000
        distributed_share = data.groupby(['code', 'year'])['distributed_share'].sum()
        distributed_share.rename = 'distributed_share_3year'
        distributed_share = pd.DataFrame(distributed_share).reset_index()

        def cal(x):
            x['distributed_share_3year'] = x['distributed_share'].rolling(3, min_periods=1).sum()
            return x[['code', 'year', 'distributed_share_3year']]

        distributed_share = distributed_share.groupby('code').apply(lambda x: cal(x))
        distributed_share['distributed_share_3year'] = np.where(distributed_share['distributed_share_3year'] < 0, 0,
                                                                distributed_share['distributed_share_3year'])
        return distributed_share
    def income(self):
        conn = pymysql.connect(host='192.168.110.66', user='develop', password='haikuan_2025',
                               database='factor_research_full_a_share')
        data = pd.read_sql('select * from income_stk', conn)

        data['year'] = data['end_date'].apply(lambda x: x.year)
        data['month'] = data['end_date'].apply(lambda x: x.month)
        data = data.sort_values(by=['year', 'month'], ascending=True)
        data = data[data['month'] == 12]
        data = data.groupby(['year', 'code']).apply(lambda x: x.tail(1)).reset_index(drop=True)

        income_data = data[['trade_date', 'year', 'code', 'net_profit', 'operating_revenue', 'rd_expenses']]
        income_data['rd_expenses'] = income_data['rd_expenses'].fillna(0)

        def cal(x):
            x['net_profit_3year'] = x['net_profit'].rolling(3, min_periods=1).mean()
            x['total_operating_revenue_3year'] = x['operating_revenue'].rolling(3, min_periods=1).sum()
            x['rd_expenses_3year'] = x['rd_expenses'].rolling(3, min_periods=1).sum()
            return x

        income_data = income_data.groupby('code').apply(lambda x: cal(x))
        return income_data
    def BS(self):
        conn = pymysql.connect(host='192.168.110.66', user='develop', password='haikuan_2025',
                               database='factor_research_full_a_share')
        data = pd.read_sql('select * from balance_stk', conn)
        data['year'] = data['end_date'].apply(lambda x: x.year)
        data['month'] = data['end_date'].apply(lambda x: x.month)
        data = data.sort_values(by=['year', 'month'], ascending=True)
        data = data[data['month'] == 12]
        data = data.groupby(['year', 'code']).apply(lambda x: x.tail(1)).reset_index(drop=True)
        BS_data = data[['trade_date', 'year', 'code', 'retained_profit']]
        BS_data['retained_profit'] = BS_data['retained_profit'].fillna(0)
        return BS_data


class Strategy():
    def __init__(self, data, contain_kechuang):
        self.contain_kechuang = contain_kechuang
        self.raw = data

    def distributed_share(self):
        conn = pymysql.connect(host='192.168.110.66', user='develop', password='haikuan_2025',
                               database='factor_research_full_a_share')
        alldata = pd.read_sql('select * from xr_xd_stk', conn)

        data = alldata[
            ['trade_date', 'end_date', 'code', 'distributed_share_base_board', 'bonus_ratio_rmb', 'plan_progress']]
        data['year'] = data['end_date'].apply(lambda x: x.year)
        data = data.sort_values(by='year', ascending=True)

        data['distributed_share_base_board'] = data.groupby('code')['distributed_share_base_board'].fillna(
            method='ffill')
        data['bonus_ratio_rmb'] = data['bonus_ratio_rmb'].fillna(0)

        data['distributed_share'] = data['distributed_share_base_board'] * data['bonus_ratio_rmb'] * 1000
        distributed_share = data.groupby(['code', 'year'])['distributed_share'].sum()
        distributed_share.rename = 'distributed_share_3year'
        distributed_share = pd.DataFrame(distributed_share).reset_index()

        def cal(x):
            x['distributed_share_3year'] = x['distributed_share'].rolling(3, min_periods=1).sum()
            return x[['code', 'year', 'distributed_share_3year']]

        distributed_share = distributed_share.groupby('code').apply(lambda x: cal(x))
        distributed_share['distributed_share_3year'] = np.where(distributed_share['distributed_share_3year'] < 0, 0,
                                                                distributed_share['distributed_share_3year'])
        return distributed_share

    def model(self, data, fac_cols):
        SQL_Data = pd.DataFrame()
        data = data.sort_values(by='trade_date')
        dates = data['trade_date'].unique()
        SQL_Data_lst = []
        for i in range(3, len(dates[1:]) + 1):
            print(dates[i], '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            linneardata = data[
                (data['trade_date'] <= dates[i]) & (data['trade_date'] >= dates[i - 3])]
            linneardata = linnear_model(linneardata, 'label1', fac_cols).compute_score()
            linneardata1 = data[
                (data['trade_date'] <= dates[i]) & (data['trade_date'] >= dates[i - 2])]
            linneardata1 = linnear_model(linneardata1, 'label1', fac_cols).compute_score()
            linneardata = pd.merge(linneardata, linneardata1[['trade_date', 'code', 'score']],
                                   on=['trade_date', 'code'])
            y = linneardata['score_x'] + linneardata['score_y']
            x = linneardata['mv_rank']
            x = sm.add_constant(x)
            model = sm.OLS(y, x)
            results = model.fit()
            linneardata['score_nonmv'] = results.resid

            temp_res = linneardata
            print(temp_res[['score_nonmv', 'label1']].corr())
            SQL_Data_lst.append(temp_res)
        SQL_Data = pd.concat(SQL_Data_lst, axis=0)
        return SQL_Data

    def cal_weight(self, SQL_Data):
        SQL_Data = SQL_Data.groupby('trade_date').apply(
            lambda x: x.sort_values(by='score_nonmv').tail(120)).reset_index(
            drop=True)
        SQL_Data['weight_1'] = SQL_Data.groupby('trade_date')['money'].apply(lambda x: x / (x.sum()))
        SQL_Data['weight_2'] = SQL_Data.groupby('trade_date')['var20'].apply(lambda x: x / (x.sum()))
        SQL_Data['weight'] = (SQL_Data['weight_1'] + SQL_Data['weight_2']) / 2
        # SQL_Data['weight'] = 1
        SQL_Data['weight'] = SQL_Data.groupby('trade_date')['weight'].apply(lambda x: x / len(x))
        SQL_Data['index_weight'] = 1
        return SQL_Data

    def get_kind(self):
        # 获取
        conn = pymysql.connect(host='192.168.110.66', user='develop', password='haikuan_2025',
                               database='factor_research_full_a_share')
        all_data = pd.read_sql('select * from daily_trading_data', conn)
        all_data['return'] = all_data['close'] / all_data['pre_close'] - 1
        idx_money_series = all_data.groupby('trade_date')['return'].mean()
        idx_money_series = idx_money_series.rolling(17).std()
        idx_money_mean_series = idx_money_series.rolling(1000).quantile(0.75)
        idx_money_mean_series = idx_money_mean_series[idx_money_mean_series.index.isin(self.raw['trade_date'].unique())]

        idx_money_mean_series.name = 'mean'
        idx_df = pd.concat([idx_money_series, idx_money_mean_series], axis=1)
        idx_df = idx_df.dropna(how='any', axis=0)
        idx_df['flag'] = np.where(idx_df['return'] > idx_df['mean'], 1, 0)
        self.b_money_dates = idx_df[idx_df['flag'] == 1].index
        self.s_money_dates = idx_df[idx_df['flag'] == 0].index

    def compute(self):
        distributed_share = self.distributed_share()
        self.get_kind()

        code_limit = pd.read_pickle('Li_Dividend_limit_code.pkl')
        self.raw['year'] = pd.to_datetime(self.raw['trade_date'], format="%Y%m%d").apply(lambda x: x.year)
        self.raw = self.raw.sort_values(by='trade_date')
        self.raw = pd.merge(self.raw, distributed_share, how='left', on=['year', 'code'])
        self.raw['distributed_share_3year'] = self.raw['distributed_share_3year'].fillna(0)

        self.raw = pd.merge(self.raw, code_limit, on=['code', 'year'], how='left')
        self.raw['flag'] = self.raw['flag'].fillna(0)
        self.raw = self.raw[self.raw['flag'] == 0]

        self.raw['normal_distributed_share_3year'] = self.raw.groupby('trade_date')['distributed_share_3year'].apply(
            lambda x: (x - x.min()) / (x.max() - x.min()))

        data = self.raw[(self.raw['public_time'] > 1000) & (self.raw['real_price'] >= 3)]
        data = data.sort_values(by='trade_date')
        data = data[data['trade_date'] > 20150101]
        data['var20_opt'] = data['var20_opt'] * (-1)
        data['money_reverse_noramal'] = data.groupby('code')['money_weekly'].apply(
            lambda x: -1 * (x - x.rolling(10).min()) / (x.rolling(10).max() - x.rolling(10).min()))
        data = data.dropna(subset=['money_reverse_noramal'], axis=0, how='any')
        label = 'label1'
        # 高波
        big_data = data[data['trade_date'].isin(self.b_money_dates)]
        fac_cols = ['ValueFactor', 'LiquidityFactor', 'var20_opt',
                    'OverallMomentumFactor', 'money_reverse_noramal', 'GrowthFactor', 'normal_distributed_share_3year'
                    ]
        SQL_Data1 = self.model(big_data, fac_cols)
        # 低波
        small_data = data[data['trade_date'].isin(self.s_money_dates)]
        fac_cols = ['QualityFactor', 'ValueFactor', 'LiquidityFactor', 'var20_opt'
            , 'GrowthFactor', 'normal_distributed_share_3year']
        SQL_Data2 = self.model(small_data, fac_cols)
        SQL_Data = pd.concat([SQL_Data1, SQL_Data2])
        SQL_Data = self.cal_weight(SQL_Data)
        con = create_engine("mysql+pymysql://develop:haikuan_2025@localhost:3306/factor_compute_new")
        SQL_Data[['trade_date', 'code', 'weight', 'index_weight', 'GicsIndustryName']].to_sql('smallsize8_8_yrs', con,
                                                                                              if_exists='replace')
        return SQL_Data

