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
con = "mysql+pymysql://develop:haikuan_2025@localhost/factor_research_full_a_share"
otherfilter ={'field': 'paused', 'type': 'not_equal','param':1}




class DataProcess():
    def __init__(self):
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
        alldata = pd.read_sql('select * from all_data_test_all_mkt_indicator', conn)
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
        dates = alldata[alldata['trade_date'] >= 20220101]['trade_date'].unique()

        begintime = 20220101  # 20150101
        endtime = dates[-1]  # 更新到最新日期
        raw, ll = self.getdata()

        raw = raw[(raw['trade_date'] > begintime) & (raw['trade_date'] <= endtime)]
        raw['return'] = (raw['close'] - raw['pre_close']) / raw['pre_close']
        # data =raw
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
        data = data[data['mv_rank'] <= 1000]
        data['GicsIndustryCode'] = data['GicsIndustryCode'].fillna(20)
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
            {"feature_name": "label1", "limit_value": 0.05, "reverse": False,
             "sort_func": ECDF,
             "output_name": "label1_ecdf", "industry_name": "GicsIndustryCode"}
        ]
        params = {'common': {'read_engine': con, 'save_engine': con,
                             'rolling_n': 20,
                             'hist_year': 0,
                             'start_date': 20150101, 'end_date': 20251231,
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


class FiveOne():
    def __init__(self, data, contain_kechuang):
        self.contain_kechuang = contain_kechuang
        self.raw = data

    def industry(self, SQL_Data):
        con = create_engine("mysql+pymysql://develop:haikuan_2025@localhost:3306/factor_compute_new")
        csi500_industry = pd.read_sql(
            'select trade_date,code,CSI500MonthlyMvWeight,GicsIndustryName from all_data_test_all_mkt_indicator where trade_date>20160101',
            con)
        csi500_industry = csi500_industry.groupby(['GicsIndustryName', 'trade_date'])[
            'CSI500MonthlyMvWeight'].sum().reset_index().sort_values(by='trade_date')
        csi500_industry = csi500_industry.rename({'GicsIndustryName': 'industry'}, axis=1)
        csi500_industry = csi500_industry.rename({'CSI500MonthlyMvWeight': 'industry_weight'}, axis=1)

        def cal_weight(x):
            sum_weight = x['weight'].sum()
            x['weight'] = x['industry_weight'] * x['weight'] / sum_weight
            return x

        SQL_Data = pd.merge(SQL_Data, csi500_industry, on=['industry', 'trade_date'])
        SQL_Data = SQL_Data.groupby(['trade_date', 'industry']).apply(lambda x: cal_weight(x))
        return SQL_Data

    def small_size_5_5(self, data):
        label = 'label1_ecdf'
        # fac_cols = [ 'QualityFactor', 'ValueFactor', 'LiquidityFactor',
        #        'LeverageFactor', 'GrowthFactor', 'OverallMomentumFactor','var20']
        fac_cols = ['QualityFactor', 'ValueFactor', 'LiquidityFactor',
                    'GrowthFactor', 'var20_opt']
        length_lst = [10, 20, 40]
        SQL_Data = pd.DataFrame()
        dates = data['trade_date'].unique()

        for i in range(40, len(dates[1:]) + 1):
            print(dates[i], '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            MAX = 0
            for length in length_lst:

                res = pd.DataFrame()
                for j in range(2):
                    count_df = data[
                        (data['trade_date'] < dates[i - 1 - j]) & (data['trade_date'] >= dates[i - 1 - length - j])]
                    count_df = count_df[['code', 'size100_muliti', ]].groupby('code').sum()
                    black_list = count_df[count_df['size100_muliti'] > length / 2].index

                    count_df = data[
                        (data['trade_date'] < dates[i - 1 - j]) & (data['trade_date'] >= dates[i - length - 1 - j])]
                    count_df = count_df[['code', 'size50', ]].groupby('code').sum()
                    white_list = count_df[count_df['size50'] < length / 5].index

                    temp_data = data[data['trade_date'] == dates[i - 1 - j]]
                    temp_res = temp_data[temp_data['mv_rank'] <= 100]
                    temp_res = temp_res[~temp_res['code'].isin(set(black_list) - set(white_list))]
                    #     temp_res = temp_res[~temp_res['code'].isin(set(black_list))]

                    res = pd.concat([res, temp_res], axis=0)
                if res['label1'].mean() > MAX:
                    MAX = res['label1'].mean()
                    better_length = length

            print('寻找较优周期：', better_length)

            count_df = data[(data['trade_date'] < dates[i]) & (data['trade_date'] >= dates[i - better_length])]
            count_df = count_df[['code', 'size100_muliti', ]].groupby('code').sum()

            black_list = count_df[count_df['size100_muliti'] > better_length / 2].index

            count_df = data[(data['trade_date'] < dates[i]) & (data['trade_date'] >= dates[i - better_length])]
            count_df = count_df[['code', 'size50', ]].groupby('code').sum()
            white_list = count_df[count_df['size50'] < better_length / 5].index

            linneardata = data[
                (data['trade_date'] <= dates[i]) & (data['trade_date'] >= dates[i - 5]) & (data['mv_rank'] <= 200)]
            #     pca.fit(linneardata[fac_cols])
            linneardata = linnear_model(linneardata, label, fac_cols).compute_score()
            linneardata1 = data[
                (data['trade_date'] <= dates[i]) & (data['trade_date'] >= dates[i - 40]) & (data['mv_rank'] <= 200)]
            #     pca.fit(linneardata[fac_cols])
            linneardata1 = linnear_model(linneardata1, label, fac_cols).compute_score()
            linneardata = pd.merge(linneardata, linneardata1[['trade_date', 'code', 'score']],
                                   on=['trade_date', 'code'])
            y = linneardata['score_x'] + linneardata['score_y']

            x = linneardata['mv_rank']
            x = sm.add_constant(x)
            model = sm.OLS(y, x)
            results = model.fit()
            linneardata['score_nonmv'] = results.resid
            temp_res = linneardata[(linneardata['mv_rank'] <= 200)]
            temp_res = temp_res[~temp_res['code'].isin(set(black_list) - set(white_list))]
            temp_res = temp_res[temp_res['code'] != '688217.XSHG']

            #     temp_res = temp_res[~temp_res['code'].isin(set(black_list))]

            SQL_Data = pd.concat([SQL_Data, temp_res], axis=0)
        SQL_Data = SQL_Data.groupby(['trade_date', 'code']).apply(lambda x: x.tail(1)).reset_index(drop=True)
        SQL_Data = SQL_Data.groupby('trade_date').apply(lambda x: x.sort_values(by='score_nonmv').tail(20)).reset_index(
            drop=True)
        SQL_Data['weight_1'] = SQL_Data.groupby('trade_date')['money'].apply(lambda x: x / x.sum())
        SQL_Data['weight_2'] = SQL_Data.groupby('trade_date')['var20'].apply(lambda x: x / x.sum())
        SQL_Data['weight'] = (SQL_Data['weight_1'] + SQL_Data['weight_2']) / 2

        #         SQL_Data['weight'] = np.where(SQL_Data['weight']<0.025,0,SQL_Data['weight'])
        # SQL_Data['weight'] = 1
        SQL_Data['index_weight'] = 0.05
        con = create_engine(
            "mysql+pymysql://develop:haikuan_2025@localhost:3306/factor_compute_new")
        SQL_Data[['trade_date', 'code', 'weight', 'index_weight']].to_sql('smallsize5_5_yrs', con, if_exists='replace')

    def small_size_6_1(self, data):
        fac_cols = ['QualityFactor', 'ValueFactor', 'LiquidityFactor',
                    'GrowthFactor', 'var20_opt', 'OverallMomentumFactor']
        label = 'label1'
        dates = data['trade_date'].unique()
        length_lst = [10, 20, 40]
        SQL_Data = pd.DataFrame()
        for i in range(40, len(dates[1:]) + 1):
            print(dates[i], '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            MAX = 0
            for length in length_lst:

                res = pd.DataFrame()
                for j in range(2):
                    count_df = data[
                        (data['trade_date'] < dates[i - 1 - j]) & (data['trade_date'] >= dates[i - 1 - length - j])]
                    count_df = count_df[['code', 'size100_muliti', ]].groupby('code').sum()
                    black_list = count_df[count_df['size100_muliti'] > length / 2].index

                    count_df = data[
                        (data['trade_date'] < dates[i - 1 - j]) & (data['trade_date'] >= dates[i - length - 1 - j])]
                    count_df = count_df[['code', 'size50', ]].groupby('code').sum()
                    white_list = count_df[count_df['size50'] < length / 5].index

                    temp_data = data[data['trade_date'] == dates[i - 1 - j]]
                    temp_res = temp_data[temp_data['mv_rank'] <= 100]
                    temp_res = temp_res[~temp_res['code'].isin(set(black_list) - set(white_list))]

                    res = pd.concat([res, temp_res], axis=0)
                if res['label1'].mean() > MAX:
                    MAX = res['label1'].mean()
                    better_length = length

            print('寻找较优周期：', better_length)

            count_df = data[(data['trade_date'] < dates[i]) & (data['trade_date'] >= dates[i - better_length])]
            count_df = count_df[['code', 'size100_muliti', ]].groupby('code').sum()

            black_list = count_df[count_df['size100_muliti'] > better_length / 2].index

            count_df = data[(data['trade_date'] < dates[i]) & (data['trade_date'] >= dates[i - better_length])]
            count_df = count_df[['code', 'size50', ]].groupby('code').sum()
            white_list = count_df[count_df['size50'] < better_length / 5].index

            linneardata = data[
                (data['trade_date'] <= dates[i]) & (data['trade_date'] >= dates[i - 5]) & (data['mv_rank'] <= 400)]
            linneardata = linnear_model(linneardata, label, fac_cols).compute_score()
            linneardata1 = data[
                (data['trade_date'] <= dates[i]) & (data['trade_date'] >= dates[i - 40]) & (data['mv_rank'] <= 400)]
            linneardata1 = linnear_model(linneardata1, label, fac_cols).compute_score()
            linneardata = pd.merge(linneardata, linneardata1[['trade_date', 'code', 'score']],
                                   on=['trade_date', 'code'])
            y = linneardata['score_x'] + linneardata['score_y']

            x = linneardata['mv_rank']
            x = sm.add_constant(x)
            model = sm.OLS(y, x)
            results = model.fit()
            linneardata['score_nonmv'] = results.resid
            temp_res = linneardata[(linneardata['mv_rank'] <= 400)]
            temp_res = temp_res[~temp_res['code'].isin(set(black_list) - set(white_list))]
            temp_res = temp_res[temp_res['code'] != '688217.XSHG']

            SQL_Data = pd.concat([SQL_Data, temp_res], axis=0)
        SQL_Data = SQL_Data.groupby(['trade_date', 'code']).apply(lambda x: x.tail(1)).reset_index(drop=True)
        SQL_Data = SQL_Data.groupby('trade_date').apply(lambda x: x.sort_values(by='score_nonmv').tail(80)).reset_index(
            drop=True)

        SQL_Data['weight_1'] = SQL_Data.groupby('trade_date')['money'].apply(lambda x: x / (x.sum()))
        SQL_Data['weight_2'] = SQL_Data.groupby('trade_date')['var20'].apply(lambda x: x / (x.sum()))
        SQL_Data['weight'] = (SQL_Data['weight_1'] + SQL_Data['weight_2']) / 2
        # SQL_Data['weight']=1

        SQL_Data['index_weight'] = 0.05
        con = create_engine("mysql+pymysql://develop:haikuan_2025@localhost:3306/factor_compute_new")
        SQL_Data = SQL_Data.rename({'GicsIndustryName': 'industry'}, axis=1)
        #         SQL_Data = self.industry(SQL_Data)
        SQL_Data[['trade_date', 'code', 'weight', 'index_weight']].to_sql('smallsize6_1_yrs', con,
                                                                          if_exists='replace')

    def compute(self):
        self.raw['real_price'] = self.raw['close'] / self.raw['factor']
        data = self.raw[(self.raw['public_time'] > 1000) & (self.raw['real_price'] >= 3)]

        data = data.iloc[:, 1:][data['trade_date'] > 20150101]
        data['size100'] = np.where((data['mv_rank'] > 0) & (data['mv_rank'] <= 100), 1, 0)
        data['size50'] = np.where((data['mv_rank'] > 0) & (data['mv_rank'] <= 50), 1, 0)
        data['size100_muliti'] = data['size100'] * (data.groupby('code')['size100'].shift(1))
        data['var20'] = data['var20'] * (-1)
        data['var20_opt'] = data['var20_opt'] * (-1)

        label = 'label1'
        # fac_cols = [ 'QualityFactor', 'ValueFactor', 'LiquidityFactor',
        #        'LeverageFactor', 'GrowthFactor', 'OverallMomentumFactor','var20']
        fac_cols = ['QualityFactor', 'ValueFactor', 'LiquidityFactor',
                    'GrowthFactor', 'var20_opt']
        length_lst = [10, 20, 40]
        SQL_Data = pd.DataFrame()
        data = data[data['trade_date'] > 20220101]
        dates = data['trade_date'].unique()

        for i in range(40, len(dates[1:]) + 1):
            print(dates[i], '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            MAX = 0
            for length in length_lst:

                res = pd.DataFrame()
                for j in range(2):
                    count_df = data[
                        (data['trade_date'] < dates[i - 1 - j]) & (data['trade_date'] >= dates[i - 1 - length - j])]
                    count_df = count_df[['code', 'size100_muliti', ]].groupby('code').sum()
                    black_list = count_df[count_df['size100_muliti'] > length / 2].index

                    count_df = data[
                        (data['trade_date'] < dates[i - 1 - j]) & (data['trade_date'] >= dates[i - length - 1 - j])]
                    count_df = count_df[['code', 'size50', ]].groupby('code').sum()
                    white_list = count_df[count_df['size50'] < length / 5].index

                    temp_data = data[data['trade_date'] == dates[i - 1 - j]]
                    temp_res = temp_data[temp_data['mv_rank'] <= 100]
                    temp_res = temp_res[~temp_res['code'].isin(set(black_list) - set(white_list))]
                    #     temp_res = temp_res[~temp_res['code'].isin(set(black_list))]

                    res = pd.concat([res, temp_res], axis=0)
                if res['label1'].mean() > MAX:
                    MAX = res['label1'].mean()
                    better_length = length

            print('寻找较优周期：', better_length)

            count_df = data[(data['trade_date'] < dates[i]) & (data['trade_date'] >= dates[i - better_length])]
            count_df = count_df[['code', 'size100_muliti', ]].groupby('code').sum()

            black_list = count_df[count_df['size100_muliti'] > better_length / 2].index

            count_df = data[(data['trade_date'] < dates[i]) & (data['trade_date'] >= dates[i - better_length])]
            count_df = count_df[['code', 'size50', ]].groupby('code').sum()
            white_list = count_df[count_df['size50'] < better_length / 5].index

            linneardata = linnear_model(data[(data['trade_date'] <= dates[i]) & (data['trade_date'] >= dates[i - 5]) & (
                    data['mv_rank'] <= 200)], label, fac_cols).compute_score()
            linneardata1 = linnear_model(data[(data['trade_date'] <= dates[i]) & (
                    data['trade_date'] >= dates[i - 40]) & (data['mv_rank'] <= 200)], label,
                                         fac_cols).compute_score()
            linneardata = pd.merge(linneardata, linneardata1[['trade_date', 'code', 'score']],
                                   on=['trade_date', 'code'])
            temp_res = linneardata[(linneardata['mv_rank'] <= 100)]
            temp_res = temp_res[~temp_res['code'].isin(set(black_list) - set(white_list))]

            #     temp_res = temp_res[~temp_res['code'].isin(set(black_list))]

            SQL_Data = pd.concat([SQL_Data, temp_res], axis=0)
        SQL_Data['score'] = SQL_Data['score_x'] + SQL_Data['score_y']
        SQL_Data = SQL_Data.groupby(['trade_date', 'code']).apply(lambda x: x.tail(1)).reset_index(drop=True)
        SQL_Data = SQL_Data.groupby('trade_date').apply(lambda x: x.sort_values(by='score').tail(20)).reset_index(
            drop=True)
        SQL_Data['weight_1'] = SQL_Data.groupby('trade_date')['money'].apply(lambda x: x / x.sum())
        SQL_Data['weight_2'] = SQL_Data.groupby('trade_date')['var20'].apply(lambda x: x / x.sum())
        SQL_Data['weight'] = SQL_Data['weight_1']
        # SQL_Data['weight'] = np.where(SQL_Data['weight']<0.025,0,SQL_Data['weight'])
        # SQL_Data['weight']=1
        SQL_Data['index_weight'] = 0.05
        con = create_engine(
            "mysql+pymysql://develop:haikuan_2025@localhost:3306/factor_compute_new")  # mysql+pymysql的意思为：指定引擎为pymysql
        SQL_Data[['trade_date', 'code', 'weight', 'index_weight']].to_sql('smallsize5_1_yrs', con, if_exists='replace')
        SQL_Data['weight'] = SQL_Data['weight_2']
        SQL_Data[['trade_date', 'code', 'weight', 'index_weight']].to_sql('smallsize5_2_yrs', con, if_exists='replace')
        SQL_Data['weight'] = (SQL_Data['weight_1'] + SQL_Data['weight_2']) / 2
        SQL_Data[['trade_date', 'code', 'weight', 'index_weight']].to_sql('smallsize5_3_yrs', con, if_exists='replace')

        SQL_Data = pd.DataFrame()
        for i in range(40, len(dates[1:]) + 1):
            print(dates[i], '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

            linneardata = linnear_model(data[(data['trade_date'] <= dates[i]) & (data['trade_date'] >= dates[i - 5]) & (
                    data['mv_rank'] <= 200)], label, fac_cols).compute_score()
            linneardata1 = linnear_model(data[(data['trade_date'] <= dates[i]) & (
                    data['trade_date'] >= dates[i - 40]) & (data['mv_rank'] <= 200)], label,
                                         fac_cols).compute_score()
            linneardata = pd.merge(linneardata, linneardata1[['trade_date', 'code', 'score']],
                                   on=['trade_date', 'code'])
            temp_res = linneardata[(linneardata['mv_rank'] <= 100)]

            #     temp_res = temp_res[~temp_res['code'].isin(set(black_list))]

            SQL_Data = pd.concat([SQL_Data, temp_res], axis=0)
        SQL_Data['score'] = SQL_Data['score_x'] + SQL_Data['score_y']
        SQL_Data = SQL_Data.groupby(['trade_date', 'code']).apply(lambda x: x.tail(1)).reset_index(drop=True)
        SQL_Data = SQL_Data.groupby('trade_date').apply(lambda x: x.sort_values(by='score').tail(20)).reset_index(
            drop=True)
        SQL_Data['weight'] = 1
        SQL_Data['index_weight'] = 0.05
        SQL_Data[['trade_date', 'code', 'weight', 'index_weight']].to_sql('smallsize5_4_yrs', con, if_exists='replace')

        self.small_size_5_5(data)
        self.small_size_6_1(data)
