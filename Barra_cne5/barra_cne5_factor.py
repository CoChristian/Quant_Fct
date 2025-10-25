# import pandas_market_calendars as mcal
import os
import pandas as pd
from datetime import datetime, timedelta
import akshare as ak
import numpy as np
import statsmodels.api as sm
from scipy.stats import zscore

START_DATE = '20150101'
END_DATE = '20251014'
File_Path_Juqing = "F:\\work\\Data\\Database_to_csv\\"
File_Path_HomeCp = ("F:\\ECO\\Database_fromJQ\\Database_to_csv\\")
File_Path = File_Path_HomeCp


def getdatetimecol(df):
    if 'date' in df.columns:
        df['datetime'] = pd.to_datetime(df['date'])
    elif 'day' in df.columns:
        df['datetime'] = pd.to_datetime(df['day'])
    else:
        return None
    df['trade_date'] = df['datetime'].dt.date
    df['month'] = df['datetime'].dt.month
    df['week'] = df['datetime'].dt.isocalendar().week
    df['year'] = df['datetime'].dt.year
    df['stock_code'] = df['code']
    df = df.sort_values(by = ['trade_date','stock_code'],ascending=True).copy()
    df = df.drop_duplicates(subset=['trade_date','stock_code'], keep='first')
    return df

class GetData():


        # self.MARKET_DATA = getdatetimecol(pd.read_csv("F:\\quant_factor\\factor_backtest-main\\data\\MAfct_data\\index.csv"))
        # self.PRICE_DATA = getdatetimecol(pd.read_csv("F:\\work\\Data\\Database_to_csv\\stock_price_history_total\\stock_price_history251014.csv"))
        # self.INDUS_DATA = getdatetimecol(pd.read_csv("F:\\work\\Data\\Database_to_csv\\industry_total\\industry251015.csv"))
        # self.STATUS_DATA = getdatetimecol(pd.read_csv("F:\\work\\Data\\Database_to_csv\\stock_status_total\\stock_status251015.csv"))
        # self.INDEX_DATA = getdatetimecol(pd.read_csv("F:\\work\\Data\\Database_to_csv\\index_price_history_total\\index_price_history251015.csv"))
        # self.VALUATION_DATA = getdatetimecol(pd.read_csv("F:\\work\\Data\\Database_to_csv\\valuation_total\\valuation251014.csv"))

    @staticmethod
    def _get_price_():
        return getdatetimecol(pd.read_csv(File_Path + "stock_price_history_total\\stock_price_history251014.csv"))

    @staticmethod
    def _get_ind_():
        return getdatetimecol(pd.read_csv(File_Path + "industry_total\\industry251015.csv"))

    @staticmethod
    def _get_status():
        return getdatetimecol(pd.read_csv(File_Path + "stock_status_total\\stock_status251015.csv"))

    @staticmethod
    def _get_index():
        return getdatetimecol(pd.read_csv(File_Path + "index_price_history_total\\index_price_history251015.csv"))

    @staticmethod
    def _get_valuation():
        return getdatetimecol(pd.read_csv(File_Path + "valuation_total\\valuation251014.csv"))

    @staticmethod
    def risk_free(START_DATE, END_DATE):
        """
        获取无风险利率（十年国债收益率）
        :return: 无风险利率数据框 格式：日期，年化收益
        """
        current_df_start_time = datetime.strptime(START_DATE, "%Y%m%d")
        end_date_time = datetime.strptime(END_DATE, "%Y%m%d")
        yield10yr_df = pd.DataFrame()
        if os.path.exists(File_Path + "rf\\risk_free.csv"):
            yield10yr_df = pd.read_csv(File_Path + "rf\\risk_free.csv")
        else:
            while current_df_start_time < end_date_time:
                cal_end_date = current_df_start_time + timedelta(days=365)
                while cal_end_date.day != 31:
                    cal_end_date = cal_end_date - timedelta(days=1)
                current_df_end_time = min(cal_end_date, end_date_time)

                bond_china_yield_df = ak.bond_china_yield(
                    start_date=current_df_start_time.strftime("%Y%m%d"),
                    end_date=current_df_end_time.strftime("%Y%m%d")
                )

                filtered_df = bond_china_yield_df[
                    (bond_china_yield_df['曲线名称'] == '中债国债收益率曲线')
                ][['日期', '10年']]

                yield10yr_df = pd.concat([yield10yr_df, filtered_df])

                current_df_start_time = current_df_end_time + timedelta(days=1)

        yield10yr_df.reset_index(drop=True, inplace=True)
        yield10yr_df['RF_RETURN_ANN'] = yield10yr_df['10年'] / 100
        yield10yr_df['datetime'] = pd.to_datetime(yield10yr_df['日期'])
        yield10yr_df['RF_RETURN'] = (1 + yield10yr_df['RF_RETURN_ANN']) ** (1 / 252) - 1
        yield10yr_df['trade_date'] = yield10yr_df['datetime'].dt.date
        yield10yr_df['month'] = yield10yr_df['datetime'].dt.month
        yield10yr_df['week'] = yield10yr_df['datetime'].dt.isocalendar().week
        yield10yr_df['year'] = yield10yr_df['datetime'].dt.year

        rf = yield10yr_df[['trade_date', 'RF_RETURN']]

        return rf






class Calculation:
    """
    计算工具类，包含常用的因子预处理、回归和加权计算方法
    """

    @staticmethod
    def _exp_weight(window: int, half_life: int) -> np.ndarray:
        """
        计算指数加权权重
        :param window: 滑动窗口大小 例: 252
        :param half_life: 半衰期 例: 63
        :return: 归一化后的权重数组
        """
        exp_weight = np.asarray([0.5 ** (1 / half_life)] * window) ** np.arange(window)
        return exp_weight[::-1] / np.sum(exp_weight)

    @staticmethod
    def _winsorize(x: pd.Series) -> pd.Series:
        """
        去极值处理，使因子值在均值的3个标准差范围内
        :param x: 输入因子序列
        :return: 去极值后的因子序列
        """
        x = x.replace([np.inf, -np.inf], np.nan)
        mean = x.dropna().mean()
        std = x.dropna().std()
        winsorized = x.copy()
        winsorized[x < mean - 3 * std] = mean - 3 * std
        winsorized[x > mean + 3 * std] = mean + 3 * std
        return winsorized

    @staticmethod
    def _standardize(x: pd.Series, market_value: pd.Series) -> pd.Series:
        """
        市值加权标准化处理
        :param x: 输入因子序列 例: pd.Series
        :param market_value: 对应的市值序列 例: pd.Series
        :return: 标准化后的因子序列 例: pd.Series
        """
        if market_value.dtype != np.float64:
            market_value = market_value.astype(np.float64)
        x = x.replace([np.inf, -np.inf], np.nan)
        w_mean = np.sum(x.dropna() * market_value) / np.sum(market_value)
        std = x.dropna().std()
        standardized = (x - w_mean) / std
        return standardized

    def _preprocess(self, data: pd.DataFrame, factor_column: str) -> pd.DataFrame:
        """
        因子预处理函数，包含去极值和标准化步骤
        :param data: 输入的因子数据表
        :param factor_column: 需要处理的因子列名 例: 'LEVERAGE'
        :return: 预处理后的数据表
        """
        if data[f'{factor_column}'].dtype != np.float64:
            data[f'{factor_column}'] = data[f'{factor_column}'].astype('float')
        data[f'{factor_column}_wsr'] = data.groupby('TRADE_DT')[f'{factor_column}'].transform(lambda x: self._winsorize(x))
        data[f'{factor_column}_pp'] = data.groupby('TRADE_DT').apply(lambda g: self._standardize(g[f'{factor_column}_wsr'], g['S_VAL_MV'])).reset_index(level=0, drop=True)
        data[f'{factor_column}'] = data[f'{factor_column}_pp']
        data.drop(columns=[f'{factor_column}_wsr', f'{factor_column}_pp'], inplace=True)
        return data

    @staticmethod
    def _cumulative_range(x: pd.Series) -> float:
        """
        计算累积区间的范围，基于最近12个月的数据
        :param x: 输入的时间序列数据
        :return: 最大累积值与最小累积值的差值
        """
        T = np.arange(1, 13)
        cumulative_ranges = [x[-(t * 21):].sum() for t in T]
        return np.max(cumulative_ranges) - np.min(cumulative_ranges)

    @staticmethod
    def _weighted_regress(df: pd.DataFrame, weight = 1) -> tuple:
        """
        进行加权回归，计算Alpha和Beta
        :param df: 输入数据表，需包含‘STOCK_RETURN’, ‘RF_RETURN’, ‘MKT_RETURN’等列
        :param weight: 回归权重 例: 1.0
        :return: 回归得到的Alpha和Beta值
        """
        y = df['STOCK_RETURN'] - df['RF_RETURN']
        X = df[['CONSTANT', 'MKT_RETURN']]
        model = sm.WLS(y, X, weights=weight).fit()
        alpha, beta = model.params.iloc[0], model.params.iloc[1]
        return alpha, beta

    @staticmethod
    def _regress_w_time(x, n_time):
        """
        随时间进行回归，返回时间斜率与因子的平均值比率
        :param x: 输入的时间序列数据
        :param n_time: 用于回归的时间段数 例: 5
        :return: 回归的斜率值与因子均值的比率
        """
        if len(x) < n_time:
            return np.nan
        else:
            T = np.arange(1, n_time + 1, 1)
            T = sm.add_constant(T)
            model = sm.OLS(x, T).fit()
        if np.sum(x) != 0:
            return model.params.iloc[1] / np.mean(x)
        else:
            return model.params.iloc[1] / np.mean(x[1:])





class Beta(Calculation):
    """
    计算个股市场暴露（Beta值）的类，继承自 Calculation 类，提供个股与市场回报的回归分析
    """
    def __init__(self, df):
        """
        初始化 Beta 类实例，加载无风险收益率和市场指数数据，并计算初始 Beta 值

        :param df: 包含个股交易数据的 DataFrame，需包含以下列:
                    - 'TRADE_DT': 交易日期 例: '2022-08-31'
                    - 'S_INFO_WINDCODE': 个股代码 例: '000001.SZ'
                    - 'S_DQ_CLOSE': 收盘价 例: 10.5
                    - 'S_DQ_PRECLOSE': 前一日收盘价 例: 10.0
        """
        self.rf_df = GetData.risk_free(START_DATE, END_DATE)
        self.csi_df = GetData._get_index()
        self.csi_df = self.csi_df[self.csi_df['stock_code']=='000985.XSHG']
        # self.csi_df['trade_date'] = pd.to_datetime(self.csi_df['date'])
        self.csi_df['MKT_RETURN'] = self.csi_df['close'] / self.csi_df['pre_close'].shift() - 1
        self.csi_df['MKT_RETURN'] = self.csi_df['MKT_RETURN'].astype(float)
        self.beta_df = self.BETA(df)

    def BETA(self, df):
        """
        计算每个个股的Alpha和Beta值，并将结果存储回DataFrame中。

        :param df: 包含个股交易数据的 DataFrame，需包含以下列:
                    - 'TRADE_DT': 交易日期 例: '2022-08-31'
                    - 'S_INFO_WINDCODE': 个股代码 例: '000001.SZ'
                    - 'S_DQ_CLOSE': 收盘价 例: 10.5
                    - 'S_DQ_PRECLOSE': 前一日收盘价 例: 10.0
        :return: 包含新增列 'ALPHA', 'BETA', 'SIGMA' 的 DataFrame
        """
        df['STOCK_RETURN'] = df['close'] / df['pre_close'] - 1
        df['STOCK_RETURN'] = df['STOCK_RETURN'].astype('float')

        # 合并市场收益
        df = df.merge(self.csi_df[['trade_date', 'MKT_RETURN']], on='trade_date', how='left')
        # 合并无风险收益
        if 'RF_RETURN' not in df.columns:
            df = df.merge(self.rf_df, on='trade_date', how='left')

        exp_weight = self._exp_weight(window=252, half_life=63)
        df['ALPHA'] = np.nan
        df['BETA'] = np.nan
        df
        grouped = df.groupby('stock_code')

        # def cal_WLS_beta_alpha(df):
        #     df

        for stock_code, group in grouped:
            if group[group['trade_date'] > pd.to_datetime('2010-01-01').date()].empty:
                continue

            elif len(group) < 252:
                continue

            else:
                group['CONSTANT'] = 1
                alphas = []
                betas = []

                for i in range(251, len(group)):
                    window_data = group.iloc[i - 251:i + 1]
                    alpha, beta = self._weighted_regress(window_data, exp_weight)
                    alphas.append(alpha)
                    betas.append(beta)

                # Store the results, shifting by one period
                original_df_index = grouped.indices[f'{stock_code}']
                df.loc[original_df_index[251:], 'ALPHA'] = np.array(alphas)
                df.loc[original_df_index[251:], 'BETA'] = np.array(betas)
                df.loc[original_df_index, 'ALPHA'] = df.loc[original_df_index, 'ALPHA'].shift(1)
                df.loc[original_df_index, 'BETA'] = df.loc[original_df_index, 'BETA'].shift(1)

        df['SIGMA'] = df['STOCK_RETURN']- df['RF_RETURN'] - (df['ALPHA'] + df['BETA'] * df['MKT_RETURN'])

        return df


class Momentum(Calculation):
    """
    计算个股动量因子的类，继承自 Calculation 类
    """

    def RSTR(self, df: pd.DataFrame, raw: bool = False) -> pd.DataFrame:
        """
        计算个股的RSTR

        :param df: 包含个股交易数据的 DataFrame，需包含以下列:
                   - 'TRADE_DT': 交易日期 例: '2022-08-31'
                   - 'S_INFO_WINDCODE': 个股代码 例: '000001.SZ'
                   - 'S_DQ_CLOSE': 收盘价 例: 10.5
                   - 'S_DQ_PRECLOSE': 前一日收盘价 例: 10.0
        :param raw: 是否返回未经预处理的原始数据，默认为 False，即返回预处理后的数据
        :return: 返回包含新增列 'RSTR' 的 DataFrame
        """

        # 检查无风险收益率数据并合并
        if 'RF_RETURN' not in df.columns:
            rf_df = GetData.risk_free()
            df = df.merge(rf_df, on='TRADE_DT', how='left')

        # 计算个股日收益率
        if 'STOCK_RETURN' not in df.columns:
            df['STOCK_RETURN'] = df['close'] / df['pre_close'] - 1
            df['STOCK_RETURN'] = df['STOCK_RETURN'].astype('float')

        # 生成指数加权权重
        exp_weight = self._exp_weight(window=504, half_life=126)

        # 按个股代码分组
        grouped = df.groupby('stock_code')

        for stock_code, group in grouped:
            # 检查数据长度是否足够计算动量因子
            if len(group) < 504 + 21:
                print(f'Not enough data to calculate Momentum for {stock_code}')
                df.loc[group.index, 'RSTR'] = np.nan

            else:
                # 计算超额对数收益率
                group['EX_LG_RT'] = np.log(1 + group['STOCK_RETURN']) - np.log(1 + group['RF_RETURN'])

                try:
                    # 使用指数加权和滚动窗口计算 RSTR
                    group['RSTR'] = group['EX_LG_RT'].shift(21).rolling(window=504).apply(
                        lambda x: np.sum(x * exp_weight))

                    # 将计算结果存入原始数据框
                    df.loc[group.index, 'RSTR'] = group['RSTR']

                except Exception as e:
                    print(f'Error processing {stock_code}: {e}')
                    df.loc[group.index, 'RSTR'] = np.nan

        # 如果 raw 为 False，则进行预处理
        if not raw:
            df = self._preprocess(df, factor_column='RSTR')

        return df


class Size(Calculation):
    """
    计算个股市值相关因子的类，包括对数市值（LNCAP）和非线性市值因子（NLSIZE）。
    """

    def LNCAP(self, df, raw = False):
        """
        计算对数市值因子（LNCAP），并根据需要进行预处理。

        :param df: 包含股票市值数据的 DataFrame，需包含 'S_VAL_MV' 列。
                    - 'S_VAL_MV': 股票市值 例: 10000000 (市值单位需为浮点数)
        :param raw: 是否返回未经预处理的原始数据，默认为 False。
        :return: 包含 'LNCAP' 因子的 DataFrame。
        """

        if df['capitalization'].dtype != np.float64:
            df['capitalization'] = df['capitalization'].astype(np.float64)

        df['LNCAP'] = np.log(df['capitalization'])

        if not raw:
            df = self._preprocess(df, factor_column = 'LNCAP')

        return df

    def NLSIZE(self, df, raw = False):
        """
        计算非线性市值因子（NLSIZE），通过回归对数市值的三次方与常数项，并返回残差作为因子。

        :param df: 包含股票市值数据的 DataFrame，需包含 'S_VAL_MV' 列。
        :param raw: 是否返回未经预处理的原始数据，默认为 False。
        :return: 包含 'NLSIZE' 因子的 DataFrame。
        """
        df = self.LNCAP(df)
        df['SIZE_CUBED'] = np.power(df['LNCAP'], 3)
        df['CONSTANT'] = 1

        df = df.dropna(how='any')

        X = df[['LNCAP','CONSTANT']]
        y = df['SIZE_CUBED']

        model = sm.OLS(y, X).fit()
        df['NLSIZE'] = model.resid

        if not raw:
            df = self._preprocess(df, factor_column = 'NLSIZE')

        df.drop(columns=['SIZE_CUBED', 'LNCAP', 'CONSTANT'], inplace=True)

        return df


# def risk_free():
#     """
#     获取无风险利率（十年国债收益率）
#     :return: 无风险利率数据框 格式：日期，年化收益
#     """
#     current_df_start_time = datetime.strptime(START_DATE, "%Y%m%d")
#     end_date_time = datetime.strptime(END_DATE, "%Y%m%d")
#     yield10yr_df = pd.DataFrame()
#
#     while current_df_start_time < end_date_time:
#         current_df_end_time = min(current_df_start_time + timedelta(days=365), end_date_time)
#
#         bond_china_yield_df = ak.bond_china_yield(
#             start_date=current_df_start_time.strftime("%Y%m%d"),
#             end_date=current_df_end_time.strftime("%Y%m%d")
#         )
#
#         filtered_df = bond_china_yield_df[
#             (bond_china_yield_df['曲线名称'] == '中债国债收益率曲线')
#         ][['日期', '10年']]
#
#         yield10yr_df = pd.concat([yield10yr_df, filtered_df])
#
#         current_df_start_time = current_df_end_time + timedelta(days=1)
#
#     yield10yr_df.reset_index(drop=True, inplace=True)
#     yield10yr_df['RF_RETURN_ANN'] = yield10yr_df['10年'] / 100
#     yield10yr_df['TRADE_DT'] = pd.to_datetime(yield10yr_df['日期'])
#     yield10yr_df['RF_RETURN'] = (1 + yield10yr_df['RF_RETURN_ANN']) ** (1 / 252) - 1
#
#     rf = yield10yr_df[['TRADE_DT', 'RF_RETURN']]
#
#     return rf

if __name__ == '__main__':
    total_data = GetData._get_price_()
    total_data = total_data[total_data['year']>=2016]
    valuation_data = GetData._get_valuation()
    size_calculator = Size()

    Beta_data = Beta(total_data)
    beta_df = Beta_data.beta_df
    print("test")