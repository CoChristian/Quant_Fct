#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 14:00:38 2021

@author: Yitao Hu
"""
import pandas as pd
if not hasattr(pd.DataFrame, 'iteritems'):
    pd.DataFrame.iteritems = pd.DataFrame.items
import alphalens as al
import datetime

import numpy as np
import matplotlib.pyplot as plt
import tqdm
import model.factor_pipeline as fp
import jqdatasdk as jd
import pdb
jd.auth("13764432461", "Nfhq12345")
from model.util import get_data_from_multi_source, ts_code_to_jq_code

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


class AlphaEvaluator():
    # def _iteritems(self, df):
    #     """兼容 pandas 新旧版本的列迭代器"""
    #     if hasattr(df, 'iteritems'):
    #         return df.iteritems()
    #     else:
    #         return df.items()


    """Evaluate single factor to check out its validness"""
    def __init__(self,factor,quantiles=10,demean = True,equal_weight=False,eval_child_factors = True,prices = None, freq='W_Tue'):
        
        """
        Read in trade_date, stock specific factor values and prices, preprocess
        and compute forward returns for each stock for each period for further research
        The prices info must cover all the trade dates of the factors.
        Args:
        ----------
        factor: factor_pipline Factor instance after running pipline 
            factor to be evaluated 
        eval_child_factors : bool indicating whether evaluate the factor itself or its children factors 
        quantiles: int 
            number of quantiles cut in alpha factors
        
        """
        # set some flags
        self.freq = freq
        self.demeaned = demean
        self.equal_weight = equal_weight
        self.factor = factor
        alpha_factors = self.factor.copy()
        # change factor date type 
        # if eval_child_factors:
        #     alpha_factors = self.factor.children_factor_value.reset_index().copy()
        # else:
        #     alpha_factors = self.factor.fac_value.reset_index().copy()
        # alpha_factors['trade_date'] = pd.to_datetime(alpha_factors['trade_date'].astype(str))
        alpha_factors['trade_date'] = pd.to_datetime(alpha_factors['trade_date'].astype(str))
        self.alpha_factors = alpha_factors.copy()
        # self.alpha_factors = alpha_factors
        # add benchamrk return 
        self.benchmark_rtn = self.get_benchmark_return(benchmark_code = '000905.sh')
        time_index = pd.to_datetime(self.alpha_factors['trade_date'].unique().astype(str))
        if prices is None:
            # get the price info 
            # self.price_factor = fp.AdjClosePrice(start_date = self.factor.trade_date.min(),
            #                                      end_date = self.factor.trade_date.max(),
            #                                      freq = self.freq).compute()
            #
            # prices = self.price_factor.fac_value.reset_index().copy()

            prices = get_price_table(start_date = 20231201,end_date = 20260120)
            prices['trade_date'] = pd.to_datetime(prices['trade_date'].astype(str))
            self.prices = prices

        else:
            self.prices = prices

        # if self.freq == 'W_Tue':
        #     factor_name = self.factor.columns[-1]
        #     date_table = get_opt2trade_table(start_date=self.factor.trade_date.min(),
        #                                      end_date=self.factor.trade_date.max())
        #     date_table['opt_date'] = pd.to_datetime(date_table['opt_date'].astype(str))
        #     date_table['trade_date_price'] = pd.to_datetime(date_table['trade_date_price'].astype(str))
        #
        #     self.alpha_factors = pd.merge(self.alpha_factors, date_table, left_on='trade_date', right_on='opt_date', how='inner')
        #     self.alpha_factors = self.alpha_factors[['trade_date','code',factor_name]].copy()


        self.factor_returns = None
        self.qr_factor_returns = None
        self.FRA = None
        self.RIC = None
        # prepare the demeaned alpha format to use in alphien 
        self.alpha_factors = self.alpha_factors.set_index('trade_date')
        self.alpha_factors.index = self.alpha_factors.index.tz_localize(tz='Asia/Shanghai')
        self.alpha_factors= self.alpha_factors.reset_index().set_index(['trade_date','code'])
        
        # prepare the price data format to use in alphien

        # self.prices = self.prices.set_index('trade_date')
        # self.prices.index = self.prices.index.tz_localize(tz='Asia/Shanghai')
        # self.prices = self.prices.reset_index().groupby(['trade_date','code']).mean().unstack()
        # self.prices.columns = self.prices.columns.get_level_values(1)
        # self.prices = self.prices.dropna(how = 'all')
        # # align the time stock index with that of alpha factors
        # self.prices = self.prices.stack().\
        #         reset_index().set_index(['trade_date','code']).\
        #             reindex(self.alpha_factors.index).unstack()
        # self.prices = self.prices.dropna(how='all')

        self.prices = self.prices.groupby(['trade_date', 'code'], as_index=False)['close0935'].mean()
        # 转为宽表：索引=日期，列=股票代码
        self.prices = self.prices.pivot(index='trade_date', columns='code', values='close0935')
        # 时区处理
        self.prices.index = self.prices.index.tz_localize('Asia/Shanghai')
        # 删除全为 NaN 的行（可选）
        self.prices = self.prices.dropna(how='all')


        # self.prices = self.prices.groupby(['trade_date', 'code'], as_index=False)['close0935'].mean()
        # # 转为宽表
        # self.prices = self.prices.pivot(index='trade_date', columns='code', values='close0935')
        # # 添加时区
        # self.prices.index = self.prices.index.tz_localize('Asia/Shanghai')
        # # 删除全空行
        # self.prices = self.prices.dropna(how='all')

        # self.prices = self.prices.stack().\
        #         reset_index().set_index(['trade_date','code']).unstack()      
        # self.prices.columns = self.prices.columns.get_level_values(1)
        
        # preprocess all the factor and forward returns with alphien

        factor_column = self.alpha_factors.columns[-1]  # 因子列名
        # factor_series = self.alpha_factors.set_index(['trade_date', 'code'])[factor_column]
        # 确保索引名称为 alphalens 要求的 'date' 和 'asset'
        # factor_series.index.names = ['date', 'asset']
        # # 时区处理（索引中 date 需要时区）
        # factor_series.index = factor_series.index.set_levels(
        #     factor_series.index.levels[0].tz_localize('Asia/Shanghai'), level=0
        # )

        # 直接调用 alphalens
        clean_data = al.utils.get_clean_factor_and_forward_returns(
            factor=self.alpha_factors,
            prices=self.prices,
            periods=[5],
            quantiles=quantiles,
            max_loss=1.0
        )
        self.clean_factor_data = {factor_column: clean_data}




        # self.clean_factor_data = {
        # factor: al.utils.get_clean_factor_and_forward_returns(factor=factor_data, \
        #                                                       prices=self.prices, \
        #                                                           periods=[5],
        #                                                      quantiles=quantiles,
        #                                                      max_loss=1.0)
        # for factor, factor_data in self.alpha_factors.iteritems()}


            
    def get_benchmark_return(self,benchmark_code = '000905.sh'):
        """
        Compute the benchmark return for the same factor test period 

        Parameters
        ----------
        benchmark_code : str, optional
            code for benchmark stock index. The default is '000905.sh'.

        Returns
        -------
        benchmark_rtn : pd.Series
            stock index return.

        """
        
        # get the time index 
        time_index = pd.to_datetime(self.alpha_factors['trade_date'].unique().astype(str))
        benchmark_price = jd.get_price(jd.normalize_code('000905.sh'),start_date=time_index.min(),end_date=time_index.max(),fields=['close'])
        # compute benchamrk return
        benchmark_rtn = benchmark_price.reindex(time_index).pct_change()
        # localize the time zone
        benchmark_rtn.index = benchmark_rtn.index.tz_localize(tz='Asia/Shanghai')
        # change columns name and get pd.Series
        benchmark_rtn = benchmark_rtn.rename(columns = {'close':'benchmark'}).iloc[:,0]
        benchmark_rtn = benchmark_rtn.shift(-1).dropna()
        return benchmark_rtn
    
    def cpt_fac_rtn(self):
        """
        Compute factor returns, 
        if self.demeaned is True, compute returns of dollar-neutral long-short portfolio of level 1
        
        if self.demeaned is True, compute returns of long-only portfolio with factor values as porpotional weight and short benchamrk 
        
        """
        # compute factor returns
        self.factor_returns = pd.DataFrame()
        
        for factor, factor_data in tqdm.tqdm(self.clean_factor_data.items()):
            if self.demeaned:
                self.factor_returns[factor] = al.performance.\
                    factor_returns(factor_data,demeaned = self.demeaned,equal_weight=self.equal_weight)\
                        .iloc[:, 0]
            else:
                self.factor_returns[factor] = al.performance.\
                    factor_returns(factor_data,demeaned = self.demeaned,equal_weight=self.equal_weight)\
                        .iloc[:, 0] - self.benchmark_rtn
    def show_cum_rtn(self,demean = True,equal_weight=False):
        
        """Plot the cumulative returns for all the factors-weighted Portfolios"""
        if self.factor_returns is None:
            self.cpt_fac_rtn()
        (1+self.factor_returns).cumprod().plot(figsize = (12,5))
        plt.title('Cumulative returns');
        
    def cpt_mean_quantile_rtn(self,by_date = False):
        """Compute mean returns by quantiles"""
        self.qr_factor_returns = pd.DataFrame()

        for factor, factor_data in tqdm.tqdm(self.clean_factor_data.items()):
            self.qr_factor_returns[factor] = al.performance.mean_return_by_quantile(factor_data,by_date = by_date)[0].iloc[:, 0]
    
    def show_quantile_plot(self):
        """Show quantile returns for all the factors-weighted Portfolios"""
        self.cpt_mean_quantile_rtn()
        
        
        n_fac = self.alpha_factors.shape[1] 
        
        (10000*self.qr_factor_returns).plot.bar(
        subplots=True,
        sharey=True,
        layout=(int(n_fac/2)+1,2),
        figsize=(14, 14),
        legend=False);
        
        
    def cpt_FRA(self):
        """Compute Factor Ranking AutoCorrelation as a proxy of Turnover Analysis"""
        self.FRA = pd.DataFrame()

        for factor, factor_data in tqdm.tqdm(self.clean_factor_data.items()):
            self.FRA[factor] = al.performance.factor_rank_autocorrelation(factor_data)
    
    def show_FRA(self):
        """
        Show the plot of Factor Ranking AutoCorrelation
        """
        self.cpt_FRA()
        
        
        n_fac = self.alpha_factors.shape[1] 
        
        (self.FRA).plot(
        title="Factor Rank Autocorrelation",
        subplots=True,
        sharey=True,
        layout=(int(n_fac/2)+1,2),
        figsize=(14, 14));
    
    def cpt_RIC(self):
        """Compute the Rank Information Coefficient for each factor to check out 
        its predictabity"""
        self.RIC = pd.DataFrame()

        for factor, factor_data in tqdm.tqdm(self.clean_factor_data.items()):
            self.RIC[factor] = al.performance.factor_information_coefficient(factor_data).reset_index().iloc[:,1]
            
    def show_RIC(self):
        self.cpt_RIC()
        
        n_fac = self.alpha_factors.shape[1] 
        
        (self.RIC).plot(
        title="Factor Information Coefficient",
        subplots=True,
        sharey=True,
        layout=(int(n_fac/2)+1,2),
        figsize=(14, 14));
    def show_factor_corr(self):
        self.factor_corr = self.alpha_factors.groupby(level = 0).corr().unstack(level = 1).mean().unstack()
        return self.factor_corr
    def show_tear_sheet(self):
        if self.factor_returns is  None:
            self.cpt_fac_rtn()
        if self.FRA is  None:
            self.cpt_FRA()
        if self.RIC is  None:
            self.cpt_RIC()
        
        
        self.tear_sheet = pd.DataFrame()
        self.tear_sheet['AnnualizedMean'] = self.factor_returns.mean()* 52
        self.tear_sheet['AnnualizedVol'] = self.factor_returns.std()* np.sqrt(52)
        self.tear_sheet['AnnualizedSharpeRatio'] = self.tear_sheet['AnnualizedMean'] /self.tear_sheet['AnnualizedVol']
        self.tear_sheet['MeanFRA'] = self.FRA.mean()
        self.tear_sheet['MeanIC'] = self.RIC.mean()
        
        return self.tear_sheet
                        
                
# if __name__ == "__main__":
#
#
#     Alpha_calculator = AlphaEvaluator(factor=)