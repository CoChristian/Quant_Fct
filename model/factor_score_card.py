import pandas as pd
from tqdm import tqdm
import math
from scipy.optimize import curve_fit
import numpy as np
from sklearn.linear_model import LinearRegression
from indicator_operator import FactorCompute, save_data_to_table
from func_operator import discretize
from factor_neutral import get_data_from_multi_source, transfer_data_to_valid_and_not_valid
from factor_regression import generate_one_term_return, process_not_valid_data_and_merge, process_one_term_return

def gen_rank_info(data, bin_infos, y_name):
    new_datas = []
    rank_return_infos = {bin_info['name']: [] for bin_info in bin_infos}

    for trade_date, tmp_data in data.groupby('trade_date'):
        tmp_data['alpha'] = tmp_data[y_name] - tmp_data[y_name].mean()
        tmp_data.sort_values('alpha', inplace=True)
        tmp_data['alpha_rank'] = [int(j*10/len(tmp_data)) for j in range(len(tmp_data))]
        alpha_rank_score = tmp_data.groupby('alpha_rank').mean()['alpha'].to_dict()
        tmp_data['alpha'] = tmp_data['alpha_rank'].map(alpha_rank_score)
        for bin_info in bin_infos:
            type_ = bin_info['type']
            feature_name = bin_info['name']
            if type_ == "continuous":
                bin_count = bin_info['bin_count']
                tmp_data.sort_values(feature_name, inplace=True)
                batch_size = len(tmp_data) / bin_count
                tmp_data['%s_rank' % feature_name] = [int(j / batch_size) for j in range(len(tmp_data))]
                tmp_data['%s_rank' % feature_name] = tmp_data['%s_rank' % feature_name].map(lambda x: x - bin_count / 2 + 0.5)
                rank_return = tmp_data.groupby('%s_rank' % feature_name).mean()['alpha']
                tmp_data["%s_alpha" % feature_name] = tmp_data['%s_rank' % feature_name].map(rank_return.to_dict())
                rank_return_infos[feature_name].append(rank_return)
            elif type_ == "discrete":
                bin_to_rank = bin_info['bin_to_rank']
                tmp_data['%s_rank' % feature_name] = tmp_data[feature_name].map(bin_to_rank)
                rank_return = tmp_data.groupby('%s_rank' % feature_name).mean()['r']
                tmp_data["%s_alpha" % feature_name] = tmp_data['%s_rank' % feature_name].map(rank_return.to_dict())
                rank_return_infos[feature_name].append(rank_return)
        new_datas.append(tmp_data)
    new_data = pd.concat(new_datas)
    return new_data, rank_return_infos

def poly1d_func(x, a, b):
    return a*x+b

def poly2d_func(x, a, b, c):
    return a*x**2+b*x+c

def generate_score_card(all_factor_data, start_date, end_date, window_size, r_name, overall_factor_name, bin_infos, lr_clf):
    all_factor_data['LiquidityVolatilityFactor'] = all_factor_data['LiquidityFactor'] +all_factor_data['VolatilityFactor']

    new_data, rank_return_infos = gen_rank_info(all_factor_data, bin_infos, r_name)

    new_data = new_data.reset_index()
    trade_dates = sorted(new_data['trade_date'].unique())
    factors = [bin_info['name'] for bin_info in bin_infos]
    score_infos = []
    for i, date in enumerate(tqdm(trade_dates)):
        if i >= window_size and date > start_date and date <= end_date:
            hist_dates = trade_dates[i-window_size: i]
            hist_factor_data = new_data[new_data.trade_date.map(lambda x: x in hist_dates)].copy()
            tmp_factor = new_data[new_data.trade_date == date].copy()
            try:
                lr_clf.fit(hist_factor_data[["%s_alpha" % _ for _ in factors]].values, hist_factor_data['alpha'])
            except Exception as e:
                import pdb
                pdb.set_trace()
                pass
            beta_s = lr_clf.coef_
            beta_info = dict(zip(factors, beta_s))
            for bin_info in bin_infos:
                factor = bin_info['name']
                type_ = bin_info['type']
                rank_scores = rank_return_infos[factor][i-window_size: i]
                rank_score = pd.concat(rank_scores, axis=1).T.mean()
                if type_ == "continuous":
                    bin_count = bin_info['bin_count']
                    shift_rank = [j-bin_count/2+0.5 for j in range(bin_count)]
                    if factor == "GrowthFactor":
                        param, pcov = curve_fit(poly1d_func, rank_score.index.values.astype(np.float), rank_score.values, p0=[1e-4,0], bounds=([0, -np.inf], [np.inf,np.inf]))
                        scores = {j: poly1d_func(j, *param) for j in shift_rank}
                    elif factor == "QualityFactor":
                        param, pcov = curve_fit(poly1d_func, rank_score.index.values.astype(np.float), rank_score.values, p0=[1e-4,0], bounds=([0, -np.inf], [np.inf,np.inf]))

                        scores = {j: poly1d_func(j, *param) for j in shift_rank}
                    elif factor == "ValueFactor":
                        param, pcov = curve_fit(poly1d_func, rank_score.index.values.astype(np.float), rank_score.values, p0=[1e-4,0], bounds=([0, -np.inf], [np.inf,np.inf]))

                        scores = {j: poly1d_func(j, *param) for j in shift_rank}

                    elif factor == "LeverageFactor":
                        param, pcov = curve_fit(poly1d_func, rank_score.index.values.astype(np.float), rank_score.values, p0=[0,0], bounds=([-np.inf, -np.inf], [np.inf,np.inf]))

                        scores = {j: poly1d_func(j, *param) for j in shift_rank}
                    elif factor == "OverallMomentumFactor":
                        param, pcov = curve_fit(poly2d_func, rank_score.index.values.astype(np.float), rank_score.values, p0=[-1e-4,0, 0], bounds=([-np.inf, -np.inf, -np.inf], [0, 0, np.inf]))
                        scores = {j: poly2d_func(j, *param) for j in shift_rank}
                    elif factor == "LiquidityFactor":

                        param, pcov = curve_fit(poly2d_func, rank_score.index.values.astype(np.float), rank_score.values, p0=[-1e-4,0, 0], bounds=([-np.inf, -np.inf, -np.inf], [0,np.inf, np.inf]))

                        scores = {j: poly2d_func(j, *param) for j in shift_rank}
                    elif factor == "VolatilityFactor":
                        param, pcov = curve_fit(poly1d_func, rank_score.index.values.astype(np.float), rank_score.values, p0=[1e-4,0], bounds=([0, -np.inf], [np.inf,np.inf]))
                        scores = rank_score.values
                    elif factor == "LiquidityVolatilityFactor":
                        param, pcov = curve_fit(poly2d_func, rank_score.index.values.astype(np.float), rank_score.values, p0=[-1e-4,0, 0], bounds=([-np.inf, 0, -np.inf], [0,np.inf, np.inf]))

                        scores = {j: poly2d_func(j, *param) for j in shift_rank}

                    else:
                        pass
                elif type_ == "discrete":
                    scores = rank_score.values
                tmp_factor["%s_pred_score" % factor] = tmp_factor["%s_rank" % factor].map(lambda x: scores[x] * beta_info[factor])
            tmp_factor[overall_factor_name] = tmp_factor[["%s_pred_score" % _ for _ in factors]].sum(axis=1)
            score_infos.append(tmp_factor)
    overall_score_df= pd.concat(score_infos)
    return overall_score_df


def _OLS_estimate_fac_premium(fac_values, stock_returns):
    """Perform cross-sectional OLS Regression to estimate factor premiums and alphas.
        Factor premiums are estimated by OLS analytical solution.

        Args:
        ----------
            fac_values: pd.DataFrame
                Factor exposure for each stock at time t-1
                N by f matrix where N is the number of stock and f is the number of factor
                - Example:
                              Log_mkt_Cap  winsorized_Log_Cap
                        0       24.258083           24.258083
                        1       22.527445           22.527445
                        2       24.496254           24.496254
                        3       23.340478           23.340478
                        4       22.067832           22.067832

            stock_returns: pd.Series
                returns for each stock from t-1 to t
                N by 1 vector
                - Example:
                        0       0.001961
                        1       0.002232
                        2      -0.022283
                        3      -0.007605
                        4       0.009877
        Returns:
        ----------
            fac_premum: pd.Series
                       (f+1) by 1 vector. The first element is the alpha and the rest are factor premium for each factor
                        - Example:
                        0       0.001961
                        1       0.002232
                        2      -0.022283
                        3      -0.007605
                        4       0.009877
            """
    # extract the number of stock and number of fac
    fac_values = pd.DataFrame(fac_values)
    num_stock = fac_values.shape[0]
    # print("num stock {}".format(num_stock))
    num_fac = fac_values.shape[1]
    # reshape fac_values and stock_returns for linear alg operation
    S_mat = fac_values.values.reshape((num_stock, num_fac))
    r_vec = stock_returns.values.reshape(num_stock)
    ## add bias terms
    bias_term = np.ones((num_stock, 1))
    S_mat = np.concatenate((bias_term, S_mat), axis=1)
    # compute OLS analytical solution
    try:
        lambda_est = pd.Series(np.linalg.inv(S_mat.T @ S_mat) @ S_mat.T @ r_vec)
        # print(lambda_est)
    except Exception as e:
        result = np.linalg.lstsq(S_mat, r_vec)
        lambda_est = pd.Series(result[0])

    return lambda_est


def weekly_performance_attribution(factor_data, factors, industry):
    factor_premium_infos = []
    processed_factor_infos = []
    if "trade_date" not in factor_data.columns:
        factor_data = factor_data.reset_index()

    for trade_date, tmp_df in factor_data.groupby('trade_date'):
        for factor in factors:
            if factor in ['GrowthFactor', 'LeverageFactor', 'ValueFactor', 'QualityFactor']:
                tmp_df[factor] = tmp_df[factor].rank(pct=True).map(lambda x: math.ceil(3 * x) / 2 - 1)
            elif factor in ['LogMktCap', 'SizeFactor']:
#                 tmp_df[factor] = tmp_df[factor].rank(pct=True, ascending=False).map(lambda x: math.ceil(3 * x) / 2 - 1)
#                 tmp_df[factor] = tmp_df[factor].rank(pct=True, ascending=False).map(lambda x: math.ceil(10 * x) / 10 - 0.5)
                tmp_df[factor] = tmp_df[factor].rank(pct=True, ascending=False).map(lambda x: 0.5 if x > 0.1 else -0.5)

            elif factor in ['LiquidityFactor', 'ShortMomentumFactorReverse']:
                tmp_df[factor] = tmp_df[factor].rank(pct=True).map(lambda x: 0.5 if x > 0.2 else -0.5)
            elif factor in ['VolatilityFactor']:
                tmp_df[factor] = tmp_df[factor].rank(pct=True).map(lambda x: -0.5 if x > 0.8 else 0.5)
            elif factor in ['LongMomentumFactorReverse']:
                def std_long_momentum(x):
                    if x > 0.7:
                        return -0.5
                    elif x > 0.4:
                        return 0
                    elif x > 0.1:
                        return 0.5
                    else:
                        return -0.5

                tmp_df[factor] = tmp_df[factor].rank(pct=True).map(lambda x: std_long_momentum(x))
            elif factor in ['Weeks50CountLog']:
#                 tmp_df["{}Std".format(factor)] = tmp_df.groupby(industry)[factor].apply(lambda x: (x-x.mean())/(x.std()+1e-8))
#                 tmp_df["{}Std".format(factor)] = tmp_df["{}Std".format(factor)].fillna(0)
                top_90 = tmp_df['Weeks50CountLog'].quantile(0.9)
                def std_leading_factor(x, top_90):
                    if x > top_90:
                        return 0.5
                    elif x > -1:
                        return 0
                    else:
                        return -0.5
                tmp_df['Weeks50CountLog'] = tmp_df['Weeks50CountLog'].map(lambda x: std_leading_factor(x, top_90))
            else:
                pass
        processed_factor_infos.append(tmp_df.copy())
        if tmp_df['OneTermReturn'].notnull().sum() != 0:
            factor_premium = _OLS_estimate_fac_premium(tmp_df[factors], tmp_df['OneTermReturn'].fillna(0))
            factor_premium.index = ['bias'] + factors

            factor_premium_info = factor_premium.to_dict()
            factor_premium_info.update({'trade_date': trade_date})
            factor_premium_infos.append(factor_premium_info)

    factor_premium_df = pd.DataFrame(factor_premium_infos)
    processed_factor_df = pd.concat(processed_factor_infos)
#     factor_premium_df.to_excel("factor_premium_all_mkt.xlsx")
#     processed_factor_df.to_pickle("processed_factor_all_mkt.pkl")

    return factor_premium_df, processed_factor_df


def weekly_performance_attribution_bin_replace(factor_data, factors, industry):
    factor_premium_infos = []
    processed_factor_infos = []
    if "trade_date" not in factor_data.columns:
        factor_data = factor_data.reset_index()

    for trade_date, tmp_df in factor_data.groupby('trade_date'):
        for factor in factors:
            if factor in ['GrowthFactor', 'LeverageFactor', 'ValueFactor', 'QualityFactor']:
#                 tmp_df[factor] = tmp_df[factor].rank(pct=True).map(lambda x: math.ceil(3 * x) / 2 - 1)
                tmp_df['bin'] = tmp_df[factor].rank(pct=True).map(lambda x: math.ceil(3*x))
                tmp_df[factor] = tmp_df[factor].rank(pct=True).map(lambda x: x-0.5)
                bin_2_value = tmp_df.groupby('bin').mean()[factor].to_dict()
                tmp_df[factor] = tmp_df['bin'].map(bin_2_value)
                
            elif factor in ['LogMktCap', 'SizeFactor']:
#                 tmp_df[factor] = tmp_df[factor].rank(pct=True, ascending=False).map(lambda x: math.ceil(3 * x) / 2 - 1)
#                 tmp_df[factor] = tmp_df[factor].rank(pct=True, ascending=False).map(lambda x: math.ceil(10 * x) / 10 - 0.5)
#                 tmp_df[factor] = tmp_df[factor].rank(pct=True, ascending=False).map(lambda x: 0.5 if x > 0.1 else -0.5)
                tmp_df['bin'] = tmp_df[factor].rank(pct=True, ascending=False).map(lambda x: math.ceil(3*x))
                tmp_df[factor] = tmp_df[factor].rank(pct=True, ascending=False).map(lambda x: x-0.5)
                bin_2_value = tmp_df.groupby('bin').mean()[factor].to_dict()
                tmp_df[factor] = tmp_df['bin'].map(bin_2_value)
            elif factor in ['LiquidityFactor', 'ShortMomentumFactorReverse']:
#                 tmp_df[factor] = tmp_df[factor].rank(pct=True).map(lambda x: 0.5 if x > 0.2 else -0.5)
                tmp_df['bin'] = tmp_df[factor].rank(pct=True).map(lambda x: 1 if x > 0.2 else 0)
                tmp_df[factor] = tmp_df[factor].rank(pct=True).map(lambda x: x-0.5)
                bin_2_value = tmp_df.groupby('bin').mean()[factor].to_dict()
                tmp_df[factor] = tmp_df['bin'].map(bin_2_value)
                
            elif factor in ['VolatilityFactor']:
#                 tmp_df[factor] = tmp_df[factor].rank(pct=True).map(lambda x: -0.5 if x > 0.8 else 0.5)
                tmp_df['bin'] = tmp_df[factor].rank(pct=True, ascending=False).map(lambda x: 1 if x > 0.2 else 0)
                tmp_df[factor] = tmp_df[factor].rank(pct=True, ascending=False).map(lambda x: x-0.5)
                bin_2_value = tmp_df.groupby('bin').mean()[factor].to_dict()
                tmp_df[factor] = tmp_df['bin'].map(bin_2_value)
                
            elif factor in ['LongMomentumFactorReverse']:
#                 def std_long_momentum(x):
#                     if x > 0.7:
#                         return -0.5
#                     elif x > 0.4:
#                         return 0
#                     elif x > 0.1:
#                         return 0.5
#                     else:
#                         return -0.5

#                 tmp_df[factor] = tmp_df[factor].rank(pct=True).map(lambda x: std_long_momentum(x))
                tmp_df['bin'] = tmp_df[factor].rank(pct=True, ascending=False).map(lambda x: math.ceil(3*x))
                tmp_df[factor] = tmp_df[factor].rank(pct=True, ascending=False).map(lambda x: x-0.5)
                bin_2_value = tmp_df.groupby('bin').mean()[factor].to_dict()
                tmp_df[factor] = tmp_df['bin'].map(bin_2_value)
            elif factor in ['Weeks50CountLog']:
#                 tmp_df["{}Std".format(factor)] = tmp_df.groupby(industry)[factor].apply(lambda x: (x-x.mean())/(x.std()+1e-8))
#                 tmp_df["{}Std".format(factor)] = tmp_df["{}Std".format(factor)].fillna(0)
                top_90 = tmp_df['Weeks50CountLog'].quantile(0.9)
                def std_leading_factor(x, top_90):
                    if x > top_90:
                        return 1
                    elif x > -1:
                        return 0
                    else:
                        return -1
                tmp_df['bin'] = tmp_df['Weeks50CountLog'].map(lambda x: std_leading_factor(x, top_90))
                tmp_df[factor] = tmp_df[factor].rank(pct=True).map(lambda x: x-0.5)
                bin_2_value = tmp_df.groupby('bin').mean()[factor].to_dict()
                tmp_df[factor] = tmp_df['bin'].map(bin_2_value)     
            else:
                pass
        processed_factor_infos.append(tmp_df.copy())
        if tmp_df['OneTermReturn'].notnull().sum() != 0:
            factor_premium = _OLS_estimate_fac_premium(tmp_df[factors], tmp_df['OneTermReturn'].fillna(0))
            factor_premium.index = ['bias'] + factors

            factor_premium_info = factor_premium.to_dict()
            factor_premium_info.update({'trade_date': trade_date})
            factor_premium_infos.append(factor_premium_info)

    factor_premium_df = pd.DataFrame(factor_premium_infos)
    processed_factor_df = pd.concat(processed_factor_infos)

#     factor_premium_df.to_excel("factor_premium_all_mkt.xlsx")
#     processed_factor_df.to_pickle("processed_factor_all_mkt.pkl")

    return factor_premium_df, processed_factor_df

def weekly_performance_attribution_balance(factor_data, factors, industry):
    factor_premium_infos = []
    processed_factor_infos = []
    if "trade_date" not in factor_data.columns:
        factor_data = factor_data.reset_index()

    for trade_date, tmp_df in factor_data.groupby('trade_date'):
        for factor in factors:
            if factor == "GrowthFactor":
                tmp_df[factor] = tmp_df[factor].rank(pct=True).map(lambda x: math.ceil(11 * x) / 10 - 0.6)

            if factor in ['GrowthFactor', 'LeverageFactor', 'ValueFactor', 'QualityFactor']:
                tmp_df[factor] = tmp_df[factor].rank(pct=True).map(lambda x: math.ceil(3 * x) / 2 - 1)
            elif factor in ['LogMktCap', 'SizeFactor']:
#                 tmp_df[factor] = tmp_df[factor].rank(pct=True, ascending=False).map(lambda x: math.ceil(3 * x) / 2 - 1)
#                 tmp_df[factor] = tmp_df[factor].rank(pct=True, ascending=False).map(lambda x: math.ceil(10 * x) / 10 - 0.5)
                tmp_df[factor] = tmp_df[factor].rank(pct=True, ascending=False).map(lambda x: 0.5 if x > 0.1 else -0.5)

            elif factor in ['LiquidityFactor', 'ShortMomentumFactorReverse']:
                tmp_df[factor] = tmp_df[factor].rank(pct=True).map(lambda x: 0.5 if x > 0.2 else -0.5)
            elif factor in ['VolatilityFactor']:
                tmp_df[factor] = tmp_df[factor].rank(pct=True).map(lambda x: -0.5 if x > 0.8 else 0.5)
            elif factor in ['LongMomentumFactorReverse']:
                def std_long_momentum(x):
                    if x > 0.7:
                        return -0.5
                    elif x > 0.4:
                        return 0
                    elif x > 0.1:
                        return 0.5
                    else:
                        return -0.5

                tmp_df[factor] = tmp_df[factor].rank(pct=True).map(lambda x: std_long_momentum(x))
            elif factor in ['Weeks50CountLog']:
#                 tmp_df["{}Std".format(factor)] = tmp_df.groupby(industry)[factor].apply(lambda x: (x-x.mean())/(x.std()+1e-8))
#                 tmp_df["{}Std".format(factor)] = tmp_df["{}Std".format(factor)].fillna(0)
#                 top_90 = tmp_df['Weeks50CountLog'].quantile(0.9)
#                 def std_leading_factor(x, top_90):
#                     if x > top_90:
#                         return 0.5
#                     elif x > -1:
#                         return 0
#                     else:
#                         return -0.5
                tmp_df_1 = tmp_df[tmp_df['Weeks50CountLog'] > -1]
                tmp_df_2 = tmp_df[tmp_df['Weeks50CountLog'] == -1]
                tmp_df_1['Weeks50CountLog'] = tmp_df_1['Weeks50CountLog'].rank(pct=True).map(lambda x: math.ceil(11 * x) / 10 - 0.6)
                tmp_df = pd.concat([tmp_df_1, tmp_df_2])
            else:
                pass
        processed_factor_infos.append(tmp_df.copy())
        if tmp_df['OneTermReturn'].notnull().sum() != 0:
            factor_premium = _OLS_estimate_fac_premium(tmp_df[factors], tmp_df['OneTermReturn'].fillna(0))
            factor_premium.index = ['bias'] + factors

            factor_premium_info = factor_premium.to_dict()
            factor_premium_info.update({'trade_date': trade_date})
            factor_premium_infos.append(factor_premium_info)

    factor_premium_df = pd.DataFrame(factor_premium_infos)
    processed_factor_df = pd.concat(processed_factor_infos)
    # factor_premium_df.to_excel("factor_premium_all_mkt.xlsx")
    # processed_factor_df.to_pickle("processed_factor_all_mkt.pkl")

    return factor_premium_df, processed_factor_df

def weekly_performance_attribution_stable(factor_data, factors, industry):
    factor_premium_infos = []
    processed_factor_infos = []
    if "trade_date" not in factor_data.columns:
        factor_data = factor_data.reset_index()

    for trade_date, tmp_df in factor_data.groupby('trade_date'):
        for factor in factors:
            if factor == "GrowthFactor":
                tmp_df[factor] = tmp_df[factor].rank(pct=True).map(lambda x: math.ceil(11 * x) / 10 - 0.6)

            if factor in ['LeverageFactor', 'ValueFactor', 'QualityFactor']:
                tmp_df[factor] = tmp_df[factor].rank(pct=True).map(lambda x: math.ceil(3 * x) / 2 - 1)
            elif factor in ['LogMktCap', 'SizeFactor']:
#                 tmp_df[factor] = tmp_df[factor].rank(pct=True, ascending=False).map(lambda x: math.ceil(3 * x) / 2 - 1)
#                 tmp_df[factor] = tmp_df[factor].rank(pct=True, ascending=False).map(lambda x: math.ceil(10 * x) / 10 - 0.5)
                tmp_df[factor] = tmp_df[factor].rank(pct=True, ascending=False).map(lambda x: 0.5 if x > 0.1 else -0.5)

            elif factor in ['LiquidityFactor', 'ShortMomentumFactorReverse']:
                tmp_df[factor] = tmp_df[factor].rank(pct=True).map(lambda x: 0.5 if x > 0.2 else -0.5)
            elif factor in ['VolatilityFactor']:
                tmp_df[factor] = tmp_df[factor].rank(pct=True).map(lambda x: -0.5 if x > 0.8 else 0.5)
            elif factor in ['LongMomentumFactorReverse']:
                def std_long_momentum(x):
                    if x > 0.7:
                        return -0.5
                    elif x > 0.4:
                        return 0
                    elif x > 0.1:
                        return 0.5
                    else:
                        return -0.5

                tmp_df[factor] = tmp_df[factor].rank(pct=True).map(lambda x: std_long_momentum(x))
            elif factor in ['Weeks50CountLog']:
#                 tmp_df["{}Std".format(factor)] = tmp_df.groupby(industry)[factor].apply(lambda x: (x-x.mean())/(x.std()+1e-8))
#                 tmp_df["{}Std".format(factor)] = tmp_df["{}Std".format(factor)].fillna(0)
#                 top_90 = tmp_df['Weeks50CountLog'].quantile(0.9)
#                 def std_leading_factor(x, top_90):
#                     if x > top_90:
#                         return 0.5
#                     elif x > -1:
#                         return 0
#                     else:
#                         return -0.5
                tmp_df_1 = tmp_df[tmp_df['Weeks50CountLog'] > -1]
                tmp_df_2 = tmp_df[tmp_df['Weeks50CountLog'] == -1]
                tmp_df_1['Weeks50CountLog'] = tmp_df_1['Weeks50CountLog'].rank(pct=True).map(lambda x: math.ceil(3 * x) / 2 - 1)
                tmp_df = pd.concat([tmp_df_1, tmp_df_2])
            else:
                pass
        processed_factor_infos.append(tmp_df.copy())
        if tmp_df['OneTermReturn'].notnull().sum() != 0:
            factor_premium = _OLS_estimate_fac_premium(tmp_df[factors], tmp_df['OneTermReturn'].fillna(0))
            factor_premium.index = ['bias'] + factors

            factor_premium_info = factor_premium.to_dict()
            factor_premium_info.update({'trade_date': trade_date})
            factor_premium_infos.append(factor_premium_info)

    factor_premium_df = pd.DataFrame(factor_premium_infos)
    processed_factor_df = pd.concat(processed_factor_infos)
    # factor_premium_df.to_excel("factor_premium_all_mkt.xlsx")
    # processed_factor_df.to_pickle("processed_factor_all_mkt.pkl")

    return factor_premium_df, processed_factor_df

def weekly_performance_attribution_hs300(factor_data, factors, industry):
    factor_premium_infos = []
    factor_premium_last_week_infos = []
    processed_factor_infos = []
    if "trade_date" not in factor_data.columns:
        factor_data = factor_data.reset_index()
    factor_data[factors] = factor_data[factors].fillna(0.5)
    factor_data = factor_data[factor_data.GrowthFactor.notnull()]
    for trade_date, tmp_df in factor_data.groupby('trade_date'):
        
        for factor in factors:
            if factor == "GrowthFactor":
#                 tmp_df[factor] = tmp_df[factor].rank(pct=True).map(lambda x: math.ceil(3 * x) / 2 - 1)
#                 tmp_df[factor] = tmp_df[factor].rank(pct=True).map(lambda x: math.ceil(2 * x) - 1.5)
                tmp_df[factor] = tmp_df[factor].rank(pct=True).map(lambda x: math.ceil(11 * x) / 10 - 0.6)

            if factor in ['LeverageFactor', 'ValueFactor', 'QualityFactor']:
#                 tmp_df[factor] = tmp_df[factor].rank(pct=True).map(lambda x: math.ceil(2 * x) - 1.5)

                tmp_df[factor] = tmp_df[factor].rank(pct=True).map(lambda x: math.ceil(3 * x)/2 - 1)
            elif factor in ['LogMktCap', 'SizeFactor']:
                tmp_df[factor] = tmp_df[factor].rank(pct=True, ascending=False).map(lambda x: math.ceil(3 * x) / 2 - 1)
#                 tmp_df[factor] = tmp_df[factor].rank(pct=True, ascending=False).map(lambda x: math.ceil(10 * x) / 10 - 0.5)
#                 tmp_df[factor] = tmp_df[factor].rank(pct=True, ascending=False).map(lambda x: 0.5 if x > 0.1 else -0.5)

            elif factor in ['LiquidityFactor', 'ShortMomentumFactorReverse']:
                tmp_df[factor] = tmp_df[factor].rank(pct=True).map(lambda x: 0.5 if x > 0.2 else -0.5)
            elif factor in ['VolatilityFactor', 'OverallMomentumFactor']:
                tmp_df[factor] = tmp_df[factor].rank(pct=True).map(lambda x: -0.5 if x > 0.8 else 0.5)
            elif factor in ['LongMomentumFactorReverse']:
                def std_long_momentum(x):
                    if x > 0.7:
                        return -0.5
                    elif x > 0.4:
                        return 0
                    elif x > 0.1:
                        return 0.5
                    else:
                        return -0.5

                tmp_df[factor] = tmp_df[factor].rank(pct=True).map(lambda x: std_long_momentum(x))
            elif factor in ['Weeks50CountLog']:
#                 tmp_df["{}Std".format(factor)] = tmp_df.groupby(industry)[factor].apply(lambda x: (x-x.mean())/(x.std()+1e-8))
#                 tmp_df["{}Std".format(factor)] = tmp_df["{}Std".format(factor)].fillna(0)
#                 top_90 = tmp_df['Weeks50CountLog'].quantile(0.9)
#                 def std_leading_factor(x, top_90):
#                     if x > top_90:
#                         return 0.5
#                     elif x > -1:
#                         return 0
#                     else:
#                         return -0.5
                tmp_df_1 = tmp_df[tmp_df['Weeks50CountLog'] > -1]
                tmp_df_2 = tmp_df[tmp_df['Weeks50CountLog'] == -1]
                tmp_df_1['Weeks50CountLog'] = tmp_df_1['Weeks50CountLog'].rank(pct=True).map(lambda x: math.ceil(11 * x) / 10 - 0.6)
#                 tmp_df[factor] = tmp_df[factor].rank(pct=True).map(lambda x: math.ceil(2 * x) - 1.5)
#                 tmp_df_1['Weeks50CountLog'] = tmp_df_1['Weeks50CountLog'].rank(pct=True).map(lambda x: math.ceil(2 * x) - 1.5)

                tmp_df = pd.concat([tmp_df_1, tmp_df_2])
            else:
                pass
        processed_factor_infos.append(tmp_df.copy())
        if tmp_df['_10amOneTermReturn'].notnull().sum() != 0:
            factor_premium = _OLS_estimate_fac_premium(tmp_df[factors], tmp_df['_10amOneTermReturn'].fillna(0))
            factor_premium.index = ['bias'] + factors

            factor_premium_info = factor_premium.to_dict()
            factor_premium_info.update({'trade_date': trade_date})
            factor_premium_infos.append(factor_premium_info)
        if tmp_df['_10amOneTermReturn4LastWeek'].notnull().sum() != 0:
            factor_premium = _OLS_estimate_fac_premium(tmp_df[factors], tmp_df['_10amOneTermReturn4LastWeek'].fillna(0))
            factor_premium.index = ['bias'] + factors

            factor_premium_info = factor_premium.to_dict()
            factor_premium_info.update({'trade_date': trade_date})
            factor_premium_last_week_infos.append(factor_premium_info)
    factor_premium_df = pd.DataFrame(factor_premium_infos)
    factor_premium_last_week_df = pd.DataFrame(factor_premium_last_week_infos)
    processed_factor_df = pd.concat(processed_factor_infos)

    return factor_premium_df, factor_premium_last_week_df, processed_factor_df

def weekly_performance_attribution_hs300_bin_replace(factor_data, factors, industry):
    factor_premium_infos = []
    factor_premium_last_week_infos = []
    processed_factor_infos = []
    if "trade_date" not in factor_data.columns:
        factor_data = factor_data.reset_index()
    factor_data[factors] = factor_data[factors].fillna(0.5)
    factor_data = factor_data[factor_data.GrowthFactor.notnull()]
    for trade_date, tmp_df in factor_data.groupby('trade_date'):
        
        for factor in factors:
            if factor == "GrowthFactor":
#                 tmp_df[factor] = tmp_df[factor].rank(pct=True).map(lambda x: math.ceil(11 * x) / 10 - 0.6)
                tmp_df['bin'] = tmp_df[factor].rank(pct=True).map(lambda x: math.ceil(10*x))
                tmp_df[factor] = tmp_df[factor].rank(pct=True).map(lambda x: x-0.5)
                bin_2_value = tmp_df.groupby('bin').mean()[factor].to_dict()
                tmp_df[factor] = tmp_df[factor].map(bin_2_value)
            if factor in ['LeverageFactor', 'ValueFactor', 'QualityFactor']:
#                 tmp_df[factor] = tmp_df[factor].rank(pct=True).map(lambda x: math.ceil(2 * x) - 1.5)
                tmp_df['bin'] = tmp_df[factor].rank(pct=True).map(lambda x: math.ceil(3*x))
                tmp_df[factor] = tmp_df[factor].rank(pct=True).map(lambda x: x-0.5)
                bin_2_value = tmp_df.groupby('bin').mean()[factor].to_dict()
                tmp_df[factor] = tmp_df[factor].map(bin_2_value)
#                 tmp_df[factor] = tmp_df[factor].rank(pct=True).map(lambda x: math.ceil(3 * x)/2 - 1)
            elif factor in ['LogMktCap', 'SizeFactor']:
#                 tmp_df[factor] = tmp_df[factor].rank(pct=True, ascending=False).map(lambda x: math.ceil(3 * x) / 2 - 1)
#                 tmp_df[factor] = tmp_df[factor].rank(pct=True, ascending=False).map(lambda x: math.ceil(10 * x) / 10 - 0.5)
#                 tmp_df[factor] = tmp_df[factor].rank(pct=True, ascending=False).map(lambda x: 0.5 if x > 0.1 else -0.5)
                tmp_df['bin'] = tmp_df[factor].rank(pct=True, ascending=False).map(lambda x: math.ceil(3*x))
                tmp_df[factor] = tmp_df[factor].rank(pct=True, ascending=False).map(lambda x: x-0.5)
                bin_2_value = tmp_df.groupby('bin').mean()[factor].to_dict()
                tmp_df[factor] = tmp_df[factor].map(bin_2_value)
            elif factor in ['LiquidityFactor', 'ShortMomentumFactorReverse']:
#                 tmp_df[factor] = tmp_df[factor].rank(pct=True).map(lambda x: 0.5 if x > 0.2 else -0.5)
                tmp_df['bin'] = tmp_df[factor].rank(pct=True).map(lambda x: 1 if x > 0.2 else 0)
                tmp_df[factor] = tmp_df[factor].rank(pct=True).map(lambda x: x-0.5)
                bin_2_value = tmp_df.groupby('bin').mean()[factor].to_dict()
                tmp_df[factor] = tmp_df[factor].map(bin_2_value)
            elif factor in ['VolatilityFactor', 'OverallMomentumFactor']:
#                 tmp_df[factor] = tmp_df[factor].rank(pct=True).map(lambda x: -0.5 if x > 0.8 else 0.5)
                tmp_df['bin'] = tmp_df[factor].rank(pct=True, ascending=False).map(lambda x: 1 if x > 0.2 else 0)
                tmp_df[factor] = tmp_df[factor].rank(pct=True, ascending=False).map(lambda x: x-0.5)
                bin_2_value = tmp_df.groupby('bin').mean()[factor].to_dict()
                tmp_df[factor] = tmp_df[factor].map(bin_2_value)
                
            elif factor in ['LongMomentumFactorReverse']:
#                 def std_long_momentum(x):
#                     if x > 0.7:
#                         return -0.5
#                     elif x > 0.4:
#                         return 0
#                     elif x > 0.1:
#                         return 0.5
#                     else:
#                         return -0.5

#                 tmp_df[factor] = tmp_df[factor].rank(pct=True).map(lambda x: std_long_momentum(x))
                tmp_df['bin'] = tmp_df[factor].rank(pct=True, ascending=False).map(lambda x: math.ceil(3*x))
                tmp_df[factor] = tmp_df[factor].rank(pct=True, ascending=False).map(lambda x: x-0.5)
                bin_2_value = tmp_df.groupby('bin').mean()[factor].to_dict()
                tmp_df[factor] = tmp_df[factor].map(bin_2_value)
                
            elif factor in ['Weeks50CountLog']:
#                 tmp_df["{}Std".format(factor)] = tmp_df.groupby(industry)[factor].apply(lambda x: (x-x.mean())/(x.std()+1e-8))
#                 tmp_df["{}Std".format(factor)] = tmp_df["{}Std".format(factor)].fillna(0)
#                 top_90 = tmp_df['Weeks50CountLog'].quantile(0.9)
#                 def std_leading_factor(x, top_90):
#                     if x > top_90:
#                         return 0.5
#                     elif x > -1:
#                         return 0
#                     else:
#                         return -0.5
                tmp_df_1 = tmp_df[tmp_df['Weeks50CountLog'] > -1]
                tmp_df_2 = tmp_df[tmp_df['Weeks50CountLog'] == -1]
                tmp_df_1['Weeks50CountLog'] = tmp_df_1['Weeks50CountLog'].rank(pct=True).map(lambda x: math.ceil(11 * x) / 10 - 0.6)
#                 tmp_df[factor] = tmp_df[factor].rank(pct=True).map(lambda x: math.ceil(2 * x) - 1.5)
#                 tmp_df_1['Weeks50CountLog'] = tmp_df_1['Weeks50CountLog'].rank(pct=True).map(lambda x: math.ceil(2 * x) - 1.5)

                tmp_df = pd.concat([tmp_df_1, tmp_df_2])
            else:
                pass
        processed_factor_infos.append(tmp_df.copy())
        if tmp_df['_10amOneTermReturn'].notnull().sum() != 0:
            factor_premium = _OLS_estimate_fac_premium(tmp_df[factors], tmp_df['_10amOneTermReturn'].fillna(0))
            factor_premium.index = ['bias'] + factors

            factor_premium_info = factor_premium.to_dict()
            factor_premium_info.update({'trade_date': trade_date})
            factor_premium_infos.append(factor_premium_info)
        if tmp_df['_10amOneTermReturn4LastWeek'].notnull().sum() != 0:
            factor_premium = _OLS_estimate_fac_premium(tmp_df[factors], tmp_df['_10amOneTermReturn4LastWeek'].fillna(0))
            factor_premium.index = ['bias'] + factors

            factor_premium_info = factor_premium.to_dict()
            factor_premium_info.update({'trade_date': trade_date})
            factor_premium_last_week_infos.append(factor_premium_info)
    factor_premium_df = pd.DataFrame(factor_premium_infos)
    factor_premium_last_week_df = pd.DataFrame(factor_premium_last_week_infos)
    processed_factor_df = pd.concat(processed_factor_infos)

    return factor_premium_df, factor_premium_last_week_df, processed_factor_df


def weekly_performance_attribution_mkt_vol_tag(factor_data, factors, industry):
    factor_premium_infos = []
    processed_factor_infos = []
    if "trade_date" not in factor_data.columns:
        factor_data = factor_data.reset_index()

    for trade_date, tmp_df in factor_data.groupby('trade_date'):
        for factor in factors:
            if factor in ['GrowthFactor', 'LeverageFactor', 'ValueFactor', 'QualityFactor', 'LiquidityFactor', 'ShortMomentumFactorReverse', 'LogMktCap', 'VolatilityFactor', 'LongMomentumFactorReverse']:
                tmp_df[factor] = tmp_df[factor].rank(pct=True).map(lambda x: math.ceil(3 * x) / 2 - 1)
#             elif factor in ['LogMktCap', 'VolatilityFactor', 'LongMomentumFactorReverse']:
#                 tmp_df[factor] = tmp_df[factor].rank(pct=True, ascending=False).map(lambda x: math.ceil(3 * x) / 2 - 1)
            elif factor in ['Weeks50CountLog']:
                tmp_df["{}Std".format(factor)] = tmp_df.groupby(industry)[factor].apply(lambda x: (x-x.mean())/(x.std()+1e-8))
                tmp_df["{}Std".format(factor)] = tmp_df["{}Std".format(factor)].fillna(0)
            else:
                pass
        processed_factor_infos.append(tmp_df.copy())
        if tmp_df['OneTermReturn'].notnull().sum() != 0:
            factor_premium = _OLS_estimate_fac_premium(tmp_df[factors], tmp_df['OneTermReturn'].fillna(0))
            factor_premium.index = ['bias'] + factors

            factor_premium_info = factor_premium.to_dict()
            factor_premium_info.update({'trade_date': trade_date})
            factor_premium_infos.append(factor_premium_info)

    factor_premium_df = pd.DataFrame(factor_premium_infos)
    processed_factor_df = pd.concat(processed_factor_infos)
#     factor_premium_df.to_excel("factor_premium_all_mkt.xlsx")
#     processed_factor_df.to_pickle("processed_factor_all_mkt.pkl")

    return factor_premium_df, processed_factor_df


def generate_score_from_perf_attribution_with_IR(start_date, end_date, factors, factor_directions, factor_premium_df, processed_factor_df,
                                         long_window_size, short_window_size, is_norm=False):
    trade_dates = sorted(processed_factor_df['trade_date'].unique())
    assert len(trade_dates) > long_window_size, print("not enough history data")
#     factor_premium_df['weight_adj_factor'] = factor_premium_df[factors].applymap(lambda x: abs(x)).sum(axis=1).map(lambda x: 0.05/x)
#     for factor in factors:
#         factor_premium_df[factor] = factor_premium_df[factor]*factor_premium_df["weight_adj_factor"]
    if is_norm:
        factor_premium_df[factors] = factor_premium_df[factors].applymap(lambda x: max(x, 0))
        factor_premium_df['premium_weight_sum'] = factor_premium_df[factors].sum(axis=1)
        factor_premium_df['adj_factor'] = factor_premium_df['premium_weight_sum'].map(lambda x: 0.012/x if x > 0 else 0)
        for factor in factors:
            factor_premium_df[factor] = factor_premium_df[factor]*factor_premium_df["adj_factor"]        

    score_infos = []
    premium_pred_infos = []
    for k in range(long_window_size, len(trade_dates)):
        if trade_dates[k] >= start_date and trade_dates[k] <= end_date:
            long_hist_premium_df = factor_premium_df[factor_premium_df.trade_date.map(lambda x: x in trade_dates[k-long_window_size: k])
            ][factors]
#             long_premium_pred = long_hist_premium_df.mean()
            long_premium_pred = long_hist_premium_df.mean()/long_hist_premium_df.std()
            long_vif = long_hist_premium_df.corr().mean()
#             vif_sum = vif.sum()
            factor_weight = long_vif.map(lambda x: 1-x)
            long_premium_pred = long_premium_pred*factor_weight
            for factor, factor_direction in zip(factors, factor_directions):
                if factor_direction == 1:
                    long_premium_pred[factor] = max(long_premium_pred[factor], 0)
                elif factor_direction == -1:
                    long_premium_pred[factor] = min(long_premium_pred[factor], 0)
                else:
                    pass
                
            long_premium_pred = long_premium_pred.map(lambda x: max(x, 0))
            short_hist_premium_df = factor_premium_df[factor_premium_df.trade_date.map(lambda x: x in trade_dates[k-short_window_size: k])
            ][factors]
#             short_premium_pred = short_hist_premium_df.mean()
            short_premium_pred = short_hist_premium_df.mean()/short_hist_premium_df.std()
            short_vif = short_hist_premium_df.corr().mean()
#             vif_sum = vif.sum()
            factor_weight = short_vif.map(lambda x: 1-x)
            short_premium_pred = short_premium_pred*factor_weight
        
            for factor, factor_direction in zip(factors, factor_directions):
                if factor_direction == 1:
                    short_premium_pred[factor] = max(short_premium_pred[factor], 0)
                elif factor_direction == -1:
                    short_premium_pred[factor] = min(short_premium_pred[factor], 0)
                else:
                    pass
                
            short_premium_pred = short_premium_pred.map(lambda x: max(x, 0))
            tmp_processed_factor_df = processed_factor_df[processed_factor_df.trade_date == trade_dates[k]]
#             premium_weight_sum = long_premium_pred.sum()*0.8 + short_premium_pred.sum()*0.2
#             adj_factor = 0.011/premium_weight_sum
#             long_premium_pred = long_premium_pred*adj_factor
#             short_premium_pred = short_premium_pred*adj_factor

            
            long_score = np.matmul(tmp_processed_factor_df[factors].values, long_premium_pred.loc[factors].values.reshape(len(factors), 1))
            short_score = np.matmul(tmp_processed_factor_df[factors].values, short_premium_pred.loc[factors].values.reshape(len(factors), 1))
            tmp_processed_factor_df['short_score'] = short_score
            tmp_processed_factor_df['long_score'] = long_score
            tmp_processed_factor_df['ScoreFromPerfAtt_8_2'] = tmp_processed_factor_df['long_score']*0.8 + tmp_processed_factor_df['short_score']*0.2
#             tmp_processed_factor_df['ScoreFromPerfAtt_9_1'] = tmp_processed_factor_df['long_score']*0.9 + tmp_processed_factor_df['short_score']*0.1
            tmp_processed_factor_df['ScoreFromPerfAtt_5_5'] = tmp_processed_factor_df['long_score']*0.5 + tmp_processed_factor_df['short_score']*0.5
#             print("long_premium_sum {}".format(long_premium_pred.sum()))
#             print("short_premium_pred {}".format(short_premium_pred.sum()))
        
            premium_pred = long_premium_pred*0.8 + short_premium_pred*0.2
#             premium_pred = short_premium_pred
            premium_pred.name = trade_dates[k]
            premium_pred_infos.append(premium_pred)
            score_infos.append(tmp_processed_factor_df[['code', 'trade_date', 'ScoreFromPerfAtt_8_2', 'ScoreFromPerfAtt_5_5', 'long_score', 'short_score']])
    score_df = pd.concat(score_infos)

    all_premium_pred_df = pd.concat(premium_pred_infos, axis=1)
    all_premium_pred_df.T.to_excel("csi500_premium_pred_long_{}_short_{}.xlsx".format(long_window_size, short_window_size))
    factor_premium_df.to_excel("csi500_premium_weekly.xlsx".format(long_window_size, short_window_size))
    return score_df



def generate_score_from_perf_attribution(start_date, end_date, factors, factor_directions, factor_premium_df, processed_factor_df,
                                         long_window_size, short_window_size, is_norm=False):
    trade_dates = sorted(processed_factor_df['trade_date'].unique())
    assert len(trade_dates) > long_window_size, print("not enough history data")
#     factor_premium_df['weight_adj_factor'] = factor_premium_df[factors].applymap(lambda x: abs(x)).sum(axis=1).map(lambda x: 0.05/x)
#     for factor in factors:
#         factor_premium_df[factor] = factor_premium_df[factor]*factor_premium_df["weight_adj_factor"]
    if is_norm:
        factor_premium_df[factors] = factor_premium_df[factors].applymap(lambda x: max(x, 0))
        factor_premium_df['premium_weight_sum'] = factor_premium_df[factors].sum(axis=1)
        factor_premium_df['adj_factor'] = factor_premium_df['premium_weight_sum'].map(lambda x: 0.012/x if x > 0 else 0)
        for factor in factors:
            factor_premium_df[factor] = factor_premium_df[factor]*factor_premium_df["adj_factor"]        

    score_infos = []
    premium_pred_infos = []
    for k in range(long_window_size, len(trade_dates)):
        if trade_dates[k] >= start_date and trade_dates[k] <= end_date:
            long_hist_premium_df = factor_premium_df[factor_premium_df.trade_date.map(lambda x: x in trade_dates[k-long_window_size: k])
            ][factors]
            long_premium_pred = long_hist_premium_df.mean()
            for factor, factor_direction in zip(factors, factor_directions):
                if factor_direction == 1:
                    long_premium_pred[factor] = max(long_premium_pred[factor], 0)
                elif factor_direction == -1:
                    long_premium_pred[factor] = min(long_premium_pred[factor], 0)
                else:
                    pass
            long_premium_pred = long_premium_pred.map(lambda x: max(x, 0))
            short_hist_premium_df = factor_premium_df[factor_premium_df.trade_date.map(lambda x: x in trade_dates[k-short_window_size: k])
            ][factors]
            short_premium_pred = short_hist_premium_df.mean()
            for factor, factor_direction in zip(factors, factor_directions):
                if factor_direction == 1:
                    short_premium_pred[factor] = max(short_premium_pred[factor], 0)
                elif factor_direction == -1:
                    short_premium_pred[factor] = min(short_premium_pred[factor], 0)
                else:
                    pass
                
            short_premium_pred = short_premium_pred.map(lambda x: max(x, 0))
            tmp_processed_factor_df = processed_factor_df[processed_factor_df.trade_date == trade_dates[k]]
#             premium_weight_sum = long_premium_pred.sum()*0.8 + short_premium_pred.sum()*0.2
#             adj_factor = 0.011/premium_weight_sum
#             long_premium_pred = long_premium_pred*adj_factor
#             short_premium_pred = short_premium_pred*adj_factor

            
            long_score = np.matmul(tmp_processed_factor_df[factors].values, long_premium_pred.loc[factors].values.reshape(len(factors), 1))
            short_score = np.matmul(tmp_processed_factor_df[factors].values, short_premium_pred.loc[factors].values.reshape(len(factors), 1))
            tmp_processed_factor_df['short_score'] = short_score
            tmp_processed_factor_df['long_score'] = long_score
            tmp_processed_factor_df['ScoreFromPerfAtt_8_2'] = tmp_processed_factor_df['long_score']*0.8 + tmp_processed_factor_df['short_score']*0.2
#             tmp_processed_factor_df['ScoreFromPerfAtt_9_1'] = tmp_processed_factor_df['long_score']*0.9 + tmp_processed_factor_df['short_score']*0.1
            tmp_processed_factor_df['ScoreFromPerfAtt_5_5'] = tmp_processed_factor_df['long_score']*0.5 + tmp_processed_factor_df['short_score']*0.5
#             print("long_premium_sum {}".format(long_premium_pred.sum()))
#             print("short_premium_pred {}".format(short_premium_pred.sum()))
        
            premium_pred = long_premium_pred*0.8 + short_premium_pred*0.2
#             premium_pred = short_premium_pred
            premium_pred.name = trade_dates[k]
            premium_pred_infos.append(premium_pred)
            score_infos.append(tmp_processed_factor_df[['code', 'trade_date', 'ScoreFromPerfAtt_8_2', 'ScoreFromPerfAtt_5_5', 'long_score', 'short_score']])
    score_df = pd.concat(score_infos)

    all_premium_pred_df = pd.concat(premium_pred_infos, axis=1)
    all_premium_pred_df.T.to_excel("csi500_premium_pred_long_{}_short_{}.xlsx".format(long_window_size, short_window_size))
    factor_premium_df.to_excel("csi500_premium_weekly.xlsx".format(long_window_size, short_window_size))
    return score_df

def generate_score_from_perf_attribution_hs300(start_date, end_date, factors, factor_premium_df, factor_premium_last_week_df,processed_factor_df,
                                         long_window_size, short_window_size):
    trade_dates = sorted(processed_factor_df['trade_date'].unique())
    assert len(trade_dates) > long_window_size, print("not enough history data")
    
    score_infos = []
    premium_pred_infos = []

    for k in range(long_window_size, len(trade_dates)):
        if trade_dates[k] >= start_date and trade_dates[k] <= end_date:
            long_hist_premium_df = factor_premium_df[factor_premium_df.trade_date.map(lambda x: x in trade_dates[k-long_window_size: k-1])
            ][factors]
            long_hist_premium_df = long_hist_premium_df.append(factor_premium_last_week_df[factor_premium_last_week_df.trade_date.map(lambda x: x in trade_dates[k-1: k])][factors])
            long_premium_pred = long_hist_premium_df.mean()
            long_premium_pred = long_premium_pred.map(lambda x: max(x, 0))
            short_hist_premium_df = factor_premium_df[factor_premium_df.trade_date.map(lambda x: x in trade_dates[k-short_window_size: k-1])
            ][factors]
            short_hist_premium_df = short_hist_premium_df.append(factor_premium_last_week_df[factor_premium_last_week_df.trade_date.map(lambda x: x in trade_dates[k-1: k])][factors])
            
            print(trade_dates[k])

            short_premium_pred = short_hist_premium_df.mean()
            print(short_premium_pred)
            short_premium_pred = short_premium_pred.map(lambda x: max(x, 0))
            tmp_processed_factor_df = processed_factor_df[processed_factor_df.trade_date == trade_dates[k]]
            long_score = np.matmul(tmp_processed_factor_df[factors].values, long_premium_pred.loc[factors].values.reshape(len(factors), 1))
            short_score = np.matmul(tmp_processed_factor_df[factors].values, short_premium_pred.loc[factors].values.reshape(len(factors), 1))
            tmp_processed_factor_df['short_score'] = short_score
            # if trade_dates[k] == 20250916:
            #     import pdb
            #     pdb.set_trace()
            #     pass
            tmp_processed_factor_df['long_score'] = long_score
            tmp_processed_factor_df['ScoreFromPerfAtt_8_2'] = tmp_processed_factor_df['long_score']*0.8 + tmp_processed_factor_df['short_score']*0.2
            tmp_processed_factor_df['ScoreFromPerfAtt_9_1'] = tmp_processed_factor_df['long_score']*0.9 + tmp_processed_factor_df['short_score']*0.1
            tmp_processed_factor_df['ScoreFromPerfAtt_5_5'] = tmp_processed_factor_df['long_score']*0.5 + tmp_processed_factor_df['short_score']*0.5
#             tmp_processed_factor_df['long_score_bin'] = tmp_processed_factor_df['long_score'].rank(pct=True).map(lambda x: math.ceil(x*5))
#             tmp_processed_factor_df['short_score_bin'] = tmp_processed_factor_df['short_score'].rank(pct=True).map(lambda x: math.ceil(x*2))
#             long_bin_score = tmp_processed_factor_df.groupby('long_score_bin').mean()['long_score'].to_dict()
#             short_bin_score = tmp_processed_factor_df.groupby('short_score_bin').mean()['short_score'].to_dict()
#             tmp_processed_factor_df['long_bin_score'] = tmp_processed_factor_df['long_score_bin'].map(long_bin_score)
#             tmp_processed_factor_df['short_bin_score'] = tmp_processed_factor_df['short_score_bin'].map(short_bin_score)
#             tmp_processed_factor_df['ScoreFromPerfAtt_8_2_Bin'] = tmp_processed_factor_df['long_score']*0.8 + tmp_processed_factor_df['short_bin_score']*0.2

#             tmp_processed_factor_df['ScoreFromPerfAtt_5_5_Bin'] = tmp_processed_factor_df['long_score']*0.5 + tmp_processed_factor_df['short_bin_score']*0.5
#             tmp_processed_factor_df["GrowthFactor"] = tmp_processed_factor_df["GrowthFactor"]*8
#             tmp_processed_factor_df["LeadingFactor"] = tmp_processed_factor_df["LeadingFactor"]*4
            tmp_processed_factor_df["GrowthFactor"] = tmp_processed_factor_df["GrowthFactor"]*8
#             tmp_processed_factor_df["LeadingFactor"] = tmp_processed_factor_df["LeadingFactor"]*4
            tmp_processed_factor_df['FixedScore'] = tmp_processed_factor_df[factors].sum(axis=1)
            long_score_std = tmp_processed_factor_df['long_score'].std()
            fixed_score_std = tmp_processed_factor_df['FixedScore'].std()
            tmp_processed_factor_df['FixedScore'] = tmp_processed_factor_df['FixedScore'].map(lambda x: x*long_score_std/fixed_score_std)
            
            tmp_processed_factor_df['ScoreFromPerfAtt_Fixed_8_2'] = tmp_processed_factor_df['FixedScore']*0.8 + tmp_processed_factor_df['short_score']*0.2
            tmp_processed_factor_df['ScoreFromPerfAtt_Fixed_9_1'] = tmp_processed_factor_df['FixedScore']*0.9 + tmp_processed_factor_df['short_score']*0.1
            tmp_processed_factor_df['ScoreFromPerfAtt_Fixed_5_5'] = tmp_processed_factor_df['FixedScore']*0.5 + tmp_processed_factor_df['short_score']*0.5            
            score_infos.append(tmp_processed_factor_df[['code', 'trade_date', 'ScoreFromPerfAtt_9_1', 'ScoreFromPerfAtt_8_2', 'ScoreFromPerfAtt_5_5', 'long_score', 'short_score', "ScoreFromPerfAtt_Fixed_8_2", "ScoreFromPerfAtt_Fixed_9_1", 'ScoreFromPerfAtt_Fixed_5_5', "FixedScore"]])
            premium_pred = long_premium_pred*0.9 + short_premium_pred*0.1
            premium_pred.name = trade_dates[k]
            premium_pred_infos.append(premium_pred)
    score_df = pd.concat(score_infos)
    all_premium_pred_df = pd.concat(premium_pred_infos, axis=1)

    return score_df


def generate_score_from_perf_attribution_with_mkt_vol(start_date, end_date, factors, factor_premium_df, processed_factor_df,
                                         long_window_size, short_window_size, mkt_vol):
    factor_premium_df = pd.merge(factor_premium_df, mkt_vol, how='left', on='trade_date')
    factor_premium_df.to_excel("factor_premium_all_mkt.xlsx")
    trade_dates = sorted(processed_factor_df['trade_date'].unique())
    assert len(trade_dates) > long_window_size, print("not enough history data")
    score_infos = []
#     import pdb
#     pdb.set_trace()
    for k in range(long_window_size, len(trade_dates)):
        if trade_dates[k] >= start_date and trade_dates[k] <= end_date:
            long_hist_premium_df = factor_premium_df[factor_premium_df.trade_date.map(lambda x: x in trade_dates[k-long_window_size: k])
            ]
            now_vol_tag = mkt_vol[mkt_vol.trade_date == trade_dates[k]]['vol_20_pct_bin'].values[0]                                                                      
            long_premium_pred_info = {}
#             print(trade_dates[k])
            for factor in factors:
                
                if factor in ['ValueFactor', 'LongMomentumFactorReverse', 'VolatilityFactor']:
                    param, pcov = curve_fit(poly1d_func, long_hist_premium_df.vol_20_pct_bin.values.astype(np.float), long_hist_premium_df[factor].values, p0=[-1e-4,0], bounds=([-np.inf, -np.inf], [0,np.inf]))
                    preminum_pred = poly1d_func(now_vol_tag, *param)
                    long_premium_pred_info.update({factor: max(preminum_pred, 0)})
                    print("factor {}, param {}".format(factor, param))
                elif factor in ['ShortMomentumFactorReverse']:
                    param, pcov = curve_fit(poly1d_func, long_hist_premium_df.vol_20_pct_bin.values.astype(np.float), long_hist_premium_df[factor].values, p0=[1e-4,0], bounds=([0, -np.inf], [np.inf,np.inf]))
                    preminum_pred = poly1d_func(now_vol_tag, *param)
                    long_premium_pred_info.update({factor: max(preminum_pred, 0)})
                    print("factor {}, param {}".format(factor, param))
                else:
                    preminum_pred = long_hist_premium_df[factor].mean()
                    long_premium_pred_info.update({factor: max(preminum_pred, 0)})
            long_premium_pred = pd.Series(long_premium_pred_info)
            short_hist_premium_df = factor_premium_df[factor_premium_df.trade_date.map(lambda x: x in trade_dates[k-short_window_size: k])
            ][factors]
            short_premium_pred = short_hist_premium_df.mean()
            short_premium_pred = short_premium_pred.map(lambda x: max(x, 0))
            tmp_processed_factor_df = processed_factor_df[processed_factor_df.trade_date == trade_dates[k]]
            long_score = np.matmul(tmp_processed_factor_df[factors].values, long_premium_pred.loc[factors].values.reshape(len(factors), 1))
            short_score = np.matmul(tmp_processed_factor_df[factors].values, short_premium_pred.loc[factors].values.reshape(len(factors), 1))
            tmp_processed_factor_df['short_score'] = short_score
            tmp_processed_factor_df['long_score'] = long_score
            tmp_processed_factor_df['ScoreFromPerfAtt_8_2'] = tmp_processed_factor_df['long_score']*0.8 + tmp_processed_factor_df['short_score']*0.2
            tmp_processed_factor_df['ScoreFromPerfAtt_9_1'] = tmp_processed_factor_df['long_score']*0.9 + tmp_processed_factor_df['short_score']*0.1
            tmp_processed_factor_df['ScoreFromPerfAtt_5_5'] = tmp_processed_factor_df['long_score']*0.5 + tmp_processed_factor_df['short_score']*0.5

            score_infos.append(tmp_processed_factor_df[['code', 'trade_date', 'ScoreFromPerfAtt_9_1', 'ScoreFromPerfAtt_8_2', 'ScoreFromPerfAtt_5_5', 'long_score', 'short_score']])
    score_df = pd.concat(score_infos)

    return score_df


class ScoreCardFactor(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.start_date = param_info['start_date']
        self.end_date = param_info['end_date']
        self.source_data_infos = param_info['source_data_infos']
        self.invalid_infos = param_info['invalid_infos']
        self.window_size = param_info['window_size']
        self.bin_count_level = param_info['bin_count_level']
        self.lr_clf = LinearRegression()
        self.dependent_variable = param_info['dependent_variable']
        self.output_score_name = param_info['output_score_name']
        self.save_info = param_info['save_info']
        self.r_process_param = param_info.get("r_process_param", {})
        self.lr_clf = LinearRegression()
        # self.bin_infos = {
        #     "growth_factor": self.bin_count_level,
        #     "quality_factor": self.bin_count_level,
        #     "overall_momentum_factor": max(self.bin_count_level, 3),
        #     "leverage_factor": self.bin_count_level,
        #     "value_factor": self.bin_count_level,
        #     "liquidity_volatility_factor": max(self.bin_count_level, 3),
        # }
        self.bin_infos = [
            {"name": "GrowthFactor", 'type': "continuous", "bin_count": self.bin_count_level},
            {"name": "QualityFactor", 'type': "continuous", "bin_count": self.bin_count_level},
            {"name": "OverallMomentumFactor", 'type': "continuous", "bin_count": max(self.bin_count_level, 3)},
            {"name": "LeverageFactor", 'type': "continuous", "bin_count": self.bin_count_level},
            {"name": "ValueFactor", 'type': "continuous", "bin_count": self.bin_count_level},
            {"name": "LiquidityVolatilityFactor", 'type': "continuous", "bin_count": max(self.bin_count_level, 3)},
        ]
        self.operators = [
            {
                "func": get_data_from_multi_source,
                "param": {
                    "data_source_infos": self.source_data_infos,
                    "start_date": self.start_date,
                    "end_date": self.end_date
                },
                "input_data": {},
                "output": ['factor']
            },
            {
                "func": generate_one_term_return,
                "param": {
                    "one_week_momentum_name": "MomentumWeeks1",
                    "output_name": "OneTermReturn",
                    "limit": self.r_process_param.get("limit", 0),
                    "is_3_sigma_std": self.r_process_param.get("is_3_sigma_std", False),
                    "is_ecdf": self.r_process_param.get("is_ecdf", False)
                },
                "input_data": {'data': 'factor'},
                "output": ['factor']
            },
            {
                "func": transfer_data_to_valid_and_not_valid,
                "param": {
                    "invalid_infos": self.invalid_infos

                },
                "input_data": {"data": "factor"},
                "output": ['valid_factor', 'invalid_factor']
            },
            {
                "func": generate_score_card,
                "param": {
                    "window_size": self.window_size,
                    "start_date": self.start_date,
                    "end_date": self.end_date,
                    "r_name": self.dependent_variable,
                    "overall_factor_name": self.output_score_name,
                    "bin_infos": self.bin_infos,
                    "lr_clf": self.lr_clf
                },
                "input_data": {"all_factor_data": "valid_factor"},
                "output": ["overall_score"],
            },
            {
                "func": process_not_valid_data_and_merge,
                "param": {
                    "output_score_name": self.output_score_name,
                },
                "input_data": {
                    "overall_score": "overall_score",
                    "invalid_data": "invalid_factor"
                },
                "output": ["all_data_score"]
            },
            {
                "func": save_data_to_table,
                "param": {"engine": self.save_info['engine'], "table": self.save_info['table'],
                          "if_exists": self.save_info.get("if_exists", "append")},
                "input_data": {"data": "all_data_score"},
                "output": ["all_data_score"]
            }
        ]

        self.output_vars = ["all_data_score"]

def generate_fixed_weight_score(all_factor_data, start_date, end_date, output_score_name):
    # factor_names = [info['name'] for info in factor_bin_infos]
    factor_names = ["ValueFactor", "LiquidityFactor", "GrowthFactor", "QualityFactor", "LongMomentumFactorReverse",
                     "ShortMomentumFactorReverse", "VolatilityFactor"],
    for factor_name in factor_names:
        if factor_name == 'OverallMomentumFactor':
            all_factor_data[factor_name] = all_factor_data[factor_name].map(lambda x: x / 2)
        else:
            all_factor_data[factor_name] = all_factor_data[factor_name].map(lambda x: x - 0.5)
    all_factor_data = all_factor_data.reset_index()
    trade_dates = sorted(all_factor_data['trade_date'].unique())
    fixed_score_infos = []
    for i, date in enumerate(tqdm(trade_dates)):
        if date > start_date and date <= end_date:
            tmp_data = all_factor_data[all_factor_data.trade_date == date]
            tmp_data.sort_values("ValueFactor", inplace=True)
            valid_count = len(tmp_data)

            tmp_data.sort_values("ValueFactor", inplace=True)
            tmp_data['ValueFactorBin'] = [int(3 * j) for j in range(valid_count)]
            bin_2_value = tmp_data.groupby('ValueFactorBin')['ValueFactor'].mean().to_dict()
            tmp_data['ValueFactorBinValue'] = tmp_data['ValueFactorBin'].map(bin_2_value)

            tmp_data.sort_values("GrowthFactor", inplace=True)
            tmp_data['GrowthFactorBin'] = [int(3 * j / valid_count) for j in range(valid_count)]
            #     valid_tmp_df['growth_factor_bin'] = valid_tmp_df['growth_factor_bin'].map(lambda x: min(x, 1))
            bin_2_value = tmp_data.groupby('GrowthFactorBin')['GrowthFactor'].mean().to_dict()
            tmp_data['GrowthFactorBinValue'] = tmp_data['GrowthFactorBin'].map(bin_2_value)

            tmp_data.sort_values("QualityFactor", inplace=True)
            tmp_data['QualityFactorBin'] = [int(3 * j / valid_count) for j in range(valid_count)]
            bin_2_value = tmp_data.groupby('QualityFactorBin')['QualityFactor'].mean().to_dict()
            tmp_data['QualityFactorBinValue'] = tmp_data['QualityFactorBin'].map(bin_2_value)

            tmp_data.sort_values("LiquidityFactor", inplace=True)
            tmp_data['LiquidityFactorBin'] = [int(5 * j / valid_count) for j in range(valid_count)]
            tmp_data['LiquidityFactorBin'] = tmp_data['LiquidityFactorBin'].map(lambda x: min(x, 1))
            bin_2_value = tmp_data.groupby('LiquidityFactorBin')['LiquidityFactor'].mean().to_dict()
            tmp_data['LiquidityFactorBinValue'] = tmp_data['LiquidityFactorBin'].map(bin_2_value)

            tmp_data.sort_values("VolatilityFactor", inplace=True)
            tmp_data['VolatilityFactorBin'] = [int(5 * j / valid_count) for j in range(valid_count)]
            tmp_data['VolatilityFactorBin'] = tmp_data['VolatilityFactorBin'].map(lambda x: min(x, 1))
            bin_2_value = tmp_data.groupby('VolatilityFactorBin')['VolatilityFactor'].mean().to_dict()
            tmp_data['VolatilityFactorBinValue'] = tmp_data['VolatilityFactorBin'].map(bin_2_value)

            tmp_data.sort_values("ShortMomentumFactorReverse", inplace=True)
            tmp_data['ShortMomentumFactorReverseBin'] = [int(5 * j / valid_count) for j in range(valid_count)]
            tmp_data['ShortMomentumFactorReverseBin'] = tmp_data['ShortMomentumFactorReverseBin'].map(lambda x: min(x, 1))
            bin_2_value = tmp_data.groupby('ShortMomentumFactorReverseBin')['ShortMomentumFactorReverse'].mean().to_dict()
            tmp_data['ShortMomentumFactorReverseBinValue'] = tmp_data['ShortMomentumFactorReverseBin'].map(bin_2_value)

            tmp_data.sort_values("LongMomentumFactorReverse", inplace=True)
            tmp_data['LongMomentumFactorReverseBin'] = [int(5 * j / valid_count) for j in range(valid_count)]
            bin_2_value = tmp_data.groupby('LongMomentumFactorReverseBin')['LongMomentumFactorReverse'].mean().to_dict()

            bin_2_value[0] = bin_2_value[4]
            tmp_data['LongMomentumFactorReverseBinValue'] = tmp_data['LongMomentumFactorReverseBin'].map(
                bin_2_value)

            tmp_data[output_score_name] = tmp_data['ValueFactorBinValue'] + tmp_data[
                'LiquidityFactorBinValue'] + tmp_data['GrowthFactorBinValue'] * 2 + tmp_data[
                                                     'QualityFactorBinValue'] - tmp_data[
                                                     'LongMomentumFactorReverseBinValue'] + tmp_data[
                                                     'ShortMomentumFactorReverseBinValue'] + tmp_data[
                                                     'VolatilityFactorBinValue']

            tmp_data[output_score_name] = tmp_data[output_score_name].map(lambda x: x * 0.0016)
            fixed_score_infos.append(tmp_data)
    fixed_score_df = pd.concat(fixed_score_infos)
    return fixed_score_df.set_index(['code', 'trade_date'])[[output_score_name]]

def generate_fixed_weight_score_rpt(all_factor_data, start_date, end_date):
    # factor_names = [info['name'] for info in factor_bin_infos]
    factor_names = ["ValueFactor", "LiquidityFactor", "GrowthFactor", "QualityFactor", "LongMomentumFactorReverse",
                     "ShortMomentumFactorReverse", "VolatilityFactor"]
    # import pdb
    # pdb.set_trace()

    for factor_name in factor_names:
        if factor_name == 'OverallMomentumFactor':
            all_factor_data[factor_name] = all_factor_data[factor_name].map(lambda x: x / 2)
        else:
            all_factor_data[factor_name] = all_factor_data[factor_name].map(lambda x: x - 0.5)
    all_factor_data = all_factor_data.reset_index()
    trade_dates = sorted(all_factor_data['trade_date'].unique())
    fixed_score_infos = []
    # bin_count = 10
    for i, date in enumerate(tqdm(trade_dates)):
        if date > start_date and date <= end_date:
            tmp_data = all_factor_data[all_factor_data.trade_date == date]
            for bin_count in [2, 3, 5, 7, 10]:

                tmp_data.sort_values("ValueFactor", inplace=True)
                valid_count = len(tmp_data)

                tmp_data.sort_values("ValueFactor", inplace=True)
                tmp_data['ValueFactorBin'] = [int(bin_count * j/valid_count) for j in range(valid_count)]
                bin_2_value = tmp_data.groupby('ValueFactorBin')['ValueFactor'].mean().to_dict()
                tmp_data['ValueFactorBinValue'] = tmp_data['ValueFactorBin'].map(bin_2_value)

                tmp_data.sort_values("GrowthFactor", inplace=True)
                tmp_data['GrowthFactorBin'] = [int(bin_count * j / valid_count) for j in range(valid_count)]
                #     valid_tmp_df['growth_factor_bin'] = valid_tmp_df['growth_factor_bin'].map(lambda x: min(x, 1))
                bin_2_value = tmp_data.groupby('GrowthFactorBin')['GrowthFactor'].mean().to_dict()
                tmp_data['GrowthFactorBinValue'] = tmp_data['GrowthFactorBin'].map(bin_2_value)

                tmp_data.sort_values("QualityFactor", inplace=True)
                tmp_data['QualityFactorBin'] = [int(bin_count * j / valid_count) for j in range(valid_count)]
                bin_2_value = tmp_data.groupby('QualityFactorBin')['QualityFactor'].mean().to_dict()
                tmp_data['QualityFactorBinValue'] = tmp_data['QualityFactorBin'].map(bin_2_value)

                tmp_data.sort_values("LiquidityFactor", inplace=True)
                tmp_data['LiquidityFactorBin'] = [int(5 * j / valid_count) for j in range(valid_count)]
                tmp_data['LiquidityFactorBin'] = tmp_data['LiquidityFactorBin'].map(lambda x: min(x, 1))
                bin_2_value = tmp_data.groupby('LiquidityFactorBin')['LiquidityFactor'].mean().to_dict()
                tmp_data['LiquidityFactorBinValue'] = tmp_data['LiquidityFactorBin'].map(bin_2_value)

                tmp_data.sort_values("VolatilityFactor", inplace=True)
                tmp_data['VolatilityFactorBin'] = [int(5 * j / valid_count) for j in range(valid_count)]
                tmp_data['VolatilityFactorBin'] = tmp_data['VolatilityFactorBin'].map(lambda x: min(x, 1))
                bin_2_value = tmp_data.groupby('VolatilityFactorBin')['VolatilityFactor'].mean().to_dict()
                tmp_data['VolatilityFactorBinValue'] = tmp_data['VolatilityFactorBin'].map(bin_2_value)

                tmp_data.sort_values("ShortMomentumFactorReverse", inplace=True)
                tmp_data['ShortMomentumFactorReverseBin'] = [int(5 * j / valid_count) for j in range(valid_count)]
                tmp_data['ShortMomentumFactorReverseBin'] = tmp_data['ShortMomentumFactorReverseBin'].map(lambda x: min(x, 1))
                bin_2_value = tmp_data.groupby('ShortMomentumFactorReverseBin')['ShortMomentumFactorReverse'].mean().to_dict()
                tmp_data['ShortMomentumFactorReverseBinValue'] = tmp_data['ShortMomentumFactorReverseBin'].map(bin_2_value)

                tmp_data.sort_values("LongMomentumFactorReverse", inplace=True)
                tmp_data['LongMomentumFactorReverseBin'] = [int(5 * j / valid_count) for j in range(valid_count)]
                bin_2_value = tmp_data.groupby('LongMomentumFactorReverseBin')['LongMomentumFactorReverse'].mean().to_dict()

                bin_2_value[0] = bin_2_value[4]
                tmp_data['LongMomentumFactorReverseBinValue'] = tmp_data['LongMomentumFactorReverseBin'].map(
                    bin_2_value)

                tmp_data["CSI500FixedWeightScoreBin{}".format(bin_count)] = tmp_data['ValueFactorBinValue'] + tmp_data[
                    'LiquidityFactorBinValue'] + tmp_data['GrowthFactorBinValue'] * 2 + \
                                              + tmp_data[
                                                         'QualityFactorBinValue'] - tmp_data[
                                                         'LongMomentumFactorReverseBinValue'] + tmp_data[
                                                         'ShortMomentumFactorReverseBinValue'] + 0*tmp_data[
                                                         'VolatilityFactorBinValue']
                # tmp_data["CSI500FixedWeightScoreWithFinanceForecast"] = tmp_data['ValueFactorBinValue'] + tmp_data[
                #     'LiquidityFactorBinValue'] + tmp_data['GrowthFactorBinValue'] * 2 + 0.6*tmp_data['HistFinanceGoodPredTag']\
                #                               - 0.6*tmp_data['HistFinancePoorPredTag'] + tmp_data[
                #                                          'QualityFactorBinValue'] - tmp_data[
                #                                          'LongMomentumFactorReverseBinValue'] + tmp_data[
                #                                          'ShortMomentumFactorReverseBinValue'] + 0*tmp_data[
                #                                          'VolatilityFactorBinValue']
                tmp_data["CSI500FixedWeightScoreBin{}".format(bin_count)] = tmp_data["CSI500FixedWeightScoreBin{}".format(bin_count)].map(lambda x: x * 0.0016)
                # tmp_data["CSI500FixedWeightScoreWithFinanceForecast"] = tmp_data["CSI500FixedWeightScoreWithFinanceForecast"].map(lambda x: x * 0.0016)


            fixed_score_infos.append(tmp_data)
    fixed_score_df = pd.concat(fixed_score_infos)
    print(fixed_score_df[(fixed_score_df.code == "000400.XSHE") & (fixed_score_df.trade_date == 20250909)].T)

    return fixed_score_df.set_index(['code', 'trade_date'])[["CSI500FixedWeightScoreBin2", "CSI500FixedWeightScoreBin3",
                                                             "CSI500FixedWeightScoreBin5", "CSI500FixedWeightScoreBin7",
                                                             "CSI500FixedWeightScoreBin10"]]

def generate_fixed_weight_score_rpt_roe(all_factor_data, start_date, end_date):
    # factor_names = [info['name'] for info in factor_bin_infos]
    factor_names = ["ValueFactor", "LiquidityFactor", "GrowthFactor", "QualityFactor", "LongMomentumFactorReverse",
                     "ShortMomentumFactorReverse", "VolatilityFactor",]
    # import pdb
    # pdb.set_trace()

    for factor_name in factor_names:
        if factor_name == 'OverallMomentumFactor':
            all_factor_data[factor_name] = all_factor_data[factor_name].map(lambda x: x / 2)
        else:
            all_factor_data[factor_name] = all_factor_data[factor_name].map(lambda x: x - 0.5)
    all_factor_data['LeadingFactorValue'] = all_factor_data['Weeks50CountLog'].map(lambda x: 0.1 if x > -1 else -0.1)
    all_factor_data = all_factor_data.reset_index()
    trade_dates = sorted(all_factor_data['trade_date'].unique())
    fixed_score_infos = []
    # bin_count = 10
    for i, date in enumerate(tqdm(trade_dates)):
        if date > start_date and date <= end_date:
            tmp_data = all_factor_data[all_factor_data.trade_date == date]
            for bin_count in [2, 3, 5, 7, 10]:

                tmp_data.sort_values("ValueFactor", inplace=True)
                valid_count = len(tmp_data)

                tmp_data.sort_values("ValueFactor", inplace=True)
                tmp_data['ValueFactorBin'] = [int(bin_count * j/valid_count) for j in range(valid_count)]
                bin_2_value = tmp_data.groupby('ValueFactorBin')['ValueFactor'].mean().to_dict()
                tmp_data['ValueFactorBinValue'] = tmp_data['ValueFactorBin'].map(bin_2_value)

                tmp_data.sort_values("GrowthFactor", inplace=True)
                tmp_data['GrowthFactorBin'] = [int(bin_count * j / valid_count) for j in range(valid_count)]
                #     valid_tmp_df['growth_factor_bin'] = valid_tmp_df['growth_factor_bin'].map(lambda x: min(x, 1))
                bin_2_value = tmp_data.groupby('GrowthFactorBin')['GrowthFactor'].mean().to_dict()
                tmp_data['GrowthFactorBinValue'] = tmp_data['GrowthFactorBin'].map(bin_2_value)
                
#                 tmp_data.sort_values("LeadingMktCapNeutralFactor", inplace=True)
#                 tmp_data['LeadingFactorBin'] = [int(bin_count * j / valid_count) for j in range(valid_count)]
#                 #     valid_tmp_df['growth_factor_bin'] = valid_tmp_df['growth_factor_bin'].map(lambda x: min(x, 1))
#                 bin_2_value = tmp_data.groupby('LeadingFactorBin')['LeadingMktCapNeutralFactor'].mean().to_dict()
#                 tmp_data['LeadingFactorBinValue'] = tmp_data['LeadingFactorBin'].map(bin_2_value)
                
#                 tmp_data.sort_values("LeadingIndustryNeutralFactor", inplace=True)
#                 tmp_data['LeadingFactorBin'] = [int(bin_count * j / valid_count) for j in range(valid_count)]
#                 #     valid_tmp_df['growth_factor_bin'] = valid_tmp_df['growth_factor_bin'].map(lambda x: min(x, 1))
#                 bin_2_value = tmp_data.groupby('LeadingFactorBin')['LeadingIndustryNeutralFactor'].mean().to_dict()
#                 tmp_data['LeadingFactorBinValue'] = tmp_data['LeadingFactorBin'].map(bin_2_value)
#                 tmp_data.sort_values("ROEMktCapNeutralFactor", inplace=True)
#                 tmp_data['ROEFactorBin'] = [int(bin_count * j / valid_count) for j in range(valid_count)]
#                 #     valid_tmp_df['growth_factor_bin'] = valid_tmp_df['growth_factor_bin'].map(lambda x: min(x, 1))
#                 bin_2_value = tmp_data.groupby('ROEFactorBin')['ROEMktCapNeutralFactor'].mean().to_dict()
#                 tmp_data['ROEFactorBinValue'] = tmp_data['ROEFactorBin'].map(bin_2_value)
                
                tmp_data.sort_values("QualityFactor", inplace=True)
                tmp_data['QualityFactorBin'] = [int(bin_count * j / valid_count) for j in range(valid_count)]
                bin_2_value = tmp_data.groupby('QualityFactorBin')['QualityFactor'].mean().to_dict()
                tmp_data['QualityFactorBinValue'] = tmp_data['QualityFactorBin'].map(bin_2_value)

                tmp_data.sort_values("LiquidityFactor", inplace=True)
                tmp_data['LiquidityFactorBin'] = [int(5 * j / valid_count) for j in range(valid_count)]
                tmp_data['LiquidityFactorBin'] = tmp_data['LiquidityFactorBin'].map(lambda x: min(x, 1))
                bin_2_value = tmp_data.groupby('LiquidityFactorBin')['LiquidityFactor'].mean().to_dict()
                tmp_data['LiquidityFactorBinValue'] = tmp_data['LiquidityFactorBin'].map(bin_2_value)

                tmp_data.sort_values("VolatilityFactor", inplace=True)
                tmp_data['VolatilityFactorBin'] = [int(5 * j / valid_count) for j in range(valid_count)]
                tmp_data['VolatilityFactorBin'] = tmp_data['VolatilityFactorBin'].map(lambda x: min(x, 1))
                bin_2_value = tmp_data.groupby('VolatilityFactorBin')['VolatilityFactor'].mean().to_dict()
                tmp_data['VolatilityFactorBinValue'] = tmp_data['VolatilityFactorBin'].map(bin_2_value)

                tmp_data.sort_values("ShortMomentumFactorReverse", inplace=True)
                tmp_data['ShortMomentumFactorReverseBin'] = [int(5 * j / valid_count) for j in range(valid_count)]
                tmp_data['ShortMomentumFactorReverseBin'] = tmp_data['ShortMomentumFactorReverseBin'].map(lambda x: min(x, 1))
                bin_2_value = tmp_data.groupby('ShortMomentumFactorReverseBin')['ShortMomentumFactorReverse'].mean().to_dict()
                tmp_data['ShortMomentumFactorReverseBinValue'] = tmp_data['ShortMomentumFactorReverseBin'].map(bin_2_value)

                tmp_data.sort_values("LongMomentumFactorReverse", inplace=True)
                tmp_data['LongMomentumFactorReverseBin'] = [int(5 * j / valid_count) for j in range(valid_count)]
                bin_2_value = tmp_data.groupby('LongMomentumFactorReverseBin')['LongMomentumFactorReverse'].mean().to_dict()

                bin_2_value[0] = bin_2_value[4]
                tmp_data['LongMomentumFactorReverseBinValue'] = tmp_data['LongMomentumFactorReverseBin'].map(
                    bin_2_value)

                tmp_data["CSI500FixedWeightScoreBin{}".format(bin_count)] = tmp_data['ValueFactorBinValue'] + tmp_data[
                    'LiquidityFactorBinValue'] + tmp_data['GrowthFactorBinValue'] * 2 + \
                                              + tmp_data[
                                                         'QualityFactorBinValue'] - tmp_data[
                                                         'LongMomentumFactorReverseBinValue'] + tmp_data[
                                                         'ShortMomentumFactorReverseBinValue'] + tmp_data[
                                                         'LeadingFactorValue']
                # tmp_data["CSI500FixedWeightScoreWithFinanceForecast"] = tmp_data['ValueFactorBinValue'] + tmp_data[
                #     'LiquidityFactorBinValue'] + tmp_data['GrowthFactorBinValue'] * 2 + 0.6*tmp_data['HistFinanceGoodPredTag']\
                #                               - 0.6*tmp_data['HistFinancePoorPredTag'] + tmp_data[
                #                                          'QualityFactorBinValue'] - tmp_data[
                #                                          'LongMomentumFactorReverseBinValue'] + tmp_data[
                #                                          'ShortMomentumFactorReverseBinValue'] + 0*tmp_data[
                #                                          'VolatilityFactorBinValue']
                tmp_data["CSI500FixedWeightScoreBin{}".format(bin_count)] = tmp_data["CSI500FixedWeightScoreBin{}".format(bin_count)].map(lambda x: x * 0.0016)
                # tmp_data["CSI500FixedWeightScoreWithFinanceForecast"] = tmp_data["CSI500FixedWeightScoreWithFinanceForecast"].map(lambda x: x * 0.0016)


            fixed_score_infos.append(tmp_data)
    fixed_score_df = pd.concat(fixed_score_infos)

    return fixed_score_df.set_index(['code', 'trade_date'])[["CSI500FixedWeightScoreBin2", "CSI500FixedWeightScoreBin3",
                                                             "CSI500FixedWeightScoreBin5", "CSI500FixedWeightScoreBin7",
                                                             "CSI500FixedWeightScoreBin10"]]


def generate_fixed_weight_score_rpt_with_mkt_vol(all_factor_data, mkt_tag, start_date, end_date):
    # factor_names = [info['name'] for info in factor_bin_infos]
    factor_names = ["ValueFactor", "LiquidityFactor", "GrowthFactor", "QualityFactor", "LongMomentumFactorReverse",
                     "ShortMomentumFactorReverse", "VolatilityFactor"]
    # import pdb
    # pdb.set_trace()

    for factor_name in factor_names:
        if factor_name == 'OverallMomentumFactor':
            all_factor_data[factor_name] = all_factor_data[factor_name].map(lambda x: x / 2)
        else:
            all_factor_data[factor_name] = all_factor_data[factor_name].map(lambda x: x - 0.5)
    all_factor_data = all_factor_data.reset_index()
    trade_dates = sorted(all_factor_data['trade_date'].unique())
    fixed_score_infos = []
    mkt_vol_dict = mkt_tag.set_index('trade_date')['vol_20_pct_bin'].to_dict()
    for i, date in enumerate(tqdm(trade_dates)):
        if date > start_date and date <= end_date:
            vol_20_bin = mkt_vol_dict[date]
#             print({"vol {}, date {}".format(vol_20_bin, date)})
            tmp_data = all_factor_data[all_factor_data.trade_date == date]
            for bin_count in [2, 3, 5, 7, 10]:

                tmp_data.sort_values("ValueFactor", inplace=True)
                valid_count = len(tmp_data)

                tmp_data.sort_values("ValueFactor", inplace=True)
                tmp_data['ValueFactorBin'] = [int(bin_count * j/valid_count) for j in range(valid_count)]
                bin_2_value = tmp_data.groupby('ValueFactorBin')['ValueFactor'].mean().to_dict()
                tmp_data['ValueFactorBinValue'] = tmp_data['ValueFactorBin'].map(bin_2_value)

                tmp_data.sort_values("GrowthFactor", inplace=True)
                tmp_data['GrowthFactorBin'] = [int(bin_count * j / valid_count) for j in range(valid_count)]
                #     valid_tmp_df['growth_factor_bin'] = valid_tmp_df['growth_factor_bin'].map(lambda x: min(x, 1))
                bin_2_value = tmp_data.groupby('GrowthFactorBin')['GrowthFactor'].mean().to_dict()
                tmp_data['GrowthFactorBinValue'] = tmp_data['GrowthFactorBin'].map(bin_2_value)

                tmp_data.sort_values("QualityFactor", inplace=True)
                tmp_data['QualityFactorBin'] = [int(bin_count * j / valid_count) for j in range(valid_count)]
                bin_2_value = tmp_data.groupby('QualityFactorBin')['QualityFactor'].mean().to_dict()
                tmp_data['QualityFactorBinValue'] = tmp_data['QualityFactorBin'].map(bin_2_value)

                tmp_data.sort_values("LiquidityFactor", inplace=True)
                tmp_data['LiquidityFactorBin'] = [int(5 * j / valid_count) for j in range(valid_count)]
                tmp_data['LiquidityFactorBin'] = tmp_data['LiquidityFactorBin'].map(lambda x: min(x, 1))
                bin_2_value = tmp_data.groupby('LiquidityFactorBin')['LiquidityFactor'].mean().to_dict()
                tmp_data['LiquidityFactorBinValue'] = tmp_data['LiquidityFactorBin'].map(bin_2_value)

                tmp_data.sort_values("VolatilityFactor", inplace=True)
                tmp_data['VolatilityFactorBin'] = [int(bin_count * j / valid_count) for j in range(valid_count)]
#                 tmp_data['VolatilityFactorBin'] = tmp_data['VolatilityFactorBin'].map(lambda x: min(x, 1))
                bin_2_value = tmp_data.groupby('VolatilityFactorBin')['VolatilityFactor'].mean().to_dict()
                tmp_data['VolatilityFactorBinValue'] = tmp_data['VolatilityFactorBin'].map(bin_2_value)

                tmp_data.sort_values("ShortMomentumFactorReverse", inplace=True)
                tmp_data['ShortMomentumFactorReverseBin'] = [int(5 * j / valid_count) for j in range(valid_count)]
                tmp_data['ShortMomentumFactorReverseBin'] = tmp_data['ShortMomentumFactorReverseBin'].map(lambda x: min(x, 1))
                bin_2_value = tmp_data.groupby('ShortMomentumFactorReverseBin')['ShortMomentumFactorReverse'].mean().to_dict()
                tmp_data['ShortMomentumFactorReverseBinValue'] = tmp_data['ShortMomentumFactorReverseBin'].map(bin_2_value)

                tmp_data.sort_values("LongMomentumFactorReverse", inplace=True)
                tmp_data['LongMomentumFactorReverseBin'] = [int(5 * j / valid_count) for j in range(valid_count)]
                bin_2_value = tmp_data.groupby('LongMomentumFactorReverseBin')['LongMomentumFactorReverse'].mean().to_dict()

                bin_2_value[0] = bin_2_value[4]
                tmp_data['LongMomentumFactorReverseBinValue'] = tmp_data['LongMomentumFactorReverseBin'].map(
                    bin_2_value)

                if vol_20_bin in [1, 2]:


                    tmp_data["CSI500FixedWeightScoreBin{}".format(bin_count)] = tmp_data['ValueFactorBinValue'] + tmp_data[
                        'LiquidityFactorBinValue'] + tmp_data['GrowthFactorBinValue'] * 2 + \
                                                  + tmp_data[
                                                             'QualityFactorBinValue'] - tmp_data[
                                                             'LongMomentumFactorReverseBinValue'] + tmp_data[
                                                             'ShortMomentumFactorReverseBinValue'] + 0*tmp_data[
                                                             'VolatilityFactorBinValue']
                    tmp_data["CSI500FixedWeightScoreBin{}WithVol".format(bin_count)] =tmp_data["CSI500FixedWeightScoreBin{}".format(bin_count)]
                    tmp_data["CSI500FixedWeightScoreBin{}WithVolNeg".format(bin_count)] =tmp_data["CSI500FixedWeightScoreBin{}".format(bin_count)]
                else:
                    tmp_data.sort_values("ShortMomentumFactorReverse", inplace=True)
                    tmp_data['ShortMomentumFactorReverseBin'] = [int(bin_count * j / valid_count) for j in range(valid_count)]
                    bin_2_value = tmp_data.groupby('ShortMomentumFactorReverseBin')['ShortMomentumFactorReverse'].mean().to_dict()
                    tmp_data['ShortMomentumFactorReverseBinValue'] = tmp_data['ShortMomentumFactorReverseBin'].map(bin_2_value)
                    tmp_data["CSI500FixedWeightScoreBin{}".format(bin_count)] = \
                    tmp_data['GrowthFactorBinValue'] + \
                    tmp_data['ShortMomentumFactorReverseBinValue']*2+\
                    tmp_data['ValueFactorBinValue']*0.2+\
                    tmp_data['LiquidityFactorBinValue'] +\
                    tmp_data['QualityFactorBinValue']
                    
                    tmp_data["CSI500FixedWeightScoreBin{}WithVol".format(bin_count)] =tmp_data["CSI500FixedWeightScoreBin{}".format(bin_count)] + tmp_data['VolatilityFactorBinValue']
                    tmp_data["CSI500FixedWeightScoreBin{}WithVolNeg".format(bin_count)] =tmp_data["CSI500FixedWeightScoreBin{}".format(bin_count)] - tmp_data['VolatilityFactorBinValue']                    
                tmp_data["CSI500FixedWeightScoreBin{}".format(bin_count)] = tmp_data["CSI500FixedWeightScoreBin{}".format(bin_count)].map(lambda x: x * 0.0016)
                tmp_data["CSI500FixedWeightScoreBin{}WithVol".format(bin_count)] = tmp_data["CSI500FixedWeightScoreBin{}WithVol".format(bin_count)].map(lambda x: x * 0.0016)
                tmp_data["CSI500FixedWeightScoreBin{}WithVolNeg".format(bin_count)] = tmp_data["CSI500FixedWeightScoreBin{}WithVolNeg".format(bin_count)].map(lambda x: x * 0.0016)                
                # tmp_data["CSI500FixedWeightScoreWithFinanceForecast"] = tmp_data["CSI500FixedWeightScoreWithFinanceForecast"].map(lambda x: x * 0.0016)

            tmp_data['vol_20_bin'] = vol_20_bin
            fixed_score_infos.append(tmp_data)
    fixed_score_df = pd.concat(fixed_score_infos)

    return fixed_score_df.set_index(['code', 'trade_date'])[["CSI500FixedWeightScoreBin2", "CSI500FixedWeightScoreBin3",
                                                             "CSI500FixedWeightScoreBin5", "CSI500FixedWeightScoreBin7",
                                                             "CSI500FixedWeightScoreBin10", "vol_20_bin", "VolatilityFactorBinValue", "CSI500FixedWeightScoreBin3WithVol", "CSI500FixedWeightScoreBin3WithVolNeg"]]



def add_non_linear_penalty_factor(factor_data, penalty_factor, score_name, penalty_interval, penalty_weight):
    factor_data['{}_group'.format(penalty_factor)] = factor_data[penalty_factor].groupby(level = 'trade_date').apply(lambda x: discretize(x,penalty_interval))
    factor_data['{}_group'.format(penalty_factor)] = factor_data['{}_group'.format(penalty_factor)].astype(int)
    factor_data['{}_group'.format(penalty_factor)] = factor_data['{}_group'.format(penalty_factor)].replace({-2147483648:0})
    factor_data['{}_non_linear'.format(penalty_factor)] = factor_data['{}_group'.format(penalty_factor)] * factor_data[penalty_factor]
    factor_data["{}_non_linear".format(score_name)] = factor_data[score_name]  + penalty_weight * factor_data['{}_non_linear'.format(penalty_factor)]

    return factor_data
    
    
    



def generate_fixed_weight_score_v2(all_factor_data, start_date, end_date):
    # factor_names = [info['name'] for info in factor_bin_infos]
    factor_names = ["ValueFactor", "LiquidityFactor", "GrowthFactor", "QualityFactor", "LongMomentumFactorReverse",
                     "ShortMomentumFactorReverse", "VolatilityFactor", "HistFinanceGoodPredTag",
                    "HistFinancePoorPredTag"]
    # import pdb
    # pdb.set_trace()
    for factor_name in factor_names:
        if factor_name == 'OverallMomentumFactor':
            all_factor_data[factor_name] = all_factor_data[factor_name].map(lambda x: x / 2)
        else:
            all_factor_data[factor_name] = all_factor_data[factor_name].map(lambda x: x - 0.5)
    all_factor_data = all_factor_data.reset_index()
    trade_dates = sorted(all_factor_data['trade_date'].unique())
    fixed_score_infos = []
    bin_count = 10
    for i, date in enumerate(tqdm(trade_dates)):
        if date > start_date and date <= end_date:
            tmp_data = all_factor_data[all_factor_data.trade_date == date]
            valid_count = len(tmp_data)
            tmp_data.sort_values("ValueFactor", inplace=True)
            tmp_data['ValueFactorBin'] = [int(3 * j / valid_count) for j in range(valid_count)]
            tmp_data['ValueFactorBinValue'] = tmp_data['ValueFactorBin'].map(lambda x: (x - 1) / 2)
            #     bin_2_value = tmp_data.groupby('ValueFactorBin')['ValueFactor'].mean().to_dict()
            #     tmp_data['ValueFactorBinValue'] = tmp_data['ValueFactorBin'].map(bin_2_value)

            tmp_data.sort_values("GrowthFactor", inplace=True)
            tmp_data['GrowthFactorBin'] = [int(3 * j / valid_count) for j in range(valid_count)]
            tmp_data['GrowthFactorBinValue'] = tmp_data['GrowthFactorBin'].map(lambda x: (x - 1) / 2)
            #     valid_tmp_df['growth_factor_bin'] = valid_tmp_df['growth_factor_bin'].map(lambda x: min(x, 1))
            #     bin_2_value = tmp_data.groupby('GrowthFactorBin')['GrowthFactor'].mean().to_dict()
            #     tmp_data['GrowthFactorBinValue'] = tmp_data['GrowthFactorBin'].map(bin_2_value)

            tmp_data.sort_values("QualityFactor", inplace=True)
            tmp_data['QualityFactorBin'] = [int(3 * j / valid_count) for j in range(valid_count)]
            tmp_data['QualityFactorBinValue'] = tmp_data['QualityFactorBin'].map(lambda x: (x - 1) / 2)
            #     bin_2_value = tmp_data.groupby('QualityFactorBin')['QualityFactor'].mean().to_dict()
            #     tmp_data['QualityFactorBinValue'] = tmp_data['QualityFactorBin'].map(bin_2_value)

            tmp_data.sort_values("LiquidityFactor", inplace=True)
            tmp_data['LiquidityFactorBin'] = [int(10 * j / valid_count) for j in range(valid_count)]
            tmp_data['LiquidityFactorBin'] = tmp_data['LiquidityFactorBin'].map(lambda x: min(x, 1))
            tmp_data['LiquidityFactorBinValue'] = tmp_data['LiquidityFactorBin'].map(lambda x: x - 0.5)

            #     bin_2_value = tmp_data.groupby('LiquidityFactorBin')['LiquidityFactor'].mean().to_dict()
            #     tmp_data['LiquidityFactorBinValue'] = tmp_data['LiquidityFactorBin'].map(bin_2_value)

            tmp_data.sort_values("VolatilityFactor", inplace=True, ascending=False)

            tmp_data['VolatilityFactorBin'] = [int(10 * j / valid_count) for j in range(valid_count)]
            tmp_data['VolatilityFactorBin'] = tmp_data['VolatilityFactorBin'].map(lambda x: min(x, 1))
            tmp_data['VolatilityFactorBinValue'] = tmp_data['VolatilityFactorBin'].map(lambda x: x - 0.5)

            #     bin_2_value = tmp_data.groupby('VolatilityFactorBin')['VolatilityFactor'].mean().to_dict()
            #     tmp_data['VolatilityFactorBinValue'] = tmp_data['VolatilityFactorBin'].map(bin_2_value)

            tmp_data.sort_values("ShortMomentumFactorReverse", inplace=True)
            tmp_data['ShortMomentumFactorReverseBin'] = [int(10 * j / valid_count) for j in range(valid_count)]
            tmp_data['ShortMomentumFactorReverseBin'] = tmp_data['ShortMomentumFactorReverseBin'].map(
                lambda x: min(x, 1))
            tmp_data['ShortMomentumFactorReverseBinValue'] = tmp_data['ShortMomentumFactorReverseBin'].map(
                lambda x: x - 0.5)

            #     bin_2_value = tmp_data.groupby('ShortMomentumFactorReverseBin')['ShortMomentumFactorReverse'].mean().to_dict()
            #     tmp_data['ShortMomentumFactorReverseBinValue'] = tmp_data['ShortMomentumFactorReverseBin'].map(bin_2_value)

            tmp_data.sort_values("LongMomentumFactorReverse", inplace=True, ascending=False)
            tmp_data['LongMomentumFactorReverseBin'] = [int(3 * j / valid_count) for j in range(valid_count)]
            tmp_data['LongMomentumFactorReverseBinValue'] = tmp_data['LongMomentumFactorReverseBin'].map(
                lambda x: (x - 1) / 2)


            tmp_data['GrowthMomentumScoreBinValue'] = (tmp_data['GrowthFactorBinValue'] * 0.8 * 3 + tmp_data[
                'LongMomentumFactorReverseBinValue'] * 0.2 * 3) + (tmp_data['ValueFactorBinValue'] * 0.25 + tmp_data[
                'ShortMomentumFactorReverseBinValue'] * 0.25 + tmp_data['VolatilityFactorBinValue'] * 0.25 + tmp_data[
                                                                       'LiquidityFactorBinValue'] * 0.25) + tmp_data[
                                                          'QualityFactorBinValue'] * 0.2
            tmp_data['ValueReverseScoreBinValue'] = (tmp_data['GrowthFactorBinValue'] * 0.8 + tmp_data[
                'LongMomentumFactorReverseBinValue'] * 0.2) + (tmp_data['ValueFactorBinValue'] * 0.25 + tmp_data[
                'ShortMomentumFactorReverseBinValue'] * 0.25 + tmp_data['VolatilityFactorBinValue'] * 0.25 + tmp_data[
                                                                   'LiquidityFactorBinValue'] * 0.25) * 3 + tmp_data[
                                                        'QualityFactorBinValue'] * 0.2
            tmp_data['AllFactorScoreBinValue'] = (tmp_data['GrowthFactorBinValue'] * 0.8 + tmp_data[
                'LongMomentumFactorReverseBinValue'] * 0.2) + (tmp_data['ValueFactorBinValue'] * 0.25 + tmp_data[
                'ShortMomentumFactorReverseBinValue'] * 0.25 + tmp_data['VolatilityFactorBinValue'] * 0.25 + tmp_data[
                                                                   'LiquidityFactorBinValue'] * 0.25) + tmp_data[
                                                     'QualityFactorBinValue'] * 0.2
            tmp_data['ValueReverseScoreBinValue'] = tmp_data['ValueReverseScoreBinValue']/tmp_data['ValueReverseScoreBinValue'].std() * 0.0016
            tmp_data['GrowthMomentumScoreBinValue'] = tmp_data['GrowthMomentumScoreBinValue']/tmp_data['GrowthMomentumScoreBinValue'].std() * 0.0016

            tmp_data['AllFactorScoreBinValue'] = tmp_data['AllFactorScoreBinValue']/tmp_data['AllFactorScoreBinValue'].std() * 0.0016
            fixed_score_infos.append(tmp_data)
    fixed_score_df = pd.concat(fixed_score_infos)
    return fixed_score_df.set_index(['code', 'trade_date'])[["AllFactorScoreBinValue",
                                                             "ValueReverseScoreBinValue", "GrowthMomentumScoreBinValue"
                                                             ]]


def generate_fixed_score_4_top_leading(all_factor_data, start_date, end_date):
    """
    针对龙头股策略给出特别的打分
    """

    def std_reverse_factor(x):
        if x > 0.1:
            return 1
        else:
            return -1

    def std_momentum_factor(x):
        if x > 0.8:
            return -1
        elif x > 0.5:
            return 0
        else:
            return 2

    def std_liq_factor(x):
        if x > 0.7:
            return 2
        elif x > 0.4:
            return 0
        else:
            return -1

    def std_value_factor(x):
        if x > 0.7:
            return 1
        elif x > 0.4:
            return 0
        else:
            return -1

    def std_growth_factor(x):
        if x > 0.6:
            return 1
        elif x > 0.2:
            return 0
        else:
            return -1

    def std_rpt_factor(x):
        if x > 0.7:
            return 1
        else:
            return 0

    def std_quality_factor(x):
        if x > 0.7:
            return 1
        elif x > 0.3:
            return 0
        else:
            return -1

    all_factor_data = all_factor_data.reset_index()
    all_factor_data['all_factor'] = all_factor_data.groupby('trade_date')[
                                                 'ShortMomentumFactorReverse'].rank(pct=True).map(
        std_reverse_factor) + all_factor_data.groupby('trade_date')['LongMomentumFactorReverse'].rank(
        pct=True).map(std_momentum_factor) + all_factor_data.groupby('trade_date')['LiquidityFactor'].rank(
        pct=True).map(std_liq_factor) + all_factor_data.groupby('trade_date')['LiquidityFactor'].rank(
        pct=True).map(std_liq_factor) + all_factor_data.groupby('trade_date')['ValueFactor'].rank(
        pct=True).map(std_value_factor) + all_factor_data.groupby('trade_date')['GrowthFactor'].rank(
        pct=True).map(std_growth_factor) + all_factor_data.groupby('trade_date')['Weeks50CountLog'].rank(
        pct=True).map(std_rpt_factor) + all_factor_data.groupby('trade_date')['QualityFactor'].rank(
        pct=True).map(std_quality_factor)

    all_factor_data['all_factor_v2'] = all_factor_data.groupby('trade_date')[
                                                 'ShortMomentumFactorReverse'].rank(pct=True).map(
        std_reverse_factor) + all_factor_data.groupby('trade_date')['LongMomentumFactorReverse'].rank(
        pct=True).map(std_momentum_factor) + all_factor_data.groupby('trade_date')['LiquidityFactor'].rank(
        pct=True).map(std_liq_factor) + all_factor_data.groupby('trade_date')['ValueFactor'].rank(
        pct=True).map(std_value_factor) + all_factor_data.groupby('trade_date')['GrowthFactor'].rank(
        pct=True).map(std_growth_factor) + all_factor_data.groupby('trade_date')['Weeks50CountLog'].rank(
        pct=True).map(std_rpt_factor) + all_factor_data.groupby('trade_date')['QualityFactor'].rank(
        pct=True).map(std_quality_factor)
    all_factor_data = all_factor_data[all_factor_data.trade_date.map(lambda date: date > start_date and date <= end_date)]
    all_factor_data['all_factor_score'] = all_factor_data.groupby('trade_date')['all_factor'].apply(lambda x: (x-x.mean()/x.std()) * 0.0016)
    all_factor_data['all_factor_score_v2'] = all_factor_data.groupby('trade_date')['all_factor_v2'].apply(lambda x: (x-x.mean()/x.std()) * 0.0016)

    return all_factor_data.set_index(['code', 'trade_date'])[['all_factor_score', 'all_factor_score_v2']]


class FixedWeightFactor(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.start_date = param_info['start_date']
        self.end_date = param_info['end_date']
        self.source_data_infos = param_info['source_data_infos']
        self.invalid_infos = param_info['invalid_infos']

        # self.factor_names = param_info['factor_names']
        self.save_info = param_info['save_info']
        # self.output_score_name = param_info['output_score_name']
        self.gen_fixed_weight_score_func = param_info['gen_fixed_weight_score_func']
        self.operators = [
            {
                "func": get_data_from_multi_source,
                "param": {
                    "data_source_infos": self.source_data_infos,
                    "start_date": self.start_date,
                    "end_date": self.end_date
                },
                "input_data": {},
                "output": ['factor']
            },
            {
                "func": transfer_data_to_valid_and_not_valid,
                "param": {
                    "invalid_infos": self.invalid_infos

                },
                "input_data": {"data": "factor"},
                "output": ['valid_factor', 'invalid_factor']
            },
            {
                "func": self.gen_fixed_weight_score_func,
                "param": {
                    # "factor_names": self.factor_names,
                    "start_date": self.start_date,
                    "end_date": self.end_date,
                    # "output_score_name": self.output_score_name,

                },
                "input_data": {"all_factor_data": "valid_factor"},
                "output": ["overall_score"],
            },
            {
                "func": save_data_to_table,
                "param": {"engine": self.save_info['engine'], "table": self.save_info['table'],
                          "if_exists": self.save_info.get("if_exists", "append")},
                "input_data": {"data": "overall_score"},
                "output": ["overall_score"]
            }
        ]

        self.output_vars = ["overall_score"]

        
class AddNonLinearPenaltyFactor(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.start_date = param_info['start_date']
        self.end_date = param_info['end_date']
        self.source_data_infos = param_info['source_data_infos']
        self.invalid_infos = param_info['invalid_infos']

        # self.factor_names = param_info['factor_names']
        self.save_info = param_info['save_info']
        # self.output_score_name = param_info['output_score_name']
        self.add_penalty_func = param_info['add_penalty_func']
        
        self.penalty_param = param_info['penalty_param']
        
        self.operators = [
            {
                "func": get_data_from_multi_source,
                "param": {
                    "data_source_infos": self.source_data_infos,
                    "start_date": self.start_date,
                    "end_date": self.end_date
                },
                "input_data": {},
                "output": ['factor']
            },
            {
                "func": transfer_data_to_valid_and_not_valid,
                "param": {
                    "invalid_infos": self.invalid_infos

                },
                "input_data": {"data": "factor"},
                "output": ['valid_factor', 'invalid_factor']
            },
            {
                "func": self.add_penalty_func,
                "param": self.penalty_param,
                "input_data": {"factor_data": "valid_factor"},
                "output": ["overall_score"],
            },
            {
                "func": save_data_to_table,
                "param": {"engine": self.save_info['engine'], "table": self.save_info['table'],
                          "if_exists": self.save_info.get("if_exists", "append")},
                "input_data": {"data": "overall_score"},
                "output": ["overall_score"]
            }
        ]

        self.output_vars = ["overall_score"]

        
class FixedWeightMktTagFactor(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.start_date = param_info['start_date']
        self.end_date = param_info['end_date']
        self.source_data_infos = param_info['source_data_infos']
        self.invalid_infos = param_info['invalid_infos']

        # self.factor_names = param_info['factor_names']
        self.save_info = param_info['save_info']
        # self.output_score_name = param_info['output_score_name']
        self.gen_fixed_weight_score_func = param_info['gen_fixed_weight_score_func']
        self.gen_mkt_tag_func = param_info['gen_mkt_tag_func']
        self.index_name = param_info.get("index_name", "000905.XSHG")
        self.operators = [
            {
                "func": get_data_from_multi_source,
                "param": {
                    "data_source_infos": self.source_data_infos,
                    "start_date": self.start_date,
                    "end_date": self.end_date
                },
                "input_data": {},
                "output": ['factor']
            },
            {
                "func": transfer_data_to_valid_and_not_valid,
                "param": {
                    "invalid_infos": self.invalid_infos

                },
                "input_data": {"data": "factor"},
                "output": ['valid_factor', 'invalid_factor']
            },
            {
                "func": self.gen_mkt_tag_func,
                "param": {
                    "index_name": self.index_name
                },
                "input_data": {},
                "output": ["mkt_tag"],
            },
            {
                "func": self.gen_fixed_weight_score_func,
                "param": {
                    # "factor_names": self.factor_names,
                    "start_date": self.start_date,
                    "end_date": self.end_date,
                    # "output_score_name": self.output_score_name,

                },
                "input_data": {"all_factor_data": "valid_factor", "mkt_tag": "mkt_tag"},
                "output": ["overall_score"],
            },
            {
                "func": save_data_to_table,
                "param": {"engine": self.save_info['engine'], "table": self.save_info['table'],
                          "if_exists": self.save_info.get("if_exists", "append")},
                "input_data": {"data": "overall_score"},
                "output": ["overall_score"]
            }
        ]

        self.output_vars = ["overall_score"]

class FactorMomentumPerformanceAttributionNew(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.start_date = param_info['start_date']
        self.end_date = param_info['end_date']
        self.source_data_infos = param_info['source_data_infos']
        self.invalid_infos = param_info['invalid_infos']
        self.long_window_size = param_info['long_window_size']
        self.short_window_size = param_info['short_window_size']
        self.is_norm = param_info.get('is_norm', False)
        self.industry_name = param_info.get("industry_name", "GicsIndustryName")
        self.perf_att_func = param_info.get("perf_att_func", weekly_performance_attribution)
        self.gen_score_func = param_info.get("gen_score_func", generate_score_from_perf_attribution)
        self.save_info = param_info['save_info']
        self.factors_4_performance_attribution = param_info['factors_4_performance_attribution']
        self.factors_4_score = param_info['factors_4_score']
        self.factor_directions = param_info.get("factor_directions", [1 for _ in self.factors_4_score])
        self.r_process_param = param_info.get("r_process_param", {})

        self.operators = [
            {
                "func": get_data_from_multi_source,
                "param": {
                    "data_source_infos": self.source_data_infos,
                    "start_date": self.start_date,
                    "end_date": self.end_date
                },
                "input_data": {},
                "output": ['factor']
            },
            {
                "func": generate_one_term_return,
                "param": {
                    "one_week_momentum_name": "MomentumWeeks1",
                    "output_name": "OneTermReturn",
                    "limit": self.r_process_param.get("limit", 0),
                    "is_3_sigma_std": self.r_process_param.get("is_3_sigma_std", False),
                    "is_ecdf": self.r_process_param.get("is_ecdf", False)
                },
                "input_data": {'data': 'factor'},
                "output": ['factor']
            },
            {
                "func": transfer_data_to_valid_and_not_valid,
                "param": {
                    "invalid_infos": self.invalid_infos

                },
                "input_data": {"data": "factor"},
                "output": ['valid_factor', 'invalid_factor']
            },
            {
                "func": self.perf_att_func,
                "param": {
                    "factors": self.factors_4_performance_attribution,
                    "industry": self.industry_name,
                },
                "input_data": {"factor_data": "valid_factor"},
                "output": ["factor_premium", 'processed_data'],
            },
            {
                "func": self.gen_score_func,
                "param": {
                    "start_date": self.start_date,
                    "end_date": self.end_date,
                    "factors": self.factors_4_score,
                    "factor_directions": self.factor_directions,
                    "long_window_size": self.long_window_size,
                    "short_window_size": self.short_window_size,
                    "is_norm": self.is_norm,
                },
                "input_data": {
                    "factor_premium_df": "factor_premium",
                    "processed_factor_df": "processed_data"
                },
                "output": ["all_data_score"]
            },
            {
                "func": save_data_to_table,
                "param": {"engine": self.save_info['engine'], "table": self.save_info['table'],
                          "if_exists": self.save_info.get("if_exists", "append")},
                "input_data": {"data": "all_data_score"},
                "output": ["all_data_score"]
            }
        ]

        self.output_vars = ["all_data_score"]

        
        
class FactorMomentumPerformanceAttribution(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.start_date = param_info['start_date']
        self.end_date = param_info['end_date']
        self.source_data_infos = param_info['source_data_infos']
        self.invalid_infos = param_info['invalid_infos']
        self.long_window_size = param_info['long_window_size']
        self.short_window_size = param_info['short_window_size']
        self.is_norm = param_info.get('is_norm', False)
        self.industry_name = param_info.get("industry_name", "GicsIndustryName")
        self.perf_att_func = param_info.get("perf_att_func", weekly_performance_attribution)
        self.save_info = param_info['save_info']
        self.factors_4_performance_attribution = param_info['factors_4_performance_attribution']
        self.factors_4_score = param_info['factors_4_score']
        self.factor_directions = param_info.get("factor_directions", [1 for _ in self.factors_4_score])
        self.r_process_param = param_info.get("r_process_param", {})

        self.operators = [
            {
                "func": get_data_from_multi_source,
                "param": {
                    "data_source_infos": self.source_data_infos,
                    "start_date": self.start_date,
                    "end_date": self.end_date
                },
                "input_data": {},
                "output": ['factor']
            },
            {
                "func": generate_one_term_return,
                "param": {
                    "one_week_momentum_name": "MomentumWeeks1",
                    "output_name": "OneTermReturn",
                    "limit": self.r_process_param.get("limit", 0),
                    "is_3_sigma_std": self.r_process_param.get("is_3_sigma_std", False),
                    "is_ecdf": self.r_process_param.get("is_ecdf", False)
                },
                "input_data": {'data': 'factor'},
                "output": ['factor']
            },
            {
                "func": transfer_data_to_valid_and_not_valid,
                "param": {
                    "invalid_infos": self.invalid_infos

                },
                "input_data": {"data": "factor"},
                "output": ['valid_factor', 'invalid_factor']
            },
            {
                "func": self.perf_att_func,
                "param": {
                    "factors": self.factors_4_performance_attribution,
                    "industry": self.industry_name,
                },
                "input_data": {"factor_data": "valid_factor"},
                "output": ["factor_premium", 'processed_data'],
            },
            {
                "func": generate_score_from_perf_attribution,
                "param": {
                    "start_date": self.start_date,
                    "end_date": self.end_date,
                    "factors": self.factors_4_score,
                    "factor_directions": self.factor_directions,
                    "long_window_size": self.long_window_size,
                    "short_window_size": self.short_window_size,
                    "is_norm": self.is_norm,
                },
                "input_data": {
                    "factor_premium_df": "factor_premium",
                    "processed_factor_df": "processed_data"
                },
                "output": ["all_data_score"]
            },
            {
                "func": save_data_to_table,
                "param": {"engine": self.save_info['engine'], "table": self.save_info['table'],
                          "if_exists": self.save_info.get("if_exists", "append")},
                "input_data": {"data": "all_data_score"},
                "output": ["all_data_score"]
            }
        ]

        self.output_vars = ["all_data_score"]

class FactorMomentumPerformanceAttributionHs300(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.start_date = param_info['start_date']
        self.end_date = param_info['end_date']
        self.source_data_infos = param_info['source_data_infos']
        self.invalid_infos = param_info['invalid_infos']
        self.long_window_size = param_info['long_window_size']
        self.short_window_size = param_info['short_window_size']
        self.industry_name = param_info.get("industry_name", "GicsIndustryName")
        self.perf_att_func = param_info.get("perf_att_func", weekly_performance_attribution)
        self.save_info = param_info['save_info']
        self.factors_4_performance_attribution = param_info['factors_4_performance_attribution']
        self.factors_4_score = param_info['factors_4_score']
        self.r_process_param = param_info.get("r_process_param", {})

        self.operators = [
            {
                "func": get_data_from_multi_source,
                "param": {
                    "data_source_infos": self.source_data_infos,
                    "start_date": self.start_date,
                    "end_date": self.end_date
                },
                "input_data": {},
                "output": ['factor']
            },
            {
                "func": generate_one_term_return,
                "param": {
                    "one_week_momentum_name": "MomentumWeeks1",
                    "output_name": "OneTermReturn",
                },
                "input_data": {'data': 'factor'},
                "output": ['factor']
            },
            {
                "func": process_one_term_return,
                "param": {
                    "output_name": "_10amOneTermReturn",
                    "limit": self.r_process_param.get("limit", 0),
                    "is_3_sigma_std": self.r_process_param.get("is_3_sigma_std", False),
                    "is_ecdf": self.r_process_param.get("is_ecdf", False)
                },
                "input_data": {"data": "factor"},
                "output": ["factor"],
            },
            {
                "func": process_one_term_return,
                "param": {
                    "output_name": "_10amOneTermReturn4LastWeek",
                    "limit": self.r_process_param.get("limit", 0),
                    "is_3_sigma_std": self.r_process_param.get("is_3_sigma_std", False),
                    "is_ecdf": self.r_process_param.get("is_ecdf", False)
                },
                "input_data": {"data": "factor"},
                "output": ["factor"],
            },
            {
                "func": transfer_data_to_valid_and_not_valid,
                "param": {
                    "invalid_infos": self.invalid_infos

                },
                "input_data": {"data": "factor"},
                "output": ['valid_factor', 'invalid_factor']
            },
            {
                "func": self.perf_att_func,
                "param": {
                    "factors": self.factors_4_performance_attribution,
                    "industry": self.industry_name,
                },
                "input_data": {"factor_data": "valid_factor"},
                "output": ["factor_premium", "factor_premium_last_week", 'processed_data'],
            },
            {
                "func": generate_score_from_perf_attribution_hs300,
                "param": {
                    "start_date": self.start_date,
                    "end_date": self.end_date,
                    "factors": self.factors_4_score,
                    "long_window_size": self.long_window_size,
                    "short_window_size": self.short_window_size,
                },
                "input_data": {
                    "factor_premium_df": "factor_premium",
                    "factor_premium_last_week_df": "factor_premium_last_week",
                    "processed_factor_df": "processed_data"
                },
                "output": ["all_data_score"]
            },
            {
                "func": save_data_to_table,
                "param": {"engine": self.save_info['engine'], "table": self.save_info['table'],
                          "if_exists": self.save_info.get("if_exists", "append")},
                "input_data": {"data": "all_data_score"},
                "output": ["all_data_score"]
            }
        ]

        self.output_vars = ["all_data_score"]
                                                                                   
class MktVolFactorMomentumPerformanceAttribution(FactorCompute):
    def __init__(self, param_info, input_name_mapping, output_name_mapping):
        super().__init__(param_info, input_name_mapping, output_name_mapping)
        self.start_date = param_info['start_date']
        self.end_date = param_info['end_date']
        self.source_data_infos = param_info['source_data_infos']
        self.invalid_infos = param_info['invalid_infos']
        self.long_window_size = param_info['long_window_size']
        self.short_window_size = param_info['short_window_size']
        self.industry_name = param_info.get("industry_name", "GicsIndustryName")

        self.save_info = param_info['save_info']
        self.factors_4_performance_attribution = param_info['factors_4_performance_attribution']
        self.factors_4_score = param_info['factors_4_score']
        self.gen_mkt_tag_func = param_info['gen_mkt_tag_func']
        self.index_name = param_info.get('index_name', '000905.XSHG')
        self.bin_count = param_info.get("bin_count", 2)
        self.operators = [
            {
                "func": get_data_from_multi_source,
                "param": {
                    "data_source_infos": self.source_data_infos,
                    "start_date": self.start_date,
                    "end_date": self.end_date
                },
                "input_data": {},
                "output": ['factor']
            },
            {
                "func": generate_one_term_return,
                "param": {
                    "one_week_momentum_name": "MomentumWeeks1",
                    "output_name": "OneTermReturn",
                },
                "input_data": {'data': 'factor'},
                "output": ['factor']
            },
            {
                "func": transfer_data_to_valid_and_not_valid,
                "param": {
                    "invalid_infos": self.invalid_infos

                },
                "input_data": {"data": "factor"},
                "output": ['valid_factor', 'invalid_factor']
            },
            {
                "func": weekly_performance_attribution_mkt_vol_tag,
                "param": {
                    "factors": self.factors_4_performance_attribution,
                    "industry": self.industry_name,
                },
                "input_data": {"factor_data": "valid_factor"},
                "output": ["factor_premium", 'processed_data'],
            },
            {
                "func": self.gen_mkt_tag_func,
                "param": {
                    "index_name": self.index_name,
                    "bin_count": self.bin_count,
                },
                "input_data": {},
                "output": ["index_vol_tag"],
            },
            {
                "func": generate_score_from_perf_attribution_with_mkt_vol,
                "param": {
                    "start_date": self.start_date,
                    "end_date": self.end_date,
                    "factors": self.factors_4_score,
                    "long_window_size": self.long_window_size,
                    "short_window_size": self.short_window_size,
                },
                "input_data": {
                    "factor_premium_df": "factor_premium",
                    "processed_factor_df": "processed_data",
                    "mkt_vol": "index_vol_tag",
                },
                "output": ["all_data_score"]
            },
            {
                "func": save_data_to_table,
                "param": {"engine": self.save_info['engine'], "table": self.save_info['table'],
                          "if_exists": self.save_info.get("if_exists", "append")},
                "input_data": {"data": "all_data_score"},
                "output": ["all_data_score"]
            }
        ]

        self.output_vars = ["all_data_score"]

        
        