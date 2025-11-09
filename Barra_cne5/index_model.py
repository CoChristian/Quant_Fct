import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Dict, Tuple, Optional
from barra_cne5_factor import GetData

class IndexEnhancedOptimizer:
    """
    指数增强优化器
    在控制跟踪误差的前提下最大化因子暴露
    """

    def __init__(self, te_target: float = 0.05, w_max: float = 0.05):
        self.te_target = te_target  # 年化跟踪误差目标
        self.w_max = w_max  # 个股权重上限

    def optimize(self,
                 alpha_scores: np.ndarray,
                 benchmark_weights: np.ndarray,
                 covariance_matrix: np.ndarray,
                 industry_exposure: Optional[np.ndarray] = None,
                 style_exposure: Optional[np.ndarray] = None,
                 industry_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 style_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict:
        """
        执行优化

        参数:
        alpha_scores: 股票在Alpha因子上的暴露度 [n_assets]
        benchmark_weights: 基准权重 [n_assets]
        covariance_matrix: 协方差矩阵 [n_assets, n_assets]
        industry_exposure: 行业暴露矩阵 [n_assets, n_industries]
        style_exposure: 风格暴露矩阵 [n_assets, n_styles]
        industry_bounds: 行业暴露上下限 (lower, upper) [n_industries]
        style_bounds: 风格暴露上下限 (lower, upper) [n_styles]
        """
        n_assets = len(alpha_scores)

        # 优化变量：组合权重
        w = cp.Variable(n_assets)

        # 主动权重
        w_active = w - benchmark_weights

        # 目标函数：最大化Alpha暴露
        objective = cp.Maximize(w @ alpha_scores)

        # 约束条件
        constraints = []

        # 基础权重约束
        constraints.append(cp.sum(w) == 1)  # 完全投资
        constraints.append(w >= 0)  # 不允许做空
        constraints.append(w <= self.w_max)  # 个股权重上限

        # 跟踪误差约束 (使用二次约束)
        te_constraint = cp.quad_form(w_active, covariance_matrix) <= self.te_target ** 2
        constraints.append(te_constraint)

        # 行业暴露约束
        if industry_exposure is not None and industry_bounds is not None:
            industry_weights = industry_exposure.T @ w
            industry_lower, industry_upper = industry_bounds
            constraints.append(industry_weights >= industry_lower)
            constraints.append(industry_weights <= industry_upper)

        # 风格暴露约束
        if style_exposure is not None and style_bounds is not None:
            style_weights = style_exposure.T @ w
            style_lower, style_upper = style_bounds
            constraints.append(style_weights >= style_lower)
            constraints.append(style_weights <= style_upper)

        # 求解优化问题
        problem = cp.Problem(objective, constraints)
        problem.solve()

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"优化失败: {problem.status}")

        # 整理结果
        optimal_weights = w.value
        active_weights = optimal_weights - benchmark_weights

        results = {
            'weights': optimal_weights,
            'active_weights': active_weights,
            'alpha_exposure': optimal_weights @ alpha_scores,
            'predicted_te': np.sqrt(active_weights @ covariance_matrix @ active_weights),
            'objective_value': problem.value,
            'industry_exposure': industry_exposure.T @ optimal_weights if industry_exposure is not None else None,
            'style_exposure': style_exposure.T @ optimal_weights if style_exposure is not None else None
        }

        return results


# ============================================================================
# 示例使用
# ============================================================================

def generate_sample_data(n_assets=500, n_industries=10, n_styles=5):
    """生成示例数据"""
    np.random.seed(42)

    # 生成Alpha因子暴露 (换手率收敛因子)
    alpha_scores = np.random.normal(0, 1, n_assets)

    # 生成基准权重 (近似市值加权)
    benchmark_weights = np.random.exponential(1, n_assets)
    benchmark_weights = benchmark_weights / benchmark_weights.sum()

    # 生成协方差矩阵
    # 先生成因子结构
    factor_volatility = np.array([0.2, 0.15, 0.1])  # 3个因子的波动率
    factor_correlation = np.array([[1.0, 0.3, -0.2],
                                   [0.3, 1.0, 0.1],
                                   [-0.2, 0.1, 1.0]])
    factor_cov = np.diag(factor_volatility) @ factor_correlation @ np.diag(factor_volatility)

    # 股票对因子的暴露
    factor_exposure = np.random.normal(0, 1, (n_assets, 3))

    # 特异性风险
    specific_risk = np.random.uniform(0.1, 0.3, n_assets)

    # 完整的协方差矩阵
    covariance_matrix = factor_exposure @ factor_cov @ factor_exposure.T + np.diag(specific_risk ** 2)

    # 行业暴露矩阵 (哑变量)
    industry_assignments = np.random.choice(n_industries, n_assets)
    industry_exposure = np.eye(n_industries)[industry_assignments]

    # 风格暴露矩阵
    style_exposure = np.random.normal(0, 1, (n_assets, n_styles))

    return {
        'alpha_scores': alpha_scores,
        'benchmark_weights': benchmark_weights,
        'covariance_matrix': covariance_matrix,
        'industry_exposure': industry_exposure,
        'style_exposure': style_exposure
    }



# get index data

index_data = pd.read_parquet("F:\\work\\Projectpycharm\\MA_factor\\factor_data\\INDEX\\index_000300.parquet")
# cal index weight
valuation_df = GetData._get_valuation()[['trade_date','stock_code','circulating_market_cap']]

index_data = pd.merge(index_data, valuation_df, on=['trade_date','stock_code'], how='left')

def cal_weight(df):
    df = df.drop_duplicates(subset=['stock_code'], keep='first')
    cap_sum = df['circulating_market_cap'].sum()
    df['weight'] = df['circulating_market_cap'] / cap_sum
    return df

g_index_date = index_data.groupby('trade_date', group_keys=False)

index_weight_data = g_index_date.apply(cal_weight)










# 生成示例数据
data = generate_sample_data()

# 设置约束条件
industry_bounds = (np.full(10, -0.01), np.full(10, 0.01))  # 行业暴露 ±1%
style_bounds = (np.full(5, -0.1), np.full(5, 0.1))  # 风格暴露 ±10%

# 创建优化器并执行优化
optimizer = IndexEnhancedOptimizer(te_target=0.05, w_max=0.05)

results = optimizer.optimize(
    alpha_scores=data['alpha_scores'],
    benchmark_weights=data['benchmark_weights'],
    covariance_matrix=data['covariance_matrix'],
    industry_exposure=data['industry_exposure'],
    style_exposure=data['style_exposure'],
    industry_bounds=industry_bounds,
    style_bounds=style_bounds
)

# 打印结果
print("优化结果:")
print(f"目标函数值 (Alpha暴露): {results['objective_value']:.4f}")
print(f"预测跟踪误差: {results['predicted_te']:.4f}")
print(f"最大个股权重: {np.max(results['weights']):.4f}")
print(f"组合权重和: {np.sum(results['weights']):.4f}")

if results['industry_exposure'] is not None:
    print(f"\n行业暴露范围: [{np.min(results['industry_exposure']):.4f}, {np.max(results['industry_exposure']):.4f}]")

if results['style_exposure'] is not None:
    print(f"风格暴露范围: [{np.min(results['style_exposure']):.4f}, {np.max(results['style_exposure']):.4f}]")

# 分析主动权重
active_positions = results['active_weights']
print(f"\n主动权重统计:")
print(f"超配股票数量: {np.sum(active_positions > 0.001)}")
print(f"低配股票数量: {np.sum(active_positions < -0.001)}")
print(f"最大超配: {np.max(active_positions):.4f}")
print(f"最大低配: {np.min(active_positions):.4f}")