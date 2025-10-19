import numpy as np
from scipy.optimize import curve_fit
from typing import Dict

from skimage.future import fit_segmenter


def first_order_reaction(t, A0, k, A_inf=0):
    """
    一级反应方程
    A(t) = A_inf + (A0 - A_inf) * exp(-k*t)
    """
    return A_inf + (A0 - A_inf) * np.exp(-k * t)

def delayed_first_order_reaction(t, A0, k, A_inf=0, delay=0):
    """
    延迟一级反应方程
    A(t) = A_inf + (A0 - A_inf) * exp(-k*(t-delay)) * H(t-delay)
    """
    shifted_time = t - delay
    mask = shifted_time >= 0
    y = np.full_like(t, A0, dtype=float)
    if np.any(mask):  # 检查是否有时点 >= delay
        y[mask] = A_inf + (A0 - A_inf) * np.exp(-k * shifted_time[mask])
    return y


class ReactionFitter:
    @staticmethod
    def fit_first_order_reaction(time_points, values) -> Dict[str, float]:
        """
        对数据进行一级反应方程拟合
        Parameters:
        - time_points: 时间点数组
        - values: 对应的数值数组
        Returns:
        - 包含拟合参数和反应时间的字典
        """
        # 移除NaN值
        mask = ~np.isnan(values)
        time_points_clean = time_points[mask]
        values_clean = values[mask]

        if len(time_points_clean) < 3:
            return {
                'A0': np.nan,
                'k': np.nan,
                'A_inf': np.nan,
                't50': np.nan,
                't90': np.nan,
                'r_squared': np.nan
            }

        # 确定初始值
        A0 = values_clean[0]  # 初始值
        A_inf = values_clean[-1]   # 最终值
        k_guess = 0.1  # 初始k值猜测


        try:
            # 进行曲线拟合
            popt, pcov = curve_fit(first_order_reaction, time_points_clean, values_clean,
                                 p0=[A0, k_guess, A_inf], maxfev=5000)
            A0_fit, k_fit, A_inf_fit = popt

            # 计算拟合优度
            fitted_values = first_order_reaction(time_points_clean, A0_fit, k_fit, A_inf_fit)
            ss_res = np.sum((values_clean - fitted_values) ** 2)
            ss_tot = np.sum((values_clean - np.mean(values_clean)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0

            # 计算50%和90%反应时间
            if k_fit > 0:
                # 计算达到目标值的时间
                if A0_fit > A_inf_fit:
                    # 衰减过程
                    target_50 = A_inf_fit + 0.5 * (A0_fit - A_inf_fit)
                    target_90 = A_inf_fit + 0.1 * (A0_fit - A_inf_fit)
                else:
                    # 增长过程
                    target_50 = A0_fit + 0.5 * (A_inf_fit - A0_fit)
                    target_90 = A0_fit + 0.9 * (A_inf_fit - A0_fit)

                t50 = -np.log((target_50 - A_inf_fit) / (A0_fit - A_inf_fit)) / k_fit if k_fit > 0 and (A0_fit - A_inf_fit) != 0 else np.nan
                t90 = -np.log((target_90 - A_inf_fit) / (A0_fit - A_inf_fit)) / k_fit if k_fit > 0 and (A0_fit - A_inf_fit) != 0 else np.nan
            else:
                t50 = np.nan
                t90 = np.nan

            return {
                'A0': A0_fit,
                'k': k_fit,
                'A_inf': A_inf_fit,
                't50': t50,
                't90': t90,
                'r_squared': r_squared
            }
        except Exception as e:
            # print(f"Warning: Fitting failed for first_order: {e}") # Optional: Log warning
            return {
                'A0': np.nan,
                'k': np.nan,
                'A_inf': np.nan,
                't50': np.nan,
                't90': np.nan,
                'r_squared': np.nan
            }

    @staticmethod
    def fit_delayed_first_order_reaction(time_points, values) -> Dict[str, float]:
        """
        对数据进行延迟一级反应方程拟合（使用改进的延迟函数）
        """
        # 移除NaN值
        mask = ~np.isnan(values)
        time_points_clean = time_points[mask]
        values_clean = values[mask]

        if len(time_points_clean) < 4:  # 需要至少4个点来拟合4个参数
            return {
                'A0': np.nan,
                'k': np.nan,
                'A_inf': np.nan,
                'delay': np.nan,
                't50': np.nan,
                't90': np.nan,
                'r_squared': np.nan
            }

        # 改进的初始值估计
        A0, A_inf, k_guess, delay_guess = ReactionFitter.estimate_delayed_initial_values(
            time_points_clean, values_clean
        )

        try:
            # 设置参数边界
            lower_bounds = [-1, 1e-6, -1, 0]  # A0, k, A_inf, delay
            upper_bounds = [1, 1e6, 1, time_points_clean[-1]]

            popt, pcov = curve_fit(
                ReactionFitter.piecewise_delayed_first_order,
                time_points_clean,
                values_clean,
                p0=[A0, k_guess, A_inf, delay_guess],
                bounds=(lower_bounds, upper_bounds),
                maxfev=10000,
                method='trf'  # 使用更稳健的算法
            )
            A0_fit, k_fit, A_inf_fit, delay_fit = popt

            # 计算拟合优度
            fitted_values = ReactionFitter.piecewise_delayed_first_order(time_points_clean, A0_fit, k_fit, A_inf_fit, delay_fit)
            ss_res = np.sum((values_clean - fitted_values) ** 2)
            ss_tot = np.sum((values_clean - np.mean(values_clean)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0

            # 计算50%和90%反应时间
            t50, t90 = ReactionFitter.calculate_reaction_times_delayed(A0_fit, k_fit, A_inf_fit, delay_fit)

            return {
                'A0': A0_fit,
                'k': k_fit,
                'A_inf': A_inf_fit,
                'delay': delay_fit,
                't50': t50,
                't90': t90,
                'r_squared': r_squared
            }
        except Exception as e:
            # print(f"Warning: Fitting failed for delayed_first_order: {e}") # Optional: Log warning
            return {
                'A0': np.nan, 'k': np.nan, 'A_inf': np.nan, 'delay': np.nan,
                't50': np.nan, 't90': np.nan, 'r_squared': np.nan
            }

    @staticmethod
    def estimate_delayed_initial_values(time_points, values):
        """估计延迟拟合的初始值"""
        A0 = values[0] if len(values) > 0 else 0
        A_inf = values[-1] if len(values) > 1 else A0

        # 估计延迟：找到值开始显著变化的时间点
        if len(values) > 2:
            # 找到值开始显著变化的点
            diff_values = np.abs(np.diff(values))
            # 设置变化阈值
            threshold = np.std(diff_values) * 0.5
            # 找到第一个显著变化的点
            significant_changes = np.where(diff_values > threshold)[0]
            if len(significant_changes) > 0:
                first_change_idx = significant_changes[0]
                delay_guess = time_points[first_change_idx] * 0.8  # 稍微提前一点
                delay_guess = max(0, delay_guess)  # 确保非负
            else:
                delay_guess = 0
        else:
            delay_guess = 0

        # 估计k值：基于延迟后的主要变化
        if delay_guess < time_points[-1]:
            active_mask = time_points > delay_guess
            if np.any(active_mask):
                active_times = time_points[active_mask]
                active_values = values[active_mask]
                if len(active_values) > 1:
                    # 估算反应速率
                    if A0 > A_inf:  # 衰减过程
                        # 找到反应进行到一半的时间点
                        mid_val = A_inf + 0.5 * (A0 - A_inf)
                        mid_idx = np.argmin(np.abs(active_values - mid_val))
                        if mid_idx > 0:
                            # k ≈ ln(2) / (t_mid - delay)
                            k_guess = np.log(2) / max(0.1, active_times[mid_idx] - delay_guess)
                        else:
                            k_guess = 0.1
                    else:  # 增长过程
                        mid_val = A0 + 0.5 * (A_inf - A0)
                        mid_idx = np.argmin(np.abs(active_values - mid_val))
                        if mid_idx > 0:
                            k_guess = np.log(2) / max(0.1, active_times[mid_idx] - delay_guess)
                        else:
                            k_guess = 0.1
                else:
                    k_guess = 0.1
            else:
                k_guess = 0.1
        else:
            k_guess = 0.1

        k_guess = max(1e-6, k_guess)  # 确保k为正

        return A0, A_inf, k_guess, delay_guess

    @staticmethod
    def calculate_reaction_times_delayed(A0, k, A_inf, delay):
        """计算延迟反应的反应时间"""
        if k <= 0 or np.isnan(k):
            return np.nan, np.nan

        if A0 > A_inf:  # 衰减过程
            # 50%反应时间：A(t) = A_inf + 0.5*(A0 - A_inf)
            # 90%反应时间：A(t) = A_inf + 0.1*(A0 - A_inf)
            t50 = delay - np.log(0.5) / k  # ln(0.5) = -ln(2)
            t90 = delay - np.log(0.1) / k  # ln(0.1) = -ln(10)
        else:  # 增长过程
            # 50%反应时间：A(t) = A0 + 0.5*(A_inf - A0)
            # 90%反应时间：A(t) = A0 + 0.9*(A_inf - A0)
            t50 = delay - np.log(0.5) / k
            t90 = delay - np.log(0.1) / k

        return t50, t90

    @staticmethod
    def piecewise_delayed_first_order(t, A0, k, A_inf=0, delay=0):
        """分段延迟一级反应方程（延迟期内保持A0）"""
        shifted_time = t - delay
        result = np.full_like(t, A0, dtype=float)
        mask = shifted_time >= 0
        if np.any(mask):
            result[mask] = A_inf + (A0 - A_inf) * np.exp(-k * shifted_time[mask])
        return result


class KineticsAnalyzer:
    """分析反应动力学的类"""

    @staticmethod
    def fit_cell_kinetics(cell_metrics: Dict[str, np.ndarray], fit_model: str = 'first_order') -> Dict[str, float]:
        """
        对单个细胞的指标进行拟合。
        Parameters:
        - cell_metrics: Dict containing 'time_points', 'correlations', 'intensity1', 'intensity2'
        - fit_model: 'first_order' or 'delayed_first_order'
        Returns:
        - Dict containing all fitting parameters for this cell.
        """
        time_points = cell_metrics['time_points']
        correlations = cell_metrics['correlations']
        ch1_values = cell_metrics['intensity1']
        ch2_values = cell_metrics['intensity2']

        # 选择拟合函数
        if fit_model == 'delayed_first_order':
            fit_func = delayed_first_order_reaction
            fit_method = ReactionFitter.fit_delayed_first_order_reaction
        else: # first_order
            fit_func = first_order_reaction
            fit_method = ReactionFitter.fit_first_order_reaction

        # 检查是否有足够的有效数据点
        valid_corr_mask = ~np.isnan(correlations)
        valid_ch1_mask = ~np.isnan(ch1_values)
        valid_ch2_mask = ~np.isnan(ch2_values)

        if np.sum(valid_corr_mask) < 3:
            correlation_fit = {
                'A0': np.nan, 'k': np.nan, 'A_inf': np.nan, 't50': np.nan, 't90': np.nan, 'r_squared': np.nan
            }
            if fit_model == 'delayed_first_order':
                correlation_fit['delay'] = np.nan
        else:
            # 拟合相关系数变化
            correlation_fit = fit_method(time_points[valid_corr_mask], correlations[valid_corr_mask])

        if np.sum(valid_ch1_mask) < 3:
            ch1_fit = {
                'A0': np.nan, 'k': np.nan, 'A_inf': np.nan, 't50': np.nan, 't90': np.nan, 'r_squared': np.nan
            }
            if fit_model == 'delayed_first_order':
                ch1_fit['delay'] = np.nan
        else:
            # 拟合通道1强度变化
            ch1_fit = fit_method(time_points[valid_ch1_mask], ch1_values[valid_ch1_mask])

        if np.sum(valid_ch2_mask) < 3:
            ch2_fit = {
                'A0': np.nan, 'k': np.nan, 'A_inf': np.nan, 't50': np.nan, 't90': np.nan, 'r_squared': np.nan
            }
            if fit_model == 'delayed_first_order':
                ch2_fit['delay'] = np.nan
        else:
            # 拟合通道2强度变化
            ch2_fit = fit_method(time_points[valid_ch2_mask], ch2_values[valid_ch2_mask])

        # 合并结果
        result = {
            'correlation_A0': correlation_fit['A0'],
            'correlation_k': correlation_fit['k'],
            'correlation_A_inf': correlation_fit['A_inf'],
            'correlation_t50': correlation_fit['t50'],
            'correlation_t90': correlation_fit['t90'],
            'correlation_r_squared': correlation_fit['r_squared'],
            'channel1_A0': ch1_fit['A0'],
            'channel1_k': ch1_fit['k'],
            'channel1_A_inf': ch1_fit['A_inf'],
            'channel1_t50': ch1_fit['t50'],
            'channel1_t90': ch1_fit['t90'],
            'channel1_r_squared': ch1_fit['r_squared'],
            'channel2_A0': ch2_fit['A0'],
            'channel2_k': ch2_fit['k'],
            'channel2_A_inf': ch2_fit['A_inf'],
            'channel2_t50': ch2_fit['t50'],
            'channel2_t90': ch2_fit['t90'],
            'channel2_r_squared': ch2_fit['r_squared']
        }
        # 添加延迟参数（如果使用延迟模型）
        if fit_model == 'delayed_first_order':
            result['correlation_delay'] = correlation_fit.get('delay', np.nan)
            result['channel1_delay'] = ch1_fit.get('delay', np.nan)
            result['channel2_delay'] = ch2_fit.get('delay', np.nan)

        return result

    @classmethod
    def fit_all_kinetics(cls, coloc_results: Dict[str, Dict[int, Dict[str, np.ndarray]]], fit_model: str = 'first_order') -> 'pd.DataFrame':
        """
        批量处理所有细胞的拟合，生成最终结果表。
        Parameters:
        - coloc_results: Dict from CoLocalizationMetrics.calculate_all_metrics
        - fit_model: 'first_order' or 'delayed_first_order'
        Returns:
        - pd.DataFrame containing all fitting results.
        """
        import pandas as pd # Import here to avoid circular import if needed
        all_results = []
        for file_path, cell_data_dict in coloc_results.items():
            for cell_id, metrics in cell_data_dict.items():
                fit_result = cls.fit_cell_kinetics(metrics, fit_model)
                fit_result['file_path'] = file_path
                fit_result['cell_id'] = cell_id
                all_results.append(fit_result)

        return pd.DataFrame(all_results)
