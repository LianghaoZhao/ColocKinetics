"""
统计分析模块：红绿比值 vs T50 分析 + 多元回归分析

单独调用方式:
=============
python statistics_analysis.py <output_dir>

示例:
    python statistics_analysis.py coloc_result

    这会读取 coloc_result/ratio_t50_raw_data.csv，执行统计分析并生成图表。

参数:
    output_dir: 包含 ratio_t50_raw_data.csv 的输出目录
    --background: 本底信号（默认100，仅在从reaction_fitting重新计算时使用）

输出:
    - ratio_vs_t50_analysis.png: 红绿比值与T50关系图
    - t50_t90_distribution.png: T50和T90分布直方图
    - multiple_regression_T50.png: T50多元回归分析图
    - multiple_regression_T90.png: T90多元回归分析图
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
from typing import Optional


class StatisticsAnalyzer:
    """统计分析类：执行红绿比值与T50的关系分析和多元回归分析"""
    
    def __init__(self, output_dir: str, suffix: str = ''):
        self.output_dir = Path(output_dir)
        self.suffix = suffix  # 输出文件名后缀，如 "_488_top20"
    
    def load_data(self, csv_path: Optional[str] = None) -> pd.DataFrame:
        """
        加载原始数据
        
        Parameters:
        - csv_path: CSV文件路径（默认为 output_dir/ratio_t50_raw_data.csv）
        
        Returns:
        - DataFrame 包含分析所需的数据
        """
        if csv_path is None:
            csv_path = self.output_dir / 'ratio_t50_raw_data.csv'
        else:
            csv_path = Path(csv_path)
        
        if not csv_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {csv_path}")
        
        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df)} cells from: {csv_path}")
        return df
    
    def run_analysis(self, df: pd.DataFrame):
        """
        执行完整的统计分析
        
        Parameters:
        - df: 包含以下列的DataFrame:
            - red: 红色强度
            - green: 绿色强度
            - ratio: 红绿比值
            - t50: T50值
            - t90: T90值
            - n_pixels: 细胞面积（像素数）
            - file_stem: 文件名
            - cell_id: 细胞ID
        """
        print("\n=== Running Statistical Analysis ===")
        
        # 确保必需的列存在
        required_cols = ['red', 'green', 'ratio', 't50', 't90', 'n_pixels', 'file_stem', 'cell_id']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            # 尝试使用 1/red 和 1/green 列
            if '1/red' in df.columns and '1/green' in df.columns:
                df['red'] = 1.0 / df['1/red']
                df['green'] = 1.0 / df['1/green']
            else:
                raise ValueError(f"缺少必需的列: {missing}")
        
        # 转换为numpy数组
        red_values = df['red'].values
        green_values = df['green'].values
        ratio_values = df['ratio'].values
        t50_values = df['t50'].values
        t90_values = df['t90'].values
        area_values = df['n_pixels'].values
        file_stems = df['file_stem'].values
        cell_ids = df['cell_id'].values
        
        print(f"  Total cells with valid data: {len(df)}")
        
        # === 第1步：面积过滤（IQR方法）===
        area_q1 = np.percentile(area_values, 25)
        area_q3 = np.percentile(area_values, 75)
        area_iqr = area_q3 - area_q1
        area_lower = area_q1 - 1.5 * area_iqr
        area_upper = area_q3 + 1.5 * area_iqr
        area_valid = (area_values >= area_lower) & (area_values <= area_upper)
        n_area_removed = np.sum(~area_valid)
        print(f"  Area filter (IQR): removed {n_area_removed} cells (area < {area_lower:.0f} or > {area_upper:.0f} pixels)")
        
        # === 第2步：T50过滤（log10 + 2.5倍IQR）===
        t50_positive_mask = t50_values > 0
        t50_valid = np.zeros(len(t50_values), dtype=bool)
        if np.sum(t50_positive_mask) > 0:
            t50_log = np.log10(t50_values[t50_positive_mask])
            t50_q1 = np.percentile(t50_log, 25)
            t50_q3 = np.percentile(t50_log, 75)
            t50_iqr = t50_q3 - t50_q1
            t50_lower_log = t50_q1 - 2.5 * t50_iqr
            t50_upper_log = t50_q3 + 2.5 * t50_iqr
            t50_lower_linear = 10 ** t50_lower_log
            t50_upper_linear = 10 ** t50_upper_log
            t50_valid[t50_positive_mask] = (t50_log >= t50_lower_log) & (t50_log <= t50_upper_log)
            n_t50_removed = np.sum(t50_positive_mask) - np.sum(t50_valid)
            print(f"  T50 filter (log10 + 2.5*IQR): removed {n_t50_removed} cells (T50 < {t50_lower_linear:.2f} or > {t50_upper_linear:.2f})")
        else:
            print("  T50 filter: no positive T50 values found")
        
        # === 第3步：1/红色强度过滤（IQR方法）===
        red_inv_values = 1.0 / red_values
        red_inv_q1 = np.percentile(red_inv_values, 25)
        red_inv_q3 = np.percentile(red_inv_values, 75)
        red_inv_iqr = red_inv_q3 - red_inv_q1
        red_inv_lower = red_inv_q1 - 1.5 * red_inv_iqr
        red_inv_upper = red_inv_q3 + 1.5 * red_inv_iqr
        red_inv_valid = (red_inv_values >= red_inv_lower) & (red_inv_values <= red_inv_upper)
        n_red_inv_removed = np.sum(~red_inv_valid)
        print(f"  1/Red filter (IQR): removed {n_red_inv_removed} cells (1/red < {red_inv_lower:.6f} or > {red_inv_upper:.6f})")
        
        # === 第4步：1/绿色强度过滤（IQR方法）===
        green_inv_values = 1.0 / green_values
        green_inv_q1 = np.percentile(green_inv_values, 25)
        green_inv_q3 = np.percentile(green_inv_values, 75)
        green_inv_iqr = green_inv_q3 - green_inv_q1
        green_inv_lower = green_inv_q1 - 1.5 * green_inv_iqr
        green_inv_upper = green_inv_q3 + 1.5 * green_inv_iqr
        green_inv_valid = (green_inv_values >= green_inv_lower) & (green_inv_values <= green_inv_upper)
        n_green_inv_removed = np.sum(~green_inv_valid)
        print(f"  1/Green filter (IQR): removed {n_green_inv_removed} cells (1/green < {green_inv_lower:.6f} or > {green_inv_upper:.6f})")
        
        # === 组合所有过滤条件 ===
        valid_mask = area_valid & t50_valid & red_inv_valid & green_inv_valid
        print(f"  Valid cells after all filters: {np.sum(valid_mask)}")
        
        # 过滤后的数据
        red_filtered = red_values[valid_mask]
        green_filtered = green_values[valid_mask]
        ratio_filtered = ratio_values[valid_mask]
        t50_filtered = t50_values[valid_mask]
        t90_filtered = t90_values[valid_mask]
        file_stems_filtered = file_stems[valid_mask]
        cell_ids_filtered = cell_ids[valid_mask]
        
        # 对红绿强度取倒数用于拟合分析
        red_inv_filtered = 1.0 / red_filtered
        green_inv_filtered = 1.0 / green_filtered
        
        if len(t50_filtered) < 3:
            print("Not enough valid data points for analysis.")
            return
        
        # 生成红绿比值 vs T50 分析图
        self._generate_ratio_vs_t50_plot(
            red_inv_filtered, green_inv_filtered, ratio_filtered,
            t50_filtered, file_stems_filtered, cell_ids_filtered
        )
        
        # 生成T50/T90分布直方图
        self._generate_t50_t90_histogram(t50_filtered, t90_filtered)
        
        # 多元回归分析
        self._perform_multiple_regression_analysis(
            red_inv_filtered, green_inv_filtered, t50_filtered, t90_filtered,
            file_stems_filtered, cell_ids_filtered, 'T50')
        self._perform_multiple_regression_analysis(
            red_inv_filtered, green_inv_filtered, t90_filtered, t50_filtered,
            file_stems_filtered, cell_ids_filtered, 'T90')
    
    def _generate_t50_t90_histogram(self, t50_values, t90_values):
        """
        生成T50和T90的分布直方图
        
        Parameters:
        - t50_values: T50值数组
        - t90_values: T90值数组
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # T50 分布
        ax1 = axes[0]
        ax1.hist(t50_values, bins=30, color='#45ADA8', alpha=0.7, edgecolor='black')
        ax1.tick_params(axis='both', labelsize=14)
        ax1.axvline(x=np.median(t50_values), color='red', linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(t50_values):.2f}')
        ax1.axvline(x=np.mean(t50_values), color='orange', linestyle='--',
                   linewidth=2, label=f'Mean: {np.mean(t50_values):.2f}')
        ax1.set_xlabel('T50', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title(f'T50 Distribution (n={len(t50_values)})', fontsize=14)
        ax1.legend(loc='best', fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # T90 分布
        ax2 = axes[1]
        ax2.hist(t90_values, bins=30, color='#D96459', alpha=0.7, edgecolor='black')
        ax2.tick_params(axis='both', labelsize=14)
        ax2.axvline(x=np.median(t90_values), color='red', linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(t90_values):.2f}')
        ax2.axvline(x=np.mean(t90_values), color='orange', linestyle='--',
                   linewidth=2, label=f'Mean: {np.mean(t90_values):.2f}')
        ax2.set_xlabel('T90', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title(f'T90 Distribution (n={len(t90_values)})', fontsize=14)
        ax2.legend(loc='best', fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        fig_path = self.output_dir / f't50_t90_distribution{self.suffix}.png'
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {fig_path}")
        
        # 输出统计信息
        print(f"\n  T50/T90 Distribution Statistics:")
        print(f"    T50: Mean={np.mean(t50_values):.2f}, Median={np.median(t50_values):.2f}, Std={np.std(t50_values):.2f}")
        print(f"    T90: Mean={np.mean(t90_values):.2f}, Median={np.median(t90_values):.2f}, Std={np.std(t90_values):.2f}")
    
    def _filter_by_cooks_distance(self, x, y, x_name, file_stems_arr, cell_ids_arr):
        """
        使用Cook's Distance过滤异常点（阈值=4/n）
        返回过滤后的x, y和对应的file/cell信息
        """
        n = len(x)
        if n < 3:
            return x, y, file_stems_arr, cell_ids_arr
        
        # 使用OLS计算Cook's Distance
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        influence = model.get_influence()
        cooks_d = influence.cooks_distance[0]
        
        # 阈值: 4/n
        threshold = 4.0 / n
        outlier_mask = cooks_d > threshold
        n_outliers = np.sum(outlier_mask)
        
        if n_outliers > 0:
            print(f"\n  Cook's Distance filter for {x_name}: removed {n_outliers} points (threshold=4/{n}={threshold:.4f})")
            outlier_indices = np.where(outlier_mask)[0]
            for idx in outlier_indices:
                print(f"    - File: {file_stems_arr[idx]}, Cell ID: {cell_ids_arr[idx]}, Cook's D: {cooks_d[idx]:.4f}")
        
        # 返回过滤后的数据
        valid_mask_cook = ~outlier_mask
        return x[valid_mask_cook], y[valid_mask_cook], file_stems_arr[valid_mask_cook], cell_ids_arr[valid_mask_cook]
    
    def _generate_ratio_vs_t50_plot(self, red_inv_filtered, green_inv_filtered, ratio_filtered,
                                     t50_filtered, file_stems_filtered, cell_ids_filtered):
        """生成红绿比值与T50关系的分析图"""
        
        # 创建图表: 2x2 布局
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. 1/红色值 vs T50 (带Cook's Distance过滤)
        red_inv_clean, t50_red_clean, _, _ = self._filter_by_cooks_distance(
            red_inv_filtered, t50_filtered, '1/Red vs T50', file_stems_filtered, cell_ids_filtered)
        ax1 = axes[0, 0]
        ax1.scatter(red_inv_clean, t50_red_clean, alpha=0.6, s=40, c='#D96459', edgecolors='#B84A40')
        ax1.tick_params(axis='both', labelsize=20)
        slope1, intercept1, r1, p1, se1 = stats.linregress(red_inv_clean, t50_red_clean)
        x_line = np.linspace(red_inv_clean.min(), red_inv_clean.max(), 100)
        ax1.plot(x_line, slope1 * x_line + intercept1, 'k--', linewidth=2, 
                label=f'R={r1:.3f}, p={p1:.2e}')
        ax1.set_xlabel('1/Red Intensity', fontsize=11)
        ax1.set_ylabel('T50 (Correlation)', fontsize=11)
        ax1.set_title(f'1/Red Intensity vs T50 (n={len(red_inv_clean)})', fontsize=12)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 2. 1/绿色值 vs T50 (带Cook's Distance过滤)
        green_inv_clean, t50_green_clean, _, _ = self._filter_by_cooks_distance(
            green_inv_filtered, t50_filtered, '1/Green vs T50', file_stems_filtered, cell_ids_filtered)
        ax2 = axes[0, 1]
        ax2.scatter(green_inv_clean, t50_green_clean, alpha=0.6, s=40, c='#45ADA8', edgecolors='#358985')
        ax2.tick_params(axis='both', labelsize=20)
        slope2, intercept2, r2, p2, se2 = stats.linregress(green_inv_clean, t50_green_clean)
        x_line = np.linspace(green_inv_clean.min(), green_inv_clean.max(), 100)
        ax2.plot(x_line, slope2 * x_line + intercept2, 'k--', linewidth=2,
                label=f'R={r2:.3f}, p={p2:.2e}')
        ax2.set_xlabel('1/Green Intensity', fontsize=11)
        ax2.set_ylabel('T50 (Correlation)', fontsize=11)
        ax2.set_title(f'1/Green Intensity vs T50 (n={len(green_inv_clean)})', fontsize=12)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # 3. 红绿比值 vs T50 (带Cook's Distance过滤)
        ratio_clean, t50_ratio_clean, _, _ = self._filter_by_cooks_distance(
            ratio_filtered, t50_filtered, 'Ratio vs T50', file_stems_filtered, cell_ids_filtered)
        ax3 = axes[1, 0]
        ax3.scatter(ratio_clean, t50_ratio_clean, alpha=0.6, s=40, c='purple', edgecolors='darkviolet')
        ax3.tick_params(axis='both', labelsize=20)
        slope3, intercept3, r3, p3, se3 = stats.linregress(ratio_clean, t50_ratio_clean)
        x_line = np.linspace(ratio_clean.min(), ratio_clean.max(), 100)
        ax3.plot(x_line, slope3 * x_line + intercept3, 'k--', linewidth=2,
                label=f'R={r3:.3f}, p={p3:.2e}')
        ax3.set_xlabel('Red/Green Ratio', fontsize=11)
        ax3.set_ylabel('T50 (Correlation)', fontsize=11)
        ax3.set_title(f'Red/Green Ratio vs T50 (n={len(ratio_clean)})', fontsize=12)
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        
        # 4. 比值分布直方图 (使用过滤后的数据)
        ax4 = axes[1, 1]
        ax4.hist(ratio_clean, bins=30, color='purple', alpha=0.7, edgecolor='black')
        ax4.tick_params(axis='both', labelsize=20)
        ax4.axvline(x=np.median(ratio_clean), color='red', linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(ratio_clean):.3f}')
        ax4.axvline(x=np.mean(ratio_clean), color='blue', linestyle='--',
                   linewidth=2, label=f'Mean: {np.mean(ratio_clean):.3f}')
        ax4.set_xlabel('Red/Green Ratio', fontsize=11)
        ax4.set_ylabel('Count', fontsize=11)
        ax4.set_title(f'Red/Green Ratio Distribution (n={len(ratio_clean)})', fontsize=12)
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        fig_path = self.output_dir / f'ratio_vs_t50_analysis{self.suffix}.png'
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {fig_path}")
        
        # 输出统计结果
        print(f"\n  Analysis Results (Inverse Intensity, after Cook's Distance filter):")
        print(f"    1/Red vs T50:   R={r1:.3f}, p={p1:.2e}, n={len(red_inv_clean)}")
        print(f"    1/Green vs T50: R={r2:.3f}, p={p2:.2e}, n={len(green_inv_clean)}")
        print(f"    Ratio vs T50:   R={r3:.3f}, p={p3:.2e}, n={len(ratio_clean)}")
    
    def _perform_multiple_regression_analysis(self, red_inv, green_inv, y_main, y_other,
                                               file_stems, cell_ids, y_name):
        """
        执行多元回归分析，包括偏回归和残差分析
        
        Parameters:
        - red_inv: 1/红色强度数组
        - green_inv: 1/绿色强度数组
        - y_main: 主因变量 (T50或T90)
        - y_other: 另一个因变量 (用于参考)
        - file_stems: 文件名数组
        - cell_ids: 细胞ID数组
        - y_name: 因变量名称 ('T50'或'T90')
        """
        print(f"\n  === Multiple Regression Analysis (Y = {y_name}) ===")
        
        n = len(y_main)
        if n < 5:
            print(f"    Not enough data points for multiple regression (n={n})")
            return
        
        # 构建自变量矩阵 (1/红色, 1/绿色)
        X = np.column_stack([red_inv, green_inv])
        X_with_const = sm.add_constant(X)
        y = y_main
        
        # 拟合OLS模型
        model = sm.OLS(y, X_with_const).fit()
        
        # 计算Cook's Distance
        influence = model.get_influence()
        cooks_d = influence.cooks_distance[0]
        threshold = 4.0 / n
        outlier_mask = cooks_d > threshold
        n_outliers = np.sum(outlier_mask)
        
        if n_outliers > 0:
            print(f"    Cook's Distance filter: removed {n_outliers} points (threshold=4/{n}={threshold:.4f})")
            outlier_indices = np.where(outlier_mask)[0]
            for idx in outlier_indices:
                print(f"      - File: {file_stems[idx]}, Cell ID: {cell_ids[idx]}, Cook's D: {cooks_d[idx]:.4f}")
        
        # 过滤后重新拟合
        valid_mask = ~outlier_mask
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        red_inv_clean = red_inv[valid_mask]
        green_inv_clean = green_inv[valid_mask]
        file_stems_clean = file_stems[valid_mask]
        cell_ids_clean = cell_ids[valid_mask]
        
        if len(y_clean) < 5:
            print(f"    Not enough data points after filtering (n={len(y_clean)})")
            return
        
        X_clean_const = sm.add_constant(X_clean)
        model_clean = sm.OLS(y_clean, X_clean_const).fit()
        
        # 输出回归结果
        print(f"\n    Multiple Regression Results (n={len(y_clean)}):")
        print(f"      R² = {model_clean.rsquared:.4f}")
        print(f"      Adjusted R² = {model_clean.rsquared_adj:.4f}")
        print(f"      F-statistic = {model_clean.fvalue:.4f}, p = {model_clean.f_pvalue:.2e}")
        print(f"\n      Coefficients:")
        print(f"        Intercept:  {model_clean.params[0]:.4f} (p={model_clean.pvalues[0]:.2e})")
        print(f"        1/Red:      {model_clean.params[1]:.4f} (p={model_clean.pvalues[1]:.2e})")
        print(f"        1/Green:    {model_clean.params[2]:.4f} (p={model_clean.pvalues[2]:.2e})")
        
        # 创建图表: 2x2 布局
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 获取残差和预测值
        residuals = model_clean.resid
        fitted_values = model_clean.fittedvalues
        
        # 1. 偏回归图 - 1/红色
        ax1 = axes[0, 0]
        model_y_green = sm.OLS(y_clean, sm.add_constant(green_inv_clean)).fit()
        model_red_green = sm.OLS(red_inv_clean, sm.add_constant(green_inv_clean)).fit()
        resid_y = model_y_green.resid
        resid_red = model_red_green.resid
        ax1.scatter(resid_red, resid_y, alpha=0.6, s=40, c='#D96459', edgecolors='#B84A40')
        ax1.tick_params(axis='both', labelsize=20)
        slope_partial, intercept_partial, r_partial, p_partial, _ = stats.linregress(resid_red, resid_y)
        x_line = np.linspace(resid_red.min(), resid_red.max(), 100)
        ax1.plot(x_line, slope_partial * x_line + intercept_partial, 'k--', linewidth=2,
                label=f'Partial R={r_partial:.3f}, p={p_partial:.2e}')
        ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
        ax1.set_xlabel('1/Red (controlling for 1/Green)', fontsize=11)
        ax1.set_ylabel(f'{y_name} (controlling for 1/Green)', fontsize=11)
        ax1.set_title(f'Partial Regression: 1/Red | 1/Green', fontsize=12)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 2. 偏回归图 - 1/绿色
        ax2 = axes[0, 1]
        model_y_red = sm.OLS(y_clean, sm.add_constant(red_inv_clean)).fit()
        model_green_red = sm.OLS(green_inv_clean, sm.add_constant(red_inv_clean)).fit()
        resid_y2 = model_y_red.resid
        resid_green = model_green_red.resid
        ax2.scatter(resid_green, resid_y2, alpha=0.6, s=40, c='#45ADA8', edgecolors='#358985')
        ax2.tick_params(axis='both', labelsize=20)
        slope_partial2, intercept_partial2, r_partial2, p_partial2, _ = stats.linregress(resid_green, resid_y2)
        x_line = np.linspace(resid_green.min(), resid_green.max(), 100)
        ax2.plot(x_line, slope_partial2 * x_line + intercept_partial2, 'k--', linewidth=2,
                label=f'Partial R={r_partial2:.3f}, p={p_partial2:.2e}')
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
        ax2.set_xlabel('1/Green (controlling for 1/Red)', fontsize=11)
        ax2.set_ylabel(f'{y_name} (controlling for 1/Red)', fontsize=11)
        ax2.set_title(f'Partial Regression: 1/Green | 1/Red', fontsize=12)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # 3. 残差 vs 拟合值图
        ax3 = axes[1, 0]
        ax3.scatter(fitted_values, residuals, alpha=0.6, s=40, c='blue', edgecolors='darkblue')
        ax3.tick_params(axis='both', labelsize=20)
        ax3.axhline(y=0, color='red', linestyle='--', linewidth=2)
        # 添加LOWESS平滑线
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            smoothed = lowess(residuals, fitted_values, frac=0.5)
            ax3.plot(smoothed[:, 0], smoothed[:, 1], 'orange', linewidth=2, label='LOWESS')
        except:
            pass
        ax3.set_xlabel(f'Fitted {y_name}', fontsize=11)
        ax3.set_ylabel('Residuals', fontsize=11)
        ax3.set_title('Residuals vs Fitted Values', fontsize=12)
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        
        # 4. Q-Q图 (残差正态性检验)
        ax4 = axes[1, 1]
        from scipy import stats as scipy_stats
        scipy_stats.probplot(residuals, dist="norm", plot=ax4)
        ax4.set_title('Normal Q-Q Plot of Residuals', fontsize=12)
        ax4.tick_params(axis='both', labelsize=20)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        fig_path = self.output_dir / f'multiple_regression_{y_name}{self.suffix}.png'
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"\n    Saved: {fig_path}")
        
        # 输出偏回归结果
        print(f"\n    Partial Regression Results:")
        print(f"      1/Red | 1/Green:   Partial R = {r_partial:.3f}, p = {p_partial:.2e}")
        print(f"      1/Green | 1/Red:   Partial R = {r_partial2:.3f}, p = {p_partial2:.2e}")


def filter_top_percent(df: pd.DataFrame, channel: str, percent: float) -> pd.DataFrame:
    """
    筛选指定通道亮度排名前xx%的细胞
    
    Parameters:
    - df: 数据框
    - channel: 通道名称 ('488' 或 '561')
    - percent: 百分比 (0-100)
    
    Returns:
    - 筛选后的DataFrame
    """
    # 映射通道到列名
    channel_map = {
        '488': 'green',   # 488nm 对应绿色通道
        '561': 'red',     # 561nm 对应红色通道
        'green': 'green',
        'red': 'red'
    }
    
    col_name = channel_map.get(channel.lower())
    if col_name is None:
        raise ValueError(f"不支持的通道: {channel}，请使用 488/561/green/red")
    
    if col_name not in df.columns:
        raise ValueError(f"数据中没有 {col_name} 列")
    
    # 计算阈值
    threshold = np.percentile(df[col_name].values, 100 - percent)
    
    # 筛选
    filtered_df = df[df[col_name] >= threshold].copy()
    
    return filtered_df


def main():
    parser = argparse.ArgumentParser(
        description='统计分析模块：红绿比值 vs T50 分析 + 多元回归分析',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python statistics_analysis.py coloc_result
  python statistics_analysis.py coloc_result --csv path/to/custom_data.csv
  
  # 只分析488通道亮度前20%的细胞
  python statistics_analysis.py coloc_result --top 488:20
  
  # 分析多组：488前20% 和 561前30%
  python statistics_analysis.py coloc_result --top 488:20 --top 561:30
        """
    )
    
    parser.add_argument('output_dir', type=str,
                       help='输出目录（包含 ratio_t50_raw_data.csv 或指定CSV）')
    parser.add_argument('--csv', type=str, default=None,
                       help='指定输入CSV文件路径（默认为 output_dir/ratio_t50_raw_data.csv）')
    parser.add_argument('--top', type=str, action='append', metavar='CHANNEL:PERCENT',
                       help='按指定通道亮度筛选前xx%%细胞，格式为 "488:20" 或 "561:30"（可指定多组）')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Statistics Analysis Module")
    print("=" * 60)
    
    # 创建分析器
    analyzer = StatisticsAnalyzer(args.output_dir)
    
    # 加载数据
    try:
        df = analyzer.load_data(args.csv)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\n请确保已运行完整的分析流程，生成 ratio_t50_raw_data.csv 文件。")
        print("或使用 --csv 参数指定数据文件路径。")
        return
    
    # 如果指定了 --top 参数，执行筛选分析
    if args.top:
        for top_spec in args.top:
            # 解析格式 "CHANNEL:PERCENT"
            try:
                parts = top_spec.split(':')
                if len(parts) != 2:
                    raise ValueError(f"格式错误: {top_spec}")
                channel, percent_str = parts
                percent = float(percent_str)
                if percent <= 0 or percent > 100:
                    raise ValueError(f"百分比必须在0-100之间: {percent}")
            except Exception as e:
                print(f"\n警告: 无法解析 --top 参数 '{top_spec}': {e}")
                print("格式应为 'CHANNEL:PERCENT'，例如 '488:20' 或 '561:30'")
                continue
            
            print(f"\n{'=' * 60}")
            print(f"Filtering: {channel} channel, top {percent}%")
            print("=" * 60)
            
            # 筛选数据
            try:
                filtered_df = filter_top_percent(df, channel, percent)
                print(f"  Filtered cells: {len(filtered_df)} / {len(df)} ({len(filtered_df)/len(df)*100:.1f}%)")
                
                if len(filtered_df) < 5:
                    print(f"  警告: 筛选后数据点太少 (n={len(filtered_df)})，跳过此分析")
                    continue
                
                # 创建带后缀的分析器
                suffix = f"_{channel}_top{int(percent)}"
                filtered_analyzer = StatisticsAnalyzer(args.output_dir, suffix=suffix)
                filtered_analyzer.run_analysis(filtered_df)
                
            except ValueError as e:
                print(f"  错误: {e}")
                continue
    else:
        # 正常分析（无筛选）
        analyzer.run_analysis(df)
    
    print("\n" + "=" * 60)
    print("Statistics Analysis Completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
