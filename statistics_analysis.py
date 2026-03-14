"""
统计分析模块：log10信号强度 vs 表观速率常数(k_app)分析 + 多元回归分析

单独调用方式:
=============

基本用法（自动创建输出文件夹）:
    python statistics_analysis.py <csv_file>
    python statistics_analysis.py data.csv
    python statistics_analysis.py path/to/ratio_t50_raw_data.csv

多文件合并分析（输出到merged_output）:
    python statistics_analysis.py file1.csv file2.csv file3.csv

指定输出目录:
    python statistics_analysis.py data.csv --output my_output

筛选分析:
    python statistics_analysis.py data.csv --top 488:20
    python statistics_analysis.py data.csv --bottom 561:30
    python statistics_analysis.py data.csv --percentile-range 488:20:80
    python statistics_analysis.py data.csv --percentile-range 561:10:90

百分位分组分析 (新功能):
    python statistics_analysis.py data.csv --percentile-bins 488           # 488通道按默认每10%分组
    python statistics_analysis.py data.csv --percentile-bins 561           # 561通道按每10%分组
    python statistics_analysis.py data.csv --percentile-bins both          # 两个通道分别分析
    python statistics_analysis.py data.csv --percentile-bins 488 --bin-step 5   # 488通道按每5%分组
    python statistics_analysis.py data.csv --percentile-bins both --bin-step 20  # 两通道每20%分组

时间截取:
    python statistics_analysis.py data.csv --max-time 600

拟合质量过滤:
    python statistics_analysis.py data.csv --min-rsq 0.95

输出:
    - intensity_vs_rate_constant.png: log10信号强度与表观速率常数(k_app)关系图
    - t50_t90_distribution.png: T50和T90分布直方图
    - rate_constant_distribution.png: 表观速率常数(k_app)分布直方图
    - multiple_regression_k_app_T50.png: 表观速率常数多元回归分析图（仅T50）
    - multiple_regression_log10k_app_T50.png: 双对数多元回归分析图（仅T50）

百分位分组分析输出 (--percentile-bins):
    - percentile_bins_{channel}/
      - percentile_bin_summary_{channel}.csv: 各组统计参数汇总CSV
      - percentile_distribution_trends_{channel}.png/pdf: T50/T90/k_app分布趋势图
      - percentile_correlation_trends_{channel}.png/pdf: 相关性参数趋势图
      - percentile_regression_trends_{channel}.png/pdf: 多元回归参数趋势图
      - percentile_kinetics_trends_{channel}.png/pdf: 动力学参数趋势图
      - percentile_fitted_curves_{channel}.png/pdf: 各分组拟合回归曲线图（归一化+未归一化）

滑窗百分位分析 (新功能):
    python statistics_analysis.py data.csv --sliding-window 488                    # 488通道默认窗口20%步长10%
    python statistics_analysis.py data.csv --sliding-window 561 --window-size 30  # 561通道窗口30%
    python statistics_analysis.py data.csv --sliding-window both --window-size 20 --bin-step 5  # 两通道窗口20%步长5%

滑窗分析输出 (--sliding-window):
    - sliding_window_{channel}/
      - sliding_window_summary_{channel}.csv: 各窗口统计参数汇总CSV
      - sliding_correlation_trends_{channel}.png/pdf: 相关性参数趋势图
      - sliding_regression_trends_{channel}.png/pdf: 多元回归参数趋势图
      - sliding_kinetics_trends_{channel}.png/pdf: 动力学参数趋势图

注意: 使用--percentile-range参数时，会在输出目录下创建独立的子目录来存放结果
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
from typing import Optional

# 设置全局字体为 Arial
plt.rcParams['font.family'] = 'Arial'
# 设置字体类型为 TrueType，使文字以真实文本而非路径保存（便于编辑和搜索）
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def apply_statistical_filters(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    应用统计分析的数据清洗流程
    
    此函数将所有数据清洗步骤封装为可复用模块，包括：
    1. 面积过滤（IQR方法）
    2. T50过滤（log10 + 2.5倍IQR）
    3. log10(红色强度)过滤（IQR方法）
    4. log10(绿色强度)过滤（IQR方法）
    
    Parameters:
    - df: 包含细胞数据的DataFrame，需要包含以下列:
        - red: 红色强度
        - green: 绿色强度
        - t50: T50值
        - n_pixels: 细胞面积（像素数）
    - verbose: 是否输出过滤信息（默认True）
    
    Returns:
    - 过滤后的DataFrame副本
    
    Usage:
        # 在 StatisticsAnalyzer 或 PercentileBinAnalyzer 中调用
        filtered_df = apply_statistical_filters(df)
    """
    # 确保必需的列存在
    required_cols = ['red', 'green', 't50', 'n_pixels']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        # 尝试使用 1/red 和 1/green 列
        if '1/red' in df.columns and '1/green' in df.columns:
            df = df.copy()
            df['red'] = 1.0 / df['1/red']
            df['green'] = 1.0 / df['1/green']
        else:
            raise ValueError(f"缺少必需的列: {missing}")
    
    # 转换为numpy数组
    red_values = df['red'].values
    green_values = df['green'].values
    t50_values = df['t50'].values
    area_values = df['n_pixels'].values
    
    n_original = len(df)
    if verbose:
        print(f"  Data cleaning: starting with {n_original} cells")
    
    # === 第1步：面积过滤（IQR方法）===
    area_q1 = np.percentile(area_values, 25)
    area_q3 = np.percentile(area_values, 75)
    area_iqr = area_q3 - area_q1
    area_lower = area_q1 - 1.5 * area_iqr
    area_upper = area_q3 + 1.5 * area_iqr
    area_valid = (area_values >= area_lower) & (area_values <= area_upper)
    n_area_removed = np.sum(~area_valid)
    if verbose:
        print(f"    Area filter (IQR): removed {n_area_removed} cells (area < {area_lower:.0f} or > {area_upper:.0f} pixels)")
    
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
        if verbose:
            print(f"    T50 filter (log10 + 2.5*IQR): removed {n_t50_removed} cells (T50 < {t50_lower_linear:.2f} or > {t50_upper_linear:.2f})")
    else:
        if verbose:
            print("    T50 filter: no positive T50 values found")
    
    # === 第3步：log10(红色强度)过滤（IQR方法）===
    red_positive_mask = red_values > 0
    red_log_valid = np.zeros(len(red_values), dtype=bool)
    if np.sum(red_positive_mask) > 0:
        red_log_values = np.log10(red_values[red_positive_mask])
        red_log_q1 = np.percentile(red_log_values, 25)
        red_log_q3 = np.percentile(red_log_values, 75)
        red_log_iqr = red_log_q3 - red_log_q1
        red_log_lower = red_log_q1 - 1.5 * red_log_iqr
        red_log_upper = red_log_q3 + 1.5 * red_log_iqr
        red_log_lower_linear = 10 ** red_log_lower
        red_log_upper_linear = 10 ** red_log_upper
        red_log_valid[red_positive_mask] = (red_log_values >= red_log_lower) & (red_log_values <= red_log_upper)
        n_red_log_removed = np.sum(red_positive_mask) - np.sum(red_log_valid)
        if verbose:
            print(f"    log10(Red) filter (IQR): removed {n_red_log_removed} cells (red < {red_log_lower_linear:.2f} or > {red_log_upper_linear:.2f})")
    else:
        if verbose:
            print("    log10(Red) filter: no positive red values found")
    
    # === 第4步：log10(绿色强度)过滤（IQR方法）===
    green_positive_mask = green_values > 0
    green_log_valid = np.zeros(len(green_values), dtype=bool)
    if np.sum(green_positive_mask) > 0:
        green_log_values = np.log10(green_values[green_positive_mask])
        green_log_q1 = np.percentile(green_log_values, 25)
        green_log_q3 = np.percentile(green_log_values, 75)
        green_log_iqr = green_log_q3 - green_log_q1
        green_log_lower = green_log_q1 - 1.5 * green_log_iqr
        green_log_upper = green_log_q3 + 1.5 * green_log_iqr
        green_log_lower_linear = 10 ** green_log_lower
        green_log_upper_linear = 10 ** green_log_upper
        green_log_valid[green_positive_mask] = (green_log_values >= green_log_lower) & (green_log_values <= green_log_upper)
        n_green_log_removed = np.sum(green_positive_mask) - np.sum(green_log_valid)
        if verbose:
            print(f"    log10(Green) filter (IQR): removed {n_green_log_removed} cells (green < {green_log_lower_linear:.2f} or > {green_log_upper_linear:.2f})")
    else:
        if verbose:
            print("    log10(Green) filter: no positive green values found")
    
    # === 组合所有过滤条件 ===
    valid_mask = area_valid & t50_valid & red_log_valid & green_log_valid
    n_valid = np.sum(valid_mask)
    if verbose:
        print(f"    Valid cells after all filters: {n_valid} ({n_valid/n_original*100:.1f}%)")
    
    # 返回过滤后的DataFrame副本
    return df.iloc[valid_mask].copy()


class StatisticsAnalyzer:
    """统计分析类：执行信号强度与表观速率常数的关系分析和多元回归分析
    
    变量定义：
    - 自变量：log10(红色强度) 和 log10(绿色强度)
    - 因变量：表观速率常数 k_app = ln(2)/T50
    
    注：k_app (Apparent Rate Constant) 用于表示在细胞内复杂环境(in situ)测量的“表观”速率常数
    """
    
    def __init__(self, output_dir: str, suffix: str = '', cooks_factor: float = 4.0, max_time_override: Optional[float] = None, min_r_squared: float = 0.9, min_pearson_change: Optional[float] = None):
        self.output_dir = Path(output_dir)
        self.suffix = suffix  # 输出文件名后缀，如 "_488_top20"
        self.cooks_factor = cooks_factor  # Cook's Distance阈值系数，默认4.0（阈值=cooks_factor/n）
        self.max_time_override = max_time_override  # 手动指定的最大时间（秒），用于覆盖CSV中的max_time
        self.min_r_squared = min_r_squared  # 最小R²阈值，默认0.9
        self.min_pearson_change = min_pearson_change  # 最小Pearson变化值阈值（A0 - A_inf）
    
    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用T90和R²过滤
        
        Parameters:
        - df: 要过滤的DataFrame
        
        Returns:
        - 过滤后的DataFrame
        """
        print(f"  加载了 {len(df)} 个细胞")
        
        # 排除T90>最大时长的反应
        if 't90' in df.columns:
            before_count = len(df)
            
            # 使用手动指定的max_time或CSV中的max_time
            if self.max_time_override is not None:
                # 使用手动指定的值
                print(f"  使用手动指定的 max_time: {self.max_time_override} 秒")
                df = df[df['t90'] <= self.max_time_override].copy()
            elif 'max_time' in df.columns:
                # 使用CSV中的max_time
                df = df[df['t90'] <= df['max_time']].copy()
            
            removed_count = before_count - len(df)
            if removed_count > 0:
                print(f"  移除了 {removed_count} 个 T90 > max_time 的细胞（T90超出实验时长）")
            print(f"  剩余细胞: {len(df)}")
        
        # 排除R²<阈值的拟合结果
        if 'r_squared' in df.columns:
            before_count = len(df)
            df = df[df['r_squared'] >= self.min_r_squared].copy()
            removed_count = before_count - len(df)
            if removed_count > 0:
                print(f"  移除了 {removed_count} 个 R² < {self.min_r_squared} 的细胞（拟合质量低）")
            print(f"  R²过滤后剩余细胞: {len(df)}")
        
        # 排除Pearson变化值<阈值的细胞
        if self.min_pearson_change is not None and 'pearson_change' in df.columns:
            before_count = len(df)
            df = df[df['pearson_change'] >= self.min_pearson_change].copy()
            removed_count = before_count - len(df)
            if removed_count > 0:
                print(f"  移除了 {removed_count} 个 Pearson变化值 < {self.min_pearson_change} 的细胞")
            print(f"  Pearson变化值过滤后剩余细胞: {len(df)}")
        
        return df
    
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
        df = self._apply_filters(df)
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
                df = df.copy()
                df['red'] = 1.0 / df['1/red']
                df['green'] = 1.0 / df['1/green']
            else:
                raise ValueError(f"缺少必需的列: {missing}")
        
        print(f"  Total cells with valid data: {len(df)}")
        
        # === 调用统一的数据清洗流程 ===
        df_filtered = apply_statistical_filters(df, verbose=True)
        
        # 提取过滤后的数据
        red_filtered = df_filtered['red'].values
        green_filtered = df_filtered['green'].values
        ratio_filtered = df_filtered['ratio'].values
        t50_filtered = df_filtered['t50'].values
        t90_filtered = df_filtered['t90'].values
        file_stems_filtered = df_filtered['file_stem'].values
        cell_ids_filtered = df_filtered['cell_id'].values
        
        # 对红绿强度取log10用于拟合分析
        red_log_filtered = np.log10(red_filtered)
        green_log_filtered = np.log10(green_filtered)
        
        # 计算表观速率常数 k_app = ln(2)/T50
        k_filtered = np.log(2) / t50_filtered
        k90_filtered = np.log(2) / t90_filtered  # 仅用于分布图展示
        
        # 计算偏回归残差（用于作图）
        # 对于 k_app_T50: 控制green后的red残差，控制red后的green残差
        model_k_vs_green = sm.OLS(k_filtered, sm.add_constant(green_log_filtered)).fit()
        model_red_vs_green = sm.OLS(red_log_filtered, sm.add_constant(green_log_filtered)).fit()
        resid_k_ctrl_green = np.array(model_k_vs_green.resid)  # k控制green后的残差
        resid_red_ctrl_green = np.array(model_red_vs_green.resid)  # red控制green后的残差
        
        model_k_vs_red = sm.OLS(k_filtered, sm.add_constant(red_log_filtered)).fit()
        model_green_vs_red = sm.OLS(green_log_filtered, sm.add_constant(red_log_filtered)).fit()
        resid_k_ctrl_red = np.array(model_k_vs_red.resid)  # k控制red后的残差
        resid_green_ctrl_red = np.array(model_green_vs_red.resid)  # green控制red后的残差
        
        # 计算多元回归的Cook's Distance
        X_multi = np.column_stack([red_log_filtered, green_log_filtered])
        X_multi_const = sm.add_constant(X_multi)
        model_multi = sm.OLS(k_filtered, X_multi_const).fit()
        influence = model_multi.get_influence()
        cooks_d_multi = influence.cooks_distance[0]
        threshold_multi = self.cooks_factor / len(k_filtered)
        multi_reg_valid = cooks_d_multi <= threshold_multi  # True=用于作图
        
        # 导出k_app和参数数据到CSV
        analysis_df = pd.DataFrame({
            'file_stem': file_stems_filtered,
            'cell_id': cell_ids_filtered,
            'log10_red': red_log_filtered,
            'log10_green': green_log_filtered,
            'ratio': ratio_filtered,
            't50': t50_filtered,
            't90': t90_filtered,
            'k_app_T50': k_filtered,
            'k_app_T90': k90_filtered,
            # 偏回归残差（用于作图）
            'resid_red_ctrl_green': resid_red_ctrl_green,  # X轴: log10(Red)|控制log10(Green)
            'resid_k_ctrl_green': resid_k_ctrl_green,      # Y轴: k_app|控制log10(Green)
            'resid_green_ctrl_red': resid_green_ctrl_red,  # X轴: log10(Green)|控制log10(Red)
            'resid_k_ctrl_red': resid_k_ctrl_red,          # Y轴: k_app|控制log10(Red)
            # Cook's Distance
            'cooks_d_multi_reg': cooks_d_multi,
            'multi_reg_valid': multi_reg_valid  # True=通过Cook过滤，用于多元回归作图
        })
        analysis_csv_path = self.output_dir / f'k_app_analysis_data{self.suffix}.csv'
        analysis_df.to_csv(analysis_csv_path, index=False)
        print(f"  Saved analysis data: {analysis_csv_path}")
        
        if len(t50_filtered) < 3:
            print("Not enough valid data points for analysis.")
            return
        
        # 生成信号强度 vs 反应速度常数分析图（线性-线性）
        self._generate_intensity_vs_rate_constant_plot(
            red_filtered, green_filtered, ratio_filtered,
            k_filtered, file_stems_filtered, cell_ids_filtered
        )
        
        # 生成双对数图（log10强度 vs log10(k_app)）
        self._generate_log_log_plot(
            red_log_filtered, green_log_filtered, k_filtered,
            file_stems_filtered, cell_ids_filtered
        )
        
        # 生成T50/T90分布直方图
        self._generate_t50_t90_histogram(t50_filtered, t90_filtered)
        
        # 生成反应速度常数分布直方图
        self._generate_rate_constant_histogram(k_filtered, k90_filtered)
        
        # 生成红绿荧光强度比分布直方图
        self._generate_ratio_histogram(ratio_filtered)
        
        # 生成红绿荧光强度线性拟合图
        self._generate_red_green_correlation_plot(red_filtered, green_filtered)
        
        # 多元回归分析（仅针对T50）
        self._perform_multiple_regression_analysis(
            red_filtered, green_filtered, k_filtered, k90_filtered,
            file_stems_filtered, cell_ids_filtered, 'k_app (T50)', log_scale=False)
        
        # 双对数多元回归分析（仅针对T50）
        k_log = np.log10(k_filtered)
        self._perform_multiple_regression_analysis(
            red_log_filtered, green_log_filtered, k_log, None,
            file_stems_filtered, cell_ids_filtered, 'log10(k_app) (T50)', log_scale=True)
        
        # Kinetic Saturation Analysis (仅针对T50)
        self._perform_coupled_kinetics_analysis(
            red_filtered, green_filtered, k_filtered, 'k_app (T50)')
    
    def _generate_t50_t90_histogram(self, t50_values, t90_values):
        """
        生成T50和T90的分布直方图
        
        Parameters:
        - t50_values: T50值数组
        - t90_values: T90值数组
        """
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        
        # T50 分布
        ax1 = axes[0]
        ax1.hist(t50_values, bins=20, color='#0072B2', alpha=0.7, edgecolor='#00507D')
        ax1.tick_params(axis='both', labelsize=16)
        ax1.locator_params(axis='both', nbins=5)
        ax1.axvline(x=np.median(t50_values), color='#D55E00', linestyle='--',
                   linewidth=2, label=f'Median: {np.median(t50_values):.2f}')
        ax1.axvline(x=np.mean(t50_values), color='#E69F00', linestyle=':',
                   linewidth=2, label=f'Mean: {np.mean(t50_values):.2f}')
        ax1.set_xlabel('T50', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title(f'T50 Distribution (n={len(t50_values)})', fontsize=14)
        ax1.legend(loc='best', fontsize=15, frameon=False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # T90 分布
        ax2 = axes[1]
        ax2.hist(t90_values, bins=20, color='#0072B2', alpha=0.7, edgecolor='#00507D')
        ax2.tick_params(axis='both', labelsize=16)
        ax2.locator_params(axis='both', nbins=5)
        ax2.axvline(x=np.median(t90_values), color='#D55E00', linestyle='--',
                   linewidth=2, label=f'Median: {np.median(t90_values):.2f}')
        ax2.axvline(x=np.mean(t90_values), color='#E69F00', linestyle=':',
                   linewidth=2, label=f'Mean: {np.mean(t90_values):.2f}')
        ax2.set_xlabel('T90', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title(f'T90 Distribution (n={len(t90_values)})', fontsize=14)
        ax2.legend(loc='best', fontsize=15, frameon=False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # 保存图片 - PNG和PDF格式
        fig_path_png = self.output_dir / f't50_t90_distribution{self.suffix}.png'
        fig_path_pdf = self.output_dir / f't50_t90_distribution{self.suffix}.pdf'
        fig.savefig(fig_path_png, dpi=300, bbox_inches='tight')
        fig.savefig(fig_path_pdf, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {fig_path_png}")
        print(f"  Saved: {fig_path_pdf}")
        
        # 输出统计信息
        print(f"\n  T50/T90 Distribution Statistics:")
        print(f"    T50: Mean={np.mean(t50_values):.2f}, Median={np.median(t50_values):.2f}, Std={np.std(t50_values):.2f}")
        print(f"    T90: Mean={np.mean(t90_values):.2f}, Median={np.median(t90_values):.2f}, Std={np.std(t90_values):.2f}")
    
    def _generate_rate_constant_histogram(self, k_values, k90_values):
        """
        生成表观速率常数的分布直方图
        
        Parameters:
        - k_values: k_app = ln(2)/T50 数组
        - k90_values: k_app = ln(2)/T90 数组
        """
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        
        # k_app (ln2/T50) 分布
        ax1 = axes[0]
        ax1.hist(k_values, bins=20, color='#0072B2', alpha=0.7, edgecolor='dimgray')
        ax1.tick_params(axis='both', labelsize=16)
        ax1.locator_params(axis='both', nbins=5)
        ax1.axvline(x=np.median(k_values), color='#D55E00', linestyle='--',
                   linewidth=2, label=f'Median: {np.median(k_values):.4f}')
        ax1.axvline(x=np.mean(k_values), color='#E69F00', linestyle=':',
                   linewidth=2, label=f'Mean: {np.mean(k_values):.4f}')
        ax1.set_xlabel(r'Apparent Rate Constant ($k_{app}$, $s^{-1}$) from T50', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title(f'$k_{{app}}$ (from T50) Distribution (n={len(k_values)})', fontsize=14)
        ax1.legend(loc='best', fontsize=15, frameon=False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # k_app (ln2/T90) 分布
        ax2 = axes[1]
        ax2.hist(k90_values, bins=20, color='#0072B2', alpha=0.7, edgecolor='dimgray')
        ax2.tick_params(axis='both', labelsize=16)
        ax2.locator_params(axis='both', nbins=5)
        ax2.axvline(x=np.median(k90_values), color='#D55E00', linestyle='--',
                   linewidth=2, label=f'Median: {np.median(k90_values):.4f}')
        ax2.axvline(x=np.mean(k90_values), color='#E69F00', linestyle=':',
                   linewidth=2, label=f'Mean: {np.mean(k90_values):.4f}')
        ax2.set_xlabel(r'Apparent Rate Constant ($k_{app}$, $s^{-1}$) from T90', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title(f'$k_{{app}}$ (from T90) Distribution (n={len(k90_values)})', fontsize=14)
        ax2.legend(loc='best', fontsize=15, frameon=False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # 保存图片 - PNG和PDF格式
        fig_path_png = self.output_dir / f'rate_constant_distribution{self.suffix}.png'
        fig_path_pdf = self.output_dir / f'rate_constant_distribution{self.suffix}.pdf'
        fig.savefig(fig_path_png, dpi=300, bbox_inches='tight')
        fig.savefig(fig_path_pdf, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {fig_path_png}")
        print(f"  Saved: {fig_path_pdf}")
        
        # 输出统计信息
        print(f"\n  Apparent Rate Constant Distribution Statistics:")
        print(f"    k_app(T50): Mean={np.mean(k_values):.4f}, Median={np.median(k_values):.4f}, Std={np.std(k_values):.4f}")
        print(f"    k_app(T90): Mean={np.mean(k90_values):.4f}, Median={np.median(k90_values):.4f}, Std={np.std(k90_values):.4f}")
    
    def _generate_ratio_histogram(self, ratio_values):
        """
        生成红绿荧光强度比的分布直方图（横轴为 log10(Red/Green)）
        
        Parameters:
        - ratio_values: 红绿比值数组 (Red/Green)
        """
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        
        # 转换为 log10
        log_ratio_values = np.log10(ratio_values)
        
        # # === 截断处理（暂时关闭） ===
        # cutoff = 2.0
        # log_cutoff = np.log10(cutoff)  # log10(2.0) ≈ 0.301
        # n_over_cutoff = np.sum(ratio_values > cutoff)
        # log_ratio_clipped = np.clip(log_ratio_values, None, log_cutoff)
        # bins = np.linspace(log_ratio_clipped.min(), log_cutoff, 31)
        # ax.hist(log_ratio_clipped, bins=bins, color='#B39DDB', alpha=0.7, edgecolor='dimgray')
        # # 修改 X 轴最右侧的标签为 ">0.3"
        # xticks = ax.get_xticks()
        # xticklabels = [f'{t:.2f}' for t in xticks]
        # for i, t in enumerate(xticks):
        #     if abs(t - log_cutoff) < 0.05:
        #         xticklabels[i] = f'>{log_cutoff:.2f}'
        # ax.set_xticks(xticks)
        # ax.set_xticklabels(xticklabels)
        # ax.set_xlim(right=log_cutoff + 0.02)
        # print(f"    Cells with ratio > {cutoff}: {n_over_cutoff} ({100*n_over_cutoff/len(ratio_values):.1f}%)")
        # # === 截断处理结束 ===
        
        # 红绿比值分布 - 使用紫色
        ax.hist(log_ratio_values, bins=20, color='#0072B2', alpha=0.7, edgecolor='dimgray')
        ax.tick_params(axis='both', labelsize=16)
        ax.locator_params(axis='both', nbins=5)
        ax.axvline(x=np.median(log_ratio_values), color='#D55E00', linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(log_ratio_values):.3f}')
        ax.axvline(x=np.mean(log_ratio_values), color='#E69F00', linestyle=':',
                   linewidth=2, label=f'Mean: {np.mean(log_ratio_values):.3f}')
        ax.set_xlabel(r'$\log_{10}$(Red/Green)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'$\\log_{{10}}$(Red/Green) Distribution (n={len(ratio_values)})', fontsize=14)
        ax.legend(loc='best', fontsize=15, frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # 保存图片 - PNG和PDF格式
        fig_path_png = self.output_dir / f'red_green_ratio_distribution{self.suffix}.png'
        fig_path_pdf = self.output_dir / f'red_green_ratio_distribution{self.suffix}.pdf'
        fig.savefig(fig_path_png, dpi=300, bbox_inches='tight')
        fig.savefig(fig_path_pdf, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {fig_path_png}")
        print(f"  Saved: {fig_path_pdf}")
        
        # 输出统计信息
        print(f"\n  log10(Red/Green) Ratio Distribution Statistics:")
        print(f"    Mean={np.mean(log_ratio_values):.3f}, Median={np.median(log_ratio_values):.3f}, Std={np.std(log_ratio_values):.3f}")
        print(f"    Min={np.min(log_ratio_values):.3f}, Max={np.max(log_ratio_values):.3f}")
    
    def _generate_red_green_correlation_plot(self, red_values, green_values):
        """
        生成红绿荧光强度的线性拟合图（双对数）
        
        Parameters:
        - red_values: 红色通道强度数组
        - green_values: 绿色通道强度数组
        """
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        
        # 取对数
        log_red = np.log10(red_values)
        log_green = np.log10(green_values)
        
        # 散点图 - 深蓝色
        ax.scatter(log_red, log_green, alpha=0.15, s=40, c='steelblue', edgecolors='none')
        
        # 线性拟合
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_red, log_green)
        x_line = np.linspace(log_red.min(), log_red.max(), 100)
        y_line = slope * x_line + intercept
        
        # 拟合线 - 黑色虚线
        ax.plot(x_line, y_line, 'k--', linewidth=2,
                label=f'R={r_value:.3f}')
        
        ax.tick_params(axis='both', labelsize=20)
        ax.set_xlabel(r'$\log_{10}$(Red Intensity)' + f'\n(p={p_value:.2e})', fontsize=12)
        ax.set_ylabel(r'$\log_{10}$(Green Intensity)', fontsize=12)
        ax.set_title(f'$\\log_{{10}}$(Red) vs $\\log_{{10}}$(Green) (n={len(red_values)})', fontsize=14)
        ax.legend(loc='best', fontsize=15, frameon=False)
        
        # 设置正方形范围
        x_range = log_red.max() - log_red.min()
        y_range = log_green.max() - log_green.min()
        max_range = max(x_range, y_range) * 1.1
        x_center = (log_red.max() + log_red.min()) / 2
        y_center = (log_green.max() + log_green.min()) / 2
        ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
        ax.set_ylim(y_center - max_range/2, y_center + max_range/2)
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        # 保存图片 - PNG和PDF格式
        fig_path_png = self.output_dir / f'red_green_correlation{self.suffix}.png'
        fig_path_pdf = self.output_dir / f'red_green_correlation{self.suffix}.pdf'
        fig.savefig(fig_path_png, dpi=300, bbox_inches='tight')
        fig.savefig(fig_path_pdf, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {fig_path_png}")
        print(f"  Saved: {fig_path_pdf}")
        
        # 输出统计信息
        print(f"\n  log10(Red) vs log10(Green) Correlation:")
        print(f"    R={r_value:.3f}, p={p_value:.2e}")
        print(f"    Slope={slope:.4f}, Intercept={intercept:.2f}")
    
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
        
        # 阈值: cooks_factor/n
        threshold = self.cooks_factor / n
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
    
    def _generate_intensity_vs_rate_constant_plot(self, red_filtered, green_filtered, ratio_filtered,
                                                    k_filtered, file_stems_filtered, cell_ids_filtered):
        """
        生成信号强度与表观速率常数(k_app=ln2/T50)关系的分析图（线性-线性）
        
        Parameters:
        - red_filtered: 红色强度数组
        - green_filtered: 绿色强度数组
        - ratio_filtered: 红绿比值数组
        - k_filtered: 表观速率常数 k_app = ln(2)/T50 数组
        - file_stems_filtered: 文件名数组
        - cell_ids_filtered: 细胞ID数组
        """
        
        # 创建图表: 2x2 布局
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        
        # 1. 红色强度 vs k_app (带Cook's Distance过滤)
        red_clean, k_red_clean, _, _ = self._filter_by_cooks_distance(
            red_filtered, k_filtered, 'Red vs k_app', file_stems_filtered, cell_ids_filtered)
        ax1 = axes[0, 0]
        ax1.scatter(red_clean, k_red_clean, alpha=0.1, s=40, c='#D96459', edgecolors='#B84A40')
        ax1.tick_params(axis='both', labelsize=20)
        slope1, intercept1, r1, p1, se1 = stats.linregress(red_clean, k_red_clean)
        x_line = np.linspace(red_clean.min(), red_clean.max(), 100)
        ax1.plot(x_line, slope1 * x_line + intercept1, 'k--', linewidth=2, 
                label=f'R={r1:.3f}')
        ax1.set_xlabel(f'Red Intensity\n(p={p1:.2e})', fontsize=11)
        ax1.set_ylabel(r'$k_{app}$ ($s^{-1}$)', fontsize=11)
        ax1.set_title(f'Red Intensity vs $k_{{app}}$ (n={len(red_clean)})', fontsize=12)
        ax1.legend(loc='best', fontsize=15, frameon=False)
        ax1.grid(True, alpha=0.3)
        
        # 2. 绿色强度 vs k_app (带Cook's Distance过滤)
        green_clean, k_green_clean, _, _ = self._filter_by_cooks_distance(
            green_filtered, k_filtered, 'Green vs k_app', file_stems_filtered, cell_ids_filtered)
        ax2 = axes[0, 1]
        ax2.scatter(green_clean, k_green_clean, alpha=0.1, s=40, c='#45ADA8', edgecolors='#358985')
        ax2.tick_params(axis='both', labelsize=20)
        slope2, intercept2, r2, p2, se2 = stats.linregress(green_clean, k_green_clean)
        x_line = np.linspace(green_clean.min(), green_clean.max(), 100)
        ax2.plot(x_line, slope2 * x_line + intercept2, 'k--', linewidth=2,
                label=f'R={r2:.3f}')
        ax2.set_xlabel(f'Green Intensity\n(p={p2:.2e})', fontsize=11)
        ax2.set_ylabel(r'$k_{app}$ ($s^{-1}$)', fontsize=11)
        ax2.set_title(f'Green Intensity vs $k_{{app}}$ (n={len(green_clean)})', fontsize=12)
        ax2.legend(loc='best', fontsize=15, frameon=False)
        ax2.grid(True, alpha=0.3)
        
        # 3. 红绿比值 vs k_app (带Cook's Distance过滤)
        ratio_clean, k_ratio_clean, _, _ = self._filter_by_cooks_distance(
            ratio_filtered, k_filtered, 'Ratio vs k_app', file_stems_filtered, cell_ids_filtered)
        ax3 = axes[1, 0]
        ax3.scatter(ratio_clean, k_ratio_clean, alpha=0.1, s=40, c='steelblue', edgecolors='none')
        ax3.tick_params(axis='both', labelsize=20)
        slope3, intercept3, r3, p3, se3 = stats.linregress(ratio_clean, k_ratio_clean)
        x_line = np.linspace(ratio_clean.min(), ratio_clean.max(), 100)
        ax3.plot(x_line, slope3 * x_line + intercept3, 'k--', linewidth=2,
                label=f'R={r3:.3f}')
        ax3.set_xlabel(f'Red/Green Ratio\n(p={p3:.2e})', fontsize=11)
        ax3.set_ylabel(r'$k_{app}$ ($s^{-1}$)', fontsize=11)
        ax3.set_title(f'Red/Green Ratio vs $k_{{app}}$ (n={len(ratio_clean)})', fontsize=12)
        ax3.legend(loc='best', fontsize=15, frameon=False)
        ax3.grid(True, alpha=0.3)
        
        # 4. 比值分布直方图 (使用过滤后的数据)
        ax4 = axes[1, 1]
        ax4.hist(ratio_clean, bins=30, color='#0072B2', alpha=0.7, edgecolor='black')
        ax4.tick_params(axis='both', labelsize=20)
        ax4.locator_params(axis='both', nbins=5)
        ax4.axvline(x=np.median(ratio_clean), color='#D55E00', linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(ratio_clean):.3f}')
        ax4.axvline(x=np.mean(ratio_clean), color='#E69F00', linestyle='--',
                   linewidth=2, label=f'Mean: {np.mean(ratio_clean):.3f}')
        ax4.set_xlabel('Red/Green Ratio', fontsize=11)
        ax4.set_ylabel('Count', fontsize=11)
        ax4.set_title(f'Red/Green Ratio Distribution (n={len(ratio_clean)})', fontsize=12)
        ax4.legend(loc='best', fontsize=15, frameon=False)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片 - PNG和PDF格式
        fig_path_png = self.output_dir / f'intensity_vs_rate_constant{self.suffix}.png'
        fig_path_pdf = self.output_dir / f'intensity_vs_rate_constant{self.suffix}.pdf'
        fig.savefig(fig_path_png, dpi=300, bbox_inches='tight')
        fig.savefig(fig_path_pdf, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {fig_path_png}")
        print(f"  Saved: {fig_path_pdf}")
        
        # 输出统计结果
        print(f"\n  Analysis Results (Intensity vs k_app, after Cook's Distance filter):")
        print(f"    Red vs k_app:   R={r1:.3f}, p={p1:.2e}, n={len(red_clean)}")
        print(f"    Green vs k_app: R={r2:.3f}, p={p2:.2e}, n={len(green_clean)}")
        print(f"    Ratio vs k_app: R={r3:.3f}, p={p3:.2e}, n={len(ratio_clean)}")
    
    def _generate_log_log_plot(self, red_log_filtered, green_log_filtered, k_filtered,
                                file_stems_filtered, cell_ids_filtered):
        """
        生成双对数图：log10(强度) vs log10(k_app)
        
        Parameters:
        - red_log_filtered: log10(红色强度) 数组
        - green_log_filtered: log10(绿色强度) 数组
        - k_filtered: k_app = ln(2)/T50 数组
        - file_stems_filtered: 文件名数组
        - cell_ids_filtered: 细胞ID数组
        """
        # 计算log10(k_app)
        k_log = np.log10(k_filtered)
        
        # 创建图表: 1x2 布局
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        
        # 1. log10(红色强度) vs log10(k_app)
        red_log_clean, k_log_red_clean, _, _ = self._filter_by_cooks_distance(
            red_log_filtered, k_log, 'log10(Red) vs log10(k_app)', file_stems_filtered, cell_ids_filtered)
        ax1 = axes[0]
        ax1.scatter(red_log_clean, k_log_red_clean, alpha=0.1, s=40, c='#D96459', edgecolors='#B84A40')
        ax1.tick_params(axis='both', labelsize=14)
        slope1, intercept1, r1, p1, _ = stats.linregress(red_log_clean, k_log_red_clean)
        x_line = np.linspace(red_log_clean.min(), red_log_clean.max(), 100)
        ax1.plot(x_line, slope1 * x_line + intercept1, 'k--', linewidth=2, 
                label=f'R={r1:.3f}')
        ax1.set_xlabel(f'log10(Red Intensity)\n(p={p1:.2e})', fontsize=11)
        ax1.set_ylabel(r'log10($k_{app}$)', fontsize=11)
        ax1.set_title(f'log10(Red) vs log10($k_{{app}}$) (n={len(red_log_clean)})', fontsize=12)
        ax1.legend(loc='best', fontsize=15, frameon=False)
        ax1.grid(True, alpha=0.3)
        
        # 2. log10(绿色强度) vs log10(k_app)
        green_log_clean, k_log_green_clean, _, _ = self._filter_by_cooks_distance(
            green_log_filtered, k_log, 'log10(Green) vs log10(k_app)', file_stems_filtered, cell_ids_filtered)
        ax2 = axes[1]
        ax2.scatter(green_log_clean, k_log_green_clean, alpha=0.1, s=40, c='#45ADA8', edgecolors='#358985')
        ax2.tick_params(axis='both', labelsize=14)
        slope2, intercept2, r2, p2, _ = stats.linregress(green_log_clean, k_log_green_clean)
        x_line = np.linspace(green_log_clean.min(), green_log_clean.max(), 100)
        ax2.plot(x_line, slope2 * x_line + intercept2, 'k--', linewidth=2,
                label=f'R={r2:.3f}')
        ax2.set_xlabel(f'log10(Green Intensity)\n(p={p2:.2e})', fontsize=11)
        ax2.set_ylabel(r'log10($k_{app}$)', fontsize=11)
        ax2.set_title(f'log10(Green) vs log10($k_{{app}}$) (n={len(green_log_clean)})', fontsize=12)
        ax2.legend(loc='best', fontsize=15, frameon=False)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片 - PNG和PDF格式
        fig_path_png = self.output_dir / f'log_log_plot{self.suffix}.png'
        fig_path_pdf = self.output_dir / f'log_log_plot{self.suffix}.pdf'
        fig.savefig(fig_path_png, dpi=300, bbox_inches='tight')
        fig.savefig(fig_path_pdf, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {fig_path_png}")
        print(f"  Saved: {fig_path_pdf}")
        
        # 输出统计结果
        print(f"\n  Log-Log Analysis Results (after Cook's Distance filter):")
        print(f"    log10(Red) vs log10(k_app):   R={r1:.3f}, p={p1:.2e}, n={len(red_log_clean)}")
        print(f"    log10(Green) vs log10(k_app): R={r2:.3f}, p={p2:.2e}, n={len(green_log_clean)}")
    
    def _perform_multiple_regression_analysis(self, red_var, green_var, y_main, y_other,
                                               file_stems, cell_ids, y_name, log_scale=False):
        """
        执行多元回归分析，包括偏回归和残差分析
        
        Parameters:
        - red_var: 红色变量（强度或log10(强度)）
        - green_var: 绿色变量（强度或log10(强度)）
        - y_main: 主因变量（k_app或log10(k_app)）
        - y_other: 另一个因变量（用于参考）
        - file_stems: 文件名数组
        - cell_ids: 细胞ID数组
        - y_name: 因变量名称
        - log_scale: 是否为双对数模式
        """
        # 根据模式设置标签
        if log_scale:
            red_label = 'log10(Red)'
            green_label = 'log10(Green)'
            file_suffix = f'_loglog_{y_name.replace(" ", "_").replace("(", "").replace(")", "")}'
        else:
            red_label = 'Red'
            green_label = 'Green'
            file_suffix = f'_{y_name.replace(" ", "_").replace("(", "").replace(")", "")}'
        
        print(f"\n  === Multiple Regression Analysis (Y = {y_name}, {'Log-Log' if log_scale else 'Linear'}) ===")
        
        n = len(y_main)
        if n < 5:
            print(f"    Not enough data points for multiple regression (n={n})")
            return
        
        # 构建自变量矩阵
        X = np.column_stack([red_var, green_var])
        X_with_const = sm.add_constant(X)
        y = y_main
        
        # 拟合OLS模型
        model = sm.OLS(y, X_with_const).fit()
        
        # 计算Cook's Distance
        influence = model.get_influence()
        cooks_d = influence.cooks_distance[0]
        threshold = self.cooks_factor / n
        outlier_mask = cooks_d > threshold
        n_outliers = np.sum(outlier_mask)
        
        if n_outliers > 0:
            print(f"    Cook's Distance filter: removed {n_outliers} points (threshold={self.cooks_factor}/{n}={threshold:.4f})")
            outlier_indices = np.where(outlier_mask)[0]
            for idx in outlier_indices:
                print(f"      - File: {file_stems[idx]}, Cell ID: {cell_ids[idx]}, Cook's D: {cooks_d[idx]:.4f}")
        
        # 过滤后重新拟合
        valid_mask = ~outlier_mask
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        red_clean = red_var[valid_mask]
        green_clean = green_var[valid_mask]
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
        print(f"        Intercept:  {model_clean.params[0]:.6f} (p={model_clean.pvalues[0]:.2e})")
        print(f"        {red_label}:  {model_clean.params[1]:.6f} (p={model_clean.pvalues[1]:.2e})")
        print(f"        {green_label}: {model_clean.params[2]:.6f} (p={model_clean.pvalues[2]:.2e})")
        
        # 创建图表1: 红绿偏回归分析图 (1x2 布局)
        fig1, axes1 = plt.subplots(1, 2, figsize=(8, 4))
        
        # 获取残差和预测值
        residuals = model_clean.resid
        fitted_values = model_clean.fittedvalues
        
        # 1. 偏回归图 - 红色
        ax1 = axes1[0]
        model_y_green = sm.OLS(y_clean, sm.add_constant(green_clean)).fit()
        model_red_green = sm.OLS(red_clean, sm.add_constant(green_clean)).fit()
        resid_y = np.array(model_y_green.resid)
        resid_red = np.array(model_red_green.resid)
        
        ax1.scatter(resid_red, resid_y, alpha=0.15, s=40, c='#D96459', edgecolors='#B84A40')
        ax1.tick_params(axis='both', labelsize=20)
        
        # 线性回归
        slope_partial, intercept_partial, r_partial, p_partial, _ = stats.linregress(resid_red, resid_y)
        x_line = np.linspace(resid_red.min(), resid_red.max(), 100)
        ax1.plot(x_line, slope_partial * x_line + intercept_partial, 'k--', linewidth=2,
                label=f'Partial R={r_partial:.3f}')
        ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
        ax1.set_xlabel(f'{red_label} | {green_label} (residuals)\n(p={p_partial:.2e})', fontsize=11)
        ax1.set_ylabel(f'{y_name} | {green_label} (residuals)', fontsize=11)
        ax1.set_title(f'Partial Regression: {red_label} | {green_label} (n={len(resid_red)})', fontsize=12)
        ax1.legend(loc='best', fontsize=15, frameon=False)
        ax1.grid(True, alpha=0.3)
        
        # 2. 偏回归图 - 绿色
        ax2 = axes1[1]
        model_y_red = sm.OLS(y_clean, sm.add_constant(red_clean)).fit()
        model_green_red = sm.OLS(green_clean, sm.add_constant(red_clean)).fit()
        resid_y2 = np.array(model_y_red.resid)
        resid_green = np.array(model_green_red.resid)
        
        ax2.scatter(resid_green, resid_y2, alpha=0.15, s=40, c='#45ADA8', edgecolors='#358985')
        ax2.tick_params(axis='both', labelsize=20)
        
        # 线性回归
        slope_partial2, intercept_partial2, r_partial2, p_partial2, _ = stats.linregress(resid_green, resid_y2)
        x_line = np.linspace(resid_green.min(), resid_green.max(), 100)
        ax2.plot(x_line, slope_partial2 * x_line + intercept_partial2, 'k--', linewidth=2,
                label=f'Partial R={r_partial2:.3f}')
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
        ax2.set_xlabel(f'{green_label} | {red_label} (residuals)\n(p={p_partial2:.2e})', fontsize=11)
        ax2.set_ylabel(f'{y_name} | {red_label} (residuals)', fontsize=11)
        ax2.set_title(f'Partial Regression: {green_label} | {red_label} (n={len(resid_green)})', fontsize=12)
        ax2.legend(loc='best', fontsize=15, frameon=False)
        ax2.grid(True, alpha=0.3)
        
        # 红图独立设置正方形范围
        x_range_red = resid_red.max() - resid_red.min()
        y_range_red = resid_y.max() - resid_y.min()
        max_range_red = max(x_range_red, y_range_red) * 1.1
        x_center_red = (resid_red.max() + resid_red.min()) / 2
        y_center_red = (resid_y.max() + resid_y.min()) / 2
        ax1.set_xlim(x_center_red - max_range_red/2, x_center_red + max_range_red/2)
        ax1.set_ylim(y_center_red - max_range_red/2, y_center_red + max_range_red/2)
        ax1.set_aspect('equal', adjustable='box')
        
        # 绿图独立设置正方形范围
        x_range_green = resid_green.max() - resid_green.min()
        y_range_green = resid_y2.max() - resid_y2.min()
        max_range_green = max(x_range_green, y_range_green) * 1.1
        x_center_green = (resid_green.max() + resid_green.min()) / 2
        y_center_green = (resid_y2.max() + resid_y2.min()) / 2
        ax2.set_xlim(x_center_green - max_range_green/2, x_center_green + max_range_green/2)
        ax2.set_ylim(y_center_green - max_range_green/2, y_center_green + max_range_green/2)
        ax2.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        # 保存偏回归图 - PNG和PDF格式
        y_name_clean = y_name.replace('/', '_').replace('(', '').replace(')', '').replace(' ', '_')
        fig1_path_png = self.output_dir / f'partial_regression_{y_name_clean}{self.suffix}.png'
        fig1_path_pdf = self.output_dir / f'partial_regression_{y_name_clean}{self.suffix}.pdf'
        fig1.savefig(fig1_path_png, dpi=300, bbox_inches='tight')
        fig1.savefig(fig1_path_pdf, dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print(f"\n    Saved: {fig1_path_png}")
        print(f"    Saved: {fig1_path_pdf}")
        
        # 创建图表2: 残差分析图 (1x2 布局)
        fig2, axes2 = plt.subplots(1, 2, figsize=(8, 4))
        
        # 3. 残差 vs 拟合值图
        ax3 = axes2[0]
        ax3.scatter(fitted_values, residuals, alpha=0.1, s=40, c='blue', edgecolors='darkblue')
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
        ax3.legend(loc='best', fontsize=15, frameon=False)
        ax3.grid(True, alpha=0.3)
        
        # 4. Q-Q图 (残差正态性检验)
        ax4 = axes2[1]
        from scipy import stats as scipy_stats
        scipy_stats.probplot(residuals, dist="norm", plot=ax4)
        ax4.set_title('Normal Q-Q Plot of Residuals', fontsize=12)
        ax4.tick_params(axis='both', labelsize=20)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存残差分析图 - PNG和PDF格式
        fig2_path_png = self.output_dir / f'residual_analysis_{y_name_clean}{self.suffix}.png'
        fig2_path_pdf = self.output_dir / f'residual_analysis_{y_name_clean}{self.suffix}.pdf'
        fig2.savefig(fig2_path_png, dpi=300, bbox_inches='tight')
        fig2.savefig(fig2_path_pdf, dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print(f"    Saved: {fig2_path_png}")
        print(f"    Saved: {fig2_path_pdf}")
        
        # 输出偏回归结果
        print(f"\n    Partial Regression Results:")
        print(f"      {red_label} | {green_label}:   Partial R = {r_partial:.3f}, p = {p_partial:.2e}")
        print(f"      {green_label} | {red_label}:   Partial R = {r_partial2:.3f}, p = {p_partial2:.2e}")
        
        # 双对数模式下生成偏回归热图
        if log_scale:
            self._generate_partial_regression_heatmap(
                resid_red, resid_y, resid_green, resid_y2,
                r_partial, p_partial, r_partial2, p_partial2,
                slope_partial, intercept_partial, slope_partial2, intercept_partial2,
                red_label, green_label, y_name
            )
    
    def _generate_partial_regression_heatmap(self, resid_red, resid_y_red, resid_green, resid_y_green,
                                              r_red, p_red, r_green, p_green,
                                              slope_red, intercept_red, slope_green, intercept_green,
                                              red_label, green_label, y_name):
        """
        生成偏回归分析的热图（双对数模式）
        """
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        
        import matplotlib.colors as mcolors
        
        # 红图独立设置正方形范围
        x_range_red = resid_red.max() - resid_red.min()
        y_range_red = resid_y_red.max() - resid_y_red.min()
        max_range_red = max(x_range_red, y_range_red) * 1.1
        x_center_red = (resid_red.max() + resid_red.min()) / 2
        y_center_red = (resid_y_red.max() + resid_y_red.min()) / 2
        xlim_red = (x_center_red - max_range_red/2, x_center_red + max_range_red/2)
        ylim_red = (y_center_red - max_range_red/2, y_center_red + max_range_red/2)
        
        # 1. 红色通道偏回归热图
        ax1 = axes[0]
        red_cmap = mcolors.LinearSegmentedColormap.from_list('custom_red', ['#FFFFFF', '#F5C3C0', '#D96459', '#B84A40'])
        h1 = ax1.hist2d(resid_red, resid_y_red, bins=30, cmap=red_cmap, cmin=1,
                        range=[xlim_red, ylim_red])
        cbar1 = plt.colorbar(h1[3], ax=ax1)
        cbar1.set_label('Count', fontname='Arial', fontsize=11)
        cbar1.ax.tick_params(labelsize=12)
        for label in cbar1.ax.get_yticklabels():
            label.set_fontname('Arial')
        cbar1.outline.set_visible(False)  # 去掉边框
        
        # 添加拟合线
        x_line_red = np.linspace(xlim_red[0], xlim_red[1], 100)
        ax1.plot(x_line_red, slope_red * x_line_red + intercept_red, 'k--', linewidth=2,
                label=f'Partial R={r_red:.3f}')
        ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.7, linewidth=1)
        ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.7, linewidth=1)
        ax1.set_xlabel(f'{red_label} | {green_label} (residuals)\n(p={p_red:.2e})', fontsize=11, fontname='Arial')
        ax1.set_ylabel(f'{y_name} | {green_label} (residuals)', fontsize=11, fontname='Arial')
        ax1.set_title(f'Partial Regression Heatmap: {red_label} (n={len(resid_red)})', fontsize=12, fontname='Arial')
        ax1.legend(loc='best', fontsize=15, frameon=False, prop={'family': 'Arial'})
        ax1.tick_params(axis='both', labelsize=12)
        for label in ax1.get_xticklabels() + ax1.get_yticklabels():
            label.set_fontname('Arial')
        ax1.set_xlim(xlim_red)
        ax1.set_ylim(ylim_red)
        ax1.set_aspect('equal', adjustable='box')
        
        # 绿图独立设置正方形范围
        x_range_green = resid_green.max() - resid_green.min()
        y_range_green = resid_y_green.max() - resid_y_green.min()
        max_range_green = max(x_range_green, y_range_green) * 1.1
        x_center_green = (resid_green.max() + resid_green.min()) / 2
        y_center_green = (resid_y_green.max() + resid_y_green.min()) / 2
        xlim_green = (x_center_green - max_range_green/2, x_center_green + max_range_green/2)
        ylim_green = (y_center_green - max_range_green/2, y_center_green + max_range_green/2)
        
        # 2. 绿色通道偏回归热图
        ax2 = axes[1]
        green_cmap = mcolors.LinearSegmentedColormap.from_list('custom_green', ['#FFFFFF', '#B8E5E3', '#45ADA8', '#358985'])
        h2 = ax2.hist2d(resid_green, resid_y_green, bins=30, cmap=green_cmap, cmin=1,
                        range=[xlim_green, ylim_green])
        cbar2 = plt.colorbar(h2[3], ax=ax2)
        cbar2.set_label('Count', fontname='Arial', fontsize=11)
        cbar2.ax.tick_params(labelsize=12)
        for label in cbar2.ax.get_yticklabels():
            label.set_fontname('Arial')
        cbar2.outline.set_visible(False)  # 去掉边框
        
        # 添加拟合线
        x_line_green = np.linspace(xlim_green[0], xlim_green[1], 100)
        ax2.plot(x_line_green, slope_green * x_line_green + intercept_green, 'k--', linewidth=2,
                label=f'Partial R={r_green:.3f}')
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.7, linewidth=1)
        ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.7, linewidth=1)
        ax2.set_xlabel(f'{green_label} | {red_label} (residuals)\n(p={p_green:.2e})', fontsize=11, fontname='Arial')
        ax2.set_ylabel(f'{y_name} | {red_label} (residuals)', fontsize=11, fontname='Arial')
        ax2.set_title(f'Partial Regression Heatmap: {green_label} (n={len(resid_green)})', fontsize=12, fontname='Arial')
        ax2.legend(loc='best', fontsize=15, frameon=False, prop={'family': 'Arial'})
        ax2.tick_params(axis='both', labelsize=12)
        for label in ax2.get_xticklabels() + ax2.get_yticklabels():
            label.set_fontname('Arial')
        ax2.set_xlim(xlim_green)
        ax2.set_ylim(ylim_green)
        ax2.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        # 保存图片 - PNG和PDF格式
        y_name_clean = y_name.replace('/', '_').replace('(', '').replace(')', '').replace(' ', '_')
        fig_path_png = self.output_dir / f'partial_regression_heatmap_{y_name_clean}{self.suffix}.png'
        fig_path_pdf = self.output_dir / f'partial_regression_heatmap_{y_name_clean}{self.suffix}.pdf'
        fig.savefig(fig_path_png, dpi=300, bbox_inches='tight')
        fig.savefig(fig_path_pdf, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: {fig_path_png}")
        print(f"    Saved: {fig_path_pdf}")
    
    def _perform_coupled_kinetics_analysis(self, red_intensity, green_intensity, k_obs, k_name):
        """
        Coupled Kinetics Analysis: 基于偏回归残差的竞争/抑制模型
        
        使用偏回归残差消除红绿共线性干扰：
        - 横轴：log10(Red) 的残差（红光中无法被绿光解释的部分）
        - 纵轴：log10(k_app) 的残差（速度变化中无法被绿光解释的部分）
        
        拟合抑制模型: resid_k = -alpha * resid_red + beta
        其中 alpha > 0 表示抑制效应
        
        Parameters:
        - red_intensity: 红色通道强度
        - green_intensity: 绿色通道强度
        - k_obs: 表观速率常数
        - k_name: 速率常数名称
        """
        from scipy.optimize import curve_fit
        
        print(f"\n  === Coupled Kinetics Analysis (Residual-based Inhibition Model) - {k_name} ===")
        
        # 排除无效数据
        valid_mask = ((red_intensity > 0) & (green_intensity > 0) & (k_obs > 0) & 
                      np.isfinite(red_intensity) & np.isfinite(green_intensity) & np.isfinite(k_obs))
        I_red = red_intensity[valid_mask]
        I_green = green_intensity[valid_mask]
        k_valid = k_obs[valid_mask]
        
        n_valid = len(k_valid)
        if n_valid < 20:
            print(f"    Not enough valid data points (n={n_valid})")
            return
        
        print(f"    Valid data points: n={n_valid}")
        
        # 转换为对数
        log_red = np.log10(I_red)
        log_green = np.log10(I_green)
        log_k = np.log10(k_valid)
        
        # ========== Step 1: 计算偏回归残差 ==========
        print(f"\n    Step 1: Computing partial regression residuals...")
        
        # 1a. log10(Red) 对 log10(Green) 的残差
        # 这是红光中无法被绿光解释的部分
        model_red_green = sm.OLS(log_red, sm.add_constant(log_green)).fit()
        resid_red = np.array(model_red_green.resid)  # "纯净"的红色效应
        
        # 1b. log10(k_app) 对 log10(Green) 的残差
        # 这是速度变化中无法被绿光解释的部分
        model_k_green = sm.OLS(log_k, sm.add_constant(log_green)).fit()
        resid_k = np.array(model_k_green.resid)  # "纯净"的速度变化
        
        print(f"      Red|Green regression: R²={model_red_green.rsquared:.4f}")
        print(f"      k|Green regression: R²={model_k_green.rsquared:.4f}")
        
        # ========== Step 2: 在残差空间中拟合抑制关系 ==========
        print(f"\n    Step 2: Fitting inhibition model in residual space...")
        
        # 线性回归: resid_k = slope * resid_red + intercept
        # 如果 slope < 0，说明红色起抑制作用
        slope, intercept, r_value, p_value, std_err = stats.linregress(resid_red, resid_k)
        
        print(f"\n    Linear Inhibition Model (Residual Space):")
        print(f"      Slope (inhibition coefficient): {slope:.6f} ± {std_err:.6f}")
        print(f"      Intercept: {intercept:.6f}")
        print(f"      Partial R: {r_value:.4f}")
        print(f"      p-value: {p_value:.2e}")
        
        if slope < 0:
            print(f"\n    ✓ CONFIRMED: Red channel shows INHIBITION effect (negative slope)")
            print(f"      Each unit increase in log10(Red|Green) decreases log10(k|Green) by {abs(slope):.4f}")
        else:
            print(f"\n    ✗ NOTE: Red channel shows ACTIVATION effect (positive slope)")
        
        # ========== Step 3: 尝试拟合非线性抑制模型 (Michaelis-Menten 抑制型) ==========
        print(f"\n    Step 3: Fitting nonlinear inhibition model...")
        
        # 在残差空间中，尝试拟合双曲线抑制模型
        # resid_k = k_max_resid / (1 + resid_red_shifted / Ki_resid) + baseline
        # 注意：残差可能有负值，需要平移
        
        resid_red_shifted = resid_red - resid_red.min() + 0.01  # 平移到正数
        
        def inhibition_model(x, k_max_r, Ki_r, baseline):
            return k_max_r / (1 + x / Ki_r) + baseline
        
        try:
            # 初始估计
            k_max_r_init = resid_k.max() - resid_k.min()
            Ki_r_init = np.median(resid_red_shifted)
            baseline_init = resid_k.min()
            
            popt, pcov = curve_fit(
                inhibition_model, resid_red_shifted, resid_k,
                p0=[k_max_r_init, Ki_r_init, baseline_init],
                bounds=([-np.inf, 0.001, -np.inf], [np.inf, np.inf, np.inf]),
                maxfev=20000
            )
            k_max_r_fit, Ki_r_fit, baseline_fit = popt
            perr = np.sqrt(np.diag(pcov))
            
            # 计算 R²
            k_pred_nl = inhibition_model(resid_red_shifted, k_max_r_fit, Ki_r_fit, baseline_fit)
            ss_res_nl = np.sum((resid_k - k_pred_nl) ** 2)
            ss_tot_nl = np.sum((resid_k - np.mean(resid_k)) ** 2)
            r_squared_nl = 1 - (ss_res_nl / ss_tot_nl)
            
            print(f"\n    Nonlinear Inhibition Model (Hyperbolic):")
            print(f"      k_max (residual): {k_max_r_fit:.6f} ± {perr[0]:.6f}")
            print(f"      Ki (residual): {Ki_r_fit:.6f} ± {perr[1]:.6f}")
            print(f"      Baseline: {baseline_fit:.6f} ± {perr[2]:.6f}")
            print(f"      R²: {r_squared_nl:.4f}")
            
            nonlinear_success = True
            
        except Exception as e:
            print(f"    Nonlinear fit failed: {e}")
            nonlinear_success = False
            r_squared_nl = None
        
        # ========== Step 4: 创建图表 ==========
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        
        # 1. 偏回归散点图 + 线性拟合
        ax1 = axes[0, 0]
        ax1.scatter(resid_red, resid_k, alpha=0.1, s=30, c='#D96459', edgecolors='none')
        
        # 线性拟合线
        x_line = np.linspace(resid_red.min(), resid_red.max(), 100)
        y_line = slope * x_line + intercept
        ax1.plot(x_line, y_line, 'k-', linewidth=2,
                label=f'Linear: slope={slope:.4f}, R={r_value:.3f}')
        
        ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
        ax1.set_xlabel('log10(Red) | log10(Green)  (Residuals)', fontsize=12)
        ax1.set_ylabel(f'log10({k_name}) | log10(Green)  (Residuals)', fontsize=12)
        ax1.set_title(f'Partial Regression: Red Inhibition Effect (n={n_valid})', fontsize=12)
        ax1.legend(loc='best', fontsize=10)
        ax1.tick_params(axis='both', labelsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal', adjustable='box')
        
        # 2. 偏回归热图
        ax2 = axes[0, 1]
        h = ax2.hist2d(resid_red, resid_k, bins=30, cmap='Reds', cmin=1)
        cbar = plt.colorbar(h[3], ax=ax2, label='Count')
        cbar.outline.set_visible(False)  # 去掉边框
        ax2.plot(x_line, y_line, 'k-', linewidth=2)
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.7)
        ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.7)
        ax2.set_xlabel('log10(Red) | log10(Green)  (Residuals)', fontsize=12)
        ax2.set_ylabel(f'log10({k_name}) | log10(Green)  (Residuals)', fontsize=12)
        ax2.set_title('Density Heatmap: Residual Space', fontsize=12)
        ax2.tick_params(axis='both', labelsize=11)
        ax2.set_aspect('equal', adjustable='box')
        
        # 3. 非线性抑制模型（如果拟合成功）
        ax3 = axes[1, 0]
        ax3.scatter(resid_red_shifted, resid_k, alpha=0.1, s=30, c='#45ADA8', edgecolors='none')
        
        if nonlinear_success:
            x_nl = np.linspace(resid_red_shifted.min(), resid_red_shifted.max(), 200)
            y_nl = inhibition_model(x_nl, k_max_r_fit, Ki_r_fit, baseline_fit)
            ax3.plot(x_nl, y_nl, 'k-', linewidth=2,
                    label=f'Hyperbolic: Ki={Ki_r_fit:.3f}, R²={r_squared_nl:.3f}')
            # 标记 IC50 点
            ax3.axvline(x=Ki_r_fit, color='red', linestyle='--', alpha=0.7,
                       label=f'Ki (IC50) = {Ki_r_fit:.3f}')
        
        ax3.set_xlabel('Shifted log10(Red) | log10(Green)', fontsize=12)
        ax3.set_ylabel(f'log10({k_name}) | log10(Green)  (Residuals)', fontsize=12)
        ax3.set_title('Nonlinear Inhibition Model Fit', fontsize=12)
        ax3.legend(loc='best', fontsize=10)
        ax3.tick_params(axis='both', labelsize=11)
        ax3.grid(True, alpha=0.3)
        
        # 4. 参数摘要
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = f"""
        Residual-based Inhibition Analysis
        {'='*40}
        
        Data: n = {n_valid} cells
        
        Step 1: Partial Regression
        {'-'*40}
        Red|Green regression R²: {model_red_green.rsquared:.4f}
        k|Green regression R²: {model_k_green.rsquared:.4f}
        
        Step 2: Linear Inhibition Model
        {'-'*40}
        Slope (inhibition coef): {slope:.6f} ± {std_err:.6f}
        Partial R: {r_value:.4f}
        p-value: {p_value:.2e}
        
        Interpretation:
        {'='*40}
        """
        
        if slope < 0:
            summary_text += f"""
        ✓ Red channel INHIBITS the reaction
        • Each 10x increase in "pure" Red signal
          decreases k_app by {10**abs(slope):.2f}x
          (after removing Green covariance)
        """
        else:
            summary_text += f"""
        ✗ Red channel ACTIVATES the reaction
        • Each 10x increase in "pure" Red signal
          increases k_app by {10**slope:.2f}x
        """
        
        if nonlinear_success:
            summary_text += f"""
        
        Nonlinear Model (Hyperbolic):
        {'-'*40}
        Ki (IC50 in residual space): {Ki_r_fit:.4f}
        R²: {r_squared_nl:.4f}
        """
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # 保存图片 - PNG和PDF格式
        k_name_clean = k_name.replace(' ', '_').replace('(', '').replace(')', '')
        fig_path_png = self.output_dir / f'coupled_kinetics_{k_name_clean}{self.suffix}.png'
        fig_path_pdf = self.output_dir / f'coupled_kinetics_{k_name_clean}{self.suffix}.pdf'
        fig.savefig(fig_path_png, dpi=300, bbox_inches='tight')
        fig.savefig(fig_path_pdf, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: {fig_path_png}")
        print(f"    Saved: {fig_path_pdf}")


class PercentileBinAnalyzer:
    """百分位分组分析器：按指定通道强度的百分位将细胞分组分析
    
    将细胞按荧光强度百分位分组（默认每10%一组），对每组执行统计分析，
    并生成参数随百分位变化的趋势曲线图。
    """
    
    def __init__(self, output_dir: str, channel: str, bin_step: float = 10.0,
                 cooks_factor: float = 4.0, max_time_override: Optional[float] = None,
                 min_r_squared: float = 0.9, window_size: float = 20.0,
                 min_pearson_change: Optional[float] = None):
        """
        初始化百分位分组分析器
        
        Parameters:
        - output_dir: 输出目录
        - channel: 用于分组的通道 ('488' 或 '561')
        - bin_step: 百分位步长 (默认10，即0-10%, 10-20%, ...)
        - cooks_factor: Cook's Distance阈值系数
        - max_time_override: 手动指定的最大时间（秒）
        - min_r_squared: 最小R²阈值
        - window_size: 滑窗大小 (默认20%，用于滑窗分析)
        - min_pearson_change: 最小Pearson变化值阈值
        """
        self.output_dir = Path(output_dir)
        self.channel = channel.lower()
        self.bin_step = bin_step
        self.cooks_factor = cooks_factor
        self.max_time_override = max_time_override
        self.min_r_squared = min_r_squared
        self.window_size = window_size
        self.min_pearson_change = min_pearson_change
        
        # 映射通道到列名
        self.channel_map = {
            '488': 'green',
            '561': 'red',
            'green': 'green',
            'red': 'red'
        }
        self.col_name = self.channel_map.get(self.channel)
        if self.col_name is None:
            raise ValueError(f"不支持的通道: {channel}，请使用 488/561/green/red")
    
    def run_binned_analysis(self, df: pd.DataFrame):
        """
        执行百分位分组分析的主入口
        
        Parameters:
        - df: 包含细胞数据的DataFrame
        """
        print(f"\n{'=' * 60}")
        print(f"Percentile Bin Analysis: {self.channel} channel, {self.bin_step}% bins")
        print("=" * 60)
        
        if self.col_name not in df.columns:
            print(f"  错误: 数据中没有 {self.col_name} 列")
            return
        
        # 创建输出子目录
        bin_output_dir = self.output_dir / f"percentile_bins_{self.channel}"
        bin_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  输出目录: {bin_output_dir}")
        
        # 生成百分位边界
        bin_edges = np.arange(0, 100 + self.bin_step, self.bin_step)
        n_bins = len(bin_edges) - 1
        print(f"  分组数量: {n_bins} (步长: {self.bin_step}%)")
        
        # 计算分组边界值
        intensity_values = df[self.col_name].values
        percentile_thresholds = [np.percentile(intensity_values, p) for p in bin_edges]
        
        # 存储每组的统计结果和原始数据
        results = []
        group_data_list = []  # 存储每个分组的原始数据用于绘图
        
        for i in range(n_bins):
            pct_min = bin_edges[i]
            pct_max = bin_edges[i + 1]
            pct_mid = (pct_min + pct_max) / 2
            
            # 筛选该百分位范围内的细胞
            threshold_low = percentile_thresholds[i]
            threshold_high = percentile_thresholds[i + 1]
            
            if i == n_bins - 1:  # 最后一组包含上边界
                mask = (intensity_values >= threshold_low) & (intensity_values <= threshold_high)
            else:
                mask = (intensity_values >= threshold_low) & (intensity_values < threshold_high)
            
            group_df = df[mask].copy()
            n_cells_raw = len(group_df)
            
            print(f"\n  --- Group {i+1}: {pct_min:.0f}%-{pct_max:.0f}% (n={n_cells_raw}) ---")
            
            if n_cells_raw < 5:
                print(f"      警告: 数据点太少 (n={n_cells_raw})，跳过此组")
                # 添加空结果
                results.append({
                    'percentile_min': pct_min,
                    'percentile_max': pct_max,
                    'percentile_mid': pct_mid,
                    'n_cells_raw': n_cells_raw,
                    'n_cells_iqr': 0,
                    'n_cells_final': 0,
                    'n_cook_removed': 0,
                    'intensity_threshold_low': threshold_low,
                    'intensity_threshold_high': threshold_high
                })
                # 添加空数据占位
                group_data_list.append(None)
                continue
            
            # 计算该组的统计参数（内部会进行数据清洗）
            group_stats, df_clean = self._compute_group_statistics(group_df, pct_min, pct_max, return_data=True)
            group_stats['percentile_min'] = pct_min
            group_stats['percentile_max'] = pct_max
            group_stats['percentile_mid'] = pct_mid
            group_stats['n_cells_raw'] = n_cells_raw
            group_stats['intensity_threshold_low'] = threshold_low
            group_stats['intensity_threshold_high'] = threshold_high
            results.append(group_stats)
            # 保存清洗后的数据用于绘图
            group_data_list.append(df_clean if group_stats['n_cells_final'] >= 5 else None)
        
        # 转换为DataFrame
        results_df = pd.DataFrame(results)
        
        # 保存汇总CSV
        csv_path = bin_output_dir / f"percentile_bin_summary_{self.channel}.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"\n  Saved summary: {csv_path}")
        
        # 生成趋势图
        self._generate_distribution_trends(results_df, bin_output_dir)
        self._generate_correlation_trends(results_df, bin_output_dir)
        self._generate_regression_trends(results_df, bin_output_dir)
        self._generate_kinetics_trends(results_df, bin_output_dir)
        self._generate_fitted_curves_trends(results_df, group_data_list, bin_output_dir)
        
        print(f"\n  Percentile Bin Analysis for {self.channel} completed!")
    
    def run_sliding_window_analysis(self, df: pd.DataFrame):
        """
        执行滑窗百分位分析
        
        与普通 percentile bin 分析不同，滑窗方式的窗口有重叠：
        - 普通方式: 步长=窗口大小，如 0-10%, 10-20%, 20-30%...
        - 滑窗方式: 窗口大小>步长，如 window=20%, step=10% → 0-20%, 10-30%, 20-40%...
        
        生成的图表不包含 distribution_trends，仅包含：
        - correlation_trends
        - regression_trends  
        - kinetics_trends
        
        Parameters:
        - df: 包含细胞数据的DataFrame
        """
        print(f"\n{'=' * 60}")
        print(f"Sliding Window Analysis: {self.channel} channel")
        print(f"  Window Size: {self.window_size}%, Step: {self.bin_step}%")
        print("=" * 60)
        
        if self.col_name not in df.columns:
            print(f"  错误: 数据中没有 {self.col_name} 列")
            return
        
        # 创建输出子目录
        sw_output_dir = self.output_dir / f"sliding_window_{self.channel}"
        sw_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  输出目录: {sw_output_dir}")
        
        # 生成滑窗中心点
        # 中心点从 window_size/2 开始，到 100 - window_size/2 结束
        half_window = self.window_size / 2
        center_points = np.arange(half_window, 100 - half_window + self.bin_step, self.bin_step)
        # 确保最后一个点不超过 100 - half_window
        center_points = center_points[center_points <= 100 - half_window]
        
        n_windows = len(center_points)
        print(f"  窗口数量: {n_windows}")
        
        if n_windows < 2:
            print(f"  警告: 窗口数量不足，请调整 window_size 或 bin_step")
            return
        
        # 计算强度值的百分位分布
        intensity_values = df[self.col_name].values
        
        # 存储每个窗口的统计结果
        results = []
        
        for i, center in enumerate(center_points):
            pct_min = center - half_window
            pct_max = center + half_window
            
            # 计算强度阈值
            threshold_low = np.percentile(intensity_values, pct_min)
            threshold_high = np.percentile(intensity_values, pct_max)
            
            # 筛选该窗口范围内的细胞
            if i == n_windows - 1:  # 最后一个窗口包含上边界
                mask = (intensity_values >= threshold_low) & (intensity_values <= threshold_high)
            else:
                mask = (intensity_values >= threshold_low) & (intensity_values < threshold_high)
            
            group_df = df[mask].copy()
            n_cells_raw = len(group_df)
            
            print(f"\n  --- Window {i+1}: {pct_min:.0f}%-{pct_max:.0f}% (center={center:.0f}%, n={n_cells_raw}) ---")
            
            if n_cells_raw < 5:
                print(f"      警告: 数据点太少 (n={n_cells_raw})，跳过此窗口")
                # 添加空结果
                results.append({
                    'percentile_min': pct_min,
                    'percentile_max': pct_max,
                    'percentile_mid': center,
                    'window_size': self.window_size,
                    'n_cells_raw': n_cells_raw,
                    'n_cells_iqr': 0,
                    'n_cells_final': 0,
                    'n_cook_removed': 0,
                    'intensity_threshold_low': threshold_low,
                    'intensity_threshold_high': threshold_high
                })
                continue
            
            # 计算该组的统计参数（内部会进行数据清洗）
            group_stats = self._compute_group_statistics(group_df, pct_min, pct_max)
            group_stats['percentile_min'] = pct_min
            group_stats['percentile_max'] = pct_max
            group_stats['percentile_mid'] = center
            group_stats['window_size'] = self.window_size
            group_stats['n_cells_raw'] = n_cells_raw
            group_stats['intensity_threshold_low'] = threshold_low
            group_stats['intensity_threshold_high'] = threshold_high
            results.append(group_stats)
        
        # 转换为DataFrame
        results_df = pd.DataFrame(results)
        
        # 保存汇总CSV
        csv_path = sw_output_dir / f"sliding_window_summary_{self.channel}.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"\n  Saved summary: {csv_path}")
        
        # 生成趋势图（不包含 distribution_trends）
        self._generate_sliding_correlation_trends(results_df, sw_output_dir)
        self._generate_sliding_regression_trends(results_df, sw_output_dir)
        self._generate_sliding_kinetics_trends(results_df, sw_output_dir)
        
        print(f"\n  Sliding Window Analysis for {self.channel} completed!")
    
    def _generate_sliding_correlation_trends(self, results_df: pd.DataFrame, output_dir: Path):
        """
        生成滑窗版本的相关性参数趋势图
        """
        df = results_df[results_df['n_cells_final'] >= 5].copy()
        if len(df) < 2:
            print("      警告: 有效窗口数量不足，跳过相关性趋势图")
            return
        
        x = df['percentile_mid'].values
        
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        
        # 1. Red vs k_app
        ax = axes[0, 0]
        if 'R_red_vs_k' in df.columns:
            y = df['R_red_vs_k'].values
            valid = ~np.isnan(y)
            if np.sum(valid) > 0:
                ax.plot(x[valid], y[valid], 'o-', color='#D96459', markersize=6, linewidth=2)
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Percentile (%)', fontsize=11)
        ax.set_ylabel('R (Correlation)', fontsize=11)
        ax.set_title('Red vs $k_{app}$', fontsize=12)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', labelsize=10)
        ax.set_ylim(-1.1, 1.1)
        
        # 2. Green vs k_app
        ax = axes[0, 1]
        if 'R_green_vs_k' in df.columns:
            y = df['R_green_vs_k'].values
            valid = ~np.isnan(y)
            if np.sum(valid) > 0:
                ax.plot(x[valid], y[valid], 'o-', color='#45ADA8', markersize=6, linewidth=2)
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Percentile (%)', fontsize=11)
        ax.set_ylabel('R (Correlation)', fontsize=11)
        ax.set_title('Green vs $k_{app}$', fontsize=12)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', labelsize=10)
        ax.set_ylim(-1.1, 1.1)
        
        # 3. Ratio vs k_app
        ax = axes[0, 2]
        if 'R_ratio_vs_k' in df.columns:
            y = df['R_ratio_vs_k'].values
            valid = ~np.isnan(y)
            if np.sum(valid) > 0:
                ax.plot(x[valid], y[valid], 'o-', color='steelblue', markersize=6, linewidth=2)
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Percentile (%)', fontsize=11)
        ax.set_ylabel('R (Correlation)', fontsize=11)
        ax.set_title('Ratio vs $k_{app}$', fontsize=12)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', labelsize=10)
        ax.set_ylim(-1.1, 1.1)
        
        # 4. log10(Red) vs log10(k_app)
        ax = axes[1, 0]
        if 'R_logred_vs_logk' in df.columns:
            y = df['R_logred_vs_logk'].values
            valid = ~np.isnan(y)
            if np.sum(valid) > 0:
                ax.plot(x[valid], y[valid], 'o-', color='#B84A40', markersize=6, linewidth=2)
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Percentile (%)', fontsize=11)
        ax.set_ylabel('R (Correlation)', fontsize=11)
        ax.set_title('$\\log_{10}$(Red) vs $\\log_{10}$($k_{app}$)', fontsize=12)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', labelsize=10)
        ax.set_ylim(-1.1, 1.1)
        
        # 5. log10(Green) vs log10(k_app)
        ax = axes[1, 1]
        if 'R_loggreen_vs_logk' in df.columns:
            y = df['R_loggreen_vs_logk'].values
            valid = ~np.isnan(y)
            if np.sum(valid) > 0:
                ax.plot(x[valid], y[valid], 'o-', color='#358985', markersize=6, linewidth=2)
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Percentile (%)', fontsize=11)
        ax.set_ylabel('R (Correlation)', fontsize=11)
        ax.set_title('$\\log_{10}$(Green) vs $\\log_{10}$($k_{app}$)', fontsize=12)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', labelsize=10)
        ax.set_ylim(-1.1, 1.1)
        
        # 6. Red-Green correlation
        ax = axes[1, 2]
        if 'R_red_green' in df.columns:
            y = df['R_red_green'].values
            valid = ~np.isnan(y)
            if np.sum(valid) > 0:
                ax.plot(x[valid], y[valid], 'o-', color='steelblue', markersize=6, linewidth=2)
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Percentile (%)', fontsize=11)
        ax.set_ylabel('R (Correlation)', fontsize=11)
        ax.set_title('Red-Green Correlation', fontsize=12)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', labelsize=10)
        ax.set_ylim(-1.1, 1.1)
        
        plt.suptitle(f'Correlation Trends - Sliding Window ({self.channel} channel, window={self.window_size}%, step={self.bin_step}%)', 
                     fontsize=13, y=1.02)
        plt.tight_layout()
        
        # 保存
        fig_path_png = output_dir / f"sliding_correlation_trends_{self.channel}.png"
        fig_path_pdf = output_dir / f"sliding_correlation_trends_{self.channel}.pdf"
        fig.savefig(fig_path_png, dpi=300, bbox_inches='tight')
        fig.savefig(fig_path_pdf, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {fig_path_png}")
    
    def _generate_sliding_regression_trends(self, results_df: pd.DataFrame, output_dir: Path):
        """
        生成滑窗版本的多元回归参数趋势图
        """
        df = results_df[results_df['n_cells_final'] >= 5].copy()
        if len(df) < 2:
            print("      警告: 有效窗口数量不足，跳过回归趋势图")
            return
        
        x = df['percentile_mid'].values
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # 1. R² 趋势
        ax = axes[0]
        y_min_r2, y_max_r2 = np.inf, -np.inf
        if 'R_squared' in df.columns:
            y = df['R_squared'].values
            valid = ~np.isnan(y)
            if np.sum(valid) > 0:
                ax.plot(x[valid], y[valid], 'o-', color='steelblue', markersize=6, linewidth=2, label='R²')
                y_min_r2 = min(y_min_r2, np.nanmin(y))
                y_max_r2 = max(y_max_r2, np.nanmax(y))
            # Adjusted R²
            if 'adj_R_squared' in df.columns:
                y2 = df['adj_R_squared'].values
                valid2 = ~np.isnan(y2)
                if np.sum(valid2) > 0:
                    ax.plot(x[valid2], y2[valid2], 's--', color='steelblue', markersize=5, linewidth=1.5, alpha=0.6, label='Adj. R²')
                    y_min_r2 = min(y_min_r2, np.nanmin(y2))
                    y_max_r2 = max(y_max_r2, np.nanmax(y2))
                ax.legend(fontsize=10, frameon=False)
        ax.set_xlabel('Percentile (%)', fontsize=11)
        ax.set_ylabel('R² / Adj. R²', fontsize=11)
        ax.set_title('Model Fit Quality', fontsize=12)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', labelsize=10)
        # 根据实际数据范围设置y轴，留出10%边距
        if y_min_r2 != np.inf and y_max_r2 != -np.inf:
            y_range = y_max_r2 - y_min_r2
            margin = y_range * 0.1 if y_range > 0 else 0.05
            ax.set_ylim(y_min_r2 - margin, y_max_r2 + margin)
        
        # 2. Beta coefficients
        ax = axes[1]
        if 'beta_red' in df.columns and 'beta_green' in df.columns:
            y_red = df['beta_red'].values
            y_green = df['beta_green'].values
            valid_r = ~np.isnan(y_red)
            valid_g = ~np.isnan(y_green)
            if np.sum(valid_r) > 0:
                ax.plot(x[valid_r], y_red[valid_r], 'o-', color='#D96459', markersize=6, linewidth=2, label=r'$\beta_{Red}$')
            if np.sum(valid_g) > 0:
                ax.plot(x[valid_g], y_green[valid_g], 's-', color='#45ADA8', markersize=6, linewidth=2, label=r'$\beta_{Green}$')
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.legend(fontsize=10, frameon=False)
        ax.set_xlabel('Percentile (%)', fontsize=11)
        ax.set_ylabel('Regression Coefficient', fontsize=11)
        ax.set_title('Multiple Regression Coefficients', fontsize=12)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', labelsize=10)
        
        # 3. Partial R (合并红绿)
        ax = axes[2]
        y_max = 0  # 用于计算对称y轴范围
        if 'partial_R_red' in df.columns:
            y_red = df['partial_R_red'].values
            valid_r = ~np.isnan(y_red)
            if np.sum(valid_r) > 0:
                ax.plot(x[valid_r], y_red[valid_r], 'o-', color='#D96459', markersize=6, linewidth=2, label='Red | Green')
                y_max = max(y_max, np.nanmax(np.abs(y_red)))
        if 'partial_R_green' in df.columns:
            y_green = df['partial_R_green'].values
            valid_g = ~np.isnan(y_green)
            if np.sum(valid_g) > 0:
                ax.plot(x[valid_g], y_green[valid_g], 's-', color='#45ADA8', markersize=6, linewidth=2, label='Green | Red')
                y_max = max(y_max, np.nanmax(np.abs(y_green)))
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.legend(fontsize=10, frameon=False)
        ax.set_xlabel('Percentile (%)', fontsize=11)
        ax.set_ylabel('Partial R', fontsize=11)
        ax.set_title('Partial Correlations', fontsize=12)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', labelsize=10)
        # 设置对称y轴，按最大值为基准，留出10%边距
        if y_max > 0:
            y_lim = y_max * 1.1
            ax.set_ylim(-y_lim, y_lim)
        else:
            ax.set_ylim(-1.1, 1.1)
        
        plt.suptitle(f'Regression Trends - Sliding Window ({self.channel} channel, window={self.window_size}%, step={self.bin_step}%)', 
                     fontsize=13, y=1.02)
        plt.tight_layout()
        
        # 保存
        fig_path_png = output_dir / f"sliding_regression_trends_{self.channel}.png"
        fig_path_pdf = output_dir / f"sliding_regression_trends_{self.channel}.pdf"
        fig.savefig(fig_path_png, dpi=300, bbox_inches='tight')
        fig.savefig(fig_path_pdf, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {fig_path_png}")
    
    def _generate_sliding_kinetics_trends(self, results_df: pd.DataFrame, output_dir: Path):
        """
        生成滑窗版本的动力学参数趋势图
        """
        df = results_df[results_df['n_cells_final'] >= 5].copy()
        if len(df) < 2:
            print("      警告: 有效窗口数量不足，跳过动力学趋势图")
            return
        
        x = df['percentile_mid'].values
        
        # 根据分组通道选择颜色
        color = self._get_channel_color()
        
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        
        # 1. Inhibition slope
        ax = axes[0, 0]
        if 'inhibition_slope' in df.columns:
            y = df['inhibition_slope'].values
            valid = ~np.isnan(y)
            if np.sum(valid) > 0:
                if 'inhibition_slope_se' in df.columns:
                    yerr = df['inhibition_slope_se'].values
                    ax.errorbar(x[valid], y[valid], yerr=yerr[valid], fmt='o-', color=color, 
                               capsize=3, markersize=6, linewidth=2)
                else:
                    ax.plot(x[valid], y[valid], 'o-', color=color, markersize=6, linewidth=2)
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Percentile (%)', fontsize=11)
        ax.set_ylabel('Inhibition Slope', fontsize=11)
        ax.set_title('Linear Inhibition Coefficient', fontsize=12)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', labelsize=10)
        
        # 2. Partial R (inhibition)
        ax = axes[0, 1]
        if 'partial_R_inhibition' in df.columns:
            y = df['partial_R_inhibition'].values
            valid = ~np.isnan(y)
            if np.sum(valid) > 0:
                ax.plot(x[valid], y[valid], 'o-', color=color, markersize=6, linewidth=2)
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Percentile (%)', fontsize=11)
        ax.set_ylabel('Partial R', fontsize=11)
        ax.set_title('Inhibition Partial Correlation', fontsize=12)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', labelsize=10)
        ax.set_ylim(-1.1, 1.1)
        
        # 3. Ki (IC50)
        ax = axes[1, 0]
        if 'Ki_IC50' in df.columns:
            y = df['Ki_IC50'].values
            valid = ~np.isnan(y)
            if np.sum(valid) > 0:
                if 'Ki_IC50_se' in df.columns:
                    yerr = df['Ki_IC50_se'].values
                    valid_err = valid & ~np.isnan(yerr)
                    if np.sum(valid_err) > 0:
                        ax.errorbar(x[valid_err], y[valid_err], yerr=yerr[valid_err], fmt='o-', 
                                   color=color, capsize=3, markersize=6, linewidth=2)
                    else:
                        ax.plot(x[valid], y[valid], 'o-', color=color, markersize=6, linewidth=2)
                else:
                    ax.plot(x[valid], y[valid], 'o-', color=color, markersize=6, linewidth=2)
        ax.set_xlabel('Percentile (%)', fontsize=11)
        ax.set_ylabel('Ki (IC50)', fontsize=11)
        ax.set_title('Nonlinear Inhibition Constant', fontsize=12)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', labelsize=10)
        
        # 4. Nonlinear R²
        ax = axes[1, 1]
        if 'R_squared_nonlinear' in df.columns:
            y = df['R_squared_nonlinear'].values
            valid = ~np.isnan(y)
            if np.sum(valid) > 0:
                ax.plot(x[valid], y[valid], 'o-', color=color, markersize=6, linewidth=2)
        ax.set_xlabel('Percentile (%)', fontsize=11)
        ax.set_ylabel('R²', fontsize=11)
        ax.set_title('Nonlinear Model R²', fontsize=12)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', labelsize=10)
        ax.set_ylim(0, 1.05)
        
        plt.suptitle(f'Kinetics Trends - Sliding Window ({self.channel} channel, window={self.window_size}%, step={self.bin_step}%)', 
                     fontsize=13, y=1.02)
        plt.tight_layout()
        
        # 保存
        fig_path_png = output_dir / f"sliding_kinetics_trends_{self.channel}.png"
        fig_path_pdf = output_dir / f"sliding_kinetics_trends_{self.channel}.pdf"
        fig.savefig(fig_path_png, dpi=300, bbox_inches='tight')
        fig.savefig(fig_path_pdf, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {fig_path_png}")
    
    def _compute_group_statistics(self, df: pd.DataFrame, pct_min: float, pct_max: float, 
                                   return_data: bool = False):
        """
        计算单个百分位组的所有统计参数
        
        Parameters:
        - df: 该组的DataFrame
        - pct_min: 百分位下限
        - pct_max: 百分位上限
        - return_data: 是否同时返回清洗后的数据（默认False）
        
        Returns:
        - dict: 包含所有统计参数的字典
        - 如果return_data=True，返回 (stats, df_clean)
        """
        stats = {}
        
        # === 调用统一的数据清洗流程 (IQR过滤) ===
        n_before = len(df)
        df = apply_statistical_filters(df, verbose=False)
        n_after_iqr = len(df)
        if n_before > n_after_iqr:
            print(f"      IQR filtering: {n_before} -> {n_after_iqr} cells ({n_after_iqr/n_before*100:.0f}% retained)")
        
        # === Pearson变化值过滤 ===
        if self.min_pearson_change is not None and 'pearson_change' in df.columns:
            n_before_pearson = len(df)
            df = df[df['pearson_change'] >= self.min_pearson_change].copy()
            n_after_pearson = len(df)
            if n_before_pearson > n_after_pearson:
                print(f"      Pearson change filtering: {n_before_pearson} -> {n_after_pearson} cells (>{self.min_pearson_change})")
        
        # 如果IQR清洗后数据点太少，返回空结果
        if len(df) < 5:
            print(f"      Warning: too few cells after IQR cleaning (n={len(df)})")
            stats['n_cells_iqr'] = len(df)
            stats['n_cells_final'] = 0
            if return_data:
                return stats, df
            return stats
        
        stats['n_cells_iqr'] = n_after_iqr
        
        # 提取基本数据
        red_values = df['red'].values
        green_values = df['green'].values
        ratio_values = df['ratio'].values
        t50_values = df['t50'].values
        t90_values = df['t90'].values
        
        # 计算表观速率常数
        k_t50 = np.log(2) / t50_values
        k_t90 = np.log(2) / t90_values
        
        # 对数变换
        log_red = np.log10(red_values[red_values > 0])
        log_green = np.log10(green_values[green_values > 0])
        log_ratio = np.log10(ratio_values[ratio_values > 0])
        log_k_t50 = np.log10(k_t50[k_t50 > 0])
        
        # === 1. 分布统计 ===
        # T50 统计
        stats['T50_mean'] = np.mean(t50_values)
        stats['T50_median'] = np.median(t50_values)
        stats['T50_std'] = np.std(t50_values)
        
        # T90 统计
        stats['T90_mean'] = np.mean(t90_values)
        stats['T90_median'] = np.median(t90_values)
        stats['T90_std'] = np.std(t90_values)
        
        # k_app(T50) 统计
        stats['k_app_T50_mean'] = np.mean(k_t50)
        stats['k_app_T50_median'] = np.median(k_t50)
        stats['k_app_T50_std'] = np.std(k_t50)
        
        # k_app(T90) 统计
        stats['k_app_T90_mean'] = np.mean(k_t90)
        stats['k_app_T90_median'] = np.median(k_t90)
        stats['k_app_T90_std'] = np.std(k_t90)
        
        # log10(Red/Green) 统计
        if len(log_ratio) > 0:
            stats['log_ratio_mean'] = np.mean(log_ratio)
            stats['log_ratio_median'] = np.median(log_ratio)
            stats['log_ratio_std'] = np.std(log_ratio)
        else:
            stats['log_ratio_mean'] = np.nan
            stats['log_ratio_median'] = np.nan
            stats['log_ratio_std'] = np.nan
        
        # 强度统计
        stats['red_mean'] = np.mean(red_values)
        stats['green_mean'] = np.mean(green_values)
        
        # === 2. 相关性分析 ===
        n = len(df)
        
        # Red vs k_app (线性坐标 - 归一化)
        if n >= 3:
            try:
                slope, intercept, r, p, se = stats_module.linregress(red_values, k_t50)
                stats['R_red_vs_k'] = r
                stats['p_red_vs_k'] = p
                stats['slope_red_vs_k'] = slope
                stats['intercept_red_vs_k'] = intercept
            except:
                stats['R_red_vs_k'] = np.nan
                stats['p_red_vs_k'] = np.nan
                stats['slope_red_vs_k'] = np.nan
                stats['intercept_red_vs_k'] = np.nan
            
            # Green vs k_app (线性坐标 - 归一化)
            try:
                slope, intercept, r, p, se = stats_module.linregress(green_values, k_t50)
                stats['R_green_vs_k'] = r
                stats['p_green_vs_k'] = p
                stats['slope_green_vs_k'] = slope
                stats['intercept_green_vs_k'] = intercept
            except:
                stats['R_green_vs_k'] = np.nan
                stats['p_green_vs_k'] = np.nan
                stats['slope_green_vs_k'] = np.nan
                stats['intercept_green_vs_k'] = np.nan
            
            # Ratio vs k_app (线性坐标 - 归一化)
            try:
                slope, intercept, r, p, se = stats_module.linregress(ratio_values, k_t50)
                stats['R_ratio_vs_k'] = r
                stats['p_ratio_vs_k'] = p
                stats['slope_ratio_vs_k'] = slope
                stats['intercept_ratio_vs_k'] = intercept
            except:
                stats['R_ratio_vs_k'] = np.nan
                stats['p_ratio_vs_k'] = np.nan
                stats['slope_ratio_vs_k'] = np.nan
                stats['intercept_ratio_vs_k'] = np.nan
            
            # log10(Red) vs log10(k_app) (双对数坐标 - 未归一化)
            valid_mask_red = (red_values > 0) & (k_t50 > 0)
            if np.sum(valid_mask_red) >= 3:
                try:
                    log_r = np.log10(red_values[valid_mask_red])
                    log_k = np.log10(k_t50[valid_mask_red])
                    slope, intercept, r, p, se = stats_module.linregress(log_r, log_k)
                    stats['R_logred_vs_logk'] = r
                    stats['p_logred_vs_logk'] = p
                    stats['slope_logred_vs_logk'] = slope
                    stats['intercept_logred_vs_logk'] = intercept
                except:
                    stats['R_logred_vs_logk'] = np.nan
                    stats['p_logred_vs_logk'] = np.nan
                    stats['slope_logred_vs_logk'] = np.nan
                    stats['intercept_logred_vs_logk'] = np.nan
            else:
                stats['R_logred_vs_logk'] = np.nan
                stats['p_logred_vs_logk'] = np.nan
                stats['slope_logred_vs_logk'] = np.nan
                stats['intercept_logred_vs_logk'] = np.nan
            
            # log10(Green) vs log10(k_app) (双对数坐标 - 未归一化)
            valid_mask_green = (green_values > 0) & (k_t50 > 0)
            if np.sum(valid_mask_green) >= 3:
                try:
                    log_g = np.log10(green_values[valid_mask_green])
                    log_k = np.log10(k_t50[valid_mask_green])
                    slope, intercept, r, p, se = stats_module.linregress(log_g, log_k)
                    stats['R_loggreen_vs_logk'] = r
                    stats['p_loggreen_vs_logk'] = p
                    stats['slope_loggreen_vs_logk'] = slope
                    stats['intercept_loggreen_vs_logk'] = intercept
                except:
                    stats['R_loggreen_vs_logk'] = np.nan
                    stats['p_loggreen_vs_logk'] = np.nan
                    stats['slope_loggreen_vs_logk'] = np.nan
                    stats['intercept_loggreen_vs_logk'] = np.nan
            else:
                stats['R_loggreen_vs_logk'] = np.nan
                stats['p_loggreen_vs_logk'] = np.nan
                stats['slope_loggreen_vs_logk'] = np.nan
                stats['intercept_loggreen_vs_logk'] = np.nan
            
            # Red-Green correlation
            valid_mask = (red_values > 0) & (green_values > 0)
            if np.sum(valid_mask) >= 3:
                try:
                    log_r = np.log10(red_values[valid_mask])
                    log_g = np.log10(green_values[valid_mask])
                    slope, intercept, r, p, se = stats_module.linregress(log_r, log_g)
                    stats['R_red_green'] = r
                except:
                    stats['R_red_green'] = np.nan
            else:
                stats['R_red_green'] = np.nan
        else:
            stats['R_red_vs_k'] = np.nan
            stats['p_red_vs_k'] = np.nan
            stats['slope_red_vs_k'] = np.nan
            stats['intercept_red_vs_k'] = np.nan
            stats['R_green_vs_k'] = np.nan
            stats['p_green_vs_k'] = np.nan
            stats['slope_green_vs_k'] = np.nan
            stats['intercept_green_vs_k'] = np.nan
            stats['R_ratio_vs_k'] = np.nan
            stats['p_ratio_vs_k'] = np.nan
            stats['slope_ratio_vs_k'] = np.nan
            stats['intercept_ratio_vs_k'] = np.nan
            stats['R_logred_vs_logk'] = np.nan
            stats['p_logred_vs_logk'] = np.nan
            stats['slope_logred_vs_logk'] = np.nan
            stats['intercept_logred_vs_logk'] = np.nan
            stats['R_loggreen_vs_logk'] = np.nan
            stats['p_loggreen_vs_logk'] = np.nan
            stats['slope_loggreen_vs_logk'] = np.nan
            stats['intercept_loggreen_vs_logk'] = np.nan
            stats['R_red_green'] = np.nan
        
        # === 3. 多元回归分析 (带Cook's Distance过滤) ===
        valid_mask = (red_values > 0) & (green_values > 0) & (k_t50 > 0)
        n_valid = np.sum(valid_mask)
        
        # 默认n_cells_final等于IQR后的数量
        stats['n_cells_final'] = n_after_iqr
        
        if n_valid >= 5:
            try:
                log_red_valid = np.log10(red_values[valid_mask])
                log_green_valid = np.log10(green_values[valid_mask])
                log_k_valid = np.log10(k_t50[valid_mask])
                
                # 第一次拟合，计算Cook's Distance
                X = np.column_stack([log_red_valid, log_green_valid])
                X_const = sm.add_constant(X)
                model_initial = sm.OLS(log_k_valid, X_const).fit()
                
                # 计算Cook's Distance
                influence = model_initial.get_influence()
                cooks_d = influence.cooks_distance[0]
                cooks_threshold = 4.0 / len(log_k_valid)  # 标准阈值 4/n
                cook_valid = cooks_d <= cooks_threshold
                n_cook_removed = np.sum(~cook_valid)
                
                if n_cook_removed > 0:
                    print(f"      Cook's Distance: removed {n_cook_removed} cells (threshold=4/{len(log_k_valid)}={cooks_threshold:.4f})")
                
                # 更新n_cells_final
                n_after_cook = np.sum(cook_valid)
                stats['n_cells_final'] = n_after_cook
                stats['n_cook_removed'] = n_cook_removed
                
                # 如果Cook过滤后数据点太少，跳过回归分析
                if n_after_cook < 5:
                    print(f"      Warning: too few cells after Cook's filtering (n={n_after_cook})")
                    stats['R_squared'] = np.nan
                    stats['adj_R_squared'] = np.nan
                    stats['beta_intercept'] = np.nan
                    stats['beta_red'] = np.nan
                    stats['beta_green'] = np.nan
                    stats['p_beta_red'] = np.nan
                    stats['p_beta_green'] = np.nan
                    stats['partial_R_red'] = np.nan
                    stats['partial_p_red'] = np.nan
                    stats['partial_R_green'] = np.nan
                    stats['partial_p_green'] = np.nan
                else:
                    # 使用Cook过滤后的数据重新拟合
                    log_red_clean = log_red_valid[cook_valid]
                    log_green_clean = log_green_valid[cook_valid]
                    log_k_clean = log_k_valid[cook_valid]
                    
                    X_clean = np.column_stack([log_red_clean, log_green_clean])
                    X_clean_const = sm.add_constant(X_clean)
                    model = sm.OLS(log_k_clean, X_clean_const).fit()
                    
                    stats['R_squared'] = model.rsquared
                    stats['adj_R_squared'] = model.rsquared_adj
                    stats['beta_intercept'] = model.params[0]
                    stats['beta_red'] = model.params[1]
                    stats['beta_green'] = model.params[2]
                    stats['p_beta_red'] = model.pvalues[1]
                    stats['p_beta_green'] = model.pvalues[2]
                    
                    # 偏回归分析 (使用Cook过滤后的数据)
                    # Red | Green
                    model_k_green = sm.OLS(log_k_clean, sm.add_constant(log_green_clean)).fit()
                    model_red_green = sm.OLS(log_red_clean, sm.add_constant(log_green_clean)).fit()
                    resid_k = model_k_green.resid
                    resid_red = model_red_green.resid
                    if len(resid_k) >= 3:
                        slope, intercept, r, p, se = stats_module.linregress(resid_red, resid_k)
                        stats['partial_R_red'] = r
                        stats['partial_p_red'] = p
                    else:
                        stats['partial_R_red'] = np.nan
                        stats['partial_p_red'] = np.nan
                    
                    # Green | Red
                    model_k_red = sm.OLS(log_k_clean, sm.add_constant(log_red_clean)).fit()
                    model_green_red = sm.OLS(log_green_clean, sm.add_constant(log_red_clean)).fit()
                    resid_k2 = model_k_red.resid
                    resid_green = model_green_red.resid
                    if len(resid_k2) >= 3:
                        slope, intercept, r, p, se = stats_module.linregress(resid_green, resid_k2)
                        stats['partial_R_green'] = r
                        stats['partial_p_green'] = p
                    else:
                        stats['partial_R_green'] = np.nan
                        stats['partial_p_green'] = np.nan
            except Exception as e:
                print(f"      多元回归分析失败: {e}")
                stats['R_squared'] = np.nan
                stats['adj_R_squared'] = np.nan
                stats['beta_intercept'] = np.nan
                stats['beta_red'] = np.nan
                stats['beta_green'] = np.nan
                stats['p_beta_red'] = np.nan
                stats['p_beta_green'] = np.nan
                stats['partial_R_red'] = np.nan
                stats['partial_p_red'] = np.nan
                stats['partial_R_green'] = np.nan
                stats['partial_p_green'] = np.nan
        else:
            stats['R_squared'] = np.nan
            stats['adj_R_squared'] = np.nan
            stats['beta_intercept'] = np.nan
            stats['beta_red'] = np.nan
            stats['beta_green'] = np.nan
            stats['p_beta_red'] = np.nan
            stats['p_beta_green'] = np.nan
            stats['partial_R_red'] = np.nan
            stats['partial_p_red'] = np.nan
            stats['partial_R_green'] = np.nan
            stats['partial_p_green'] = np.nan
        
        # === 4. Coupled Kinetics 分析 ===
        if n_valid >= 20:
            try:
                log_red_valid = np.log10(red_values[valid_mask])
                log_green_valid = np.log10(green_values[valid_mask])
                log_k_valid = np.log10(k_t50[valid_mask])
                
                # 偏回归残差
                model_red_green = sm.OLS(log_red_valid, sm.add_constant(log_green_valid)).fit()
                model_k_green = sm.OLS(log_k_valid, sm.add_constant(log_green_valid)).fit()
                resid_red = np.array(model_red_green.resid)
                resid_k = np.array(model_k_green.resid)
                
                # 线性抑制模型
                slope, intercept, r, p, se = stats_module.linregress(resid_red, resid_k)
                stats['inhibition_slope'] = slope
                stats['inhibition_slope_se'] = se
                stats['partial_R_inhibition'] = r
                stats['p_inhibition'] = p
                
                # 非线性抑制模型 (尝试)
                from scipy.optimize import curve_fit
                resid_red_shifted = resid_red - resid_red.min() + 0.01
                
                def inhibition_model(x, k_max_r, Ki_r, baseline):
                    return k_max_r / (1 + x / Ki_r) + baseline
                
                try:
                    k_max_r_init = resid_k.max() - resid_k.min()
                    Ki_r_init = np.median(resid_red_shifted)
                    baseline_init = resid_k.min()
                    
                    popt, pcov = curve_fit(
                        inhibition_model, resid_red_shifted, resid_k,
                        p0=[k_max_r_init, Ki_r_init, baseline_init],
                        bounds=([-np.inf, 0.001, -np.inf], [np.inf, np.inf, np.inf]),
                        maxfev=10000
                    )
                    k_max_r_fit, Ki_r_fit, baseline_fit = popt
                    perr = np.sqrt(np.diag(pcov))
                    
                    # 计算 R²
                    k_pred = inhibition_model(resid_red_shifted, k_max_r_fit, Ki_r_fit, baseline_fit)
                    ss_res = np.sum((resid_k - k_pred) ** 2)
                    ss_tot = np.sum((resid_k - np.mean(resid_k)) ** 2)
                    r_squared_nl = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
                    
                    stats['Ki_IC50'] = Ki_r_fit
                    stats['Ki_IC50_se'] = perr[1] if len(perr) > 1 else np.nan
                    stats['R_squared_nonlinear'] = r_squared_nl
                except:
                    stats['Ki_IC50'] = np.nan
                    stats['Ki_IC50_se'] = np.nan
                    stats['R_squared_nonlinear'] = np.nan
            except Exception as e:
                print(f"      Coupled Kinetics分析失败: {e}")
                stats['inhibition_slope'] = np.nan
                stats['inhibition_slope_se'] = np.nan
                stats['partial_R_inhibition'] = np.nan
                stats['p_inhibition'] = np.nan
                stats['Ki_IC50'] = np.nan
                stats['Ki_IC50_se'] = np.nan
                stats['R_squared_nonlinear'] = np.nan
        else:
            stats['inhibition_slope'] = np.nan
            stats['inhibition_slope_se'] = np.nan
            stats['partial_R_inhibition'] = np.nan
            stats['p_inhibition'] = np.nan
            stats['Ki_IC50'] = np.nan
            stats['Ki_IC50_se'] = np.nan
            stats['R_squared_nonlinear'] = np.nan
        
        if return_data:
            return stats, df
        return stats
    
    def _get_channel_color(self) -> str:
        """
        根据分组通道返回对应的颜色
        488/green 返回绿色 #45ADA8，561/red 返回红色 #D96459
        """
        if self.channel in ['488', 'green']:
            return '#45ADA8'  # 绿色
        elif self.channel in ['561', 'red']:
            return '#D96459'  # 红色
        else:
            return 'steelblue'  # 默认
    
    def _generate_distribution_trends(self, results_df: pd.DataFrame, output_dir: Path):
        """
        生成分布统计参数随百分位变化的趋势图
        """
        # 过滤有效数据 (Cook过滤后细胞数 >= 5)
        df = results_df[results_df['n_cells_final'] >= 5].copy()
        if len(df) < 2:
            print("      警告: 有效分组数量不足，跳过分布趋势图")
            return
        
        x = df['percentile_mid'].values
        
        # 根据分组通道选择颜色
        color = self._get_channel_color()
        
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        
        # 1. T50 趋势
        ax = axes[0, 0]
        if 'T50_mean' in df.columns:
            y = df['T50_mean'].values
            yerr = df['T50_std'].values
            ax.errorbar(x, y, yerr=yerr, fmt='o-', color=color, capsize=3, markersize=6)
            ax.fill_between(x, y - yerr, y + yerr, alpha=0.2, color=color)
        ax.set_xlabel('Percentile (%)', fontsize=11)
        ax.set_ylabel('T50 (s)', fontsize=11)
        ax.set_title('T50 vs Percentile', fontsize=12)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', labelsize=10)
        
        # 2. T90 趋势
        ax = axes[0, 1]
        if 'T90_mean' in df.columns:
            y = df['T90_mean'].values
            yerr = df['T90_std'].values
            ax.errorbar(x, y, yerr=yerr, fmt='o-', color=color, capsize=3, markersize=6)
            ax.fill_between(x, y - yerr, y + yerr, alpha=0.2, color=color)
        ax.set_xlabel('Percentile (%)', fontsize=11)
        ax.set_ylabel('T90 (s)', fontsize=11)
        ax.set_title('T90 vs Percentile', fontsize=12)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', labelsize=10)
        
        # 3. k_app(T50) 趋势
        ax = axes[0, 2]
        if 'k_app_T50_mean' in df.columns:
            y = df['k_app_T50_mean'].values
            yerr = df['k_app_T50_std'].values
            ax.errorbar(x, y, yerr=yerr, fmt='o-', color=color, capsize=3, markersize=6)
            ax.fill_between(x, y - yerr, y + yerr, alpha=0.2, color=color)
        ax.set_xlabel('Percentile (%)', fontsize=11)
        ax.set_ylabel(r'$k_{app}$ (T50) ($s^{-1}$)', fontsize=11)
        ax.set_title(r'$k_{app}$ (T50) vs Percentile', fontsize=12)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', labelsize=10)
        
        # 4. k_app(T90) 趋势
        ax = axes[1, 0]
        if 'k_app_T90_mean' in df.columns:
            y = df['k_app_T90_mean'].values
            yerr = df['k_app_T90_std'].values
            ax.errorbar(x, y, yerr=yerr, fmt='o-', color=color, capsize=3, markersize=6)
            ax.fill_between(x, y - yerr, y + yerr, alpha=0.2, color=color)
        ax.set_xlabel('Percentile (%)', fontsize=11)
        ax.set_ylabel(r'$k_{app}$ (T90) ($s^{-1}$)', fontsize=11)
        ax.set_title(r'$k_{app}$ (T90) vs Percentile', fontsize=12)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', labelsize=10)
        
        # 5. log10(Red/Green) 趋势
        ax = axes[1, 1]
        if 'log_ratio_mean' in df.columns:
            y = df['log_ratio_mean'].values
            yerr = df['log_ratio_std'].values
            valid = ~np.isnan(y)
            if np.sum(valid) > 0:
                ax.errorbar(x[valid], y[valid], yerr=yerr[valid], fmt='o-', color=color, capsize=3, markersize=6)
                ax.fill_between(x[valid], (y - yerr)[valid], (y + yerr)[valid], alpha=0.2, color=color)
        ax.set_xlabel('Percentile (%)', fontsize=11)
        ax.set_ylabel(r'$\log_{10}$(Red/Green)', fontsize=11)
        ax.set_title(r'$\log_{10}$(Red/Green) vs Percentile', fontsize=12)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', labelsize=10)
        
        # 6. 细胞数量 (显示Raw -> IQR -> Cook三个阶段)
        ax = axes[1, 2]
        y_raw = df['n_cells_raw'].values if 'n_cells_raw' in df.columns else np.zeros(len(x))
        y_iqr = df['n_cells_iqr'].values if 'n_cells_iqr' in df.columns else np.zeros(len(x))
        y_final = df['n_cells_final'].values
        width = self.bin_step * 0.25
        ax.bar(x - width, y_raw, width=width, color='#CCCCCC', edgecolor='dimgray', alpha=0.7, label='Raw')
        ax.bar(x, y_iqr, width=width, color='#88CCEE', edgecolor='dimgray', alpha=0.7, label='IQR')
        ax.bar(x + width, y_final, width=width, color='#44AA99', edgecolor='dimgray', alpha=0.7, label='Final')
        ax.set_xlabel('Percentile (%)', fontsize=11)
        ax.set_ylabel('Cell Count', fontsize=11)
        ax.set_title('Cell Count: Raw -> IQR -> Cook', fontsize=12)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', labelsize=10)
        ax.legend(fontsize=9, frameon=False, loc='upper left')
        
        plt.suptitle(f'Distribution Statistics Trends ({self.channel} channel, {self.bin_step}% bins)', 
                     fontsize=14, y=1.02)
        plt.tight_layout()
        
        # 保存
        fig_path_png = output_dir / f"percentile_distribution_trends_{self.channel}.png"
        fig_path_pdf = output_dir / f"percentile_distribution_trends_{self.channel}.pdf"
        fig.savefig(fig_path_png, dpi=300, bbox_inches='tight')
        fig.savefig(fig_path_pdf, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {fig_path_png}")
    
    def _generate_correlation_trends(self, results_df: pd.DataFrame, output_dir: Path):
        """
        生成相关性参数随百分位变化的趋势图
        """
        df = results_df[results_df['n_cells_final'] >= 5].copy()
        if len(df) < 2:
            print("      警告: 有效分组数量不足，跳过相关性趋势图")
            return
        
        x = df['percentile_mid'].values
        
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        
        # 1. Red vs k_app
        ax = axes[0, 0]
        if 'R_red_vs_k' in df.columns:
            y = df['R_red_vs_k'].values
            valid = ~np.isnan(y)
            if np.sum(valid) > 0:
                ax.plot(x[valid], y[valid], 'o-', color='#D96459', markersize=6, linewidth=2)
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Percentile (%)', fontsize=11)
        ax.set_ylabel('R (Correlation)', fontsize=11)
        ax.set_title('Red vs $k_{app}$', fontsize=12)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', labelsize=10)
        ax.set_ylim(-1.1, 1.1)
        
        # 2. Green vs k_app
        ax = axes[0, 1]
        if 'R_green_vs_k' in df.columns:
            y = df['R_green_vs_k'].values
            valid = ~np.isnan(y)
            if np.sum(valid) > 0:
                ax.plot(x[valid], y[valid], 'o-', color='#45ADA8', markersize=6, linewidth=2)
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Percentile (%)', fontsize=11)
        ax.set_ylabel('R (Correlation)', fontsize=11)
        ax.set_title('Green vs $k_{app}$', fontsize=12)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', labelsize=10)
        ax.set_ylim(-1.1, 1.1)
        
        # 3. Ratio vs k_app
        ax = axes[0, 2]
        if 'R_ratio_vs_k' in df.columns:
            y = df['R_ratio_vs_k'].values
            valid = ~np.isnan(y)
            if np.sum(valid) > 0:
                ax.plot(x[valid], y[valid], 'o-', color='steelblue', markersize=6, linewidth=2)
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Percentile (%)', fontsize=11)
        ax.set_ylabel('R (Correlation)', fontsize=11)
        ax.set_title('Ratio vs $k_{app}$', fontsize=12)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', labelsize=10)
        ax.set_ylim(-1.1, 1.1)
        
        # 4. log10(Red) vs log10(k_app)
        ax = axes[1, 0]
        if 'R_logred_vs_logk' in df.columns:
            y = df['R_logred_vs_logk'].values
            valid = ~np.isnan(y)
            if np.sum(valid) > 0:
                ax.plot(x[valid], y[valid], 'o-', color='#B84A40', markersize=6, linewidth=2)
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Percentile (%)', fontsize=11)
        ax.set_ylabel('R (Correlation)', fontsize=11)
        ax.set_title('$\log_{10}$(Red) vs $\log_{10}$($k_{app}$)', fontsize=12)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', labelsize=10)
        ax.set_ylim(-1.1, 1.1)
        
        # 5. log10(Green) vs log10(k_app)
        ax = axes[1, 1]
        if 'R_loggreen_vs_logk' in df.columns:
            y = df['R_loggreen_vs_logk'].values
            valid = ~np.isnan(y)
            if np.sum(valid) > 0:
                ax.plot(x[valid], y[valid], 'o-', color='#358985', markersize=6, linewidth=2)
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Percentile (%)', fontsize=11)
        ax.set_ylabel('R (Correlation)', fontsize=11)
        ax.set_title('$\log_{10}$(Green) vs $\log_{10}$($k_{app}$)', fontsize=12)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', labelsize=10)
        ax.set_ylim(-1.1, 1.1)
        
        # 6. Red-Green correlation
        ax = axes[1, 2]
        if 'R_red_green' in df.columns:
            y = df['R_red_green'].values
            valid = ~np.isnan(y)
            if np.sum(valid) > 0:
                ax.plot(x[valid], y[valid], 'o-', color='steelblue', markersize=6, linewidth=2)
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Percentile (%)', fontsize=11)
        ax.set_ylabel('R (Correlation)', fontsize=11)
        ax.set_title('Red-Green Correlation', fontsize=12)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', labelsize=10)
        ax.set_ylim(-1.1, 1.1)
        
        plt.suptitle(f'Correlation Trends ({self.channel} channel, {self.bin_step}% bins)', 
                     fontsize=14, y=1.02)
        plt.tight_layout()
        
        # 保存
        fig_path_png = output_dir / f"percentile_correlation_trends_{self.channel}.png"
        fig_path_pdf = output_dir / f"percentile_correlation_trends_{self.channel}.pdf"
        fig.savefig(fig_path_png, dpi=300, bbox_inches='tight')
        fig.savefig(fig_path_pdf, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {fig_path_png}")
    
    def _generate_regression_trends(self, results_df: pd.DataFrame, output_dir: Path):
        """
        生成多元回归参数随百分位变化的趋势图
        """
        df = results_df[results_df['n_cells_final'] >= 5].copy()
        if len(df) < 2:
            print("      警告: 有效分组数量不足，跳过回归趋势图")
            return
        
        x = df['percentile_mid'].values
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # 1. R² 趋势
        ax = axes[0]
        y_min_r2, y_max_r2 = np.inf, -np.inf
        if 'R_squared' in df.columns:
            y = df['R_squared'].values
            valid = ~np.isnan(y)
            if np.sum(valid) > 0:
                ax.plot(x[valid], y[valid], 'o-', color='steelblue', markersize=6, linewidth=2, label='R²')
                y_min_r2 = min(y_min_r2, np.nanmin(y))
                y_max_r2 = max(y_max_r2, np.nanmax(y))
            # Adjusted R²
            if 'adj_R_squared' in df.columns:
                y2 = df['adj_R_squared'].values
                valid2 = ~np.isnan(y2)
                if np.sum(valid2) > 0:
                    ax.plot(x[valid2], y2[valid2], 's--', color='steelblue', markersize=5, linewidth=1.5, alpha=0.6, label='Adj. R²')
                    y_min_r2 = min(y_min_r2, np.nanmin(y2))
                    y_max_r2 = max(y_max_r2, np.nanmax(y2))
                ax.legend(fontsize=10, frameon=False)
        ax.set_xlabel('Percentile (%)', fontsize=11)
        ax.set_ylabel('R² / Adj. R²', fontsize=11)
        ax.set_title('Model Fit Quality', fontsize=12)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', labelsize=10)
        # 根据实际数据范围设置y轴，留出10%边距
        if y_min_r2 != np.inf and y_max_r2 != -np.inf:
            y_range = y_max_r2 - y_min_r2
            margin = y_range * 0.1 if y_range > 0 else 0.05
            ax.set_ylim(y_min_r2 - margin, y_max_r2 + margin)
        
        # 2. Beta coefficients
        ax = axes[1]
        if 'beta_red' in df.columns and 'beta_green' in df.columns:
            y_red = df['beta_red'].values
            y_green = df['beta_green'].values
            valid_r = ~np.isnan(y_red)
            valid_g = ~np.isnan(y_green)
            if np.sum(valid_r) > 0:
                ax.plot(x[valid_r], y_red[valid_r], 'o-', color='#D96459', markersize=6, linewidth=2, label=r'$\beta_{Red}$')
            if np.sum(valid_g) > 0:
                ax.plot(x[valid_g], y_green[valid_g], 's-', color='#45ADA8', markersize=6, linewidth=2, label=r'$\beta_{Green}$')
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.legend(fontsize=10, frameon=False)
        ax.set_xlabel('Percentile (%)', fontsize=11)
        ax.set_ylabel('Regression Coefficient', fontsize=11)
        ax.set_title('Multiple Regression Coefficients', fontsize=12)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', labelsize=10)
        
        # 3. Partial R (合并红绿)
        ax = axes[2]
        y_max = 0  # 用于计算对称y轴范围
        if 'partial_R_red' in df.columns:
            y_red = df['partial_R_red'].values
            valid_r = ~np.isnan(y_red)
            if np.sum(valid_r) > 0:
                ax.plot(x[valid_r], y_red[valid_r], 'o-', color='#D96459', markersize=6, linewidth=2, label='Red | Green')
                y_max = max(y_max, np.nanmax(np.abs(y_red)))
        if 'partial_R_green' in df.columns:
            y_green = df['partial_R_green'].values
            valid_g = ~np.isnan(y_green)
            if np.sum(valid_g) > 0:
                ax.plot(x[valid_g], y_green[valid_g], 's-', color='#45ADA8', markersize=6, linewidth=2, label='Green | Red')
                y_max = max(y_max, np.nanmax(np.abs(y_green)))
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.legend(fontsize=10, frameon=False)
        ax.set_xlabel('Percentile (%)', fontsize=11)
        ax.set_ylabel('Partial R', fontsize=11)
        ax.set_title('Partial Correlations', fontsize=12)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', labelsize=10)
        # 设置对称y轴，按最大值为基准，留出10%边距
        if y_max > 0:
            y_lim = y_max * 1.1
            ax.set_ylim(-y_lim, y_lim)
        else:
            ax.set_ylim(-1.1, 1.1)
        
        plt.suptitle(f'Multiple Regression Trends ({self.channel} channel, {self.bin_step}% bins)', 
                     fontsize=14, y=1.02)
        plt.tight_layout()
        
        # 保存
        fig_path_png = output_dir / f"percentile_regression_trends_{self.channel}.png"
        fig_path_pdf = output_dir / f"percentile_regression_trends_{self.channel}.pdf"
        fig.savefig(fig_path_png, dpi=300, bbox_inches='tight')
        fig.savefig(fig_path_pdf, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {fig_path_png}")
    
    def _generate_kinetics_trends(self, results_df: pd.DataFrame, output_dir: Path):
        """
        生成动力学参数随百分位变化的趋势图
        """
        df = results_df[results_df['n_cells_final'] >= 5].copy()
        if len(df) < 2:
            print("      警告: 有效分组数量不足，跳过动力学趋势图")
            return
        
        x = df['percentile_mid'].values
        
        # 根据分组通道选择颜色
        color = self._get_channel_color()
        
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        
        # 1. Inhibition slope
        ax = axes[0, 0]
        if 'inhibition_slope' in df.columns:
            y = df['inhibition_slope'].values
            valid = ~np.isnan(y)
            if np.sum(valid) > 0:
                if 'inhibition_slope_se' in df.columns:
                    yerr = df['inhibition_slope_se'].values
                    ax.errorbar(x[valid], y[valid], yerr=yerr[valid], fmt='o-', color=color, 
                               capsize=3, markersize=6, linewidth=2)
                else:
                    ax.plot(x[valid], y[valid], 'o-', color=color, markersize=6, linewidth=2)
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Percentile (%)', fontsize=11)
        ax.set_ylabel('Inhibition Slope', fontsize=11)
        ax.set_title('Linear Inhibition Coefficient', fontsize=12)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', labelsize=10)
        
        # 2. Partial R (inhibition)
        ax = axes[0, 1]
        if 'partial_R_inhibition' in df.columns:
            y = df['partial_R_inhibition'].values
            valid = ~np.isnan(y)
            if np.sum(valid) > 0:
                ax.plot(x[valid], y[valid], 'o-', color=color, markersize=6, linewidth=2)
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Percentile (%)', fontsize=11)
        ax.set_ylabel('Partial R', fontsize=11)
        ax.set_title('Inhibition Partial Correlation', fontsize=12)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', labelsize=10)
        ax.set_ylim(-1.1, 1.1)
        
        # 3. Ki (IC50)
        ax = axes[1, 0]
        if 'Ki_IC50' in df.columns:
            y = df['Ki_IC50'].values
            valid = ~np.isnan(y)
            if np.sum(valid) > 0:
                if 'Ki_IC50_se' in df.columns:
                    yerr = df['Ki_IC50_se'].values
                    valid_err = valid & ~np.isnan(yerr)
                    if np.sum(valid_err) > 0:
                        ax.errorbar(x[valid_err], y[valid_err], yerr=yerr[valid_err], fmt='o-', 
                                   color=color, capsize=3, markersize=6, linewidth=2)
                    else:
                        ax.plot(x[valid], y[valid], 'o-', color=color, markersize=6, linewidth=2)
                else:
                    ax.plot(x[valid], y[valid], 'o-', color=color, markersize=6, linewidth=2)
        ax.set_xlabel('Percentile (%)', fontsize=11)
        ax.set_ylabel('Ki (IC50)', fontsize=11)
        ax.set_title('Nonlinear Inhibition Constant', fontsize=12)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', labelsize=10)
        
        # 4. Nonlinear R²
        ax = axes[1, 1]
        if 'R_squared_nonlinear' in df.columns:
            y = df['R_squared_nonlinear'].values
            valid = ~np.isnan(y)
            if np.sum(valid) > 0:
                ax.plot(x[valid], y[valid], 'o-', color=color, markersize=6, linewidth=2)
        ax.set_xlabel('Percentile (%)', fontsize=11)
        ax.set_ylabel('R²', fontsize=11)
        ax.set_title('Nonlinear Model R²', fontsize=12)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', labelsize=10)
        ax.set_ylim(0, 1.05)
        
        plt.suptitle(f'Coupled Kinetics Trends ({self.channel} channel, {self.bin_step}% bins)', 
                     fontsize=14, y=1.02)
        plt.tight_layout()
        
        # 保存
        fig_path_png = output_dir / f"percentile_kinetics_trends_{self.channel}.png"
        fig_path_pdf = output_dir / f"percentile_kinetics_trends_{self.channel}.pdf"
        fig.savefig(fig_path_png, dpi=300, bbox_inches='tight')
        fig.savefig(fig_path_pdf, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {fig_path_png}")
    
    def _generate_fitted_curves_trends(self, results_df: pd.DataFrame, 
                                       group_data_list: list, output_dir: Path):
        """
        生成各分组的拟合回归曲线大图
        
        每个子图显示一个百分位分组的散点和拟合线，
        包含归一化（线性坐标）和未归一化（双对数坐标）两种拟合。
        所有子图共享相同的横纵坐标轴范围。
        
        Parameters:
        - results_df: 各组统计结果DataFrame
        - group_data_list: 每个分组的原始数据列表（与results_df对应）
        - output_dir: 输出目录
        """
        # 过滤有效分组
        valid_indices = []
        for i, data in enumerate(group_data_list):
            if data is not None and len(data) >= 5:
                valid_indices.append(i)
        
        if len(valid_indices) < 2:
            print("      警告: 有效分组数量不足，跳过拟合曲线图")
            return
        
        n_valid = len(valid_indices)
        
        # 计算子图布局（尽量接近正方形）
        n_cols = int(np.ceil(np.sqrt(n_valid)))
        n_rows = int(np.ceil(n_valid / n_cols))
        
        # 创建图形：2行（线性坐标 + 双对数坐标），多列（各分组）
        fig, axes = plt.subplots(2, n_valid, figsize=(3.5 * n_valid, 8))
        
        # 如果只有一个有效分组，axes会是1维，需要reshape
        if n_valid == 1:
            axes = axes.reshape(2, 1)
        
        # 收集所有数据以确定全局坐标轴范围
        all_red, all_green = [], []
        all_k_red, all_k_green = [], []
        all_log_red, all_log_green = [], []
        all_log_k_red, all_log_k_green = [], []
        
        for idx in valid_indices:
            df = group_data_list[idx]
            if df is None or len(df) < 5:
                continue
            
            red_vals = df['red'].values
            green_vals = df['green'].values
            t50_vals = df['t50'].values
            k_vals = np.log(2) / t50_vals
            
            # 线性坐标有效数据
            valid_red = (red_vals > 0) & (k_vals > 0) & np.isfinite(red_vals) & np.isfinite(k_vals)
            valid_green = (green_vals > 0) & (k_vals > 0) & np.isfinite(green_vals) & np.isfinite(k_vals)
            
            if np.sum(valid_red) >= 3:
                all_red.extend(red_vals[valid_red])
                all_k_red.extend(k_vals[valid_red])
                all_log_red.extend(np.log10(red_vals[valid_red]))
                all_log_k_red.extend(np.log10(k_vals[valid_red]))
            
            if np.sum(valid_green) >= 3:
                all_green.extend(green_vals[valid_green])
                all_k_green.extend(k_vals[valid_green])
                all_log_green.extend(np.log10(green_vals[valid_green]))
                all_log_k_green.extend(np.log10(k_vals[valid_green]))
        
        # 计算全局坐标轴范围
        if len(all_red) > 0:
            red_range = (min(all_red) * 0.95, max(all_red) * 1.05)
            k_red_range = (min(all_k_red) * 0.95, max(all_k_red) * 1.05)
            log_red_range = (min(all_log_red) - 0.05, max(all_log_red) + 0.05)
            log_k_red_range = (min(all_log_k_red) - 0.05, max(all_log_k_red) + 0.05)
        else:
            red_range = (0, 1)
            k_red_range = (0, 1)
            log_red_range = (-1, 1)
            log_k_red_range = (-1, 1)
        
        if len(all_green) > 0:
            green_range = (min(all_green) * 0.95, max(all_green) * 1.05)
            k_green_range = (min(all_k_green) * 0.95, max(all_k_green) * 1.05)
            log_green_range = (min(all_log_green) - 0.05, max(all_log_green) + 0.05)
            log_k_green_range = (min(all_log_k_green) - 0.05, max(all_log_k_green) + 0.05)
        else:
            green_range = (0, 1)
            k_green_range = (0, 1)
            log_green_range = (-1, 1)
            log_k_green_range = (-1, 1)
        
        # 根据分组通道确定要显示的通道
        if self.channel in ['488', 'green']:
            display_channel = 'green'
            color = '#45ADA8'
            intensity_range = green_range
            k_range = k_green_range
            log_intensity_range = log_green_range
            log_k_range = log_k_green_range
        else:  # 561/red
            display_channel = 'red'
            color = '#D96459'
            intensity_range = red_range
            k_range = k_red_range
            log_intensity_range = log_red_range
            log_k_range = log_k_red_range
        
        # 绘制每个分组的子图
        for col_idx, data_idx in enumerate(valid_indices):
            df = group_data_list[data_idx]
            stats_row = results_df.iloc[data_idx]
            
            pct_min = stats_row['percentile_min']
            pct_max = stats_row['percentile_max']
            n_cells = stats_row['n_cells_final']
            
            # 提取数据
            intensity_vals = df[display_channel].values
            t50_vals = df['t50'].values
            k_vals = np.log(2) / t50_vals
            
            # 有效数据掩码
            valid_mask = (intensity_vals > 0) & (k_vals > 0) & np.isfinite(intensity_vals) & np.isfinite(k_vals)
            
            if np.sum(valid_mask) < 3:
                continue
            
            x_valid = intensity_vals[valid_mask]
            k_valid = k_vals[valid_mask]
            x_log = np.log10(x_valid)
            k_log = np.log10(k_valid)
            
            # 获取回归参数
            slope_linear = stats_row.get(f'slope_{display_channel}_vs_k', np.nan)
            intercept_linear = stats_row.get(f'intercept_{display_channel}_vs_k', np.nan)
            r_linear = stats_row.get(f'R_{display_channel}_vs_k', np.nan)
            
            slope_loglog = stats_row.get(f'slope_log{display_channel}_vs_logk', np.nan)
            intercept_loglog = stats_row.get(f'intercept_log{display_channel}_vs_logk', np.nan)
            r_loglog = stats_row.get(f'R_log{display_channel}_vs_logk', np.nan)
            
            # === 第一行：线性坐标（归一化）===
            ax1 = axes[0, col_idx]
            
            # 散点
            ax1.scatter(x_valid, k_valid, alpha=0.3, s=30, c=color, edgecolors='none')
            
            # 拟合线
            if not np.isnan(slope_linear):
                x_line = np.linspace(intensity_range[0], intensity_range[1], 100)
                y_line = slope_linear * x_line + intercept_linear
                ax1.plot(x_line, y_line, 'k--', linewidth=2, 
                        label=f'R={r_linear:.3f}')
                ax1.legend(loc='best', fontsize=9, frameon=False)
            
            ax1.set_xlim(intensity_range)
            ax1.set_ylim(k_range)
            ax1.set_xlabel(f'{display_channel.capitalize()} Intensity', fontsize=10)
            if col_idx == 0:
                ax1.set_ylabel(r'$k_{app}$ ($s^{-1}$)', fontsize=10)
            ax1.set_title(f'{pct_min:.0f}%-{pct_max:.0f}% (n={n_cells})', fontsize=10)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.tick_params(axis='both', labelsize=9)
            ax1.grid(True, alpha=0.3)
            
            # === 第二行：双对数坐标（未归一化）===
            ax2 = axes[1, col_idx]
            
            # 散点
            ax2.scatter(x_log, k_log, alpha=0.3, s=30, c=color, edgecolors='none')
            
            # 拟合线
            if not np.isnan(slope_loglog):
                x_line_log = np.linspace(log_intensity_range[0], log_intensity_range[1], 100)
                y_line_log = slope_loglog * x_line_log + intercept_loglog
                ax2.plot(x_line_log, y_line_log, 'k--', linewidth=2,
                        label=f'R={r_loglog:.3f}')
                ax2.legend(loc='best', fontsize=9, frameon=False)
            
            ax2.set_xlim(log_intensity_range)
            ax2.set_ylim(log_k_range)
            ax2.set_xlabel(r'$\log_{10}$(' + display_channel.capitalize() + ')', fontsize=10)
            if col_idx == 0:
                ax2.set_ylabel(r'$\log_{10}$($k_{app}$)', fontsize=10)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.tick_params(axis='both', labelsize=9)
            ax2.grid(True, alpha=0.3)
        
        # 添加总标题
        coord_type = "Green" if self.channel in ['488', 'green'] else "Red"
        plt.suptitle(f'Fitted Regression Curves by Percentile ({coord_type} channel, {self.bin_step}% bins)\n'
                     f'Top: Linear Scale (Normalized) | Bottom: Log-Log Scale (Unnormalized)',
                     fontsize=13, y=1.02)
        
        plt.tight_layout()
        
        # 保存
        fig_path_png = output_dir / f"percentile_fitted_curves_{self.channel}.png"
        fig_path_pdf = output_dir / f"percentile_fitted_curves_{self.channel}.pdf"
        fig.savefig(fig_path_png, dpi=300, bbox_inches='tight')
        fig.savefig(fig_path_pdf, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {fig_path_png}")


# 为了避免与已导入的stats模块冲突，这里使用别名
stats_module = stats


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


def filter_bottom_percent(df: pd.DataFrame, channel: str, percent: float) -> pd.DataFrame:
    """
    筛选指定通道亮度排名后xx%的细胞
    
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
    threshold = np.percentile(df[col_name].values, percent)
    
    # 筛选
    filtered_df = df[df[col_name] <= threshold].copy()
    
    return filtered_df


def filter_percentile_range(df: pd.DataFrame, channel: str, min_percentile: float, max_percentile: float) -> pd.DataFrame:
    """
    筛选指定通道百分位范围内的细胞
    
    Parameters:
    - df: 数据框
    - channel: 通道名称 ('488' 或 '561')
    - min_percentile: 最小百分位 (0-100)
    - max_percentile: 最大百分位 (0-100)
    
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
    
    # 计算百分位对应的强度阈值
    min_threshold = np.percentile(df[col_name].values, min_percentile)
    max_threshold = np.percentile(df[col_name].values, max_percentile)
    
    # 筛选百分位范围
    filtered_df = df[(df[col_name] >= min_threshold) & (df[col_name] <= max_threshold)].copy()
    
    return filtered_df


def main():
    parser = argparse.ArgumentParser(
        description='统计分析模块：红绿比值 vs T50 分析 + 多元回归分析',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单文件输入（自动创建输出文件夹）
  python statistics_analysis.py data.csv
  python statistics_analysis.py path/to/ratio_t50_raw_data.csv
  
  # 多文件合并分析（自动输出到merged_output）
  python statistics_analysis.py data1.csv data2.csv data3.csv
  
  # 指定输出目录
  python statistics_analysis.py data.csv --output output_folder
  
  # 只分析488通道亮度前20%的细胞
  python statistics_analysis.py data.csv --top 488:20
  
  # 分析多组：488前20% 和 561前30%
  python statistics_analysis.py data.csv --top 488:20 --top 561:30
  
  # 分析488通道亮度后20%的细胞
  python statistics_analysis.py data.csv --bottom 488:20
  
  # 分析488通道百分位20-80范围的细胞（会创建独立子目录）
  python statistics_analysis.py data.csv --percentile-range 488:20:80
  
  # 分析多个百分位范围
  python statistics_analysis.py data.csv --percentile-range 488:20:80 --percentile-range 561:10:90
  
  # 手动指定最大实验时间为600秒，过滤T90>600的数据
  python statistics_analysis.py data.csv --max-time 600
  
  # 只分析R²≥0.95的高质量拟合结果
  python statistics_analysis.py data.csv --min-rsq 0.95
  
  # 只分析Pearson变化值≥0.3的细胞（过滤变化过小的细胞）
  python statistics_analysis.py data.csv --min-pearson-change 0.3
        """
    )
    
    parser.add_argument('csv_files', type=str, nargs='+',
                       help='输入CSV文件路径（支持多个文件）')
    parser.add_argument('--output', type=str, default=None,
                       help='指定输出目录（默认根据输入文件名自动创建）')
    parser.add_argument('--top', type=str, action='append', metavar='CHANNEL:PERCENT',
                       help='按指定通道亮度筛选前xx%%细胞，格式为 "488:20" 或 "561:30"（可指定多组）')
    parser.add_argument('--bottom', type=str, action='append', metavar='CHANNEL:PERCENT',
                       help='按指定通道亮度筛选后xx%%细胞，格式为 "488:20" 或 "561:30"（可指定多组）')
    parser.add_argument('--percentile-range', type=str, action='append', metavar='CHANNEL:MIN_PCT:MAX_PCT',
                       help='按指定通道百分位范围筛选细胞，格式为 "488:20:80" 或 "561:10:90"（可指定多组）')
    parser.add_argument('--cooks', type=float, default=4.0, metavar='FACTOR',
                       help="Cook's Distance阈值系数，阈值=FACTOR/n（默认4.0，越小越严格）")
    parser.add_argument('--max-time', type=float, default=None, metavar='SECONDS',
                       help='手动指定最大实验时间（秒），用于过滤T90>max_time的数据，覆盖CSV中的max_time字段')
    parser.add_argument('--min-rsq', type=float, default=0.9, metavar='VALUE',
                       help='最小R²阈值，过滤拟合质量低的细胞（默认0.9）')
    parser.add_argument('--min-pearson-change', type=float, default=None, metavar='VALUE',
                       help='最小Pearson变化值阈值（A0 - A_inf），过滤变化过小的细胞（默认不过滤）')
    parser.add_argument('--percentile-bins', type=str, default=None, metavar='CHANNEL',
                       help='按指定通道百分位分组分析，可选 488/561/both（默认None）')
    parser.add_argument('--bin-step', type=float, default=10.0, metavar='PERCENT',
                       help='百分位分组步长（默认10，即每10%%一组）')
    parser.add_argument('--sliding-window', type=str, default=None, metavar='CHANNEL',
                       help='滑窗百分位分析，可选 488/561/both（默认None），生成除distribution外的趋势图')
    parser.add_argument('--window-size', type=float, default=20.0, metavar='PERCENT',
                       help='滑窗大小（默认20%%），配合 --sliding-window 使用')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Statistics Analysis Module")
    print("=" * 60)
    
    # 检查输入文件是否存在
    for csv_file in args.csv_files:
        csv_path = Path(csv_file)
        if not csv_path.exists():
            print(f"\nError: 文件不存在: {csv_file}")
            return
    
    # 决定输出目录
    if args.output:
        output_dir = Path(args.output)
    else:
        if len(args.csv_files) == 1:
            # 单文件：根据文件名创建输出文件夹
            csv_path = Path(args.csv_files[0])
            # 去掉文件后缀，作为文件夹名
            output_dir = csv_path.parent / f"{csv_path.stem}_stats"
        else:
            # 多文件：使用merged_output
            output_dir = Path('merged_output')
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n输出目录: {output_dir}")
    
    # 加载数据（支持多文件合并）
    try:
        if len(args.csv_files) == 1:
            print(f"\n加载数据: {args.csv_files[0]}")
            df = pd.read_csv(args.csv_files[0])
        else:
            print(f"\n加载并合并 {len(args.csv_files)} 个文件:")
            dfs = []
            for csv_file in args.csv_files:
                print(f"  - {csv_file}")
                df_temp = pd.read_csv(csv_file)
                dfs.append(df_temp)
            df = pd.concat(dfs, ignore_index=True)
            print(f"\n合并后总行数: {len(df)}")
    except Exception as e:
        print(f"\nError: 无法加载数据 - {e}")
        return
    
    # 创建分析器
    analyzer = StatisticsAnalyzer(output_dir, cooks_factor=args.cooks, max_time_override=args.max_time, min_r_squared=args.min_rsq, min_pearson_change=args.min_pearson_change)
    
    # 应用过滤（T90、R²和Pearson变化值）
    print(f"\n应用数据过滤...")
    df = analyzer._apply_filters(df)
    
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
                filtered_analyzer = StatisticsAnalyzer(output_dir, suffix=suffix, cooks_factor=args.cooks, 
                                                      max_time_override=args.max_time, min_r_squared=args.min_rsq,
                                                      min_pearson_change=args.min_pearson_change)
                filtered_analyzer.run_analysis(filtered_df)
                
            except ValueError as e:
                print(f"  错误: {e}")
                continue
    
    # 如果指定了 --bottom 参数，执行筛选分析
    if args.bottom:
        for bottom_spec in args.bottom:
            # 解析格式 "CHANNEL:PERCENT"
            try:
                parts = bottom_spec.split(':')
                if len(parts) != 2:
                    raise ValueError(f"格式错误: {bottom_spec}")
                channel, percent_str = parts
                percent = float(percent_str)
                if percent <= 0 or percent > 100:
                    raise ValueError(f"百分比必须在0-100之间: {percent}")
            except Exception as e:
                print(f"\n警告: 无法解析 --bottom 参数 '{bottom_spec}': {e}")
                print("格式应为 'CHANNEL:PERCENT'，例如 '488:20' 或 '561:30'")
                continue
            
            print(f"\n{'=' * 60}")
            print(f"Filtering: {channel} channel, bottom {percent}%")
            print("=" * 60)
            
            # 筛选数据
            try:
                filtered_df = filter_bottom_percent(df, channel, percent)
                print(f"  Filtered cells: {len(filtered_df)} / {len(df)} ({len(filtered_df)/len(df)*100:.1f}%)")
                
                if len(filtered_df) < 5:
                    print(f"  警告: 筛选后数据点太少 (n={len(filtered_df)})，跳过此分析")
                    continue
                
                # 创建带后缀的分析器
                suffix = f"_{channel}_bottom{int(percent)}"
                filtered_analyzer = StatisticsAnalyzer(output_dir, suffix=suffix, cooks_factor=args.cooks, 
                                                      max_time_override=args.max_time, min_r_squared=args.min_rsq,
                                                      min_pearson_change=args.min_pearson_change)
                filtered_analyzer.run_analysis(filtered_df)
                
            except ValueError as e:
                print(f"  错误: {e}")
                continue
    
    # 如果指定了 --percentile-range 参数，执行筛选分析
    if args.percentile_range:
        for range_spec in args.percentile_range:
            # 解析格式 "CHANNEL:MIN_PCT:MAX_PCT"
            try:
                parts = range_spec.split(':')
                if len(parts) != 3:
                    raise ValueError(f"格式错误: {range_spec}")
                channel, min_str, max_str = parts
                min_percentile = float(min_str)
                max_percentile = float(max_str)
                if min_percentile < 0 or max_percentile > 100 or min_percentile >= max_percentile:
                    raise ValueError(f"百分位必须在0-100之间且最小值<最大值: {min_percentile}, {max_percentile}")
            except Exception as e:
                print(f"\n警告: 无法解析 --percentile-range 参数 '{range_spec}': {e}")
                print("格式应为 'CHANNEL:MIN_PCT:MAX_PCT'，例如 '488:20:80' 或 '561:10:90'")
                continue
            
            print(f"\n{'=' * 60}")
            print(f"Filtering: {channel} channel, percentile range [{min_percentile}, {max_percentile}]")
            print("=" * 60)
            
            # 筛选数据
            try:
                filtered_df = filter_percentile_range(df, channel, min_percentile, max_percentile)
                print(f"  Filtered cells: {len(filtered_df)} / {len(df)} ({len(filtered_df)/len(df)*100:.1f}%)")
                
                if len(filtered_df) < 5:
                    print(f"  警告: 筛选后数据点太少 (n={len(filtered_df)})，跳过此分析")
                    continue
                
                # 为percentile-range创建独立的子目录
                range_output_dir = output_dir / f"percentile_range_{channel}_{int(min_percentile)}_{int(max_percentile)}"
                range_output_dir.mkdir(parents=True, exist_ok=True)
                print(f"  输出目录: {range_output_dir}")
                
                # 创建分析器（不使用suffix，直接用子目录）
                filtered_analyzer = StatisticsAnalyzer(range_output_dir, suffix="", cooks_factor=args.cooks, 
                                                      max_time_override=args.max_time, min_r_squared=args.min_rsq,
                                                      min_pearson_change=args.min_pearson_change)
                filtered_analyzer.run_analysis(filtered_df)
                
            except ValueError as e:
                print(f"  错误: {e}")
                continue
    
    # 如果指定了 --percentile-bins 参数，执行百分位分组分析
    if args.percentile_bins:
        channels = ['488', '561'] if args.percentile_bins.lower() == 'both' else [args.percentile_bins]
        for channel in channels:
            try:
                bin_analyzer = PercentileBinAnalyzer(
                    output_dir, channel=channel, bin_step=args.bin_step,
                    cooks_factor=args.cooks, max_time_override=args.max_time,
                    min_r_squared=args.min_rsq, min_pearson_change=args.min_pearson_change
                )
                bin_analyzer.run_binned_analysis(df)
            except ValueError as e:
                print(f"\n错误: {e}")
                continue
    
    # 如果指定了 --sliding-window 参数，执行滑窗百分位分析
    if args.sliding_window:
        channels = ['488', '561'] if args.sliding_window.lower() == 'both' else [args.sliding_window]
        for channel in channels:
            try:
                sw_analyzer = PercentileBinAnalyzer(
                    output_dir, channel=channel, bin_step=args.bin_step,
                    cooks_factor=args.cooks, max_time_override=args.max_time,
                    min_r_squared=args.min_rsq, window_size=args.window_size,
                    min_pearson_change=args.min_pearson_change
                )
                sw_analyzer.run_sliding_window_analysis(df)
            except ValueError as e:
                print(f"\n错误: {e}")
                continue
    
    # 如果没有筛选参数，执行正常分析
    if not args.top and not args.bottom and not args.percentile_range and not args.percentile_bins and not args.sliding_window:
        analyzer.run_analysis(df)
    
    print("\n" + "=" * 60)
    print("Statistics Analysis Completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
