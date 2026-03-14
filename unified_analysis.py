#!/usr/bin/env python3
"""
统一分析模块：整合动力学可视化、简单回归、多元回归分析

合并了以下脚本的功能：
- plot_kinetics_summary.py: Pearson相关系数动力学汇总可视化
- simple_linear_regression.py: 单通道强度 vs k_app 简单线性回归
- statistics_analysis.py: 统计分析 + 多元回归

用法:
=====

基本用法（默认执行全部分析）:
    python unified_analysis.py data.csv
    python unified_analysis.py *.csv -o output_dir

选择分析类型:
    python unified_analysis.py data.csv --analysis stats          # 仅统计分析
    python unified_analysis.py data.csv --analysis kinetics       # 仅动力学可视化
    python unified_analysis.py data.csv --analysis regression     # 仅简单回归
    python unified_analysis.py data.csv --analysis all            # 全部（默认）

百分位分组分析:
    python unified_analysis.py data.csv --percentile-bins 488
    python unified_analysis.py data.csv --percentile-bins both --bin-step 10

滑窗分析:
    python unified_analysis.py data.csv --sliding-window 488 --window-size 20

数据过滤:
    python unified_analysis.py data.csv --min-rsq 0.95 --max-time 600
"""

import argparse
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.optimize import curve_fit
import statsmodels.api as sm
from typing import List, Tuple, Optional, Dict, Any

# ============================================================
# 全局绘图配置
# ============================================================
def setup_plot_style():
    """设置全局绘图样式"""
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

setup_plot_style()


# ============================================================
# 公共常量和配置
# ============================================================
CHANNEL_MAP = {
    '488': 'green',
    '561': 'red',
    'green': 'green',
    'red': 'red'
}

CHANNEL_CONFIG = {
    'red': {
        'column': 'red',
        'label': 'Red Intensity (561nm)',
        'log_label': r'$\log_{10}$(Red Intensity)',
        'color': '#D96459',
        'edge_color': '#B84A40'
    },
    'green': {
        'column': 'green',
        'label': 'Green Intensity (488nm)',
        'log_label': r'$\log_{10}$(Green Intensity)',
        'color': '#45ADA8',
        'edge_color': '#358985'
    },
    'ratio': {
        'column': 'ratio',
        'label': 'Red/Green Ratio',
        'log_label': r'$\log_{10}$(Red/Green)',
        'color': '#8E7CC3',
        'edge_color': '#6A5ACD'
    }
}


# ============================================================
# 公共数据加载类
# ============================================================
class DataLoader:
    """统一的数据加载和合并功能"""
    
    @staticmethod
    def classify_files(files: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """
        从输入文件中分类拟合结果、原始数据、荧光强度数据
        
        Returns:
        - fitting_files: 拟合结果文件列表 (*reaction_fitting_results*)
        - raw_files: 原始数据文件列表 (*correlation_analysis_results*)
        - intensity_files: 荧光强度数据文件列表 (*ratio_t50_raw_data*)
        """
        fitting_files = []
        raw_files = []
        intensity_files = []
        
        for f in files:
            p = Path(f)
            # 跳过目录，只处理文件
            if p.is_dir():
                continue
            # 只处理 CSV 文件
            if p.suffix.lower() != '.csv':
                continue
            
            fname = p.name.lower()
            if 'reaction_fitting_results' in fname:
                fitting_files.append(f)
            elif 'correlation_analysis_results' in fname:
                raw_files.append(f)
            elif 'ratio_t50_raw_data' in fname:
                intensity_files.append(f)
        
        return fitting_files, raw_files, intensity_files
    
    @staticmethod
    def load_fitting_data(files: List[str], min_r_squared: float = 0.9,
                          max_time: Optional[float] = None) -> pd.DataFrame:
        """
        加载并合并拟合结果数据
        
        Parameters:
        - files: reaction_fitting_results.csv 文件列表
        - min_r_squared: 最小 R^2 阈值
        - max_time: 最大时间阈值，过滤 T90 > max_time 的细胞
        """
        dfs = []
        for f in files:
            df = pd.read_csv(f)
            df['source_file'] = Path(f).parent.name
            dfs.append(df)
        
        merged = pd.concat(dfs, ignore_index=True)
        print(f"  加载了 {len(merged)} 个细胞的拟合数据（来自 {len(files)} 个文件）")
        
        # R^2 过滤
        if 'correlation_r_squared' in merged.columns:
            n_before = len(merged)
            merged = merged[merged['correlation_r_squared'] >= min_r_squared].copy()
            n_removed = n_before - len(merged)
            if n_removed > 0:
                print(f"  移除了 {n_removed} 个 R^2 < {min_r_squared} 的细胞")
        
        # T90 过滤
        if max_time is not None and 'correlation_t90' in merged.columns:
            n_before = len(merged)
            merged = merged[merged['correlation_t90'] <= max_time].copy()
            n_removed = n_before - len(merged)
            if n_removed > 0:
                print(f"  移除了 {n_removed} 个 T90 > {max_time}s 的细胞")
        
        # 过滤无效参数
        for col in ['correlation_A0', 'correlation_k', 'correlation_A_inf']:
            if col in merged.columns:
                merged = merged[~merged[col].isna()].copy()
        
        print(f"  清洗后剩余 {len(merged)} 个有效细胞")
        return merged
    
    @staticmethod
    def load_raw_data(files: List[str]) -> pd.DataFrame:
        """加载并合并原始相关系数数据"""
        dfs = []
        for f in files:
            df = pd.read_csv(f)
            df['source_file'] = Path(f).parent.name
            dfs.append(df)
        
        merged = pd.concat(dfs, ignore_index=True)
        print(f"  加载了 {len(merged)} 个原始观测点（来自 {len(files)} 个文件）")
        return merged
    
    @staticmethod
    def load_intensity_data(files: List[str]) -> pd.DataFrame:
        """加载并合并荧光强度数据（ratio_t50_raw_data）"""
        dfs = []
        for f in files:
            df = pd.read_csv(f)
            df['source_file'] = Path(f).parent.name
            dfs.append(df)
        
        merged = pd.concat(dfs, ignore_index=True)
        print(f"  加载了 {len(merged)} 个细胞的荧光强度数据（来自 {len(files)} 个文件）")
        return merged
    
    @staticmethod
    def merge_fitting_with_intensity(fitting_df: pd.DataFrame, 
                                      intensity_df: pd.DataFrame) -> pd.DataFrame:
        """将拟合结果与荧光强度数据合并"""
        fitting_df = fitting_df.copy()
        fitting_df['file_stem'] = fitting_df['file_path'].apply(lambda x: Path(x).stem)
        
        intensity_df = intensity_df.copy()
        if 'file_stem' not in intensity_df.columns and 'file_path' in intensity_df.columns:
            intensity_df['file_stem'] = intensity_df['file_path'].apply(lambda x: Path(x).stem)
        
        intensity_cols = ['file_stem', 'cell_id', 'green', 'red']
        if 'ratio' in intensity_df.columns:
            intensity_cols.append('ratio')
        intensity_subset = intensity_df[intensity_cols].drop_duplicates()
        
        merged = fitting_df.merge(intensity_subset, on=['file_stem', 'cell_id'], how='left')
        n_with_intensity = merged['green'].notna().sum()
        print(f"  合并后 {n_with_intensity}/{len(merged)} 个细胞有荧光强度数据")
        
        return merged


# ============================================================
# 公共数据清洗函数
# ============================================================
def apply_statistical_filters(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    应用统计分析的数据清洗流程（IQR方法）
    
    包括：
    1. 面积过滤（IQR方法）- 可选
    2. T50过滤（log10 + 2.5倍IQR）
    3. log10(红色强度)过滤（IQR方法）
    4. log10(绿色强度)过滤（IQR方法）
    """
    # 检查必需列
    required_cols = ['red', 'green', 't50']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        if '1/red' in df.columns and '1/green' in df.columns:
            df = df.copy()
            df['red'] = 1.0 / df['1/red']
            df['green'] = 1.0 / df['1/green']
        else:
            raise ValueError(f"缺少必需的列: {missing}")
    
    red_values = df['red'].values
    green_values = df['green'].values
    t50_values = df['t50'].values
    
    n_original = len(df)
    if verbose:
        print(f"  Data cleaning: starting with {n_original} cells")
    
    # 1. 面积过滤（可选）
    if 'n_pixels' in df.columns:
        area_values = df['n_pixels'].values
        area_q1, area_q3 = np.percentile(area_values, [25, 75])
        area_iqr = area_q3 - area_q1
        area_valid = (area_values >= area_q1 - 1.5 * area_iqr) & (area_values <= area_q3 + 1.5 * area_iqr)
        if verbose:
            print(f"    Area filter (IQR): removed {np.sum(~area_valid)} cells")
    else:
        area_valid = np.ones(len(df), dtype=bool)
        if verbose:
            print(f"    Area filter: skipped (no n_pixels column)")
    
    # 2. T50过滤
    t50_positive_mask = t50_values > 0
    t50_valid = np.zeros(len(t50_values), dtype=bool)
    if np.sum(t50_positive_mask) > 0:
        t50_log = np.log10(t50_values[t50_positive_mask])
        t50_q1, t50_q3 = np.percentile(t50_log, [25, 75])
        t50_iqr = t50_q3 - t50_q1
        t50_lower_log = t50_q1 - 2.5 * t50_iqr
        t50_upper_log = t50_q3 + 2.5 * t50_iqr
        t50_valid[t50_positive_mask] = (t50_log >= t50_lower_log) & (t50_log <= t50_upper_log)
        if verbose:
            print(f"    T50 filter (log10 + 2.5*IQR): removed {np.sum(t50_positive_mask) - np.sum(t50_valid)} cells")
    
    # 3. log10(红色强度)过滤
    red_positive_mask = red_values > 0
    red_log_valid = np.zeros(len(red_values), dtype=bool)
    if np.sum(red_positive_mask) > 0:
        red_log = np.log10(red_values[red_positive_mask])
        red_q1, red_q3 = np.percentile(red_log, [25, 75])
        red_iqr = red_q3 - red_q1
        red_log_valid[red_positive_mask] = (red_log >= red_q1 - 1.5 * red_iqr) & (red_log <= red_q3 + 1.5 * red_iqr)
        if verbose:
            print(f"    log10(Red) filter (IQR): removed {np.sum(red_positive_mask) - np.sum(red_log_valid)} cells")
    
    # 4. log10(绿色强度)过滤
    green_positive_mask = green_values > 0
    green_log_valid = np.zeros(len(green_values), dtype=bool)
    if np.sum(green_positive_mask) > 0:
        green_log = np.log10(green_values[green_positive_mask])
        green_q1, green_q3 = np.percentile(green_log, [25, 75])
        green_iqr = green_q3 - green_q1
        green_log_valid[green_positive_mask] = (green_log >= green_q1 - 1.5 * green_iqr) & (green_log <= green_q3 + 1.5 * green_iqr)
        if verbose:
            print(f"    log10(Green) filter (IQR): removed {np.sum(green_positive_mask) - np.sum(green_log_valid)} cells")
    
    # 组合所有过滤条件
    valid_mask = area_valid & t50_valid & red_log_valid & green_log_valid
    n_valid = np.sum(valid_mask)
    if verbose:
        print(f"    Valid cells after all filters: {n_valid} ({n_valid/n_original*100:.1f}%)")
    
    return df.iloc[valid_mask].copy()


# ============================================================
# 公共 Percentile 分组逻辑
# ============================================================
class PercentileGrouper:
    """百分位分组工具类"""
    
    @staticmethod
    def group_by_percentile(df: pd.DataFrame, channel: str, 
                            bin_step: float = 10.0) -> Tuple[List[Tuple], str]:
        """
        根据指定通道的荧光强度将细胞按 percentile 分组
        
        Returns:
        - List of (group_df, label, pct_min, pct_max)
        - col_name: 实际使用的列名
        """
        col_name = CHANNEL_MAP.get(channel.lower())
        if col_name is None:
            raise ValueError(f"不支持的通道: {channel}，请使用 488/561/green/red")
        
        if col_name not in df.columns:
            raise ValueError(f"数据中没有 {col_name} 列")
        
        valid_mask = ~df[col_name].isna()
        valid_df = df[valid_mask].copy()
        intensity_values = valid_df[col_name].values
        
        bin_edges = np.arange(0, 100 + bin_step, bin_step)
        n_bins = len(bin_edges) - 1
        percentile_thresholds = [np.percentile(intensity_values, p) for p in bin_edges]
        
        groups = []
        for i in range(n_bins):
            pct_min = int(bin_edges[i])
            pct_max = int(bin_edges[i + 1])
            threshold_low = percentile_thresholds[i]
            threshold_high = percentile_thresholds[i + 1]
            
            if i == n_bins - 1:
                mask = (intensity_values >= threshold_low) & (intensity_values <= threshold_high)
            else:
                mask = (intensity_values >= threshold_low) & (intensity_values < threshold_high)
            
            group_df = valid_df[mask].copy()
            if len(group_df) >= 3:
                label = f"{pct_min}-{pct_max}%"
                groups.append((group_df, label, pct_min, pct_max))
        
        return groups, col_name
    
    @staticmethod
    def sliding_window(df: pd.DataFrame, channel: str, window_size: float = 20.0,
                       step: float = 10.0) -> Tuple[List[Tuple], str]:
        """
        滑窗分组
        
        Returns:
        - List of (group_df, label, center_pct)
        - col_name: 实际使用的列名
        """
        col_name = CHANNEL_MAP.get(channel.lower())
        if col_name is None:
            raise ValueError(f"不支持的通道: {channel}")
        
        if col_name not in df.columns:
            raise ValueError(f"数据中没有 {col_name} 列")
        
        intensity_values = df[col_name].values
        half_window = window_size / 2
        center_points = np.arange(half_window, 100 - half_window + step, step)
        center_points = center_points[center_points <= 100 - half_window]
        
        groups = []
        for i, center in enumerate(center_points):
            pct_min = center - half_window
            pct_max = center + half_window
            threshold_low = np.percentile(intensity_values, pct_min)
            threshold_high = np.percentile(intensity_values, pct_max)
            
            if i == len(center_points) - 1:
                mask = (intensity_values >= threshold_low) & (intensity_values <= threshold_high)
            else:
                mask = (intensity_values >= threshold_low) & (intensity_values < threshold_high)
            
            group_df = df[mask].copy()
            if len(group_df) >= 3:
                label = f"{pct_min:.0f}-{pct_max:.0f}%"
                groups.append((group_df, label, center))
        
        return groups, col_name


# ============================================================
# 公共绘图辅助函数
# ============================================================
def get_channel_color(channel: str) -> str:
    """根据通道返回对应颜色"""
    if channel in ['488', 'green']:
        return '#45ADA8'
    elif channel in ['561', 'red']:
        return '#D96459'
    return 'steelblue'


def filter_by_cooks_distance(x: np.ndarray, y: np.ndarray, 
                              cooks_factor: float = 4.0) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    使用Cook's Distance过滤异常值
    
    Returns:
        x_clean, y_clean, n_removed
    """
    n = len(x)
    if n < 5:
        return x, y, 0
    
    slope, intercept, r, p, se = stats.linregress(x, y)
    y_pred = slope * x + intercept
    residuals = y - y_pred
    
    x_mean = np.mean(x)
    h = 1/n + (x - x_mean)**2 / np.sum((x - x_mean)**2)
    mse = np.sum(residuals**2) / (n - 2)
    cooks_d = (residuals**2 / (2 * mse)) * (h / (1 - h)**2)
    
    threshold = cooks_factor / n
    valid_mask = cooks_d <= threshold
    n_removed = np.sum(~valid_mask)
    
    return x[valid_mask], y[valid_mask], n_removed


# ============================================================
# 用于计算 delta Pearson 的辅助函数
# ============================================================
def calculate_delta_pearson_from_raw(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    从原始数据计算每个细胞的 delta Pearson
    delta = 最后时间点的 pearson_corr - 第一个时间点的 pearson_corr
    """
    results = []
    
    if 'file_path' in raw_df.columns:
        group_cols = ['file_path', 'cell_id']
    else:
        group_cols = ['cell_id']
    
    for group_key, group_data in raw_df.groupby(group_cols):
        sorted_data = group_data.sort_values('time_point')
        start_pearson = sorted_data['pearson_corr'].iloc[0]
        end_pearson = sorted_data['pearson_corr'].iloc[-1]
        delta = end_pearson - start_pearson
        
        if isinstance(group_key, tuple):
            file_path, cell_id = group_key
        else:
            file_path = None
            cell_id = group_key
        
        results.append({
            'file_path': file_path,
            'cell_id': cell_id,
            'start_pearson': start_pearson,
            'end_pearson': end_pearson,
            'delta_pearson': delta
        })
    
    return pd.DataFrame(results)


# ============================================================
# KineticsAnalyzer: 动力学曲线可视化
# ============================================================
class KineticsAnalyzer:
    """
    Pearson 相关系数动力学汇总可视化
    
    功能：
    - 原始值 + 归一化动力学汇总图
    - 分 percentile 的动力学曲线
    - Pearson 变化值直方图
    - 单细胞拟合曲线
    """
    
    def __init__(self, output_dir: str, max_time: float = 300.0, time_step: float = 1.0,
                 use_delay: bool = False, max_scatter_points: Optional[int] = None,
                 prefix: str = 'kinetics_'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.png_dir = self.output_dir / 'png'
        self.pdf_dir = self.output_dir / 'pdf'
        self.png_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.max_time = max_time
        self.time_step = time_step
        self.use_delay = use_delay
        self.max_scatter_points = max_scatter_points
        self.prefix = prefix
    
    def reconstruct_curves_on_grid(self, fitting_df: pd.DataFrame) -> np.ndarray:
        """根据拟合参数在标准时间网格上重建所有细胞的曲线"""
        time_grid = np.arange(0, self.max_time + self.time_step, self.time_step)
        n_cells = len(fitting_df)
        n_times = len(time_grid)
        matrix = np.zeros((n_cells, n_times))
        
        for i, (_, row) in enumerate(fitting_df.iterrows()):
            A0 = row['correlation_A0']
            k = row['correlation_k']
            A_inf = row['correlation_A_inf']
            
            if self.use_delay and 'correlation_delay' in row and not pd.isna(row['correlation_delay']):
                delay = row['correlation_delay']
                shifted_t = time_grid - delay
                y = np.where(shifted_t < 0, A0, A_inf + (A0 - A_inf) * np.exp(-k * shifted_t))
            else:
                y = A_inf + (A0 - A_inf) * np.exp(-k * time_grid)
            
            matrix[i, :] = y
        
        return matrix
    
    @staticmethod
    def normalize_matrix_per_cell(matrix: np.ndarray) -> np.ndarray:
        """对每个细胞的曲线进行归一化（按该细胞的最小值和最大值）"""
        n_cells = matrix.shape[0]
        normalized = np.zeros_like(matrix)
        
        for i in range(n_cells):
            row = matrix[i, :]
            row_min = np.nanmin(row)
            row_max = np.nanmax(row)
            if row_max - row_min > 1e-10:
                normalized[i, :] = (row - row_min) / (row_max - row_min)
            else:
                normalized[i, :] = 0.5
        
        return normalized
    
    @staticmethod
    def calculate_statistics(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算每个时间点的统计量（中位数、Q25、Q75）"""
        median = np.nanmedian(matrix, axis=0)
        q25 = np.nanpercentile(matrix, 25, axis=0)
        q75 = np.nanpercentile(matrix, 75, axis=0)
        return median, q25, q75
    
    def plot_kinetics_summary(self, fitting_df: pd.DataFrame, 
                               raw_df: Optional[pd.DataFrame] = None):
        """生成 Pearson 相关系数动力学汇总图（原始值 + 归一化）"""
        time_grid = np.arange(0, self.max_time + self.time_step, self.time_step)
        
        print(f"\n重建 {len(fitting_df)} 个细胞的曲线...")
        matrix = self.reconstruct_curves_on_grid(fitting_df)
        normalized_matrix = self.normalize_matrix_per_cell(matrix)
        
        median, q25, q75 = self.calculate_statistics(matrix)
        norm_median, norm_q25, norm_q75 = self.calculate_statistics(normalized_matrix)
        
        n_cells = len(fitting_df)
        
        # === 图1: 原始值 ===
        fig1, ax1 = plt.subplots(figsize=(4, 4))
        
        if raw_df is not None and len(raw_df) > 0:
            scatter_data = raw_df[['time_point', 'pearson_corr']].dropna()
            if self.max_scatter_points and len(scatter_data) > self.max_scatter_points:
                scatter_data = scatter_data.sample(n=self.max_scatter_points, random_state=42)
            ax1.scatter(scatter_data['time_point'], scatter_data['pearson_corr'],
                        c='lightgray', s=2.5, alpha=0.01, edgecolors='none', zorder=1)
        
        ax1.fill_between(time_grid, q25, q75, color='steelblue', alpha=0.3, zorder=2)
        ax1.plot(time_grid, median, color='steelblue', linewidth=2.5, zorder=3)
        
        ax1.set_xlabel('Time (s)', fontsize=12)
        ax1.set_ylabel('Pearson Correlation Coefficient', fontsize=12)
        ax1.set_title(f'Pearson Correlation Kinetics Summary\n(n = {n_cells} cells)', fontsize=14)
        ax1.set_xlim(0, self.max_time)
        ax1.set_ylim(-0.1, 1.1)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
        ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        fig1.savefig(self.png_dir / f'{self.prefix}pearson_kinetics_summary.png', dpi=300, bbox_inches='tight')
        fig1.savefig(self.pdf_dir / f'{self.prefix}pearson_kinetics_summary.pdf', dpi=300, bbox_inches='tight')
        plt.close(fig1)
        
        # === 图2: 归一化 ===
        fig2, ax2 = plt.subplots(figsize=(4, 4))
        ax2.fill_between(time_grid, norm_q25, norm_q75, color='steelblue', alpha=0.3)
        ax2.plot(time_grid, norm_median, color='steelblue', linewidth=2.5)
        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('Normalized Pearson Correlation', fontsize=12)
        ax2.set_title(f'Normalized Pearson Correlation Kinetics\n(n = {n_cells} cells)', fontsize=14)
        ax2.set_xlim(0, self.max_time)
        ax2.set_ylim(-0.05, 1.05)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        fig2.savefig(self.png_dir / f'{self.prefix}pearson_kinetics_normalized.png', dpi=300, bbox_inches='tight')
        fig2.savefig(self.pdf_dir / f'{self.prefix}pearson_kinetics_normalized.pdf', dpi=300, bbox_inches='tight')
        plt.close(fig2)
        
        # 导出统计数据
        stats_df = pd.DataFrame({
            'time': time_grid, 'median': median, 'q25': q25, 'q75': q75,
            'norm_median': norm_median, 'norm_q25': norm_q25, 'norm_q75': norm_q75
        })
        stats_df.to_csv(self.output_dir / f'{self.prefix}pearson_kinetics_statistics.csv', index=False)
        print(f"  保存: {self.png_dir / f'{self.prefix}pearson_kinetics_summary.png'}")
    
    def plot_pearson_change_histogram(self, raw_df: pd.DataFrame, 
                                        fitting_df: Optional[pd.DataFrame] = None):
        """绘制 Pearson 变化值的直方图"""
        if raw_df is None or len(raw_df) == 0:
            return
        
        delta_df = calculate_delta_pearson_from_raw(raw_df)
        
        if fitting_df is not None and len(fitting_df) > 0:
            if 'file_path' in fitting_df.columns and 'file_path' in delta_df.columns:
                fitting_df = fitting_df.copy()
                fitting_df['file_stem'] = fitting_df['file_path'].apply(lambda x: Path(x).stem)
                delta_df['file_stem'] = delta_df['file_path'].apply(lambda x: Path(x).stem if x else '')
                valid_keys = set(zip(fitting_df['file_stem'], fitting_df['cell_id']))
                delta_df = delta_df[delta_df.apply(lambda r: (r['file_stem'], r['cell_id']) in valid_keys, axis=1)]
        
        delta_pearson = delta_df['delta_pearson'].dropna()
        if len(delta_pearson) < 5:
            return
        
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.hist(delta_pearson, bins=20, color='steelblue', alpha=0.7, edgecolor='white')
        ax.axvline(x=np.median(delta_pearson), color='#D55E00', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(delta_pearson):.3f}')
        ax.axvline(x=np.mean(delta_pearson), color='#E69F00', linestyle=':', linewidth=2,
                   label=f'Mean: {np.mean(delta_pearson):.3f}')
        
        ax.set_xlabel(r'$\Delta$Pearson (end - start)', fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        ax.set_title(f'Pearson Correlation Change Distribution\n(n = {len(delta_pearson)} cells)', fontsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper right', fontsize=10)
        plt.tight_layout()
        
        fig.savefig(self.png_dir / f'{self.prefix}pearson_change_histogram.png', dpi=300, bbox_inches='tight')
        fig.savefig(self.pdf_dir / f'{self.prefix}pearson_change_histogram.pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  保存: {self.png_dir / f'{self.prefix}pearson_change_histogram.png'}")
    
    def plot_kinetics_by_percentile(self, fitting_df: pd.DataFrame, channel: str,
                                     bin_step: float = 10.0, raw_df: Optional[pd.DataFrame] = None):
        """生成分 percentile 的动力学图"""
        groups, col_name = PercentileGrouper.group_by_percentile(fitting_df, channel, bin_step)
        n_groups = len(groups)
        
        if n_groups < 2:
            print("  警告: 有效分组数量不足")
            return
        
        time_grid = np.arange(0, self.max_time + self.time_step, self.time_step)
        n_cols = int(np.ceil(np.sqrt(n_groups)))
        n_rows = int(np.ceil(n_groups / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), squeeze=False)
        color = 'steelblue'
        
        for idx, (group_df, label, pct_min, pct_max) in enumerate(groups):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]
            n_cells = len(group_df)
            
            matrix = self.reconstruct_curves_on_grid(group_df)
            median, q25, q75 = self.calculate_statistics(matrix)
            
            ax.fill_between(time_grid, q25, q75, color=color, alpha=0.3, zorder=2)
            ax.plot(time_grid, median, color=color, linewidth=1.5, zorder=3)
            
            ax.set_xlabel('Time (s)', fontsize=9)
            ax.set_ylabel('Pearson r', fontsize=9)
            ax.set_title(f'{label} (n={n_cells})', fontsize=10, fontweight='bold')
            ax.set_xlim(0, self.max_time)
            ax.set_ylim(-0.1, 1.1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        for idx in range(n_groups, n_rows * n_cols):
            axes[idx // n_cols, idx % n_cols].set_visible(False)
        
        channel_display = '488nm (Green)' if channel.lower() in ['488', 'green'] else '561nm (Red)'
        fig.suptitle(f'Pearson Correlation Kinetics by {channel_display} Intensity Percentile', fontsize=12, y=1.02)
        plt.tight_layout()
        
        fig.savefig(self.png_dir / f'{self.prefix}pearson_kinetics_by_{channel}_percentile.png', dpi=300, bbox_inches='tight')
        fig.savefig(self.pdf_dir / f'{self.prefix}pearson_kinetics_by_{channel}_percentile.pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  保存: {self.png_dir / f'{self.prefix}pearson_kinetics_by_{channel}_percentile.png'}")
    
    def plot_kinetics_by_percentile_normalized(self, fitting_df: pd.DataFrame, channel: str,
                                                 bin_step: float = 10.0):
        """生成分 percentile 的归一化动力学图"""
        groups, col_name = PercentileGrouper.group_by_percentile(fitting_df, channel, bin_step)
        n_groups = len(groups)
        
        if n_groups < 2:
            return
        
        time_grid = np.arange(0, self.max_time + self.time_step, self.time_step)
        n_cols = int(np.ceil(np.sqrt(n_groups)))
        n_rows = int(np.ceil(n_groups / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), squeeze=False)
        color = 'steelblue'
        
        for idx, (group_df, label, pct_min, pct_max) in enumerate(groups):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]
            n_cells = len(group_df)
            
            matrix = self.reconstruct_curves_on_grid(group_df)
            normalized_matrix = self.normalize_matrix_per_cell(matrix)
            median, q25, q75 = self.calculate_statistics(normalized_matrix)
            
            ax.fill_between(time_grid, q25, q75, color=color, alpha=0.3)
            ax.plot(time_grid, median, color=color, linewidth=1.5)
            
            ax.set_xlabel('Time (s)', fontsize=9)
            ax.set_ylabel('Normalized r', fontsize=9)
            ax.set_title(f'{label} (n={n_cells})', fontsize=10, fontweight='bold')
            ax.set_xlim(0, self.max_time)
            ax.set_ylim(-0.05, 1.05)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        for idx in range(n_groups, n_rows * n_cols):
            axes[idx // n_cols, idx % n_cols].set_visible(False)
        
        channel_display = '488nm (Green)' if channel.lower() in ['488', 'green'] else '561nm (Red)'
        fig.suptitle(f'Normalized Pearson Kinetics by {channel_display} Intensity Percentile', fontsize=12, y=1.02)
        plt.tight_layout()
        
        fig.savefig(self.png_dir / f'{self.prefix}pearson_kinetics_normalized_by_{channel}_percentile.png', dpi=300, bbox_inches='tight')
        fig.savefig(self.pdf_dir / f'{self.prefix}pearson_kinetics_normalized_by_{channel}_percentile.pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  保存: {self.png_dir / f'{self.prefix}pearson_kinetics_normalized_by_{channel}_percentile.png'}")
    
    def plot_pearson_change_histogram_by_percentile(self, fitting_df: pd.DataFrame, 
                                                      raw_df: pd.DataFrame,
                                                      channel: str, bin_step: float = 10.0):
        """绘制分 percentile 的 Pearson 变化值直方图"""
        if raw_df is None or len(raw_df) == 0:
            return
        
        fitting_df = fitting_df.copy()
        
        # 计算 delta_pearson
        if 'delta_pearson' not in fitting_df.columns:
            delta_df = calculate_delta_pearson_from_raw(raw_df)
            if 'file_path' in delta_df.columns:
                delta_df['file_stem'] = delta_df['file_path'].apply(lambda x: Path(x).stem if x else '')
            
            if 'file_stem' not in fitting_df.columns and 'file_path' in fitting_df.columns:
                fitting_df['file_stem'] = fitting_df['file_path'].apply(lambda x: Path(x).stem)
            
            merge_keys = ['file_stem', 'cell_id'] if 'file_stem' in fitting_df.columns else ['cell_id']
            fitting_df = fitting_df.merge(
                delta_df[merge_keys + ['delta_pearson']].drop_duplicates(subset=merge_keys),
                on=merge_keys, how='left'
            )
        
        if 'delta_pearson' not in fitting_df.columns or fitting_df['delta_pearson'].notna().sum() == 0:
            return
        
        groups, col_name = PercentileGrouper.group_by_percentile(fitting_df, channel, bin_step)
        n_groups = len(groups)
        
        if n_groups < 2:
            return
        
        n_cols = int(np.ceil(np.sqrt(n_groups)))
        n_rows = int(np.ceil(n_groups / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), squeeze=False)
        color = 'steelblue'
        
        # 统一 x 轴范围
        all_delta = []
        for group_df, _, _, _ in groups:
            all_delta.extend(group_df['delta_pearson'].dropna().tolist())
        if len(all_delta) == 0:
            return
        x_min, x_max = min(all_delta) - 0.05, max(all_delta) + 0.05
        bins = np.linspace(x_min, x_max, 21)
        
        for idx, (group_df, label, pct_min, pct_max) in enumerate(groups):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]
            
            delta_pearson = group_df['delta_pearson'].dropna()
            n_cells = len(delta_pearson)
            
            if n_cells < 3:
                ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
            else:
                ax.hist(delta_pearson, bins=bins, color=color, alpha=0.7, edgecolor='white')
                ax.axvline(x=np.median(delta_pearson), color='#D55E00', linestyle='--', linewidth=1.5,
                           label=f'Median: {np.median(delta_pearson):.3f}')
                ax.legend(loc='upper right', fontsize=8)
            
            ax.set_xlim(x_min, x_max)
            ax.set_xlabel(r'$\Delta$Pearson', fontsize=9)
            ax.set_ylabel('Count', fontsize=9)
            ax.set_title(f'{label} (n={n_cells})', fontsize=10, fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        for idx in range(n_groups, n_rows * n_cols):
            axes[idx // n_cols, idx % n_cols].set_visible(False)
        
        channel_display = '488nm (Green)' if channel.lower() in ['488', 'green'] else '561nm (Red)'
        fig.suptitle(f'Pearson Change Distribution by {channel_display} Intensity Percentile', fontsize=12, y=1.02)
        plt.tight_layout()
        
        fig.savefig(self.png_dir / f'{self.prefix}pearson_change_histogram_by_{channel}_percentile.png', dpi=300, bbox_inches='tight')
        fig.savefig(self.pdf_dir / f'{self.prefix}pearson_change_histogram_by_{channel}_percentile.pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  保存: {self.png_dir / f'{self.prefix}pearson_change_histogram_by_{channel}_percentile.png'}")
    
    def plot_individual_cells(self, fitting_df: pd.DataFrame, raw_df: pd.DataFrame,
                               max_cells: int = 100, cells_per_page: int = 25):
        """
        为每个细胞生成独立的拟合曲线图
        
        Parameters:
        - fitting_df: 拟合结果数据
        - raw_df: 原始时间序列数据
        - max_cells: 最大绘制细胞数
        - cells_per_page: 每页细胞数
        """
        print(f"\n  生成单细胞拟合曲线图...")
        
        # 创建输出目录
        cells_png_dir = self.png_dir / 'individual_cells'
        cells_pdf_dir = self.pdf_dir / 'individual_cells'
        cells_png_dir.mkdir(parents=True, exist_ok=True)
        cells_pdf_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取有效细胞列表
        valid_cells = fitting_df[fitting_df['correlation_r_squared'] >= 0.9].copy()
        if len(valid_cells) > max_cells:
            valid_cells = valid_cells.sample(n=max_cells, random_state=42)
        
        n_cells = len(valid_cells)
        if n_cells == 0:
            print("    没有有效细胞可绘制")
            return
        
        print(f"    绘制 {n_cells} 个细胞")
        
        # 分页绘制
        n_pages = int(np.ceil(n_cells / cells_per_page))
        n_cols = 5
        n_rows = int(np.ceil(cells_per_page / n_cols))
        
        cell_list = valid_cells.to_dict('records')
        
        for page_idx in range(n_pages):
            start_idx = page_idx * cells_per_page
            end_idx = min(start_idx + cells_per_page, n_cells)
            page_cells = cell_list[start_idx:end_idx]
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 2.5))
            axes = axes.flatten()
            
            for i, cell_info in enumerate(page_cells):
                ax = axes[i]
                
                # 获取该细胞的原始数据
                file_path = cell_info.get('file_path', '')
                cell_id = cell_info['cell_id']
                
                if 'file_path' in raw_df.columns:
                    cell_raw = raw_df[(raw_df['file_path'] == file_path) & 
                                      (raw_df['cell_id'] == cell_id)].sort_values('time_point')
                else:
                    cell_raw = raw_df[raw_df['cell_id'] == cell_id].sort_values('time_point')
                
                if len(cell_raw) == 0:
                    ax.set_visible(False)
                    continue
                
                time_points = cell_raw['time_point'].values
                pearson_values = cell_raw['pearson_corr'].values
                
                # 绘制原始数据点+连线
                ax.plot(time_points, pearson_values, 'o-', color='steelblue', 
                        markersize=3, linewidth=1, alpha=0.7, label='Data')
                
                # 绘制拟合曲线
                A0 = cell_info['correlation_A0']
                k = cell_info['correlation_k']
                A_inf = cell_info['correlation_A_inf']
                t50 = cell_info['correlation_t50']
                t90 = cell_info['correlation_t90']
                r_sq = cell_info['correlation_r_squared']
                
                if k > 0 and not np.isnan(k):
                    t_fit = np.linspace(time_points.min(), time_points.max(), 100)
                    y_fit = A_inf + (A0 - A_inf) * np.exp(-k * t_fit)
                    ax.plot(t_fit, y_fit, '--', color='red', linewidth=1.5, label='Fit')
                
                # T50 标注线
                if not np.isnan(t50) and t50 <= time_points.max():
                    ax.axvline(x=t50, color='orange', linestyle=':', linewidth=1.5)
                    y_t50 = A_inf + (A0 - A_inf) * 0.5
                    ax.axhline(y=y_t50, color='orange', linestyle=':', linewidth=1, alpha=0.5)
                    ax.text(t50, ax.get_ylim()[1], f'T50:{t50:.1f}', fontsize=6, 
                            color='orange', ha='center', va='bottom')
                
                # T90 标注线
                if not np.isnan(t90) and t90 <= time_points.max():
                    ax.axvline(x=t90, color='purple', linestyle=':', linewidth=1.5)
                    y_t90 = A_inf + (A0 - A_inf) * 0.1
                    ax.axhline(y=y_t90, color='purple', linestyle=':', linewidth=1, alpha=0.5)
                    ax.text(t90, ax.get_ylim()[0], f'T90:{t90:.1f}', fontsize=6,
                            color='purple', ha='center', va='top')
                
                ax.set_title(f'Cell {cell_id}\nR^2={r_sq:.3f}', fontsize=8)
                ax.set_xlabel('Time (s)', fontsize=7)
                ax.set_ylabel('Pearson r', fontsize=7)
                ax.tick_params(labelsize=6)
                ax.grid(True, alpha=0.3)
                ax.set_ylim(-0.1, 1.1)
            
            # 隐藏多余的子图
            for j in range(len(page_cells), len(axes)):
                axes[j].set_visible(False)
            
            plt.tight_layout()
            
            page_name = f'{self.prefix}individual_cells_page{page_idx + 1}'
            fig.savefig(cells_png_dir / f'{page_name}.png', dpi=200, bbox_inches='tight')
            fig.savefig(cells_pdf_dir / f'{page_name}.pdf', dpi=200, bbox_inches='tight')
            plt.close(fig)
        
        print(f"    保存到: {cells_png_dir}")
    
    def run_analysis(self, fitting_df: pd.DataFrame, raw_df: Optional[pd.DataFrame] = None,
                     percentile_bins: Optional[str] = None, bin_step: float = 10.0,
                     intensity_df: Optional[pd.DataFrame] = None,
                     plot_individual_cells: bool = False):
        """运行完整的动力学分析"""
        print("\n" + "=" * 60)
        print("Kinetics Analysis")
        print("=" * 60)
        
        # 基础动力学汇总
        self.plot_kinetics_summary(fitting_df, raw_df)
        
        # Pearson 变化直方图
        if raw_df is not None:
            self.plot_pearson_change_histogram(raw_df, fitting_df)
        
        # 分 percentile 分析
        if percentile_bins and intensity_df is not None:
            fitting_with_intensity = DataLoader.merge_fitting_with_intensity(fitting_df, intensity_df)
            channels = ['488', '561'] if percentile_bins.lower() == 'both' else [percentile_bins]
            for ch in channels:
                try:
                    self.plot_kinetics_by_percentile(fitting_with_intensity, ch, bin_step, raw_df)
                    self.plot_kinetics_by_percentile_normalized(fitting_with_intensity, ch, bin_step)
                    self.plot_pearson_change_histogram_by_percentile(fitting_with_intensity, raw_df, ch, bin_step)
                except ValueError as e:
                    print(f"  警告: {e}")
        
        # 逐个细胞作图（默认不执行，需要显式指定 --plot-individual-cells）
        if plot_individual_cells and raw_df is not None:
            self.plot_individual_cells(fitting_df, raw_df)


# ============================================================
# SimpleRegressionAnalyzer: 简单线性回归
# ============================================================
class SimpleRegressionAnalyzer:
    """
    简单线性回归分析器：单通道 vs k_app
    
    分析红色/绿色/比值与表观速率常数的关系
    """
    
    def __init__(self, output_dir: str, min_r_squared: float = 0.9, prefix: str = 'regression_'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.png_dir = self.output_dir / 'png'
        self.pdf_dir = self.output_dir / 'pdf'
        self.png_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.min_r_squared = min_r_squared
        self.prefix = prefix
        self.results = []
    
    def analyze_channel(self, df: pd.DataFrame, channel: str, save_plot: bool = True):
        """对单个通道进行线性回归分析"""
        if channel not in CHANNEL_CONFIG:
            raise ValueError(f"未知通道: {channel}")
        
        config = CHANNEL_CONFIG[channel]
        col_name = config['column']
        
        print(f"\n  === Analyzing {channel.upper()} channel ===")
        
        x_raw = df[col_name].values
        t50 = df['t50'].values
        k_app = np.log(2) / t50
        
        valid_mask = (x_raw > 0) & (k_app > 0) & np.isfinite(x_raw) & np.isfinite(k_app)
        x_valid = x_raw[valid_mask]
        k_valid = k_app[valid_mask]
        
        if len(x_valid) < 10:
            print(f"      Not enough valid data (n={len(x_valid)}), skipping...")
            return None
        
        x_log = np.log10(x_valid)
        k_log = np.log10(k_valid)
        
        # 执行回归
        x_clean, k_clean, _ = filter_by_cooks_distance(x_valid, k_valid)
        x_log_clean, k_log_clean, _ = filter_by_cooks_distance(x_log, k_log)
        
        if len(x_clean) < 3 or len(x_log_clean) < 3:
            return None
        
        slope_lin, intercept_lin, r_lin, p_lin, _ = stats.linregress(x_clean, k_clean)
        slope_log, intercept_log, r_log, p_log, _ = stats.linregress(x_log_clean, k_log_clean)
        
        if save_plot:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
            
            # 线性坐标
            ax1 = axes[0]
            ax1.scatter(x_clean, k_clean, alpha=0.15, s=40, c=config['color'], edgecolors='none')
            x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
            ax1.plot(x_line, slope_lin * x_line + intercept_lin, 'k--', linewidth=2, label=f'R = {r_lin:.3f}')
            ax1.set_xlabel(f'{config["label"]}\n(p = {p_lin:.2e})', fontsize=11)
            ax1.set_ylabel(r'$k_{app}$ ($s^{-1}$)', fontsize=11)
            ax1.set_title(f'{config["label"]} vs $k_{{app}}$\n(n = {len(x_clean)})', fontsize=12)
            ax1.legend(loc='best', fontsize=12, frameon=False)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.grid(True, alpha=0.3)
            
            # 双对数坐标
            ax2 = axes[1]
            ax2.scatter(x_log_clean, k_log_clean, alpha=0.15, s=40, c=config['color'], edgecolors='none')
            x_line_log = np.linspace(x_log_clean.min(), x_log_clean.max(), 100)
            ax2.plot(x_line_log, slope_log * x_line_log + intercept_log, 'k--', linewidth=2, label=f'R = {r_log:.3f}')
            ax2.set_xlabel(f'{config["log_label"]}\n(p = {p_log:.2e})', fontsize=11)
            ax2.set_ylabel(r'$\log_{10}$($k_{app}$)', fontsize=11)
            ax2.set_title(f'{config["log_label"]} vs $\\log_{{10}}$($k_{{app}}$)\n(n = {len(x_log_clean)})', fontsize=12)
            ax2.legend(loc='best', fontsize=12, frameon=False)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            fig.savefig(self.png_dir / f'{self.prefix}{channel}.png', dpi=300, bbox_inches='tight')
            fig.savefig(self.pdf_dir / f'{self.prefix}{channel}.pdf', dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"      Saved: {self.prefix}{channel}.png")
        
        print(f"      Linear: R = {r_lin:.4f}, p = {p_lin:.2e}")
        print(f"      Log-log: R = {r_log:.4f}, p = {p_log:.2e}")
        
        result = {
            'channel': channel,
            'n_cells_linear': len(x_clean),
            'n_cells_loglog': len(x_log_clean),
            'R_linear': r_lin, 'p_linear': p_lin, 'slope_linear': slope_lin,
            'R_loglog': r_log, 'p_loglog': p_log, 'slope_loglog': slope_log
        }
        self.results.append(result)
        return result
    
    def run_analysis(self, df: pd.DataFrame, channels: List[str] = None):
        """运行分析"""
        if channels is None:
            channels = ['red', 'green', 'ratio']
        
        print("\n" + "=" * 60)
        print("Simple Linear Regression Analysis")
        print("=" * 60)
        print(f"  Channels: {channels}")
        print(f"  Total cells: {len(df)}")
        
        # 应用数据清洗
        try:
            df_clean = apply_statistical_filters(df, verbose=True)
        except:
            df_clean = df
        
        for channel in channels:
            if CHANNEL_CONFIG[channel]['column'] in df_clean.columns:
                self.analyze_channel(df_clean, channel)
        
        if self.results:
            summary_df = pd.DataFrame(self.results)
            summary_df.to_csv(self.output_dir / f'{self.prefix}summary.csv', index=False)
            print(f"\n  Summary saved: {self.prefix}summary.csv")


# ============================================================
# StatisticsAnalyzer: 统计分析 + 多元回归
# ============================================================
class StatisticsAnalyzer:
    """
    统计分析类：信号强度与表观速率常数的关系分析和多元回归分析
    """
    
    def __init__(self, output_dir: str, prefix: str = 'stats_', cooks_factor: float = 4.0,
                 max_time_override: Optional[float] = None, min_r_squared: float = 0.9):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.png_dir = self.output_dir / 'png'
        self.pdf_dir = self.output_dir / 'pdf'
        self.png_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.cooks_factor = cooks_factor
        self.max_time_override = max_time_override
        self.min_r_squared = min_r_squared
    
    def _apply_basic_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用T90和R^2过滤"""
        print(f"  加载了 {len(df)} 个细胞")
        
        if 't90' in df.columns:
            before = len(df)
            if self.max_time_override is not None:
                df = df[df['t90'] <= self.max_time_override].copy()
            elif 'max_time' in df.columns:
                df = df[df['t90'] <= df['max_time']].copy()
            removed = before - len(df)
            if removed > 0:
                print(f"  移除了 {removed} 个 T90 > max_time 的细胞")
        
        if 'r_squared' in df.columns:
            before = len(df)
            df = df[df['r_squared'] >= self.min_r_squared].copy()
            removed = before - len(df)
            if removed > 0:
                print(f"  移除了 {removed} 个 R^2 < {self.min_r_squared} 的细胞")
        
        print(f"  过滤后剩余细胞: {len(df)}")
        return df
    
    def _generate_t50_t90_histogram(self, t50_values, t90_values):
        """生成T50和T90的分布直方图"""
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        
        for ax, values, name in [(axes[0], t50_values, 'T50'), (axes[1], t90_values, 'T90')]:
            ax.hist(values, bins=20, color='#0072B2', alpha=0.7, edgecolor='#00507D')
            ax.axvline(x=np.median(values), color='#D55E00', linestyle='--', linewidth=2,
                       label=f'Median: {np.median(values):.2f}')
            ax.axvline(x=np.mean(values), color='#E69F00', linestyle=':', linewidth=2,
                       label=f'Mean: {np.mean(values):.2f}')
            ax.set_xlabel(name, fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title(f'{name} Distribution (n={len(values)})', fontsize=14)
            ax.legend(loc='best', fontsize=10, frameon=False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        fig.savefig(self.png_dir / f'{self.prefix}t50_t90_distribution.png', dpi=300, bbox_inches='tight')
        fig.savefig(self.pdf_dir / f'{self.prefix}t50_t90_distribution.pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {self.prefix}t50_t90_distribution.png")
    
    def _generate_rate_constant_histogram(self, k_t50, k_t90):
        """生成表观速率常数的分布直方图"""
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        
        for ax, values, name in [(axes[0], k_t50, 'T50'), (axes[1], k_t90, 'T90')]:
            ax.hist(values, bins=20, color='#0072B2', alpha=0.7, edgecolor='dimgray')
            ax.axvline(x=np.median(values), color='#D55E00', linestyle='--', linewidth=2,
                       label=f'Median: {np.median(values):.4f}')
            ax.axvline(x=np.mean(values), color='#E69F00', linestyle=':', linewidth=2,
                       label=f'Mean: {np.mean(values):.4f}')
            ax.set_xlabel(r'$k_{app}$ ($s^{-1}$) from ' + name, fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title(f'$k_{{app}}$ ({name}) Distribution (n={len(values)})', fontsize=14)
            ax.legend(loc='best', fontsize=10, frameon=False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        fig.savefig(self.png_dir / f'{self.prefix}rate_constant_distribution.png', dpi=300, bbox_inches='tight')
        fig.savefig(self.pdf_dir / f'{self.prefix}rate_constant_distribution.pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {self.prefix}rate_constant_distribution.png")
    
    def _generate_intensity_vs_rate_plot(self, red, green, ratio, k, file_stems, cell_ids):
        """生成信号强度与速率常数关系图"""
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        
        # Red vs k_app
        red_clean, k_red, _ = filter_by_cooks_distance(red, k, self.cooks_factor)
        ax = axes[0, 0]
        ax.scatter(red_clean, k_red, alpha=0.1, s=40, c='#D96459', edgecolors='#B84A40')
        slope, intercept, r, p, _ = stats.linregress(red_clean, k_red)
        x_line = np.linspace(red_clean.min(), red_clean.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, 'k--', linewidth=2, label=f'R={r:.3f}')
        ax.set_xlabel(f'Red Intensity\n(p={p:.2e})', fontsize=11)
        ax.set_ylabel(r'$k_{app}$ ($s^{-1}$)', fontsize=11)
        ax.set_title(f'Red Intensity vs $k_{{app}}$ (n={len(red_clean)})', fontsize=12)
        ax.legend(loc='best', fontsize=12, frameon=False)
        ax.grid(True, alpha=0.3)
        
        # Green vs k_app
        green_clean, k_green, _ = filter_by_cooks_distance(green, k, self.cooks_factor)
        ax = axes[0, 1]
        ax.scatter(green_clean, k_green, alpha=0.1, s=40, c='#45ADA8', edgecolors='#358985')
        slope, intercept, r, p, _ = stats.linregress(green_clean, k_green)
        x_line = np.linspace(green_clean.min(), green_clean.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, 'k--', linewidth=2, label=f'R={r:.3f}')
        ax.set_xlabel(f'Green Intensity\n(p={p:.2e})', fontsize=11)
        ax.set_ylabel(r'$k_{app}$ ($s^{-1}$)', fontsize=11)
        ax.set_title(f'Green Intensity vs $k_{{app}}$ (n={len(green_clean)})', fontsize=12)
        ax.legend(loc='best', fontsize=12, frameon=False)
        ax.grid(True, alpha=0.3)
        
        # Ratio vs k_app
        ratio_clean, k_ratio, _ = filter_by_cooks_distance(ratio, k, self.cooks_factor)
        ax = axes[1, 0]
        ax.scatter(ratio_clean, k_ratio, alpha=0.1, s=40, c='steelblue', edgecolors='none')
        slope, intercept, r, p, _ = stats.linregress(ratio_clean, k_ratio)
        x_line = np.linspace(ratio_clean.min(), ratio_clean.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, 'k--', linewidth=2, label=f'R={r:.3f}')
        ax.set_xlabel(f'Red/Green Ratio\n(p={p:.2e})', fontsize=11)
        ax.set_ylabel(r'$k_{app}$ ($s^{-1}$)', fontsize=11)
        ax.set_title(f'Red/Green Ratio vs $k_{{app}}$ (n={len(ratio_clean)})', fontsize=12)
        ax.legend(loc='best', fontsize=12, frameon=False)
        ax.grid(True, alpha=0.3)
        
        # Ratio distribution
        ax = axes[1, 1]
        ax.hist(ratio_clean, bins=30, color='#0072B2', alpha=0.7, edgecolor='black')
        ax.axvline(x=np.median(ratio_clean), color='#D55E00', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(ratio_clean):.3f}')
        ax.set_xlabel('Red/Green Ratio', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'Red/Green Ratio Distribution (n={len(ratio_clean)})', fontsize=12)
        ax.legend(loc='best', fontsize=12, frameon=False)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(self.png_dir / f'{self.prefix}intensity_vs_rate_constant.png', dpi=300, bbox_inches='tight')
        fig.savefig(self.pdf_dir / f'{self.prefix}intensity_vs_rate_constant.pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {self.prefix}intensity_vs_rate_constant.png")
    
    def _perform_multiple_regression(self, red_log, green_log, k_log):
        """执行多元回归分析"""
        n = len(k_log)
        if n < 5:
            return
        
        X = np.column_stack([red_log, green_log])
        X_const = sm.add_constant(X)
        model = sm.OLS(k_log, X_const).fit()
        
        # Cook's Distance 过滤
        influence = model.get_influence()
        cooks_d = influence.cooks_distance[0]
        threshold = self.cooks_factor / n
        valid_mask = cooks_d <= threshold
        n_removed = np.sum(~valid_mask)
        
        if n_removed > 0:
            print(f"  Cook's Distance: removed {n_removed} points")
        
        if np.sum(valid_mask) < 5:
            return
        
        X_clean = X[valid_mask]
        k_clean = k_log[valid_mask]
        X_clean_const = sm.add_constant(X_clean)
        model_clean = sm.OLS(k_clean, X_clean_const).fit()
        
        print(f"\n  Multiple Regression Results (n={np.sum(valid_mask)}):")
        print(f"    R^2 = {model_clean.rsquared:.4f}")
        print(f"    Adj. R^2 = {model_clean.rsquared_adj:.4f}")
        print(f"    log10(Red): coef={model_clean.params[1]:.6f}, p={model_clean.pvalues[1]:.2e}")
        print(f"    log10(Green): coef={model_clean.params[2]:.6f}, p={model_clean.pvalues[2]:.2e}")
        
        # 生成偏回归图
        red_clean = X_clean[:, 0]
        green_clean = X_clean[:, 1]
        
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        
        # Red | Green
        model_k_green = sm.OLS(k_clean, sm.add_constant(green_clean)).fit()
        model_red_green = sm.OLS(red_clean, sm.add_constant(green_clean)).fit()
        resid_k = model_k_green.resid
        resid_red = model_red_green.resid
        
        ax = axes[0]
        ax.scatter(resid_red, resid_k, alpha=0.15, s=40, c='#D96459', edgecolors='#B84A40')
        slope, intercept, r, p, _ = stats.linregress(resid_red, resid_k)
        x_line = np.linspace(resid_red.min(), resid_red.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, 'k--', linewidth=2, label=f'Partial R={r:.3f}')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
        ax.set_xlabel(f'log10(Red) | log10(Green)\n(p={p:.2e})', fontsize=11)
        ax.set_ylabel('log10(k) | log10(Green)', fontsize=11)
        ax.set_title('Partial Regression: Red | Green', fontsize=12)
        ax.legend(loc='best', fontsize=12, frameon=False)
        ax.grid(True, alpha=0.3)
        
        # Green | Red
        model_k_red = sm.OLS(k_clean, sm.add_constant(red_clean)).fit()
        model_green_red = sm.OLS(green_clean, sm.add_constant(red_clean)).fit()
        resid_k2 = model_k_red.resid
        resid_green = model_green_red.resid
        
        ax = axes[1]
        ax.scatter(resid_green, resid_k2, alpha=0.15, s=40, c='#45ADA8', edgecolors='#358985')
        slope, intercept, r, p, _ = stats.linregress(resid_green, resid_k2)
        x_line = np.linspace(resid_green.min(), resid_green.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, 'k--', linewidth=2, label=f'Partial R={r:.3f}')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
        ax.set_xlabel(f'log10(Green) | log10(Red)\n(p={p:.2e})', fontsize=11)
        ax.set_ylabel('log10(k) | log10(Red)', fontsize=11)
        ax.set_title('Partial Regression: Green | Red', fontsize=12)
        ax.legend(loc='best', fontsize=12, frameon=False)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(self.png_dir / f'{self.prefix}partial_regression_log10k_app.png', dpi=300, bbox_inches='tight')
        fig.savefig(self.pdf_dir / f'{self.prefix}partial_regression_log10k_app.pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {self.prefix}partial_regression_log10k_app.png")
    
    def run_analysis(self, df: pd.DataFrame):
        """执行完整的统计分析"""
        print("\n" + "=" * 60)
        print("Statistics Analysis")
        print("=" * 60)
        
        # 应用基础过滤
        df = self._apply_basic_filters(df)
        
        # 确保必需的列存在
        df = df.copy()
        
        # 处理 red/green 列
        if 'red' not in df.columns and '1/red' in df.columns:
            df['red'] = 1.0 / df['1/red']
            df['green'] = 1.0 / df['1/green']
        
        # 处理 ratio 列
        if 'ratio' not in df.columns and 'red' in df.columns and 'green' in df.columns:
            df['ratio'] = df['red'] / df['green']
        
        # 处理 file_stem 列
        if 'file_stem' not in df.columns:
            if 'file_path' in df.columns:
                df['file_stem'] = df['file_path'].apply(lambda x: Path(x).stem if pd.notna(x) else '')
            elif 'source_file' in df.columns:
                df['file_stem'] = df['source_file']
            else:
                df['file_stem'] = 'unknown'
        
        # 处理 cell_id 列
        if 'cell_id' not in df.columns:
            df['cell_id'] = range(len(df))
        
        # 检查核心列
        core_required = ['red', 'green', 't50', 't90']
        missing = [c for c in core_required if c not in df.columns]
        if missing:
            raise ValueError(f"缺少必需的列: {missing}")
        
        # 应用IQR数据清洗
        df_filtered = apply_statistical_filters(df, verbose=True)
        
        if len(df_filtered) < 3:
            print("Not enough valid data points.")
            return
        
        red = df_filtered['red'].values
        green = df_filtered['green'].values
        ratio = df_filtered['ratio'].values
        t50 = df_filtered['t50'].values
        t90 = df_filtered['t90'].values
        file_stems = df_filtered['file_stem'].values
        cell_ids = df_filtered['cell_id'].values
        
        k_t50 = np.log(2) / t50
        k_t90 = np.log(2) / t90
        
        # 生成各种图表
        self._generate_t50_t90_histogram(t50, t90)
        self._generate_rate_constant_histogram(k_t50, k_t90)
        self._generate_intensity_vs_rate_plot(red, green, ratio, k_t50, file_stems, cell_ids)
        
        # 双对数多元回归
        red_log = np.log10(red)
        green_log = np.log10(green)
        k_log = np.log10(k_t50)
        self._perform_multiple_regression(red_log, green_log, k_log)
        
        # 导出分析数据
        analysis_df = pd.DataFrame({
            'file_stem': file_stems, 'cell_id': cell_ids,
            'log10_red': red_log, 'log10_green': green_log,
            'ratio': ratio, 't50': t50, 't90': t90,
            'k_app_T50': k_t50, 'k_app_T90': k_t90
        })
        analysis_df.to_csv(self.output_dir / f'{self.prefix}k_app_analysis_data.csv', index=False)
        print(f"  Saved: {self.prefix}k_app_analysis_data.csv")


# ============================================================
# PercentileBinAnalyzer: 百分位分组分析器
# ============================================================
class PercentileBinAnalyzer:
    """
    对特定通道进行percentile分组，分析每个分组的动力学参数
    
    支持：
    - 固定分组（如 0-10%, 10-20%, ...）
    - 滑窗分组（如 window_size=20, step=10）
    """
    
    def __init__(self, output_dir: str, bin_step: float = 10.0, window_size: float = 20.0,
                 min_r_squared: float = 0.9, cooks_factor: float = 4.0,
                 max_time_override: Optional[float] = None, prefix: str = 'percentile_'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.png_dir = self.output_dir / 'png'
        self.pdf_dir = self.output_dir / 'pdf'
        self.png_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.bin_step = bin_step
        self.window_size = window_size
        self.min_r_squared = min_r_squared
        self.cooks_factor = cooks_factor
        self.max_time_override = max_time_override
        self.prefix = prefix
    
    def _compute_group_statistics(self, group_df: pd.DataFrame) -> Dict[str, Any]:
        """计算组内的统计量（带 IQR 异常值过滤）"""
        n = len(group_df)
        if n < 3:
            return {'n': n}
        
        t50_raw = group_df['t50'].values
        t90_raw = group_df['t90'].values
        
        # IQR 异常值过滤 (T90 使用 2.5*IQR)
        t90_q1, t90_q3 = np.percentile(t90_raw, [25, 75])
        t90_iqr = t90_q3 - t90_q1
        t90_lower, t90_upper = t90_q1 - 2.5 * t90_iqr, t90_q3 + 2.5 * t90_iqr
        valid_mask = (t90_raw >= t90_lower) & (t90_raw <= t90_upper) & (t90_raw > 0)
        
        t50 = t50_raw[valid_mask]
        t90 = t90_raw[valid_mask]
        n_valid = len(t90)
        
        if n_valid < 3:
            return {'n': n, 'n_valid': 0}
        
        k_t50 = np.log(2) / t50
        k_t90 = np.log(2) / t90
        
        red_raw = group_df['red'].values if 'red' in group_df.columns else None
        green_raw = group_df['green'].values if 'green' in group_df.columns else None
        red = red_raw[valid_mask] if red_raw is not None else None
        green = green_raw[valid_mask] if green_raw is not None else None
        
        # 计算 IQR
        t90_q25, t90_q75 = np.percentile(t90, [25, 75])
        
        stats_dict = {
            'n': n, 'n_valid': n_valid,
            't50_median': np.median(t50), 't50_mean': np.mean(t50), 't50_std': np.std(t50),
            't90_median': np.median(t90), 't90_mean': np.mean(t90), 't90_std': np.std(t90),
            't90_q25': t90_q25, 't90_q75': t90_q75,
            'k_t50_median': np.median(k_t50), 'k_t50_mean': np.mean(k_t50), 'k_t50_std': np.std(k_t50),
            'k_t90_median': np.median(k_t90), 'k_t90_mean': np.mean(k_t90), 'k_t90_std': np.std(k_t90)
        }
        
        if red is not None:
            stats_dict['red_median'] = np.median(red)
            stats_dict['red_mean'] = np.mean(red)
        if green is not None:
            stats_dict['green_median'] = np.median(green)
            stats_dict['green_mean'] = np.mean(green)
        
        # 简单相关性 (Linear R)
        if red is not None and len(red) >= 5:
            r_red, p_red = stats.pearsonr(red, k_t50)
            stats_dict['corr_red_k'] = r_red
            stats_dict['corr_red_k_p'] = p_red
        if green is not None and len(green) >= 5:
            r_green, p_green = stats.pearsonr(green, k_t50)
            stats_dict['corr_green_k'] = r_green
            stats_dict['corr_green_k_p'] = p_green
        
        # 偏相关性 (Partial R) - 控制另一个通道
        if red is not None and green is not None and n_valid >= 10:
            try:
                valid_pos = (red > 0) & (green > 0)
                if np.sum(valid_pos) >= 10:
                    log_red = np.log10(red[valid_pos])
                    log_green = np.log10(green[valid_pos])
                    log_k = np.log10(k_t50[valid_pos])
                    
                    # Partial R: Red | Green
                    model_k_green = sm.OLS(log_k, sm.add_constant(log_green)).fit()
                    model_red_green = sm.OLS(log_red, sm.add_constant(log_green)).fit()
                    resid_k = model_k_green.resid
                    resid_red = model_red_green.resid
                    slope, intercept, r_partial_red, p_partial_red, _ = stats.linregress(resid_red, resid_k)
                    stats_dict['partial_R_red'] = r_partial_red
                    stats_dict['partial_p_red'] = p_partial_red
                    
                    # Partial R: Green | Red
                    model_k_red = sm.OLS(log_k, sm.add_constant(log_red)).fit()
                    model_green_red = sm.OLS(log_green, sm.add_constant(log_red)).fit()
                    resid_k2 = model_k_red.resid
                    resid_green = model_green_red.resid
                    slope, intercept, r_partial_green, p_partial_green, _ = stats.linregress(resid_green, resid_k2)
                    stats_dict['partial_R_green'] = r_partial_green
                    stats_dict['partial_p_green'] = p_partial_green
            except Exception:
                pass
        
        return stats_dict
    
    def run_percentile_analysis(self, df: pd.DataFrame, channel: str):
        """对指定通道进行percentile分组分析"""
        print(f"\n  Percentile Analysis: grouping by {channel} intensity")
        
        # 获取分组
        groups, col_name = PercentileGrouper.group_by_percentile(df, channel, self.bin_step)
        print(f"    Created {len(groups)} valid groups")
        
        # 对每个组计算统计量
        results = []
        print(f"    IQR 过滤后各组细胞数:")
        for group_df, label, pct_min, pct_max in groups:
            stats_dict = self._compute_group_statistics(group_df)
            stats_dict['group'] = label
            stats_dict['pct_min'] = pct_min
            stats_dict['pct_max'] = pct_max
            stats_dict['intensity_median'] = group_df[col_name].median()
            results.append(stats_dict)
            n_raw = stats_dict.get('n', 0)
            n_valid = stats_dict.get('n_valid', n_raw)
            print(f"      {label}: {n_raw} → {n_valid} cells (IQR过滤后)")
        
        # 生成统计结果表格
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_dir / f'{self.prefix}stats_{channel}.csv', index=False)
        
        # 生成可视化
        self._plot_percentile_trends(results_df, channel, col_name)
        
        print(f"    Saved: {self.prefix}stats_{channel}.csv")
        return results_df
    
    def run_sliding_window_analysis(self, df: pd.DataFrame, channel: str):
        """滑窗分析"""
        print(f"\n  Sliding Window Analysis: {channel} (window={self.window_size}%)")
        
        groups, col_name = PercentileGrouper.sliding_window(df, channel, self.window_size, step=self.bin_step)
        print(f"    Created {len(groups)} sliding windows")
        
        results = []
        for group_df, label, center_pct in groups:
            stats_dict = self._compute_group_statistics(group_df)
            stats_dict['window'] = label
            stats_dict['center_pct'] = center_pct
            stats_dict['intensity_median'] = group_df[col_name].median()
            results.append(stats_dict)
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_dir / f'{self.prefix}stats_{channel}.csv', index=False)
        
        self._plot_sliding_window_trends(results_df, channel, col_name)
        
        print(f"    Saved: {self.prefix}stats_{channel}.csv")
        return results_df
    
    def _plot_percentile_trends(self, results_df: pd.DataFrame, channel: str, col_name: str):
        """绘制 percentile 组的趋势图（4个子图）"""
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        color = get_channel_color(channel)
        x = (results_df['pct_min'] + results_df['pct_max']) / 2
        
        # 1. T90 vs percentile (带 IQR 阴影)
        ax = axes[0, 0]
        y = results_df['t90_median']
        n_valid = results_df['n_valid'] if 'n_valid' in results_df.columns else results_df['n']
        if 't90_q25' in results_df.columns and 't90_q75' in results_df.columns:
            q25, q75 = results_df['t90_q25'], results_df['t90_q75']
            ax.fill_between(x, q25, q75, alpha=0.2, color=color)
        yerr = results_df['t90_std'] / np.sqrt(n_valid)
        ax.errorbar(x, y, yerr=yerr, fmt='o-', color=color, capsize=3, markersize=6)
        ax.set_xlabel('Percentile (%)', fontsize=11)
        ax.set_ylabel(r'$T_{90}$ (s)', fontsize=11)
        ax.set_title(r'$T_{90}$ vs Percentile', fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3)
        
        # 2. k_app vs percentile
        ax = axes[0, 1]
        y = results_df['k_t50_median']
        yerr = results_df['k_t50_std'] / np.sqrt(n_valid)
        ax.errorbar(x, y, yerr=yerr, fmt='o-', color=color, capsize=3, markersize=6)
        ax.set_xlabel('Percentile (%)', fontsize=11)
        ax.set_ylabel(r'$k_{app}$ ($s^{-1}$)', fontsize=11)
        ax.set_title(r'$k_{app}$ vs Percentile', fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3)
        
        # 3. Linear R vs percentile (红绿双线)
        ax = axes[1, 0]
        if 'corr_red_k' in results_df.columns:
            y_red = results_df['corr_red_k'].values
            valid = ~np.isnan(y_red)
            if np.sum(valid) > 0:
                ax.plot(x[valid], y_red[valid], 'o-', color='#D96459', markersize=6, linewidth=2, label='Red')
        if 'corr_green_k' in results_df.columns:
            y_green = results_df['corr_green_k'].values
            valid = ~np.isnan(y_green)
            if np.sum(valid) > 0:
                ax.plot(x[valid], y_green[valid], 's-', color='#45ADA8', markersize=6, linewidth=2, label='Green')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.legend(fontsize=10, frameon=False)
        ax.set_xlabel('Percentile (%)', fontsize=11)
        ax.set_ylabel('R (Correlation)', fontsize=11)
        ax.set_title('Linear R vs Percentile', fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(-1.1, 1.1)
        ax.grid(True, alpha=0.3)
        
        # 4. Partial R vs percentile (红绿双线)
        ax = axes[1, 1]
        y_max = 0.5
        if 'partial_R_red' in results_df.columns:
            y_red = results_df['partial_R_red'].values
            valid = ~np.isnan(y_red)
            if np.sum(valid) > 0:
                ax.plot(x[valid], y_red[valid], 'o-', color='#D96459', markersize=6, linewidth=2, label='Red | Green')
                y_max = max(y_max, np.nanmax(np.abs(y_red)))
        if 'partial_R_green' in results_df.columns:
            y_green = results_df['partial_R_green'].values
            valid = ~np.isnan(y_green)
            if np.sum(valid) > 0:
                ax.plot(x[valid], y_green[valid], 's-', color='#45ADA8', markersize=6, linewidth=2, label='Green | Red')
                y_max = max(y_max, np.nanmax(np.abs(y_green)))
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.legend(fontsize=10, frameon=False)
        ax.set_xlabel('Percentile (%)', fontsize=11)
        ax.set_ylabel('Partial R', fontsize=11)
        ax.set_title('Partial R vs Percentile', fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        y_lim = y_max * 1.1 if y_max > 0 else 1.1
        ax.set_ylim(-y_lim, y_lim)
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Percentile Trends ({channel} channel)', fontsize=13, y=1.02)
        plt.tight_layout()
        fig.savefig(self.png_dir / f'{self.prefix}trends_{channel}.png', dpi=300, bbox_inches='tight')
        fig.savefig(self.pdf_dir / f'{self.prefix}trends_{channel}.pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: {self.prefix}trends_{channel}.png")
    
    def _plot_sliding_window_trends(self, results_df: pd.DataFrame, channel: str, col_name: str):
        """绘制滑窗分析趋势（4个子图）"""
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        color = get_channel_color(channel)
        x = results_df['center_pct']
        
        # 1. T90 trend (带 IQR 阴影)
        ax = axes[0, 0]
        y = results_df['t90_median']
        n_valid = results_df['n_valid'] if 'n_valid' in results_df.columns else results_df['n']
        if 't90_q25' in results_df.columns and 't90_q75' in results_df.columns:
            q25, q75 = results_df['t90_q25'], results_df['t90_q75']
            ax.fill_between(x, q25, q75, alpha=0.2, color=color)
        yerr = results_df['t90_std'] / np.sqrt(n_valid)
        ax.errorbar(x, y, yerr=yerr, fmt='o-', color=color, capsize=3, markersize=5, linewidth=2)
        ax.set_xlabel('Percentile (%)', fontsize=11)
        ax.set_ylabel(r'$T_{90}$ (s)', fontsize=11)
        ax.set_title(r'$T_{90}$ vs Percentile (Sliding)', fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3)
        
        # 2. k_app trend
        ax = axes[0, 1]
        y = results_df['k_t50_median']
        yerr = results_df['k_t50_std'] / np.sqrt(n_valid)
        ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.2)
        ax.plot(x, y, 'o-', color=color, markersize=5, linewidth=2)
        ax.set_xlabel('Percentile (%)', fontsize=11)
        ax.set_ylabel(r'$k_{app}$ ($s^{-1}$)', fontsize=11)
        ax.set_title(r'$k_{app}$ vs Percentile (Sliding)', fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3)
        
        # 3. Linear R trend (红绿双线)
        ax = axes[1, 0]
        if 'corr_red_k' in results_df.columns:
            y_red = results_df['corr_red_k'].values
            valid = ~np.isnan(y_red)
            if np.sum(valid) > 0:
                ax.plot(x[valid], y_red[valid], 'o-', color='#D96459', markersize=5, linewidth=2, label='Red')
        if 'corr_green_k' in results_df.columns:
            y_green = results_df['corr_green_k'].values
            valid = ~np.isnan(y_green)
            if np.sum(valid) > 0:
                ax.plot(x[valid], y_green[valid], 's-', color='#45ADA8', markersize=5, linewidth=2, label='Green')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.legend(fontsize=10, frameon=False)
        ax.set_xlabel('Percentile (%)', fontsize=11)
        ax.set_ylabel('R (Correlation)', fontsize=11)
        ax.set_title('Linear R vs Percentile (Sliding)', fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(-1.1, 1.1)
        ax.grid(True, alpha=0.3)
        
        # 4. Partial R trend (红绿双线)
        ax = axes[1, 1]
        y_max = 0.5
        if 'partial_R_red' in results_df.columns:
            y_red = results_df['partial_R_red'].values
            valid = ~np.isnan(y_red)
            if np.sum(valid) > 0:
                ax.plot(x[valid], y_red[valid], 'o-', color='#D96459', markersize=5, linewidth=2, label='Red | Green')
                y_max = max(y_max, np.nanmax(np.abs(y_red)))
        if 'partial_R_green' in results_df.columns:
            y_green = results_df['partial_R_green'].values
            valid = ~np.isnan(y_green)
            if np.sum(valid) > 0:
                ax.plot(x[valid], y_green[valid], 's-', color='#45ADA8', markersize=5, linewidth=2, label='Green | Red')
                y_max = max(y_max, np.nanmax(np.abs(y_green)))
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.legend(fontsize=10, frameon=False)
        ax.set_xlabel('Percentile (%)', fontsize=11)
        ax.set_ylabel('Partial R', fontsize=11)
        ax.set_title('Partial R vs Percentile (Sliding)', fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        y_lim = y_max * 1.1 if y_max > 0 else 1.1
        ax.set_ylim(-y_lim, y_lim)
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Sliding Window Trends ({channel} channel, window={self.window_size}%)', fontsize=13, y=1.02)
        plt.tight_layout()
        fig.savefig(self.png_dir / f'{self.prefix}trends_{channel}.png', dpi=300, bbox_inches='tight')
        fig.savefig(self.pdf_dir / f'{self.prefix}trends_{channel}.pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: {self.prefix}trends_{channel}.png")


# ============================================================
# 统一的主分析函数
# ============================================================
def run_unified_analysis(input_files: List[str], output_dir: str, args):
    """执行统一分析"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("Unified Analysis Pipeline")
    print("=" * 60)
    print(f"Output directory: {output_path}")
    print(f"Analysis types: {args.analysis}")
    
    # 分类文件
    fitting_files, raw_files, intensity_files = DataLoader.classify_files(input_files)
    
    print(f"\n找到的文件:")
    print(f"  拟合结果文件: {len(fitting_files)}")
    print(f"  原始数据文件: {len(raw_files)}")
    print(f"  荧光强度文件: {len(intensity_files)}")
    
    fitting_df = None
    raw_df = None
    intensity_df = None
    
    # 加载数据
    if fitting_files:
        fitting_df = DataLoader.load_fitting_data(fitting_files, args.min_rsq, args.max_time)
    if raw_files:
        raw_df = DataLoader.load_raw_data(raw_files)
    if intensity_files:
        intensity_df = DataLoader.load_intensity_data(intensity_files)
    
    analyses = args.analysis.lower().split(',')
    
    # ==== Kinetics Analysis ====
    if 'all' in analyses or 'kinetics' in analyses:
        if fitting_df is not None:
            kinetics = KineticsAnalyzer(
                output_dir=str(output_path),
                max_time=args.max_time or 300.0,
                use_delay=getattr(args, 'use_delay', False),
                prefix='kinetics_'
            )
            # 默认运行时也执行 percentile 分组动力学图
            pct_bins = getattr(args, 'percentile_bins', None)
            if pct_bins is None and 'all' in analyses:
                pct_bins = '488'  # 默认用 488 通道
            kinetics.run_analysis(
                fitting_df, raw_df,
                percentile_bins=pct_bins,
                bin_step=args.bin_step,
                intensity_df=intensity_df,
                plot_individual_cells=getattr(args, 'plot_individual_cells', False)
            )
        else:
            print("\n警告: 没有拟合结果文件，跳过动力学分析")
    
    # ==== Simple Regression Analysis ====
    if 'all' in analyses or 'regression' in analyses:
        if intensity_df is not None:
            # 合并拟合结果和荧光强度数据
            if fitting_df is not None:
                merged_df = DataLoader.merge_fitting_with_intensity(fitting_df, intensity_df)
                # 确保有t50列
                if 'correlation_t50' in merged_df.columns:
                    merged_df['t50'] = merged_df['correlation_t50']
            else:
                merged_df = intensity_df
            
            if 't50' in merged_df.columns:
                regression = SimpleRegressionAnalyzer(
                    output_dir=str(output_path),
                    min_r_squared=args.min_rsq,
                    prefix='regression_'
                )
                channels = ['red', 'green', 'ratio'] if 'ratio' in merged_df.columns else ['red', 'green']
                regression.run_analysis(merged_df, channels=channels)
            else:
                print("\n警告: 数据中没有 t50 列，跳过回归分析")
        else:
            print("\n警告: 没有荧光强度数据，跳过回归分析")
    
    # ==== Statistics Analysis ====
    if 'all' in analyses or 'stats' in analyses:
        if intensity_df is not None:
            # 合并数据
            if fitting_df is not None:
                merged_df = DataLoader.merge_fitting_with_intensity(fitting_df, intensity_df)
                if 'correlation_t50' in merged_df.columns:
                    merged_df['t50'] = merged_df['correlation_t50']
                if 'correlation_t90' in merged_df.columns:
                    merged_df['t90'] = merged_df['correlation_t90']
                if 'correlation_r_squared' in merged_df.columns:
                    merged_df['r_squared'] = merged_df['correlation_r_squared']
            else:
                merged_df = intensity_df
            
            if 't50' in merged_df.columns and 't90' in merged_df.columns:
                stats_analyzer = StatisticsAnalyzer(
                    output_dir=str(output_path),
                    prefix='stats_',
                    cooks_factor=args.cooks_factor,
                    max_time_override=args.max_time,
                    min_r_squared=args.min_rsq
                )
                stats_analyzer.run_analysis(merged_df)
            else:
                print("\n警告: 数据中缺少 t50 或 t90 列，跳过统计分析")
        else:
            print("\n警告: 没有荧光强度数据，跳过统计分析")
    
    # ==== Percentile Bin Analysis ====
    # 默认在 'all' 模式下执行，或显式指定 --percentile-bins
    run_percentile = ('all' in analyses) or getattr(args, 'percentile_bins', None)
    if run_percentile and intensity_df is not None and fitting_df is not None:
        merged_df = DataLoader.merge_fitting_with_intensity(fitting_df, intensity_df)
        if 'correlation_t50' in merged_df.columns:
            merged_df['t50'] = merged_df['correlation_t50']
        if 'correlation_t90' in merged_df.columns:
            merged_df['t90'] = merged_df['correlation_t90']
        
        if 't50' in merged_df.columns:
            bin_analyzer = PercentileBinAnalyzer(
                output_dir=str(output_path),
                bin_step=args.bin_step,
                window_size=args.window_size,
                min_r_squared=args.min_rsq,
                cooks_factor=args.cooks_factor,
                prefix='percentile_'
            )
            
            # 默认用 488，如果显式指定则用指定的
            if getattr(args, 'percentile_bins', None):
                channels = ['488', '561'] if args.percentile_bins.lower() == 'both' else [args.percentile_bins]
            else:
                channels = ['488']  # 默认只分析 488
            
            for ch in channels:
                try:
                    bin_analyzer.run_percentile_analysis(merged_df, ch)
                except ValueError as e:
                    print(f"  警告: {e}")
    
    # ==== Sliding Window Analysis ====
    # 默认在 'all' 模式下执行，或显式指定 --sliding-window
    run_sliding = ('all' in analyses) or getattr(args, 'sliding_window', None)
    if run_sliding and intensity_df is not None and fitting_df is not None:
        merged_df = DataLoader.merge_fitting_with_intensity(fitting_df, intensity_df)
        if 'correlation_t50' in merged_df.columns:
            merged_df['t50'] = merged_df['correlation_t50']
        if 'correlation_t90' in merged_df.columns:
            merged_df['t90'] = merged_df['correlation_t90']
        
        if 't50' in merged_df.columns:
            window_analyzer = PercentileBinAnalyzer(
                output_dir=str(output_path),
                bin_step=args.bin_step,
                window_size=args.window_size,
                min_r_squared=args.min_rsq,
                prefix='sliding_'
            )
            
            # 默认用 488，如果显式指定则用指定的
            if getattr(args, 'sliding_window', None):
                channels = ['488', '561'] if args.sliding_window.lower() == 'both' else [args.sliding_window]
            else:
                channels = ['488']  # 默认只分析 488
            
            for ch in channels:
                try:
                    window_analyzer.run_sliding_window_analysis(merged_df, ch)
                except ValueError as e:
                    print(f"  警告: {e}")
    
    print("\n" + "=" * 60)
    print("分析完成!")
    print("=" * 60)


# ============================================================
# CLI 入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Unified Analysis Pipeline for Colocalization Kinetics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
用法示例:
  基本用法（执行全部分析）:
    python unified_analysis.py data.csv -o output_dir
    python unified_analysis.py *.csv -o output_dir
    
  指定分析类型:
    python unified_analysis.py data.csv --analysis kinetics
    python unified_analysis.py data.csv --analysis stats,regression
    
  百分位分组分析:
    python unified_analysis.py data.csv --percentile-bins 488
    python unified_analysis.py data.csv --percentile-bins both --bin-step 10
    
  滑窗分析:
    python unified_analysis.py data.csv --sliding-window 488 --window-size 20
    
  数据过滤:
    python unified_analysis.py data.csv --min-rsq 0.95 --max-time 600
        """
    )
    
    # 输入输出
    parser.add_argument('input_files', nargs='+', help='输入CSV文件（支持通配符）')
    parser.add_argument('-o', '--output', default='unified_analysis_output', help='输出目录')
    
    # 分析类型
    parser.add_argument('--analysis', default='all',
                        help='分析类型，逗号分隔: all, kinetics, regression, stats')
    
    # 数据过滤参数
    parser.add_argument('--min-rsq', type=float, default=0.9, help='R^2最小阈值 (default: 0.9)')
    parser.add_argument('--max-time', type=float, default=None, help='T90最大时间阈值')
    parser.add_argument('--cooks-factor', type=float, default=4.0, help="Cook's Distance 因子 (default: 4.0)")
    
    # 百分位分组参数
    parser.add_argument('--percentile-bins', type=str, default=None,
                        help='按指定通道进行 percentile 分组: 488, 561, both')
    parser.add_argument('--bin-step', type=float, default=10.0, help='分组步长 (default: 10)')
    
    # 滑窗分析参数
    parser.add_argument('--sliding-window', type=str, default=None,
                        help='滑窗分析通道: 488, 561, both')
    parser.add_argument('--window-size', type=float, default=30.0, help='滑窗大小 (default: 30)')
    
    # 动力学分析参数
    parser.add_argument('--use-delay', action='store_true', help='在拟合时使用 delay 参数')
    
    # 逐个细胞作图（默认不执行）
    parser.add_argument('--plot-individual-cells', action='store_true',
                        help='绘制逐个细胞的拟合曲线图（默认不执行）')
    
    args = parser.parse_args()
    
    # 展开通配符
    all_files = []
    for pattern in args.input_files:
        matches = glob.glob(pattern)
        if matches:
            all_files.extend(matches)
        elif Path(pattern).exists():
            all_files.append(pattern)
    
    if not all_files:
        print("错误: 未找到任何输入文件")
        return
    
    all_files = list(set(all_files))
    print(f"\n找到 {len(all_files)} 个输入文件")
    
    run_unified_analysis(all_files, args.output, args)


if __name__ == '__main__':
    main()
