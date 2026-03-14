"""
简单线性回归分析：单通道强度 vs 表观速率常数(k_app)

此脚本对指定通道进行简单线性回归分析，不进行偏回归（不控制其他变量）。
适用于初步探索单个变量与反应速度的关系。

使用方式:
=========

基本用法（默认对所有通道分别做回归）:
    python simple_linear_regression.py <csv_file>
    python simple_linear_regression.py ratio_t50_raw_data.csv

指定单个通道:
    python simple_linear_regression.py data.csv --channel red
    python simple_linear_regression.py data.csv --channel green
    python simple_linear_regression.py data.csv --channel ratio

指定多个通道:
    python simple_linear_regression.py data.csv --channel red green
    python simple_linear_regression.py data.csv --channel red ratio

百分位分组分析:
    python simple_linear_regression.py data.csv --percentile-bins 488           # 按488分组，分析green vs k_app
    python simple_linear_regression.py data.csv --percentile-bins 561           # 按561分组，分析red vs k_app
    python simple_linear_regression.py data.csv --percentile-bins 488 --bin-step 5    # 每5%分组
    python simple_linear_regression.py data.csv --percentile-bins both                # 两个通道分别分析

指定输出目录:
    python simple_linear_regression.py data.csv --output my_output

拟合质量过滤:
    python simple_linear_regression.py data.csv --min-rsq 0.95

输出:
    - simple_regression_{channel}.png/pdf: 单通道线性回归图（2x1布局：线性+双对数）
    - simple_regression_summary.csv: 回归参数汇总表

百分位分组输出 (--percentile-bins):
    - percentile_bins_{488|561}/
      - simple_regression_bin_summary_{channel}.csv: 各组回归参数汇总
      - simple_regression_trends_{channel}.png/pdf: 统计量和R值趋势图

百分位分组逻辑:
    - --percentile-bins 488  → 按绿色通道分组，分析 green vs k_app
    - --percentile-bins 561  → 按红色通道分组，分析 red vs k_app
    - --percentile-bins both → 两者分别进行

通道说明:
    - red: 红色通道强度 (561nm)
    - green: 绿色通道强度 (488nm)  
    - ratio: 红绿比值 (Red/Green)
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from typing import List, Optional

# 复用现有的数据过滤函数
from statistics_analysis import apply_statistical_filters

# 设置全局字体为 Arial
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


class SimpleLinearRegressionAnalyzer:
    """简单线性回归分析器：单通道 vs k_app"""
    
    # 通道配置
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
    
    def __init__(self, output_dir: str, min_r_squared: float = 0.9):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_r_squared = min_r_squared
        self.results = []  # 存储回归结果
    
    def load_and_filter_data(self, csv_path: str) -> pd.DataFrame:
        """加载并过滤数据"""
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # R²过滤
        if 'r_squared' in df.columns:
            n_before = len(df)
            df = df[df['r_squared'] >= self.min_r_squared].copy()
            n_after = len(df)
            print(f"  R² filter (>= {self.min_r_squared}): {n_before} -> {n_after} cells")
        
        # 复用统计分析的数据清洗流程
        df_filtered = apply_statistical_filters(df, verbose=True)
        
        return df_filtered
    
    def _filter_by_cooks_distance(self, x: np.ndarray, y: np.ndarray, 
                                   name: str, cooks_factor: float = 4.0):
        """
        使用Cook's Distance过滤异常值
        
        Returns:
            x_clean, y_clean, n_removed
        """
        n = len(x)
        if n < 5:
            return x, y, 0
        
        # 计算线性回归
        slope, intercept, r, p, se = stats.linregress(x, y)
        y_pred = slope * x + intercept
        residuals = y - y_pred
        
        # 计算杠杆值 (leverage)
        x_mean = np.mean(x)
        h = 1/n + (x - x_mean)**2 / np.sum((x - x_mean)**2)
        
        # 计算MSE
        mse = np.sum(residuals**2) / (n - 2)
        
        # 计算Cook's Distance
        cooks_d = (residuals**2 / (2 * mse)) * (h / (1 - h)**2)
        
        # 过滤
        threshold = cooks_factor / n
        valid_mask = cooks_d <= threshold
        n_removed = np.sum(~valid_mask)
        
        if n_removed > 0:
            print(f"      Cook's Distance ({name}): removed {n_removed} points (threshold={threshold:.4f})")
        
        return x[valid_mask], y[valid_mask], n_removed
    
    def _do_regression(self, x: np.ndarray, y: np.ndarray, name: str, apply_cooks: bool = True):
        """
        执行线性回归并返回结果
        
        Returns:
            dict with slope, intercept, r, p, se, n
        """
        if len(x) < 3:
            return {'slope': np.nan, 'intercept': np.nan, 'r': np.nan, 
                    'p': np.nan, 'se': np.nan, 'n': len(x)}
        
        if apply_cooks:
            x_clean, y_clean, _ = self._filter_by_cooks_distance(x, y, name)
        else:
            x_clean, y_clean = x, y
        
        if len(x_clean) < 3:
            return {'slope': np.nan, 'intercept': np.nan, 'r': np.nan, 
                    'p': np.nan, 'se': np.nan, 'n': len(x_clean)}
        
        slope, intercept, r, p, se = stats.linregress(x_clean, y_clean)
        return {'slope': slope, 'intercept': intercept, 'r': r, 
                'p': p, 'se': se, 'n': len(x_clean)}
    
    def analyze_channel(self, df: pd.DataFrame, channel: str, save_plot: bool = True):
        """
        对单个通道进行线性回归分析
        
        Parameters:
            df: 过滤后的数据
            channel: 'red', 'green', 或 'ratio'
            save_plot: 是否保存图片
            
        Returns:
            dict: 回归结果
        """
        if channel not in self.CHANNEL_CONFIG:
            raise ValueError(f"未知通道: {channel}. 可选: {list(self.CHANNEL_CONFIG.keys())}")
        
        config = self.CHANNEL_CONFIG[channel]
        col_name = config['column']
        
        print(f"\n  === Analyzing {channel.upper()} channel ===")
        
        # 提取数据
        x_raw = df[col_name].values
        t50 = df['t50'].values
        k_app = np.log(2) / t50  # 表观速率常数
        
        # 过滤无效值
        valid_mask = (x_raw > 0) & (k_app > 0) & np.isfinite(x_raw) & np.isfinite(k_app)
        x_valid = x_raw[valid_mask]
        k_valid = k_app[valid_mask]
        
        if len(x_valid) < 10:
            print(f"      Not enough valid data (n={len(x_valid)}), skipping...")
            return None
        
        x_log = np.log10(x_valid)
        k_log = np.log10(k_valid)
        
        # 执行回归
        result_linear = self._do_regression(x_valid, k_valid, f'{channel} (linear)')
        result_loglog = self._do_regression(x_log, k_log, f'{channel} (log-log)')
        
        if save_plot:
            # === 创建图表：2x1 布局 ===
            fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
            
            # --- Panel 1: 线性坐标 ---
            ax1 = axes[0]
            x_clean, k_clean, _ = self._filter_by_cooks_distance(x_valid, k_valid, f'{channel} (linear)')
            
            ax1.scatter(x_clean, k_clean, alpha=0.15, s=40, 
                       c=config['color'], edgecolors='none')
            
            if not np.isnan(result_linear['slope']):
                x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
                ax1.plot(x_line, result_linear['slope'] * x_line + result_linear['intercept'], 
                        'k--', linewidth=2, label=f'R = {result_linear["r"]:.3f}')
            
            ax1.set_xlabel(f'{config["label"]}\n(p = {result_linear["p"]:.2e})', fontsize=11)
            ax1.set_ylabel(r'$k_{app}$ ($s^{-1}$)', fontsize=11)
            ax1.set_title(f'{config["label"]} vs $k_{{app}}$\n(n = {result_linear["n"]})', fontsize=12)
            ax1.legend(loc='best', fontsize=12, frameon=False)
            ax1.tick_params(axis='both', labelsize=11)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.grid(True, alpha=0.3)
            
            # --- Panel 2: 双对数坐标 ---
            ax2 = axes[1]
            x_log_clean, k_log_clean, _ = self._filter_by_cooks_distance(x_log, k_log, f'{channel} (log-log)')
            
            ax2.scatter(x_log_clean, k_log_clean, alpha=0.15, s=40,
                       c=config['color'], edgecolors='none')
            
            if not np.isnan(result_loglog['slope']):
                x_line_log = np.linspace(x_log_clean.min(), x_log_clean.max(), 100)
                ax2.plot(x_line_log, result_loglog['slope'] * x_line_log + result_loglog['intercept'],
                        'k--', linewidth=2, label=f'R = {result_loglog["r"]:.3f}')
            
            ax2.set_xlabel(f'{config["log_label"]}\n(p = {result_loglog["p"]:.2e})', fontsize=11)
            ax2.set_ylabel(r'$\log_{10}$($k_{app}$)', fontsize=11)
            ax2.set_title(f'{config["log_label"]} vs $\\log_{{10}}$($k_{{app}}$)\n(n = {result_loglog["n"]})', fontsize=12)
            ax2.legend(loc='best', fontsize=12, frameon=False)
            ax2.tick_params(axis='both', labelsize=11)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图片
            fig_path_png = self.output_dir / f'simple_regression_{channel}.png'
            fig_path_pdf = self.output_dir / f'simple_regression_{channel}.pdf'
            fig.savefig(fig_path_png, dpi=300, bbox_inches='tight')
            fig.savefig(fig_path_pdf, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"      Saved: {fig_path_png}")
            print(f"      Saved: {fig_path_pdf}")
        
        # 输出统计结果
        print(f"\n      Linear scale:  R = {result_linear['r']:.4f}, p = {result_linear['p']:.2e}, slope = {result_linear['slope']:.6f}")
        print(f"      Log-log scale: R = {result_loglog['r']:.4f}, p = {result_loglog['p']:.2e}, slope = {result_loglog['slope']:.4f}")
        
        # 存储结果
        result = {
            'channel': channel,
            'n_cells_linear': result_linear['n'],
            'n_cells_loglog': result_loglog['n'],
            'R_linear': result_linear['r'],
            'p_linear': result_linear['p'],
            'slope_linear': result_linear['slope'],
            'intercept_linear': result_linear['intercept'],
            'R_loglog': result_loglog['r'],
            'p_loglog': result_loglog['p'],
            'slope_loglog': result_loglog['slope'],
            'intercept_loglog': result_loglog['intercept']
        }
        self.results.append(result)
        return result
    
    def run_analysis(self, df: pd.DataFrame, channels: List[str]):
        """
        运行分析
        
        Parameters:
            df: 过滤后的数据
            channels: 要分析的通道列表
        """
        print("\n=== Simple Linear Regression Analysis ===")
        print(f"  Channels to analyze: {channels}")
        print(f"  Total cells: {len(df)}")
        
        for channel in channels:
            self.analyze_channel(df, channel)
        
        # 导出汇总表
        if self.results:
            summary_df = pd.DataFrame(self.results)
            summary_path = self.output_dir / 'simple_regression_summary.csv'
            summary_df.to_csv(summary_path, index=False)
            print(f"\n  Summary saved: {summary_path}")
            
            # 打印汇总表
            print("\n  === Summary ===")
            print(f"  {'Channel':<8} | {'R (linear)':<12} | {'p (linear)':<12} | {'R (log-log)':<12} | {'p (log-log)':<12}")
            print("  " + "-" * 70)
            for r in self.results:
                print(f"  {r['channel']:<8} | {r['R_linear']:>10.4f}   | {r['p_linear']:>10.2e}   | {r['R_loglog']:>10.4f}   | {r['p_loglog']:>10.2e}")


class PercentileBinRegressionAnalyzer:
    """百分位分组线性回归分析器"""
    
    # 分组通道映射
    BIN_CHANNEL_MAP = {
        '488': 'green',
        '561': 'red',
        'green': 'green',
        'red': 'red'
    }
    
    # 分析通道配置（与 SimpleLinearRegressionAnalyzer 相同）
    CHANNEL_CONFIG = SimpleLinearRegressionAnalyzer.CHANNEL_CONFIG
    
    def __init__(self, output_dir: str, bin_channel: str, bin_step: int = 10, 
                 min_r_squared: float = 0.9):
        """
        Parameters:
            output_dir: 输出目录
            bin_channel: 用于分组的通道 (488/561/green/red)
            bin_step: 百分位步长 (默认10%)
            min_r_squared: 最小R²阈值
        """
        self.output_dir = Path(output_dir)
        self.bin_channel = bin_channel
        self.bin_step = bin_step
        self.min_r_squared = min_r_squared
        
        # 映射分组通道
        self.bin_col = self.BIN_CHANNEL_MAP.get(bin_channel)
        if self.bin_col is None:
            raise ValueError(f"不支持的分组通道: {bin_channel}，请使用 488/561/green/red")
    
    def run_binned_analysis(self, df: pd.DataFrame, channels: List[str]):
        """
        执行百分位分组回归分析
        
        Parameters:
            df: 过滤后的数据
            channels: 要分析的通道列表
        """
        print(f"\n{'=' * 60}")
        print(f"Percentile Bin Regression Analysis")
        print(f"  Bin by: {self.bin_channel} ({self.bin_col}), Step: {self.bin_step}%")
        print(f"  Analyze channels: {channels}")
        print("=" * 60)
        
        if self.bin_col not in df.columns:
            print(f"  错误: 数据中没有 {self.bin_col} 列")
            return
        
        # 创建输出子目录
        bin_output_dir = self.output_dir / f"percentile_bins_{self.bin_channel}"
        bin_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  输出目录: {bin_output_dir}")
        
        # 生成百分位边界
        bin_edges = np.arange(0, 100 + self.bin_step, self.bin_step)
        n_bins = len(bin_edges) - 1
        print(f"  分组数量: {n_bins} (步长: {self.bin_step}%)")
        
        # 计算分组边界值
        intensity_values = df[self.bin_col].values
        percentile_thresholds = [np.percentile(intensity_values, p) for p in bin_edges]
        
        # 对每个分析通道分别处理
        for channel in channels:
            print(f"\n  ===== Analyzing {channel.upper()} channel by {self.bin_channel} bins =====")
            
            bin_results = []
            
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
                
                print(f"\n    --- Bin {i+1}: {pct_min:.0f}%-{pct_max:.0f}% (n={n_cells_raw}) ---")
                
                # 计算该组的回归参数
                group_result = self._compute_bin_regression(group_df, channel, pct_mid)
                group_result['percentile_min'] = pct_min
                group_result['percentile_max'] = pct_max
                group_result['percentile_mid'] = pct_mid
                group_result['n_cells_raw'] = n_cells_raw
                group_result['intensity_threshold_low'] = threshold_low
                group_result['intensity_threshold_high'] = threshold_high
                bin_results.append(group_result)
            
            # 转换为DataFrame
            results_df = pd.DataFrame(bin_results)
            
            # 保存汇总CSV
            csv_path = bin_output_dir / f"simple_regression_bin_summary_{channel}.csv"
            results_df.to_csv(csv_path, index=False)
            print(f"\n    Saved: {csv_path}")
            
            # 生成趋势图
            self._generate_trends_plot(results_df, channel, bin_output_dir)
        
        print(f"\n  Percentile Bin Analysis completed!")
    
    def _compute_bin_regression(self, df: pd.DataFrame, channel: str, pct_mid: float) -> dict:
        """计算单个分组的回归参数和统计量"""
        result = {
            'channel': channel,
            'n_cells_final': 0,
            # 分布统计量
            'intensity_mean': np.nan,
            'intensity_sd': np.nan,
            't50_mean': np.nan,
            't50_sd': np.nan,
            't90_mean': np.nan,
            't90_sd': np.nan,
            'k_app_mean': np.nan,
            'k_app_sd': np.nan,
            # 回归参数
            'R_linear': np.nan,
            'p_linear': np.nan,
            'slope_linear': np.nan,
            'R_loglog': np.nan,
            'p_loglog': np.nan,
            'slope_loglog': np.nan
        }
        
        if len(df) < 5:
            print(f"        数据点太少 (n={len(df)}), 跳过")
            return result
        
        # 应用IQR过滤
        try:
            df_clean = apply_statistical_filters(df, verbose=False)
        except:
            df_clean = df
        
        if len(df_clean) < 5:
            print(f"        过滤后数据点太少 (n={len(df_clean)}), 跳过")
            return result
        
        config = self.CHANNEL_CONFIG.get(channel)
        if config is None:
            return result
        
        col_name = config['column']
        
        # 提取数据
        x_raw = df_clean[col_name].values
        t50 = df_clean['t50'].values
        t90 = df_clean['t90'].values if 't90' in df_clean.columns else np.full_like(t50, np.nan)
        k_app = np.log(2) / t50
        
        # 过滤无效值
        valid_mask = (x_raw > 0) & (k_app > 0) & np.isfinite(x_raw) & np.isfinite(k_app)
        x_valid = x_raw[valid_mask]
        k_valid = k_app[valid_mask]
        t50_valid = t50[valid_mask]
        t90_valid = t90[valid_mask]
        
        if len(x_valid) < 5:
            return result
        
        result['n_cells_final'] = len(x_valid)
        
        # === 分布统计量 ===
        result['intensity_mean'] = np.mean(x_valid)
        result['intensity_sd'] = np.std(x_valid)
        result['t50_mean'] = np.mean(t50_valid)
        result['t50_sd'] = np.std(t50_valid)
        # T90统计（排除NaN）
        t90_finite = t90_valid[np.isfinite(t90_valid)]
        if len(t90_finite) > 0:
            result['t90_mean'] = np.mean(t90_finite)
            result['t90_sd'] = np.std(t90_finite)
        result['k_app_mean'] = np.mean(k_valid)
        result['k_app_sd'] = np.std(k_valid)
        
        # === 线性回归 ===
        try:
            slope, intercept, r, p, se = stats.linregress(x_valid, k_valid)
            result['R_linear'] = r
            result['p_linear'] = p
            result['slope_linear'] = slope
        except:
            pass
        
        # === 双对数回归 ===
        try:
            x_log = np.log10(x_valid)
            k_log = np.log10(k_valid)
            slope, intercept, r, p, se = stats.linregress(x_log, k_log)
            result['R_loglog'] = r
            result['p_loglog'] = p
            result['slope_loglog'] = slope
        except:
            pass
        
        print(f"        n={result['n_cells_final']}, T50={result['t50_mean']:.1f}±{result['t50_sd']:.1f}, T90={result['t90_mean']:.1f}±{result['t90_sd']:.1f}, R(log-log)={result['R_loglog']:.3f}")
        
        return result
    
    def _generate_trends_plot(self, results_df: pd.DataFrame, channel: str, output_dir: Path):
        """生成统计量和R值随百分位变化的趋势图"""
        # 过滤有效数据
        df = results_df[results_df['n_cells_final'] >= 5].copy()
        if len(df) < 2:
            print(f"      警告: 有效分组数量不足，跳过趋势图")
            return
        
        x = df['percentile_mid'].values
        
        # 根据分组通道选择颜色
        if self.bin_channel in ['488', 'green']:
            color = '#45ADA8'  # 488绿色通道
        elif self.bin_channel in ['561', 'red']:
            color = '#D96459'  # 561红色通道
        else:
            color = 'steelblue'  # 默认
        
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))  # 每个子图4x4英寸
        
        # --- Panel 1: T50 均值±SD 趋势 ---
        ax1 = axes[0, 0]
        if 't50_mean' in df.columns:
            y_mean = df['t50_mean'].values
            y_sd = df['t50_sd'].values
            valid = ~np.isnan(y_mean)
            if np.sum(valid) > 0:
                ax1.errorbar(x[valid], y_mean[valid], yerr=y_sd[valid], 
                            fmt='o-', color=color, capsize=4, markersize=7, linewidth=2)
                ax1.fill_between(x[valid], (y_mean - y_sd)[valid], (y_mean + y_sd)[valid], 
                                color=color, alpha=0.2)
        ax1.set_xlabel(f'Percentile (%) - binned by {self.bin_channel}', fontsize=11)
        ax1.set_ylabel('T50 (s)', fontsize=11)
        ax1.set_title('T50 Mean ± SD', fontsize=12)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.tick_params(axis='both', labelsize=10)
        
        # --- Panel 2: T90 均值±SD 趋势 ---
        ax2 = axes[0, 1]
        if 't90_mean' in df.columns:
            y_mean = df['t90_mean'].values
            y_sd = df['t90_sd'].values
            valid = ~np.isnan(y_mean)
            if np.sum(valid) > 0:
                ax2.errorbar(x[valid], y_mean[valid], yerr=y_sd[valid], 
                            fmt='o-', color=color, capsize=4, markersize=7, linewidth=2)
                ax2.fill_between(x[valid], (y_mean - y_sd)[valid], (y_mean + y_sd)[valid], 
                                color=color, alpha=0.2)
        ax2.set_xlabel(f'Percentile (%) - binned by {self.bin_channel}', fontsize=11)
        ax2.set_ylabel('T90 (s)', fontsize=11)
        ax2.set_title('T90 Mean ± SD', fontsize=12)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.tick_params(axis='both', labelsize=10)
        
        # --- Panel 3: R值趋势 ---
        ax3 = axes[1, 0]
        
        # 线性回归R值
        if 'R_linear' in df.columns:
            y_linear = df['R_linear'].values
            valid = ~np.isnan(y_linear)
            if np.sum(valid) > 0:
                ax3.plot(x[valid], y_linear[valid], 'o-', color=color, 
                        markersize=8, linewidth=2, label='Linear')
        
        # 双对数回归R值
        if 'R_loglog' in df.columns:
            y_loglog = df['R_loglog'].values
            valid = ~np.isnan(y_loglog)
            if np.sum(valid) > 0:
                ax3.plot(x[valid], y_loglog[valid], 's--', color=color, 
                        markersize=7, linewidth=2, alpha=0.7, label='Log-Log')
        
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax3.set_xlabel(f'Percentile (%) - binned by {self.bin_channel}', fontsize=11)
        ax3.set_ylabel('R (Correlation)', fontsize=11)
        ax3.set_title(f'{channel.upper()} vs $k_{{app}}$: R Trend', fontsize=12)
        ax3.legend(loc='best', fontsize=10, frameon=False)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.tick_params(axis='both', labelsize=10)
        ax3.set_ylim(-1.1, 1.1)
        
        # --- Panel 4: 细胞数量 ---
        ax4 = axes[1, 1]
        
        y_n = df['n_cells_final'].values
        width = self.bin_step * 0.6
        ax4.bar(x, y_n, width=width, color=color, edgecolor='dimgray', alpha=0.7)
        
        ax4.set_xlabel(f'Percentile (%) - binned by {self.bin_channel}', fontsize=11)
        ax4.set_ylabel('Cell Count', fontsize=11)
        ax4.set_title('Cell Count per Bin', fontsize=12)
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.tick_params(axis='both', labelsize=10)
        
        plt.suptitle(f'Simple Regression Trends: {channel.upper()} (binned by {self.bin_channel}, {self.bin_step}% step)', 
                     fontsize=13, y=1.02)
        plt.tight_layout()
        
        # 保存
        fig_path_png = output_dir / f"simple_regression_trends_{channel}.png"
        fig_path_pdf = output_dir / f"simple_regression_trends_{channel}.pdf"
        fig.savefig(fig_path_png, dpi=300, bbox_inches='tight')
        fig.savefig(fig_path_pdf, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: {fig_path_png}")


def main():
    parser = argparse.ArgumentParser(
        description='简单线性回归分析：单通道强度 vs k_app',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('csv_file', type=str, help='输入CSV文件路径')
    parser.add_argument('--channel', '-c', type=str, nargs='+', 
                       default=['red', 'green', 'ratio'],
                       choices=['red', 'green', 'ratio'],
                       help='要分析的通道 (默认: 全部)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='输出目录 (默认: 与输入文件同目录下创建 simple_regression/)')
    parser.add_argument('--min-rsq', type=float, default=0.9,
                       help='最小R²阈值 (默认: 0.9)')
    
    # 百分位分组参数
    parser.add_argument('--percentile-bins', type=str, default=None,
                       choices=['488', '561', 'red', 'green', 'both'],
                       help='按指定通道进行百分位分组分析 (488/561/red/green/both)')
    parser.add_argument('--bin-step', type=int, default=10,
                       help='百分位分组步长 (默认: 10%%)')
    
    args = parser.parse_args()
    
    # 确定输出目录
    csv_path = Path(args.csv_file)
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = csv_path.parent / 'simple_regression'
    
    print(f"\nInput: {csv_path}")
    print(f"Output: {output_dir}")
    print(f"Channels: {args.channel}")
    
    # 加载数据（仅做R²过滤，IQR过滤在分析时做）
    df_raw = pd.read_csv(csv_path)
    if 'r_squared' in df_raw.columns:
        n_before = len(df_raw)
        df_raw = df_raw[df_raw['r_squared'] >= args.min_rsq].copy()
        print(f"  R² filter (>= {args.min_rsq}): {n_before} -> {len(df_raw)} cells")
    
    # 百分位分组分析
    if args.percentile_bins:
        if args.percentile_bins == 'both':
            # 两个通道分别分析：488分组分析green，561分组分析red
            bin_channel_pairs = [('488', ['green']), ('561', ['red'])]
        else:
            # 单通道分组：自动匹配分析通道
            bin_ch = args.percentile_bins
            if bin_ch in ['488', 'green']:
                analysis_channels = ['green']
            elif bin_ch in ['561', 'red']:
                analysis_channels = ['red']
            else:
                analysis_channels = args.channel  # fallback
            bin_channel_pairs = [(bin_ch, analysis_channels)]
        
        for bin_ch, analysis_channels in bin_channel_pairs:
            analyzer = PercentileBinRegressionAnalyzer(
                output_dir=str(output_dir),
                bin_channel=bin_ch,
                bin_step=args.bin_step,
                min_r_squared=args.min_rsq
            )
            analyzer.run_binned_analysis(df_raw, analysis_channels)
    
    else:
        # 常规分析（全部数据）
        analyzer = SimpleLinearRegressionAnalyzer(
            output_dir=str(output_dir),
            min_r_squared=args.min_rsq
        )
        
        # 加载并过滤数据
        df = analyzer.load_and_filter_data(str(csv_path))
        
        # 运行分析
        analyzer.run_analysis(df, args.channel)
    
    print("\n=== Analysis Complete ===")


if __name__ == '__main__':
    main()
