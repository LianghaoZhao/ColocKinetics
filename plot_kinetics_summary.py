#!/usr/bin/env python3
"""
Pearson 相关系数动力学汇总可视化脚本

生成所有细胞的 Pearson 相关系数随时间变化的汇总图，包括：
1. 原始值图：中位数轨迹 + IQR 阴影带
2. 归一化图：每个细胞按自身最小/最大值归一化后的中位数轨迹 + IQR 阴影带
3. 分 percentile 图：按指定通道荧光强度（488/561）分组，生成各组的动力学曲线大图

用法:
    python plot_kinetics_summary.py fitting1.csv
    python plot_kinetics_summary.py *_reaction_fitting_results.csv -o merged_output
    python plot_kinetics_summary.py *.csv --percentile-bins 488 --bin-step 10
"""

import argparse
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional

# 设置全局字体为 Arial
plt.rcParams['font.family'] = 'Arial'
# 设置字体类型为 TrueType，使文字以真实文本而非路径保存（便于编辑和搜索）
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def load_and_merge_fitting_data(fitting_files: List[str], min_r_squared: float = 0.9, 
                                  max_time: Optional[float] = None) -> pd.DataFrame:
    """
    加载并合并拟合结果数据，应用数据清洗
    
    Parameters:
    - fitting_files: reaction_fitting_results.csv 文件列表
    - min_r_squared: 最小 R² 阈值（默认 0.9）
    - max_time: 最大时间阈值，用于过滤 T90 > max_time 的细胞
    
    Returns:
    - 合并并清洗后的 DataFrame
    """
    dfs = []
    for f in fitting_files:
        df = pd.read_csv(f)
        # 添加来源文件标记（用于区分不同实验）
        df['source_file'] = Path(f).parent.name
        dfs.append(df)
    
    merged = pd.concat(dfs, ignore_index=True)
    print(f"  加载了 {len(merged)} 个细胞的拟合数据（来自 {len(fitting_files)} 个文件）")
    
    # === 数据清洗（强制执行）===
    n_before = len(merged)
    
    # 1. R² 过滤
    if 'correlation_r_squared' in merged.columns:
        merged = merged[merged['correlation_r_squared'] >= min_r_squared].copy()
        n_removed = n_before - len(merged)
        if n_removed > 0:
            print(f"  移除了 {n_removed} 个 R² < {min_r_squared} 的细胞")
    
    # 2. T90 过滤（如果提供了 max_time）
    if max_time is not None and 'correlation_t90' in merged.columns:
        n_before_t90 = len(merged)
        merged = merged[merged['correlation_t90'] <= max_time].copy()
        n_removed = n_before_t90 - len(merged)
        if n_removed > 0:
            print(f"  移除了 {n_removed} 个 T90 > {max_time}s 的细胞")
    
    # 3. 过滤无效参数
    required_cols = ['correlation_A0', 'correlation_k', 'correlation_A_inf']
    for col in required_cols:
        if col in merged.columns:
            merged = merged[~merged[col].isna()].copy()
    
    print(f"  清洗后剩余 {len(merged)} 个有效细胞")
    return merged


def load_and_merge_raw_data(raw_files: List[str]) -> pd.DataFrame:
    """
    加载并合并原始相关系数数据（用于背景散点）
    
    Parameters:
    - raw_files: correlation_analysis_results.csv 文件列表
    
    Returns:
    - 合并后的 DataFrame
    """
    dfs = []
    for f in raw_files:
        df = pd.read_csv(f)
        df['source_file'] = Path(f).parent.name
        dfs.append(df)
    
    merged = pd.concat(dfs, ignore_index=True)
    print(f"  加载了 {len(merged)} 个原始观测点（来自 {len(raw_files)} 个文件）")
    return merged


def reconstruct_curves_on_grid(fitting_df: pd.DataFrame, 
                                time_grid: np.ndarray,
                                use_delay: bool = False) -> np.ndarray:
    """
    根据拟合参数在标准时间网格上重建所有细胞的曲线
    
    Parameters:
    - fitting_df: 包含拟合参数的 DataFrame
    - time_grid: 标准化时间网格
    - use_delay: 是否使用延迟模型
    
    Returns:
    - 矩阵 [n_cells × n_timepoints]，每行是一个细胞的预测值
    """
    n_cells = len(fitting_df)
    n_times = len(time_grid)
    matrix = np.zeros((n_cells, n_times))
    
    for i, (_, row) in enumerate(fitting_df.iterrows()):
        A0 = row['correlation_A0']
        k = row['correlation_k']
        A_inf = row['correlation_A_inf']
        
        if use_delay and 'correlation_delay' in row and not pd.isna(row['correlation_delay']):
            delay = row['correlation_delay']
            # 延迟模型：t < delay 时保持 A0
            shifted_t = time_grid - delay
            y = np.where(shifted_t < 0, A0, A_inf + (A0 - A_inf) * np.exp(-k * shifted_t))
        else:
            # 标准一级反应
            y = A_inf + (A0 - A_inf) * np.exp(-k * time_grid)
        
        matrix[i, :] = y
    
    return matrix


def normalize_matrix_per_cell(matrix: np.ndarray) -> np.ndarray:
    """
    对每个细胞的曲线进行归一化（按该细胞的最小值和最大值）
    
    normalized = (value - min) / (max - min)
    
    Parameters:
    - matrix: [n_cells × n_timepoints] 原始矩阵
    
    Returns:
    - [n_cells × n_timepoints] 归一化矩阵，每行范围为 [0, 1]
    """
    n_cells = matrix.shape[0]
    normalized = np.zeros_like(matrix)
    
    for i in range(n_cells):
        row = matrix[i, :]
        row_min = np.nanmin(row)
        row_max = np.nanmax(row)
        
        if row_max - row_min > 1e-10:  # 避免除以零
            normalized[i, :] = (row - row_min) / (row_max - row_min)
        else:
            normalized[i, :] = 0.5  # 如果没有变化，设为0.5
    
    return normalized


def calculate_statistics(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算每个时间点的统计量
    
    Parameters:
    - matrix: [n_cells × n_timepoints] 矩阵
    
    Returns:
    - median: 中位数
    - q25: 25% 分位数
    - q75: 75% 分位数
    """
    median = np.nanmedian(matrix, axis=0)
    q25 = np.nanpercentile(matrix, 25, axis=0)
    q75 = np.nanpercentile(matrix, 75, axis=0)
    return median, q25, q75


def group_cells_by_percentile(fitting_df: pd.DataFrame, 
                               channel: str = '488',
                               bin_step: float = 10.0) -> List[Tuple[pd.DataFrame, str, int, int]]:
    """
    根据指定通道的荧光强度将细胞按 percentile 分组
    
    Parameters:
    - fitting_df: 拟合参数 DataFrame
    - channel: 用于分组的通道 ('488' 或 '561')
    - bin_step: 百分位步长（默认 10，即 0-10%, 10-20%, ...）
    
    Returns:
    - List of (group_df, label, pct_min, pct_max)
    """
    # 映射通道到列名
    channel_map = {
        '488': 'green',
        '561': 'red',
        'green': 'green',
        'red': 'red'
    }
    
    col_name = channel_map.get(channel.lower())
    if col_name is None:
        raise ValueError(f"不支持的通道: {channel}，请使用 488/561/green/red")
    
    if col_name not in fitting_df.columns:
        raise ValueError(f"数据中没有 {col_name} 列")
    
    # 获取有效数据（排除 NaN）
    valid_mask = ~fitting_df[col_name].isna()
    valid_df = fitting_df[valid_mask].copy()
    intensity_values = valid_df[col_name].values
    
    # 生成 percentile 边界
    bin_edges = np.arange(0, 100 + bin_step, bin_step)
    n_bins = len(bin_edges) - 1
    
    # 计算分组边界值
    percentile_thresholds = [np.percentile(intensity_values, p) for p in bin_edges]
    
    groups = []
    for i in range(n_bins):
        pct_min = int(bin_edges[i])
        pct_max = int(bin_edges[i + 1])
        threshold_low = percentile_thresholds[i]
        threshold_high = percentile_thresholds[i + 1]
        
        # 筛选该百分位范围内的细胞
        if i == n_bins - 1:  # 最后一组包含上边界
            mask = (intensity_values >= threshold_low) & (intensity_values <= threshold_high)
        else:
            mask = (intensity_values >= threshold_low) & (intensity_values < threshold_high)
        
        group_df = valid_df[mask].copy()
        
        if len(group_df) >= 3:  # 至少需要 3 个细胞才有意义
            label = f"{pct_min}-{pct_max}%"
            groups.append((group_df, label, pct_min, pct_max))
    
    return groups, col_name


def plot_kinetics_summary(fitting_df: pd.DataFrame, 
                           output_path: Path,
                           max_time: float = 300.0,
                           time_step: float = 1.0,
                           use_delay: bool = False,
                           raw_df: Optional[pd.DataFrame] = None,
                           max_scatter_points: Optional[int] = None):
    """
    生成 Pearson 相关系数动力学汇总图（原始值 + 归一化）
    
    Parameters:
    - fitting_df: 拟合参数 DataFrame
    - output_path: 输出目录
    - max_time: 时间网格最大值（秒）
    - time_step: 时间网格步长（秒）
    - use_delay: 是否使用延迟模型
    - raw_df: 原始数据 DataFrame（用于散点，可选）
    - max_scatter_points: 每个子图最大散点数（默认 None 不限制，超出则随机抽样）
    """
    # 1. 创建标准时间网格
    time_grid = np.arange(0, max_time + time_step, time_step)
    
    # 2. 重建曲线矩阵
    print(f"\n重建 {len(fitting_df)} 个细胞的曲线...")
    matrix = reconstruct_curves_on_grid(fitting_df, time_grid, use_delay)
    
    # 3. 计算归一化矩阵
    print("计算归一化曲线...")
    normalized_matrix = normalize_matrix_per_cell(matrix)
    
    # 4. 计算统计量
    median, q25, q75 = calculate_statistics(matrix)
    norm_median, norm_q25, norm_q75 = calculate_statistics(normalized_matrix)
    
    n_cells = len(fitting_df)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # ========== 图1: 原始值 ==========
    fig1, ax1 = plt.subplots(figsize=(4, 4))
    
    # 背景散点（原始数据，可选）
    if raw_df is not None and len(raw_df) > 0:
        scatter_data = raw_df[['time_point', 'pearson_corr']].dropna()
        # 随机抽样（如果指定了最大点数）
        if max_scatter_points is not None and len(scatter_data) > max_scatter_points:
            scatter_data = scatter_data.sample(n=max_scatter_points, random_state=42)
        ax1.scatter(scatter_data['time_point'], scatter_data['pearson_corr'],
                    c='lightgray', s=2.5, alpha=0.1,
                    edgecolors='none', zorder=1)
        print(f"  绘制了 {len(scatter_data)} 个散点")
    
    # IQR 阴影带
    ax1.fill_between(time_grid, q25, q75, 
                     color='steelblue', alpha=0.3, zorder=2)
    
    # 中位数实线
    ax1.plot(time_grid, median, 
             color='steelblue', linewidth=2.5, zorder=3)
    
    # 图表美化
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Pearson Correlation Coefficient', fontsize=12)
    ax1.set_title(f'Pearson Correlation Kinetics Summary\n(n = {n_cells} cells)', fontsize=14)
    ax1.set_xlim(0, max_time)
    ax1.set_ylim(-0.1, 1.1)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(axis='both', labelsize=10)
    
    # 保存图1
    fig1_png = output_path / 'pearson_kinetics_summary.png'
    fig1_pdf = output_path / 'pearson_kinetics_summary.pdf'
    fig1.savefig(fig1_png, dpi=300, bbox_inches='tight')
    fig1.savefig(fig1_pdf, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # ========== 图2: 归一化 ==========
    fig2, ax2 = plt.subplots(figsize=(4, 4))
    
    # IQR 阴影带
    ax2.fill_between(time_grid, norm_q25, norm_q75, 
                     color='steelblue', alpha=0.3)
    
    # 中位数实线
    ax2.plot(time_grid, norm_median, 
             color='steelblue', linewidth=2.5)
    
    # 图表美化
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Normalized Pearson Correlation', fontsize=12)
    ax2.set_title(f'Normalized Pearson Correlation Kinetics\n(n = {n_cells} cells, per-cell min-max normalization)', fontsize=14)
    ax2.set_xlim(0, max_time)
    ax2.set_ylim(-0.05, 1.05)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.tick_params(axis='both', labelsize=10)
    
    # 保存图2
    fig2_png = output_path / 'pearson_kinetics_normalized.png'
    fig2_pdf = output_path / 'pearson_kinetics_normalized.pdf'
    fig2.savefig(fig2_png, dpi=300, bbox_inches='tight')
    fig2.savefig(fig2_pdf, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    print(f"\n图表已保存:")
    print(f"  {fig1_png}")
    print(f"  {fig1_pdf}")
    print(f"  {fig2_png}")
    print(f"  {fig2_pdf}")
    
    # 导出统计数据
    stats_df = pd.DataFrame({
        'time': time_grid,
        'median': median,
        'q25': q25,
        'q75': q75,
        'norm_median': norm_median,
        'norm_q25': norm_q25,
        'norm_q75': norm_q75
    })
    stats_path = output_path / 'pearson_kinetics_statistics.csv'
    stats_df.to_csv(stats_path, index=False)
    print(f"  {stats_path}")


def calculate_delta_pearson_from_raw(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    从原始数据计算每个细胞的 delta Pearson
    
    delta = 最后时间点的 pearson_corr - 第一个时间点的 pearson_corr
    正值表示共定位增加，负值表示共定位减少
    
    Parameters:
    - raw_df: correlation_analysis_results.csv 的数据
    
    Returns:
    - DataFrame 包含 cell_id, file_path, start_pearson, end_pearson, delta_pearson
    """
    results = []
    
    # 按 file_path 和 cell_id 分组
    if 'file_path' in raw_df.columns:
        group_cols = ['file_path', 'cell_id']
    else:
        group_cols = ['cell_id']
    
    for group_key, group_data in raw_df.groupby(group_cols):
        # 按时间排序
        sorted_data = group_data.sort_values('time_point')
        
        # 获取第一个和最后一个时间点的 pearson
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


def plot_pearson_change_histogram(raw_df: pd.DataFrame, output_path: Path,
                                   fitting_df: Optional[pd.DataFrame] = None):
    """
    绘制 Pearson 变化值的直方图（不分 percentile）
    
    变化值 = 最后时间点的 pearson_corr - 第一个时间点的 pearson_corr
    正值表示共定位增加，负值表示共定位减少
    
    Parameters:
    - raw_df: 原始数据 DataFrame (correlation_analysis_results.csv)
    - output_path: 输出目录
    - fitting_df: 拟合结果 DataFrame（用于筛选有效细胞，可选）
    """
    print("\n绘制 Pearson 变化值直方图...")
    
    if raw_df is None or len(raw_df) == 0:
        print("  警告: 没有原始数据，跳过直方图")
        return
    
    # 从原始数据计算每个细胞的 delta pearson
    delta_df = calculate_delta_pearson_from_raw(raw_df)
    print(f"  计算了 {len(delta_df)} 个细胞的 delta Pearson")
    
    # 如果提供了 fitting_df，只保留有效拟合的细胞
    if fitting_df is not None and len(fitting_df) > 0:
        # 通过 file_path + cell_id 筛选
        if 'file_path' in fitting_df.columns and 'file_path' in delta_df.columns:
            fitting_df = fitting_df.copy()
            fitting_df['file_stem'] = fitting_df['file_path'].apply(lambda x: Path(x).stem)
            delta_df['file_stem'] = delta_df['file_path'].apply(lambda x: Path(x).stem if x else '')
            valid_keys = set(zip(fitting_df['file_stem'], fitting_df['cell_id']))
            delta_df = delta_df[delta_df.apply(lambda r: (r['file_stem'], r['cell_id']) in valid_keys, axis=1)]
        elif 'cell_id' in fitting_df.columns:
            valid_cell_ids = set(fitting_df['cell_id'])
            delta_df = delta_df[delta_df['cell_id'].isin(valid_cell_ids)]
        print(f"  筛选后剩余 {len(delta_df)} 个有效细胞")
    
    delta_pearson = delta_df['delta_pearson'].dropna()
    
    if len(delta_pearson) < 5:
        print("  警告: 有效数据不足，跳过直方图")
        return
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # 绘制直方图
    ax.hist(delta_pearson, bins=20, color='steelblue', alpha=0.7, edgecolor='white')
    
    # 添加中位数和均值线
    median_val = np.median(delta_pearson)
    mean_val = np.mean(delta_pearson)
    ax.axvline(x=median_val, color='#D55E00', linestyle='--', linewidth=2,
               label=f'Median: {median_val:.3f}')
    ax.axvline(x=mean_val, color='#E69F00', linestyle=':', linewidth=2,
               label=f'Mean: {mean_val:.3f}')
    
    # 图表美化
    ax.set_xlabel(r'$\Delta$Pearson (end - start)', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title(f'Pearson Correlation Change Distribution\n(n = {len(delta_pearson)} cells)', fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    # 保存
    fig_png = output_path / 'pearson_change_histogram.png'
    fig_pdf = output_path / 'pearson_change_histogram.pdf'
    fig.savefig(fig_png, dpi=300, bbox_inches='tight')
    fig.savefig(fig_pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  保存: {fig_png}")


def plot_pearson_change_histogram_by_percentile(fitting_df: pd.DataFrame,
                                                  output_path: Path,
                                                  raw_df: pd.DataFrame,
                                                  channel: str = '488',
                                                  bin_step: float = 10.0):
    """
    绘制分 percentile 的 Pearson 变化值直方图
    
    每个子图 4x4 寸，展示一个 percentile 组的变化值分布
    
    Parameters:
    - fitting_df: 拟合参数 DataFrame（含 green/red 列，用于 percentile 分组）
    - output_path: 输出目录
    - raw_df: 原始数据 DataFrame (correlation_analysis_results.csv)
    - channel: 用于分组的通道 ('488' 或 '561')
    - bin_step: 百分位步长
    """
    print(f"\n绘制分 percentile 的 Pearson 变化值直方图（按 {channel} 通道，{bin_step}% 步长）...")
    
    if raw_df is None or len(raw_df) == 0:
        print("  警告: 没有原始数据，跳过 percentile 直方图")
        return
    
    fitting_df = fitting_df.copy()
    
    # 检查 fitting_df 是否已经有 delta_pearson 列
    if 'delta_pearson' not in fitting_df.columns:
        # 从原始数据计算每个细胞的 delta pearson
        delta_df = calculate_delta_pearson_from_raw(raw_df)
        
        # 为 delta_df 添加 file_stem
        if 'file_path' in delta_df.columns:
            delta_df = delta_df.copy()
            delta_df['file_stem'] = delta_df['file_path'].apply(lambda x: Path(x).stem if x else '')
        
        # 确定合并键
        if 'file_stem' in fitting_df.columns and 'file_stem' in delta_df.columns:
            merge_keys = ['file_stem', 'cell_id']
        elif 'file_path' in fitting_df.columns:
            fitting_df['file_stem'] = fitting_df['file_path'].apply(lambda x: Path(x).stem)
            merge_keys = ['file_stem', 'cell_id']
        else:
            merge_keys = ['cell_id']
        
        fitting_df = fitting_df.merge(
            delta_df[merge_keys + ['delta_pearson']].drop_duplicates(subset=merge_keys), 
            on=merge_keys, 
            how='left'
        )
    
    # 检查合并结果
    if 'delta_pearson' not in fitting_df.columns:
        print("  警告: 没有 delta_pearson 列，跳过直方图")
        return
    
    n_valid = fitting_df['delta_pearson'].notna().sum()
    if n_valid == 0:
        print("  警告: 没有有效的 delta_pearson 数据")
        return
    
    # 分组
    groups, col_name = group_cells_by_percentile(fitting_df, channel, bin_step)
    n_groups = len(groups)
    
    if n_groups < 2:
        print("  警告: 有效分组数量不足，跳过 percentile 直方图")
        return
    
    print(f"  有效分组数: {n_groups}")
    
    # 计算子图布局
    n_cols = int(np.ceil(np.sqrt(n_groups)))
    n_rows = int(np.ceil(n_groups / n_cols))
    
    # 每个子图 4x4 寸
    subplot_size = 4
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * subplot_size, n_rows * subplot_size),
                              squeeze=False)
    
    # 字体参数
    title_fontsize = 10
    label_fontsize = 12
    tick_fontsize = 10
    color = 'steelblue'
    
    # 收集所有组的数据范围，用于统一 x 轴
    all_delta = []
    for group_df, _, _, _ in groups:
        delta = group_df['delta_pearson'].dropna()
        all_delta.extend(delta.tolist())
    
    if len(all_delta) == 0:
        print("  警告: 没有有效数据")
        return
    
    x_min = min(all_delta) - 0.05
    x_max = max(all_delta) + 0.05
    bins = np.linspace(x_min, x_max, 21)
    
    # 绘制每个分组
    for idx, (group_df, label, pct_min, pct_max) in enumerate(groups):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # 获取变化值
        delta_pearson = group_df['delta_pearson'].dropna()
        n_cells = len(delta_pearson)
        
        if n_cells < 3:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{label} (n={n_cells})', fontsize=title_fontsize, fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            continue
        
        # 绘制直方图
        ax.hist(delta_pearson, bins=bins, color=color, alpha=0.7, edgecolor='white')
        
        # 添加中位数线
        median_val = np.median(delta_pearson)
        ax.axvline(x=median_val, color='#D55E00', linestyle='--', linewidth=1.5,
                   label=f'Median: {median_val:.3f}')
        
        # 图表美化
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel(r'$\Delta$Pearson', fontsize=label_fontsize)
        ax.set_ylabel('Count', fontsize=label_fontsize)
        ax.set_title(f'{label} (n={n_cells})', fontsize=title_fontsize, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        ax.legend(loc='upper right', fontsize=8)
    
    # 隐藏多余的子图
    for idx in range(n_groups, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    # 添加总标题
    channel_display = '488nm (Green)' if channel.lower() in ['488', 'green'] else '561nm (Red)'
    fig.suptitle(f'Pearson Correlation Change Distribution by {channel_display} Intensity Percentile\n'
                 f'(Total n={len(fitting_df)}, {bin_step:.0f}% bins)',
                 fontsize=12, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # 保存
    fig_png = output_path / f'pearson_change_histogram_by_{channel}_percentile.png'
    fig_pdf = output_path / f'pearson_change_histogram_by_{channel}_percentile.pdf'
    fig.savefig(fig_png, dpi=300, bbox_inches='tight')
    fig.savefig(fig_pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  保存: {fig_png}")


def plot_kinetics_by_percentile_raw(fitting_df: pd.DataFrame,
                                     output_path: Path,
                                     channel: str = '488',
                                     bin_step: float = 10.0,
                                     max_time: float = 300.0,
                                     time_step: float = 1.0,
                                     use_delay: bool = False,
                                     raw_df: Optional[pd.DataFrame] = None,
                                     max_scatter_points: Optional[int] = None):
    """
    生成分 percentile 的 Pearson 相关系数动力学大图（原始值版本）
    
    按荧光强度百分位分组，每个子图 4x4 寸
    
    Parameters:
    - fitting_df: 拟合参数 DataFrame
    - output_path: 输出目录
    - channel: 用于分组的通道 ('488' 或 '561')
    - bin_step: 百分位步长
    - max_time: 时间网格最大值（秒）
    - time_step: 时间网格步长（秒）
    - use_delay: 是否使用延迟模型
    - raw_df: 原始数据 DataFrame（用于散点，可选）
    - max_scatter_points: 每个子图最大散点数（默认 None 不限制）
    """
    print(f"\n生成分 percentile 的原始值图（按 {channel} 通道荧光强度分组，{bin_step}% 步长）...")
    
    # 分组
    groups, col_name = group_cells_by_percentile(fitting_df, channel, bin_step)
    n_groups = len(groups)
    
    if n_groups < 2:
        print("  警告: 有效分组数量不足，跳过 percentile 分组图")
        return
    
    print(f"  有效分组数: {n_groups}")
    
    # 计算子图布局（尽量接近正方形）
    n_cols = int(np.ceil(np.sqrt(n_groups)))
    n_rows = int(np.ceil(n_groups / n_cols))
    
    # 创建时间网格
    time_grid = np.arange(0, max_time + time_step, time_step)
    
    # 每个子图 4x4 寸
    subplot_size = 4
    fig, axes = plt.subplots(n_rows, n_cols, 
                              figsize=(n_cols * subplot_size, n_rows * subplot_size),
                              squeeze=False)
    
    # 字体大小根据子图大小调整
    title_fontsize = 10
    label_fontsize = 9
    tick_fontsize = 8
    legend_fontsize = 7
    linewidth = 1.5
    scatter_size = 1.5
    
    # 统一颜色
    color = 'steelblue'
    
    # 绘制每个分组
    for idx, (group_df, label, pct_min, pct_max) in enumerate(groups):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        n_cells = len(group_df)
        
        # 重建该组的曲线矩阵
        matrix = reconstruct_curves_on_grid(group_df, time_grid, use_delay)
        median, q25, q75 = calculate_statistics(matrix)
        
        # 背景散点（如果有原始数据）
        if raw_df is not None and len(raw_df) > 0:
            # 筛选属于该组的原始数据点
            group_cell_ids = set(group_df['cell_id'].values) if 'cell_id' in group_df.columns else set()
            if group_cell_ids and 'cell_id' in raw_df.columns:
                group_raw = raw_df[raw_df['cell_id'].isin(group_cell_ids)]
                scatter_data = group_raw[['time_point', 'pearson_corr']].dropna()
                # 随机抽样（如果指定了最大点数）
                if max_scatter_points is not None and len(scatter_data) > max_scatter_points:
                    scatter_data = scatter_data.sample(n=max_scatter_points, random_state=42 + idx)
                if len(scatter_data) > 0:
                    ax.scatter(scatter_data['time_point'], scatter_data['pearson_corr'],
                              c='lightgray', s=scatter_size, alpha=0.1,
                              edgecolors='none', zorder=1)
        
        # IQR 阴影带
        ax.fill_between(time_grid, q25, q75, 
                        color=color, alpha=0.3, zorder=2)
        
        # 中位数实线
        ax.plot(time_grid, median, 
                color=color, linewidth=linewidth, zorder=3)
        
        # 图表美化
        ax.set_xlabel('Time (s)', fontsize=label_fontsize)
        ax.set_ylabel('Pearson r', fontsize=label_fontsize)
        ax.set_title(f'{label} (n={n_cells})', fontsize=title_fontsize, fontweight='bold')
        ax.set_xlim(0, max_time)
        ax.set_ylim(-0.1, 1.1)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.4, linewidth=0.5)
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.4, linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        ax.locator_params(axis='both', nbins=5)
    
    # 隐藏多余的子图
    for idx in range(n_groups, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    # 添加总标题
    channel_display = '488nm (Green)' if channel.lower() in ['488', 'green'] else '561nm (Red)'
    fig.suptitle(f'Pearson Correlation Kinetics by {channel_display} Intensity Percentile\n'
                 f'(Total n={len(fitting_df)}, {bin_step:.0f}% bins)',
                 fontsize=12, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # 保存
    fig_png = output_path / f'pearson_kinetics_by_{channel}_percentile.png'
    fig_pdf = output_path / f'pearson_kinetics_by_{channel}_percentile.pdf'
    fig.savefig(fig_png, dpi=300, bbox_inches='tight')
    fig.savefig(fig_pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  保存: {fig_png}")
    print(f"  保存: {fig_pdf}")


def plot_kinetics_by_percentile_normalized(fitting_df: pd.DataFrame,
                                            output_path: Path,
                                            channel: str = '488',
                                            bin_step: float = 10.0,
                                            max_time: float = 300.0,
                                            time_step: float = 1.0,
                                            use_delay: bool = False):
    """
    生成分 percentile 的归一化 Pearson 相关系数动力学大图
    
    按荧光强度百分位分组，每个子图 4x4 寸
    
    Parameters:
    - fitting_df: 拟合参数 DataFrame
    - output_path: 输出目录
    - channel: 用于分组的通道 ('488' 或 '561')
    - bin_step: 百分位步长
    - max_time: 时间网格最大值（秒）
    - time_step: 时间网格步长（秒）
    - use_delay: 是否使用延迟模型
    """
    print(f"\n生成分 percentile 的归一化图（按 {channel} 通道荧光强度分组，{bin_step}% 步长）...")
    
    # 分组
    groups, col_name = group_cells_by_percentile(fitting_df, channel, bin_step)
    n_groups = len(groups)
    
    if n_groups < 2:
        print("  警告: 有效分组数量不足，跳过 percentile 分组图")
        return
    
    print(f"  有效分组数: {n_groups}")
    
    # 计算子图布局
    n_cols = int(np.ceil(np.sqrt(n_groups)))
    n_rows = int(np.ceil(n_groups / n_cols))
    
    # 创建时间网格
    time_grid = np.arange(0, max_time + time_step, time_step)
    
    # 每个子图 4x4 寸
    subplot_size = 4
    fig, axes = plt.subplots(n_rows, n_cols, 
                              figsize=(n_cols * subplot_size, n_rows * subplot_size),
                              squeeze=False)
    
    # 字体大小根据子图大小调整
    title_fontsize = 10
    label_fontsize = 9
    tick_fontsize = 8
    linewidth = 1.5
    
    # 统一颜色
    color = 'steelblue'
    
    # 绘制每个分组
    for idx, (group_df, label, pct_min, pct_max) in enumerate(groups):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        n_cells = len(group_df)
        
        # 重建该组的曲线矩阵并归一化
        matrix = reconstruct_curves_on_grid(group_df, time_grid, use_delay)
        normalized_matrix = normalize_matrix_per_cell(matrix)
        median, q25, q75 = calculate_statistics(normalized_matrix)
        
        # IQR 阴影带
        ax.fill_between(time_grid, q25, q75, 
                        color=color, alpha=0.3)
        
        # 中位数实线
        ax.plot(time_grid, median, 
                color=color, linewidth=linewidth)
        
        # 图表美化
        ax.set_xlabel('Time (s)', fontsize=label_fontsize)
        ax.set_ylabel('Normalized r', fontsize=label_fontsize)
        ax.set_title(f'{label} (n={n_cells})', fontsize=title_fontsize, fontweight='bold')
        ax.set_xlim(0, max_time)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.4, linewidth=0.5)
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.4, linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        ax.locator_params(axis='both', nbins=5)
    
    # 隐藏多余的子图
    for idx in range(n_groups, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    # 添加总标题
    channel_display = '488nm (Green)' if channel.lower() in ['488', 'green'] else '561nm (Red)'
    fig.suptitle(f'Normalized Pearson Correlation Kinetics by {channel_display} Intensity Percentile\n'
                 f'(Total n={len(fitting_df)}, {bin_step:.0f}% bins, per-cell min-max normalization)',
                 fontsize=12, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # 保存
    fig_png = output_path / f'pearson_kinetics_normalized_by_{channel}_percentile.png'
    fig_pdf = output_path / f'pearson_kinetics_normalized_by_{channel}_percentile.pdf'
    fig.savefig(fig_png, dpi=300, bbox_inches='tight')
    fig.savefig(fig_pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  保存: {fig_png}")
    print(f"  保存: {fig_pdf}")


def plot_individual_cell_curves_by_percentile(fitting_df: pd.DataFrame,
                                               output_path: Path,
                                               channel: str = '488',
                                               bin_step: float = 10.0,
                                               max_time: float = 300.0,
                                               time_step: float = 1.0,
                                               use_delay: bool = False,
                                               cells_per_page: int = 30,
                                               raw_df: Optional[pd.DataFrame] = None):
    """
    绘制每个 percentile 组内每个细胞的拟合曲线
    
    每 cells_per_page 个细胞输出一张大图，布局 5×6
    所有大图共享相同的 xy 坐标轴范围
    
    每个小图包含：
    - 原始数据散点 + 折线
    - 拟合曲线
    - T50、T90 标记（水平线 + 垂直线 + 数值）
    
    Parameters:
    - fitting_df: 拟合参数 DataFrame（需包含 green/red 列）
    - output_path: 输出目录
    - channel: 用于分组的通道 ('488' 或 '561')
    - bin_step: 百分位步长
    - max_time: 时间网格最大值（秒）
    - time_step: 时间网格步长（秒）
    - use_delay: 是否使用延迟模型
    - cells_per_page: 每页细胞数（默认 30）
    - raw_df: 原始数据 DataFrame（包含 time_point, pearson_corr）
    """
    print(f"\n绘制每个细胞的拟合曲线（按 {channel} 通道分 percentile，{bin_step}% 步长）...")
    
    # 分组
    groups, col_name = group_cells_by_percentile(fitting_df, channel, bin_step)
    n_groups = len(groups)
    
    if n_groups < 1:
        print("  警告: 没有有效分组")
        return
    
    # 创建输出子目录
    cells_output_dir = output_path / f"individual_cells_by_{channel}_percentile"
    cells_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建时间网格
    time_grid = np.arange(0, max_time + time_step, time_step)
    
    # 统一的坐标轴范围
    x_range = (0, max_time)
    y_range = (-0.1, 1.1)
    
    # 布局：5行×6列 = 30 个子图
    n_rows, n_cols = 5, 6
    subplot_size = 4
    
    # 字体和线条参数（参考 plotting.py 的动力学拟合样式）
    title_fontsize = 9
    label_fontsize = 16
    tick_fontsize = 14
    annotation_fontsize = 14
    fit_linewidth = 2        # plotting.py: linewidth=2
    raw_linewidth = 2        # plotting.py: linewidth=2
    marker_size = 6          # plotting.py: markersize=6
    color_raw = 'C0'         # plotting.py: color='C0'
    color_fit = 'red'        # plotting.py: color='red'
    color_t50 = 'orange'     # plotting.py: color='orange'
    color_t90 = 'purple'     # plotting.py: color='purple'
    
    # 预处理原始数据（如果有）
    raw_data_dict = {}
    if raw_df is not None and len(raw_df) > 0:
        # 为 raw_df 添加 file_stem
        if 'file_stem' not in raw_df.columns and 'file_path' in raw_df.columns:
            raw_df = raw_df.copy()
            raw_df['file_stem'] = raw_df['file_path'].apply(lambda x: Path(x).stem)
        
        # 按 file_stem 和 cell_id 分组
        if 'file_stem' in raw_df.columns and 'cell_id' in raw_df.columns:
            for (fs, cid), group in raw_df.groupby(['file_stem', 'cell_id']):
                raw_data_dict[(fs, cid)] = group[['time_point', 'pearson_corr']].dropna().sort_values('time_point')
    
    has_raw_data = len(raw_data_dict) > 0
    if has_raw_data:
        print(f"  加载了 {len(raw_data_dict)} 个细胞的原始数据")
    else:
        print("  未提供原始数据，仅绘制拟合曲线")
    
    # 遍历每个 percentile 组
    for group_idx, (group_df, label, pct_min, pct_max) in enumerate(groups):
        n_cells = len(group_df)
        n_pages = int(np.ceil(n_cells / cells_per_page))
        
        print(f"  {label}: {n_cells} 个细胞，输出 {n_pages} 页")
        
        # 遍历每页
        for page_idx in range(n_pages):
            start_idx = page_idx * cells_per_page
            end_idx = min(start_idx + cells_per_page, n_cells)
            page_cells = group_df.iloc[start_idx:end_idx]
            n_cells_on_page = len(page_cells)
            
            # 创建图形
            fig, axes = plt.subplots(n_rows, n_cols,
                                      figsize=(n_cols * subplot_size, n_rows * subplot_size),
                                      squeeze=False)
            
            # 绘制每个细胞
            for cell_idx, (_, row) in enumerate(page_cells.iterrows()):
                ax_row = cell_idx // n_cols
                ax_col = cell_idx % n_cols
                ax = axes[ax_row, ax_col]
                
                # 获取拟合参数
                A0 = row['correlation_A0']
                k = row['correlation_k']
                A_inf = row['correlation_A_inf']
                t50 = row.get('correlation_t50', np.nan)
                t90 = row.get('correlation_t90', np.nan)
                
                # 获取细胞标识
                cell_id = row.get('cell_id', '')
                file_stem = row.get('file_stem', '')
                if not file_stem and 'file_path' in row:
                    file_stem = Path(row['file_path']).stem
                
                # 1. 绘制原始数据散点 + 折线（如果有）
                if has_raw_data and (file_stem, cell_id) in raw_data_dict:
                    cell_raw = raw_data_dict[(file_stem, cell_id)]
                    t_raw = cell_raw['time_point'].values
                    y_raw = cell_raw['pearson_corr'].values
                    # 散点 + 折线（参考 plotting.py 的 'o-' 样式）
                    ax.plot(t_raw, y_raw, 'o-', color=color_raw, 
                           linewidth=raw_linewidth, markersize=marker_size, 
                           label='Correlation', zorder=1)
                
                # 2. 绘制拟合曲线（参考 plotting.py: 红色虚线）
                if use_delay and 'correlation_delay' in row and not pd.isna(row['correlation_delay']):
                    delay = row['correlation_delay']
                    shifted_t = time_grid - delay
                    y_fit = np.where(shifted_t < 0, A0, A_inf + (A0 - A_inf) * np.exp(-k * shifted_t))
                else:
                    y_fit = A_inf + (A0 - A_inf) * np.exp(-k * time_grid)
                
                ax.plot(time_grid, y_fit, '--', color=color_fit, linewidth=fit_linewidth, 
                       label='Fitted curve', zorder=3)
                
                # 3. 绘制 T50 标记（参考 plotting.py: 橙色虚线，只有垂直线）
                if not np.isnan(t50) and t50 <= max_time:
                    # 垂直线
                    ax.axvline(x=t50, color=color_t50, linestyle=':', linewidth=2,
                              label=f't50: {t50:.1f}s', zorder=4)
                    # 标注
                    ax.annotate(f'T50={t50:.1f}s', xy=(t50, y_range[1] * 0.9), 
                               xytext=(5, 0), textcoords='offset points',
                               fontsize=annotation_fontsize, color=color_t50, fontweight='bold')
                
                # 4. 绘制 T90 标记（参考 plotting.py: 紫色虚线，只有垂直线）
                if not np.isnan(t90) and t90 <= max_time:
                    # 垂直线
                    ax.axvline(x=t90, color=color_t90, linestyle=':', linewidth=2,
                              label=f't90: {t90:.1f}s', zorder=4)
                    # 标注
                    ax.annotate(f'T90={t90:.1f}s', xy=(t90, y_range[1] * 0.8),
                               xytext=(5, 0), textcoords='offset points',
                               fontsize=annotation_fontsize, color=color_t90, fontweight='bold')
                
                # 图表美化
                ax.set_xlim(x_range)
                ax.set_ylim(y_range)
                ax.set_xlabel('Time (s)', fontsize=label_fontsize)
                ax.set_ylabel('Pearson r', fontsize=label_fontsize)
                ax.set_title(f'{file_stem}\nCell {cell_id}', fontsize=title_fontsize)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.tick_params(axis='both', labelsize=tick_fontsize)
                ax.locator_params(axis='both', nbins=4)
            
            # 隐藏多余的子图
            for idx in range(n_cells_on_page, n_rows * n_cols):
                ax_row = idx // n_cols
                ax_col = idx % n_cols
                axes[ax_row, ax_col].set_visible(False)
            
            # 添加总标题
            channel_display = '488nm (Green)' if channel.lower() in ['488', 'green'] else '561nm (Red)'
            page_info = f"Page {page_idx + 1}/{n_pages}" if n_pages > 1 else ""
            fig.suptitle(f'Individual Cell Curves - {label} ({channel_display} Intensity Percentile)\n'
                         f'Cells {start_idx + 1}-{end_idx} of {n_cells} {page_info}',
                         fontsize=12, fontweight='bold', y=1.01)
            
            plt.tight_layout()
            
            # 保存
            page_suffix = f"_page{page_idx + 1}" if n_pages > 1 else ""
            pct_label = f"{pct_min:02d}-{pct_max:02d}pct"
            fig_png = cells_output_dir / f"cells_{pct_label}{page_suffix}.png"
            fig_pdf = cells_output_dir / f"cells_{pct_label}{page_suffix}.pdf"
            fig.savefig(fig_png, dpi=300, bbox_inches='tight')
            fig.savefig(fig_pdf, dpi=300, bbox_inches='tight')
            plt.close(fig)
    
    print(f"  输出目录: {cells_output_dir}")


def classify_files(files: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    从输入文件中分类拟合结果文件、原始数据文件和荧光强度数据文件
    
    支持带前缀/后缀的文件名，通过关键词匹配：
    - *reaction_fitting_results* -> 拟合结果
    - *correlation_analysis_results* -> 原始数据
    - *ratio_t50_raw_data* -> 荧光强度数据
    
    Returns:
    - fitting_files: 拟合结果文件列表
    - raw_files: 原始数据文件列表
    - intensity_files: 荧光强度数据文件列表
    """
    fitting_files = []
    raw_files = []
    intensity_files = []
    
    for f in files:
        fname = Path(f).name.lower()
        
        # 通过关键词匹配（支持前缀/后缀）
        if 'reaction_fitting_results' in fname:
            fitting_files.append(f)
        elif 'correlation_analysis_results' in fname:
            raw_files.append(f)
        elif 'ratio_t50_raw_data' in fname:
            intensity_files.append(f)
        else:
            print(f"  跳过: {fname}（未匹配关键词）")
    
    return fitting_files, raw_files, intensity_files


def load_and_merge_intensity_data(intensity_files: List[str]) -> pd.DataFrame:
    """
    加载并合并荧光强度数据（用于 percentile 分组）
    
    Parameters:
    - intensity_files: ratio_t50_raw_data.csv 文件列表
    
    Returns:
    - 合并后的 DataFrame，包含 green, red, t50, file_stem, cell_id 等列
    """
    dfs = []
    for f in intensity_files:
        df = pd.read_csv(f)
        df['source_file'] = Path(f).parent.name
        dfs.append(df)
    
    merged = pd.concat(dfs, ignore_index=True)
    print(f"  加载了 {len(merged)} 个细胞的荧光强度数据（来自 {len(intensity_files)} 个文件）")
    return merged


def merge_fitting_with_intensity(fitting_df: pd.DataFrame, intensity_df: pd.DataFrame) -> pd.DataFrame:
    """
    将拟合结果与荧光强度数据合并
    
    通过 cell_id 和 file_stem/file_path 关联
    
    Parameters:
    - fitting_df: 拟合结果 DataFrame
    - intensity_df: 荧光强度 DataFrame
    
    Returns:
    - 合并后的 DataFrame
    """
    # 从 fitting_df 的 file_path 提取 file_stem
    fitting_df = fitting_df.copy()
    fitting_df['file_stem'] = fitting_df['file_path'].apply(lambda x: Path(x).stem)
    
    # 确保 intensity_df 有 file_stem 列
    intensity_df = intensity_df.copy()
    if 'file_stem' not in intensity_df.columns and 'file_path' in intensity_df.columns:
        intensity_df['file_stem'] = intensity_df['file_path'].apply(lambda x: Path(x).stem)
    
    # 只保留需要的列
    intensity_cols = ['file_stem', 'cell_id', 'green', 'red']
    if 'ratio' in intensity_df.columns:
        intensity_cols.append('ratio')
    intensity_subset = intensity_df[intensity_cols].drop_duplicates()
    
    # 合并
    merged = fitting_df.merge(intensity_subset, on=['file_stem', 'cell_id'], how='left')
    
    # 统计合并结果
    n_with_intensity = merged['green'].notna().sum()
    print(f"  合并后 {n_with_intensity}/{len(merged)} 个细胞有荧光强度数据")
    
    return merged


def main():
    parser = argparse.ArgumentParser(
        description='生成 Pearson 相关系数动力学汇总图（原始值 + 归一化）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单个实验
  python plot_kinetics_summary.py results/reaction_fitting_results.csv

  # 多个实验合并（通配符）
  python plot_kinetics_summary.py *_reaction_fitting_results.csv -o merged_output

  # 指定参数
  python plot_kinetics_summary.py *.csv -o merged_output --max-time 400 --min-rsq 0.95

  # 按 488nm 通道荧光强度分 percentile 绘图（默认每 10% 一组）
  python plot_kinetics_summary.py *.csv --percentile-bins 488

  # 按 561nm 通道荧光强度分 percentile 绘图，每 20% 一组
  python plot_kinetics_summary.py *.csv --percentile-bins 561 --bin-step 20

  # 两个通道都分析
  python plot_kinetics_summary.py *.csv --percentile-bins both --bin-step 10
        """
    )
    
    parser.add_argument('csv_files', nargs='+',
                        help='CSV 文件（自动筛选 fitting 文件）')
    parser.add_argument('-o', '--output', type=str, default='kinetics_summary_output',
                        help='输出目录（默认: kinetics_summary_output）')
    parser.add_argument('--max-time', type=float, default=300.0,
                        help='时间网格最大值（秒，默认: 300）')
    parser.add_argument('--time-step', type=float, default=1.0,
                        help='时间网格步长（秒，默认: 1）')
    parser.add_argument('--min-rsq', type=float, default=0.9,
                        help='最小 R² 阈值（默认: 0.9）')
    parser.add_argument('--t90-max', type=float, default=None,
                        help='T90 最大值阈值（秒），超过此值的细胞将被排除')
    parser.add_argument('--min-delta-pearson', type=float, default=None,
                        help='最小 delta Pearson 阈值（绝对值），过滤变化幅度太小的细胞')
    parser.add_argument('--use-delay', action='store_true',
                        help='使用延迟模型（如果数据包含 delay 参数）')
    parser.add_argument('--max-scatter-points', type=int, default=None,
                        help='每个子图最大散点数，超出则随机抽样（默认: 不限制）')
    parser.add_argument('--percentile-bins', type=str, default=None,
                        help='按指定通道荧光强度分 percentile 绘图（488/561/both）')
    parser.add_argument('--bin-step', type=float, default=10.0,
                        help='Percentile 分组步长（默认: 10，即 0-10%%, 10-20%%, ...）')
    parser.add_argument('--plot-individual-cells', action='store_true',
                        help='绘制每个细胞的拟合曲线（需要 --percentile-bins）')
    parser.add_argument('--cells-per-page', type=int, default=30,
                        help='每页显示的细胞数（默认: 30）')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Pearson Correlation Kinetics Summary Plot")
    print("=" * 60)
    
    # Windows 不会自动展开通配符，需要在 Python 内部处理
    expanded_files = []
    for pattern in args.csv_files:
        if '*' in pattern or '?' in pattern:
            matched = glob.glob(pattern)
            if matched:
                expanded_files.extend(matched)
            else:
                print(f"Warning: 通配符 '{pattern}' 未匹配到任何文件")
        else:
            expanded_files.append(pattern)
    
    # 去重
    expanded_files = list(set(expanded_files))
    
    if not expanded_files:
        print("Error: 未找到任何匹配的 CSV 文件")
        return
    
    # 分类文件
    print(f"\n识别输入文件...")
    print(f"  找到 {len(expanded_files)} 个 CSV 文件")
    fitting_files, raw_files, intensity_files = classify_files(expanded_files)
    
    if not fitting_files:
        print("Error: 未找到拟合结果文件（包含 'reaction_fitting_results' 关键词）")
        return
    
    print(f"\n拟合文件 ({len(fitting_files)}):")
    for f in fitting_files:
        print(f"  - {f}")
    
    if raw_files:
        print(f"\n原始数据文件 ({len(raw_files)}):")
        for f in raw_files:
            print(f"  - {f}")
    else:
        print("\nWarning: 未找到原始数据文件，无法显示散点")
    
    if intensity_files:
        print(f"\n荧光强度文件 ({len(intensity_files)}):")
        for f in intensity_files:
            print(f"  - {f}")
    elif args.percentile_bins:
        print("\nWarning: 未找到荧光强度文件（包含 'ratio_t50_raw_data' 关键词），无法进行 percentile 分组")
    
    # 加载原始数据（先加载，用于计算 delta pearson）
    raw_df = None
    if raw_files:
        print(f"\n加载原始数据...")
        raw_df = load_and_merge_raw_data(raw_files)
    
    # 加载拟合数据（基本清洗）
    print(f"\n加载拟合数据...")
    fitting_df = load_and_merge_fitting_data(
        fitting_files, 
        min_r_squared=args.min_rsq,
        max_time=args.t90_max
    )
    
    # Delta Pearson 过滤（基于原始数据计算，在其他过滤之后）
    if args.min_delta_pearson is not None and raw_df is not None:
        print(f"\n计算并过滤 delta Pearson...")
        delta_df = calculate_delta_pearson_from_raw(raw_df)
        print(f"  计算了 {len(delta_df)} 个细胞的 delta Pearson")
        
        # 为 fitting_df 添加 file_stem
        fitting_df = fitting_df.copy()
        if 'file_stem' not in fitting_df.columns and 'file_path' in fitting_df.columns:
            fitting_df['file_stem'] = fitting_df['file_path'].apply(lambda x: Path(x).stem)
        
        # 为 delta_df 添加 file_stem
        if 'file_stem' not in delta_df.columns and 'file_path' in delta_df.columns:
            delta_df['file_stem'] = delta_df['file_path'].apply(lambda x: Path(x).stem if x else '')
        
        # 合并 delta_pearson 到 fitting_df
        if 'file_stem' in fitting_df.columns and 'file_stem' in delta_df.columns:
            fitting_df = fitting_df.merge(
                delta_df[['file_stem', 'cell_id', 'delta_pearson']], 
                on=['file_stem', 'cell_id'], 
                how='left'
            )
        else:
            fitting_df = fitting_df.merge(
                delta_df[['cell_id', 'delta_pearson']], 
                on=['cell_id'], 
                how='left'
            )
        
        # 过滤（只保留正值且 >= 阈值，即共定位增加的细胞）
        n_before = len(fitting_df)
        fitting_df = fitting_df[fitting_df['delta_pearson'] >= args.min_delta_pearson].copy()
        n_removed = n_before - len(fitting_df)
        if n_removed > 0:
            print(f"  移除了 {n_removed} 个 delta Pearson < {args.min_delta_pearson} 的细胞（含负值）")
        print(f"  过滤后剩余 {len(fitting_df)} 个细胞")
    
    if len(fitting_df) == 0:
        print("\nError: 清洗后没有有效细胞数据")
        return
    
    # 生成图表
    output_path = Path(args.output)
    plot_kinetics_summary(
        fitting_df=fitting_df,
        output_path=output_path,
        max_time=args.max_time,
        time_step=args.time_step,
        use_delay=args.use_delay,
        raw_df=raw_df,
        max_scatter_points=args.max_scatter_points
    )
    
    # Pearson 变化值直方图（不分 percentile）
    plot_pearson_change_histogram(raw_df=raw_df, output_path=output_path, fitting_df=fitting_df)
    
    # 分 percentile 绘图（如果指定）
    if args.percentile_bins:
        if not intensity_files:
            print("\nError: percentile 分组需要荧光强度数据（*ratio_t50_raw_data*.csv）")
        else:
            # 加载荧光强度数据
            print(f"\n加载荧光强度数据...")
            intensity_df = load_and_merge_intensity_data(intensity_files)
            
            # 合并拟合结果与荧光强度数据
            print(f"\n合并数据...")
            fitting_df_with_intensity = merge_fitting_with_intensity(fitting_df, intensity_df)
            
            # 确定要分析的通道
            channels = ['488', '561'] if args.percentile_bins.lower() == 'both' else [args.percentile_bins]
            
            for channel in channels:
                try:
                    # 原始值版本
                    plot_kinetics_by_percentile_raw(
                        fitting_df=fitting_df_with_intensity,
                        output_path=output_path,
                        channel=channel,
                        bin_step=args.bin_step,
                        max_time=args.max_time,
                        time_step=args.time_step,
                        use_delay=args.use_delay,
                        raw_df=raw_df,
                        max_scatter_points=args.max_scatter_points
                    )
                    
                    # 归一化版本
                    plot_kinetics_by_percentile_normalized(
                        fitting_df=fitting_df_with_intensity,
                        output_path=output_path,
                        channel=channel,
                        bin_step=args.bin_step,
                        max_time=args.max_time,
                        time_step=args.time_step,
                        use_delay=args.use_delay
                    )
                    
                    # Pearson 变化值直方图（分 percentile）
                    plot_pearson_change_histogram_by_percentile(
                        fitting_df=fitting_df_with_intensity,
                        output_path=output_path,
                        raw_df=raw_df,
                        channel=channel,
                        bin_step=args.bin_step
                    )
                    
                    # 每个细胞的拟合曲线（如果指定）
                    if args.plot_individual_cells:
                        plot_individual_cell_curves_by_percentile(
                            fitting_df=fitting_df_with_intensity,
                            output_path=output_path,
                            channel=channel,
                            bin_step=args.bin_step,
                            max_time=args.max_time,
                            time_step=args.time_step,
                            use_delay=args.use_delay,
                            cells_per_page=args.cells_per_page,
                            raw_df=raw_df
                        )
                except ValueError as e:
                    print(f"  警告: {e}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
