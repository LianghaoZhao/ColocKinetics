import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
from multiprocessing import Pool
from analyzer import FileData
from reaction_kinetics import first_order_reaction, delayed_first_order_reaction, ReactionFitter
from coloc_metrics import CoLocalizationMetrics
from typing import Tuple, List, Dict
from scipy import stats
from sklearn.mixture import GaussianMixture

class Visualizer:
    """处理可视化结果的类"""
    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def generate_visualizations(self, analyses: list, fit_model: str, include_individual_plots: bool, include_scatter: bool):
        """生成所有可视化图表"""
        if not analyses:
            print("No analyses to visualize.")
            return

        # 1. 为每个文件中所有细胞绘制一个大图 (相关系数随时间变化)
        print("Generating summary plots (all cells per file)...")
        pbar_viz = tqdm(total=len(analyses), desc="Generating summary plots", unit="file")
        for analysis in analyses:
            self._plot_all_cells_summary(analysis, fit_model)
            pbar_viz.update(1)
        pbar_viz.close()

        # 2. 为每个细胞单独保存分析图 (如果用户选择)
        if include_individual_plots:
            print("Generating individual cell plots...")
            total_individual_plots = sum(len(analysis.cells.keys()) for analysis in analyses)
            pbar_indiv = tqdm(total=total_individual_plots, desc="Generating individual plots", unit="plot")
            for analysis in analyses:
                for cell_id in analysis.cells.keys():
                    self._plot_single_cell_detailed(analysis, cell_id, fit_model, include_scatter)
                    pbar_indiv.update(1)
            pbar_indiv.close()

    def _plot_all_cells_summary(self, analysis: FileData, fit_model: str):
        """为单个文件的所有细胞生成汇总图（相关系数随时间变化）"""
        original_filename = Path(analysis.file_path).stem
        all_cell_ids = list(analysis.cells.keys())
        n_cells = len(all_cell_ids)
        if n_cells == 0:
            return

        # 计算子图布局 (5列)
        n_cols = 5
        n_rows = (n_cells + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_cols > 1 else [axes]
        elif n_cols == 1:
            axes = axes if n_rows > 1 else [axes]
        else:
            axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

        # 为每个细胞绘制相关系数随时间变化的图
        for idx, cell_id in enumerate(all_cell_ids):
            time_points, correlations, p_values = CoLocalizationMetrics.get_correlation_over_time_of_a_cell(analysis, cell_id)
            # Get fit results for this cell
            # We need to calculate metrics and then fit for the specific cell to get results
            # This requires calling the kinetics analyzer logic or having fit results available
            # For simplicity in this module, let's assume we can call the analyzer's fit method on single data
            # Or we re-implement the fitting call here based on the cell's data
            cell_metrics = CoLocalizationMetrics.calculate_metrics_for_file(analysis)[cell_id]
            fit_results = ReactionFitter.fit_first_order_reaction if fit_model == 'first_order' else ReactionFitter.fit_delayed_first_order_reaction
            # This is complex, as fit_results is from the KineticsAnalyzer. We need to get it.
            # A better way is to pass fit results from the main process or re-calculate here.
            # Let's re-calculate the fit for the cell within the plotting function.
            # Calculate fit
            valid_corr_mask = ~np.isnan(correlations)
            if np.sum(valid_corr_mask) >= 3:
                 fit_results = ReactionFitter.fit_first_order_reaction(time_points[valid_corr_mask], correlations[valid_corr_mask]) if fit_model == 'first_order' else ReactionFitter.fit_delayed_first_order_reaction(time_points[valid_corr_mask], correlations[valid_corr_mask])
            else:
                 fit_results = {
                     'A0': np.nan, 'k': np.nan, 'A_inf': np.nan, 't50': np.nan, 't90': np.nan, 'r_squared': np.nan,
                     'delay': np.nan if fit_model == 'delayed_first_order' else None
                 }

            ax = axes[idx]
            ax.plot(time_points, correlations, 'o-', label='Correlation', linewidth=2, markersize=6)

            # 选择拟合函数
            fit_func = delayed_first_order_reaction if fit_model == 'delayed_first_order' else first_order_reaction

            # 绘制拟合曲线
            if not np.isnan(fit_results['k']) and fit_results['k'] > 0:
                t_fit = np.linspace(time_points.min(), time_points.max(), 100)
                if fit_model == 'delayed_first_order':
                    y_fit = fit_func(t_fit, fit_results['A0'],
                                    fit_results['k'], fit_results['A_inf'],
                                    fit_results.get('delay', 0))
                else:
                    y_fit = fit_func(t_fit, fit_results['A0'],
                                    fit_results['k'], fit_results['A_inf'])
                ax.plot(t_fit, y_fit, '--', label='Fitted curve', color='red', linewidth=2)
                # 标记50%和90%反应时间
                if not np.isnan(fit_results['t50']):
                    ax.axvline(x=fit_results['t50'], color='orange', linestyle=':',
                               label=f't50: {fit_results["t50"]:.2f}', linewidth=2)
                if not np.isnan(fit_results['t90']):
                    ax.axvline(x=fit_results['t90'], color='purple', linestyle=':',
                               label=f't90: {fit_results["t90"]:.2f}', linewidth=2)

            ax.set_xlabel('Time Point')
            ax.set_ylabel('Pearson Correlation')
            # 添加延迟参数到标题（如果使用延迟模型）
            if fit_model == 'delayed_first_order' and not np.isnan(fit_results.get('delay', np.nan)):
                ax.set_title(f'Cell {cell_id}\nR²: {fit_results["r_squared"]:.3f}, Delay: {fit_results["delay"]:.3f}')
            else:
                ax.set_title(f'Cell {cell_id}\nR²: {fit_results["r_squared"]:.3f}')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)

        # 隐藏多余的子图
        for idx in range(len(all_cell_ids), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        # 保存图片，使用原始文件名作为前缀
        fig_filename = f"{original_filename}_all_cells_analysis.png"
        fig_path = Path(self.output_dir) / fig_filename
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)  # 关闭图形以释放内存

    def _plot_single_cell_detailed(self, analysis: FileData, cell_id: int, fit_model: str, include_scatter: bool):
        """为单个细胞生成详细分析图"""
        original_filename = Path(analysis.file_path).stem
        time_points, correlations, p_values = CoLocalizationMetrics.get_correlation_over_time_of_a_cell(analysis, cell_id)

        # Re-calculate fit for this specific cell
        valid_corr_mask = ~np.isnan(correlations)
        if np.sum(valid_corr_mask) >= 3:
             corr_fit_results = ReactionFitter.fit_first_order_reaction(time_points[valid_corr_mask], correlations[valid_corr_mask]) if fit_model == 'first_order' else ReactionFitter.fit_delayed_first_order_reaction(time_points[valid_corr_mask], correlations[valid_corr_mask])
        else:
             corr_fit_results = {
                 'A0': np.nan, 'k': np.nan, 'A_inf': np.nan, 't50': np.nan, 't90': np.nan, 'r_squared': np.nan,
                 'delay': np.nan if fit_model == 'delayed_first_order' else None
             }

        # Get intensity data and fit for channels
        ch1_time, ch1_values = self._get_intensity_over_time(analysis, cell_id, 'channel1')
        ch2_time, ch2_values = self._get_intensity_over_time(analysis, cell_id, 'channel2')

        if np.sum(~np.isnan(ch1_values)) >= 3:
             ch1_fit_results = ReactionFitter.fit_first_order_reaction(ch1_time[~np.isnan(ch1_values)], ch1_values[~np.isnan(ch1_values)]) if fit_model == 'first_order' else ReactionFitter.fit_delayed_first_order_reaction(ch1_time[~np.isnan(ch1_values)], ch1_values[~np.isnan(ch1_values)])
        else:
             ch1_fit_results = {
                 'A0': np.nan, 'k': np.nan, 'A_inf': np.nan, 't50': np.nan, 't90': np.nan, 'r_squared': np.nan,
                 'delay': np.nan if fit_model == 'delayed_first_order' else None
             }

        if np.sum(~np.isnan(ch2_values)) >= 3:
             ch2_fit_results = ReactionFitter.fit_first_order_reaction(ch2_time[~np.isnan(ch2_values)], ch2_values[~np.isnan(ch2_values)]) if fit_model == 'first_order' else ReactionFitter.fit_delayed_first_order_reaction(ch2_time[~np.isnan(ch2_values)], ch2_values[~np.isnan(ch2_values)])
        else:
             ch2_fit_results = {
                 'A0': np.nan, 'k': np.nan, 'A_inf': np.nan, 't50': np.nan, 't90': np.nan, 'r_squared': np.nan,
                 'delay': np.nan if fit_model == 'delayed_first_order' else None
             }

        # Create plots
        if include_scatter:
            fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Correlation vs Time
        ax1 = axes[0, 0] if include_scatter else axes[0, 0]
        ax1.plot(time_points, correlations, 'o-', label='Correlation', linewidth=2, markersize=6)
        if not np.isnan(corr_fit_results['k']) and corr_fit_results['k'] > 0:
            t_fit = np.linspace(time_points.min(), time_points.max(), 100)
            fit_func = delayed_first_order_reaction if fit_model == 'delayed_first_order' else first_order_reaction
            y_fit = fit_func(t_fit, corr_fit_results['A0'], corr_fit_results['k'], corr_fit_results['A_inf'], corr_fit_results.get('delay', 0))
            ax1.plot(t_fit, y_fit, '--', label='Fitted curve', color='red', linewidth=2)
            if not np.isnan(corr_fit_results['t50']):
                ax1.axvline(x=corr_fit_results['t50'], color='orange', linestyle=':', label=f't50: {corr_fit_results["t50"]:.2f}', linewidth=2)
            if not np.isnan(corr_fit_results['t90']):
                ax1.axvline(x=corr_fit_results['t90'], color='purple', linestyle=':', label=f't90: {corr_fit_results["t90"]:.2f}', linewidth=2)
        ax1.set_xlabel('Time Point')
        ax1.set_ylabel('Pearson Correlation')
        if fit_model == 'delayed_first_order' and not np.isnan(corr_fit_results.get('delay', np.nan)):
            ax1.set_title(f'Cell {cell_id} - Correlation Over Time\nR²: {corr_fit_results["r_squared"]:.3f}, Delay: {corr_fit_results["delay"]:.3f}')
        else:
            ax1.set_title(f'Cell {cell_id} - Correlation Over Time\nR²: {corr_fit_results["r_squared"]:.3f}')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)

        # 2. P-value vs Time
        ax2 = axes[0, 1] if include_scatter else axes[0, 1]
        ax2.plot(time_points, p_values, 's-', color='red', linewidth=2, markersize=6)
        ax2.set_xlabel('Time Point')
        ax2.set_ylabel('P-value')
        ax2.set_yscale('log')
        ax2.set_title(f'Cell {cell_id} - P-value Over Time')
        ax2.grid(True, alpha=0.3)

        # 3. Channel 1 Intensity vs Time
        ax3 = axes[1, 0] if include_scatter else axes[1, 0]
        ax3.plot(ch1_time, ch1_values, 'o-', label='Channel 1', linewidth=2, markersize=6)
        if not np.isnan(ch1_fit_results['k']) and ch1_fit_results['k'] > 0:
            t_fit = np.linspace(ch1_time.min(), ch1_time.max(), 100)
            y_fit = fit_func(t_fit, ch1_fit_results['A0'], ch1_fit_results['k'], ch1_fit_results['A_inf'], ch1_fit_results.get('delay', 0))
            ax3.plot(t_fit, y_fit, '--', label='Fitted curve', color='red', linewidth=2)
        ax3.set_xlabel('Time Point')
        ax3.set_ylabel('Intensity')
        if fit_model == 'delayed_first_order' and not np.isnan(ch1_fit_results.get('delay', np.nan)):
            ax3.set_title(f'Cell {cell_id} - Channel 1 Intensity\nR²: {ch1_fit_results["r_squared"]:.3f}, Delay: {ch1_fit_results["delay"]:.3f}')
        else:
            ax3.set_title(f'Cell {cell_id} - Channel 1 Intensity\nR²: {ch1_fit_results["r_squared"]:.3f}')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # 4. Channel 2 Intensity vs Time
        ax4 = axes[1, 1] if include_scatter else axes[1, 1]
        ax4.plot(ch2_time, ch2_values, 'o-', label='Channel 2', linewidth=2, markersize=6, color='green')
        if not np.isnan(ch2_fit_results['k']) and ch2_fit_results['k'] > 0:
            t_fit = np.linspace(ch2_time.min(), ch2_time.max(), 100)
            y_fit = fit_func(t_fit, ch2_fit_results['A0'], ch2_fit_results['k'], ch2_fit_results['A_inf'], ch2_fit_results.get('delay', 0))
            ax4.plot(t_fit, y_fit, '--', label='Fitted curve', color='red', linewidth=2)
        ax4.set_xlabel('Time Point')
        ax4.set_ylabel('Intensity')
        if fit_model == 'delayed_first_order' and not np.isnan(ch2_fit_results.get('delay', np.nan)):
            ax4.set_title(f'Cell {cell_id} - Channel 2 Intensity\nR²: {ch2_fit_results["r_squared"]:.3f}, Delay: {ch2_fit_results["delay"]:.3f}')
        else:
            ax4.set_title(f'Cell {cell_id} - Channel 2 Intensity\nR²: {ch2_fit_results["r_squared"]:.3f}')
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        # 5. Scatter Plot (if needed)
        if include_scatter:
            ax5 = axes[2, 0]
            # Get data from first timepoint after skip_initial_frames for this cell
            first_time_data = next((cd for cd in analysis.all_cells if cd.cell_id == cell_id and cd.time_point >= analysis.skip_initial_frames), None)
            if first_time_data:
                mask = ~(np.isnan(first_time_data.intensity1) | np.isnan(first_time_data.intensity2))
                ch1_clean = first_time_data.intensity1[mask]
                ch2_clean = first_time_data.intensity2[mask]
                ax5.scatter(ch1_clean, ch2_clean, alpha=0.6, s=20)
                ax5.set_xlabel('Channel 1 Intensity')
                ax5.set_ylabel('Channel 2 Intensity')
                # Calculate correlation for this specific data point for the title
                if len(ch1_clean) > 1 and len(ch2_clean) > 1:
                    scatter_corr, _ = CoLocalizationMetrics._calculate_single_timepoint_correlation(ch1_clean, ch2_clean)
                else:
                    scatter_corr = np.nan
                ax5.set_title(f'Cell {cell_id} - Scatter Plot\nCorr: {scatter_corr:.3f}')
                ax5.grid(True, alpha=0.3)

            # 6. Summary Text (if needed)
            ax6 = axes[2, 1]
            ax6.axis('off')
            summary_text = f"Cell {cell_id} - Fitting Summary\n"
            summary_text += f"Correlation:\n"
            summary_text += f"  k = {corr_fit_results['k']:.4f}\n"
            if fit_model == 'delayed_first_order':
                summary_text += f"  delay = {corr_fit_results.get('delay', np.nan):.4f}\n"
            summary_text += f"  t50 = {corr_fit_results['t50']:.2f}\n"
            summary_text += f"  t90 = {corr_fit_results['t90']:.2f}\n"
            summary_text += f"  R² = {corr_fit_results['r_squared']:.3f}\n"
            summary_text += f"Channel1:\n"
            summary_text += f"  k = {ch1_fit_results['k']:.4f}\n"
            if fit_model == 'delayed_first_order':
                summary_text += f"  delay = {ch1_fit_results.get('delay', np.nan):.4f}\n"
            summary_text += f"  t50 = {ch1_fit_results['t50']:.2f}\n"
            summary_text += f"  t90 = {ch1_fit_results['t90']:.2f}\n" # Note: Original typo corrected
            summary_text += f"  R² = {ch1_fit_results['r_squared']:.3f}\n"
            summary_text += f"Channel2:\n"
            summary_text += f"  k = {ch2_fit_results['k']:.4f}\n"
            if fit_model == 'delayed_first_order':
                summary_text += f"  delay = {ch2_fit_results.get('delay', np.nan):.4f}\n"
            summary_text += f"  t50 = {ch2_fit_results['t50']:.2f}\n"
            summary_text += f"  t90 = {ch2_fit_results['t90']:.2f}\n"
            summary_text += f"  R² = {ch2_fit_results['r_squared']:.3f}"
            ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

        plt.tight_layout()
        # Save detailed plot
        fig_filename = f"{original_filename}_cell_{cell_id}_analysis.png"
        fig_path = Path(self.output_dir) / fig_filename
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)  # Close figure to free memory

    def generate_ratio_vs_t50_analysis(self, analyses: List[FileData], reaction_df, 
                                       fit_model: str, background: float = 100.0):
        """
        生成红绿比值与T50关系的分析图
        使用GMM拟合来提取红绿通道的代表性强度
            
        Parameters:
        - analyses: FileData 列表
        - reaction_df: 反应拟合结果 DataFrame
        - fit_model: 拟合模型
        - background: 本底信号（默认100）
        """
        print("\nGenerating Red/Green ratio vs T50 analysis (GMM fitting)...")
            
        # 创建GMM拟合可视化输出目录
        gmm_viz_dir = Path(self.output_dir) / 'gmm_fitting_viz'
        gmm_viz_dir.mkdir(parents=True, exist_ok=True)
            
        # 收集所有细胞的数据
        ratio_data = []
        gmm_results = []  # 存储GMM拟合结果用于可视化
            
        for analysis in analyses:
            file_path = analysis.file_path
            file_stem = Path(file_path).stem
            skip_frames = analysis.skip_initial_frames
                
            for cell_id, cell_list in analysis.cells.items():
                # 获取第一个有效时间点的数据
                first_cell_data = None
                for cd in cell_list:
                    if cd.time_point >= skip_frames:
                        first_cell_data = cd
                        break
                    
                if first_cell_data is None:
                    continue
                    
                # 获取原始像素值（通道0=红色，通道1=绿色）
                red_pixels = first_cell_data.intensity1.flatten()
                green_pixels = first_cell_data.intensity2.flatten()
                    
                # 过滤无效值
                red_pixels = red_pixels[~np.isnan(red_pixels)]
                green_pixels = green_pixels[~np.isnan(green_pixels)]
                    
                if len(red_pixels) < 50 or len(green_pixels) < 50:
                    continue
                    
                # GMM拟合
                try:
                    red_result = self._fit_gmm_2components(red_pixels)
                    green_result = self._fit_gmm_2components(green_pixels)
                except Exception as e:
                    print(f"  GMM fitting failed for {file_stem} cell {cell_id}: {e}")
                    continue
                    
                # 红色：取权重小的峰（信号峰）
                red_value = red_result['minor_peak_mean'] - background
                # 绿色：取权重大的峰（主峰）
                green_value = green_result['major_peak_mean'] - background
                    
                # 检查有效性
                if red_value <= 0 or green_value <= 0:
                    continue
                    
                ratio = red_value / green_value
                    
                # 获取T50
                mask = (reaction_df['file_path'] == file_path) & (reaction_df['cell_id'] == cell_id)
                if mask.sum() == 0:
                    continue
                    
                t50 = reaction_df.loc[mask, 'correlation_t50'].values[0]
                if np.isnan(t50):
                    continue
                    
                ratio_data.append({
                    'file_path': file_path,
                    'file_stem': file_stem,
                    'cell_id': cell_id,
                    'red': red_value,
                    'green': green_value,
                    'ratio': ratio,
                    't50': t50
                })
                    
                # 存储GMM结果用于可视化
                gmm_results.append({
                    'file_stem': file_stem,
                    'cell_id': cell_id,
                    'red_pixels': red_pixels,
                    'green_pixels': green_pixels,
                    'red_gmm': red_result,
                    'green_gmm': green_result,
                    'red_selected': red_value + background,  # 选中的峰值（未扣除本底）
                    'green_selected': green_value + background,
                    'background': background
                })
            
        if len(ratio_data) == 0:
            print("No valid data for ratio analysis.")
            return
            
        # 生成GMM拟合可视化图
        print(f"  Generating GMM fitting visualizations for {len(gmm_results)} cells...")
        self._generate_gmm_visualizations(gmm_results, gmm_viz_dir)
            
        # 转换为 numpy 数组
        red_values = np.array([d['red'] for d in ratio_data])
        green_values = np.array([d['green'] for d in ratio_data])
        ratio_values = np.array([d['ratio'] for d in ratio_data])
        t50_values = np.array([d['t50'] for d in ratio_data])
            
        # 剔除极端比值（使用 IQR 方法）
        q1 = np.percentile(ratio_values, 25)
        q3 = np.percentile(ratio_values, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
            
        valid_mask = (ratio_values >= lower_bound) & (ratio_values <= upper_bound)
        n_removed = np.sum(~valid_mask)
            
        print(f"  Total cells with valid GMM fit: {len(ratio_data)}")
        print(f"  Removed outliers: {n_removed} (ratio < {lower_bound:.3f} or > {upper_bound:.3f})")
        print(f"  Valid cells for analysis: {np.sum(valid_mask)}")
            
        # 过滤后的数据
        red_filtered = red_values[valid_mask]
        green_filtered = green_values[valid_mask]
        ratio_filtered = ratio_values[valid_mask]
        t50_filtered = t50_values[valid_mask]
            
        if len(t50_filtered) < 3:
            print("Not enough valid data points for analysis.")
            return
            
        # 创建图表: 2x2 布局
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            
        # 1. 红色值 vs T50
        ax1 = axes[0, 0]
        ax1.scatter(red_filtered, t50_filtered, alpha=0.6, s=40, c='red', edgecolors='darkred')
        slope1, intercept1, r1, p1, se1 = stats.linregress(red_filtered, t50_filtered)
        x_line = np.linspace(red_filtered.min(), red_filtered.max(), 100)
        ax1.plot(x_line, slope1 * x_line + intercept1, 'k--', linewidth=2, 
                label=f'R={r1:.3f}, p={p1:.2e}')
        ax1.set_xlabel('Red Intensity (GMM signal peak)', fontsize=11)
        ax1.set_ylabel('T50 (Correlation)', fontsize=11)
        ax1.set_title('Red Intensity vs T50', fontsize=12)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
            
        # 2. 绿色值 vs T50
        ax2 = axes[0, 1]
        ax2.scatter(green_filtered, t50_filtered, alpha=0.6, s=40, c='green', edgecolors='darkgreen')
        slope2, intercept2, r2, p2, se2 = stats.linregress(green_filtered, t50_filtered)
        x_line = np.linspace(green_filtered.min(), green_filtered.max(), 100)
        ax2.plot(x_line, slope2 * x_line + intercept2, 'k--', linewidth=2,
                label=f'R={r2:.3f}, p={p2:.2e}')
        ax2.set_xlabel('Green Intensity (GMM main peak)', fontsize=11)
        ax2.set_ylabel('T50 (Correlation)', fontsize=11)
        ax2.set_title('Green Intensity vs T50', fontsize=12)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
            
        # 3. 红绿比值 vs T50
        ax3 = axes[1, 0]
        ax3.scatter(ratio_filtered, t50_filtered, alpha=0.6, s=40, c='purple', edgecolors='darkviolet')
        slope3, intercept3, r3, p3, se3 = stats.linregress(ratio_filtered, t50_filtered)
        x_line = np.linspace(ratio_filtered.min(), ratio_filtered.max(), 100)
        ax3.plot(x_line, slope3 * x_line + intercept3, 'k--', linewidth=2,
                label=f'R={r3:.3f}, p={p3:.2e}')
        ax3.set_xlabel('Red/Green Ratio (GMM)', fontsize=11)
        ax3.set_ylabel('T50 (Correlation)', fontsize=11)
        ax3.set_title('Red/Green Ratio vs T50', fontsize=12)
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
            
        # 4. 比值分布直方图
        ax4 = axes[1, 1]
        ax4.hist(ratio_filtered, bins=30, color='purple', alpha=0.7, edgecolor='black')
        ax4.axvline(x=np.median(ratio_filtered), color='red', linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(ratio_filtered):.3f}')
        ax4.axvline(x=np.mean(ratio_filtered), color='blue', linestyle='--',
                   linewidth=2, label=f'Mean: {np.mean(ratio_filtered):.3f}')
        ax4.set_xlabel('Red/Green Ratio (GMM)', fontsize=11)
        ax4.set_ylabel('Count', fontsize=11)
        ax4.set_title(f'Red/Green Ratio Distribution (n={len(ratio_filtered)})', fontsize=12)
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)
            
        plt.tight_layout()
            
        # 保存图片
        fig_path = Path(self.output_dir) / 'ratio_vs_t50_analysis.png'
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {fig_path}")
            
        # 输出统计结果
        print(f"\n  Analysis Results (GMM-based):")
        print(f"    Red vs T50:   R={r1:.3f}, p={p1:.2e}")
        print(f"    Green vs T50: R={r2:.3f}, p={p2:.2e}")
        print(f"    Ratio vs T50: R={r3:.3f}, p={p3:.2e}")
        
    def _fit_gmm_2components(self, pixels: np.ndarray) -> Dict:
        """
        对像素强度进行2分量GMM拟合
            
        Returns:
        - dict: 包含拟合结果的字典
            - means: 两个分量的均值
            - weights: 两个分量的权重
            - stds: 两个分量的标准差
            - major_peak_idx: 权重大的峰索引
            - minor_peak_idx: 权重小的峰索引
            - major_peak_mean: 权重大的峰均值
            - minor_peak_mean: 权重小的峰均值
        """
        # 重塑为2D数组
        X = pixels.reshape(-1, 1)
            
        # 拟合GMM
        gmm = GaussianMixture(n_components=2, random_state=42, n_init=5)
        gmm.fit(X)
            
        means = gmm.means_.flatten()
        weights = gmm.weights_.flatten()
        stds = np.sqrt(gmm.covariances_.flatten())
            
        # 确定主峰（权重大）和次峰（权重小）
        major_idx = np.argmax(weights)
        minor_idx = np.argmin(weights)
            
        return {
            'means': means,
            'weights': weights,
            'stds': stds,
            'major_peak_idx': major_idx,
            'minor_peak_idx': minor_idx,
            'major_peak_mean': means[major_idx],
            'minor_peak_mean': means[minor_idx],
            'major_peak_weight': weights[major_idx],
            'minor_peak_weight': weights[minor_idx],
            'gmm': gmm
        }
        
    def _generate_gmm_visualizations(self, gmm_results: List[Dict], output_dir: Path):
        """
        为每个细胞生成GMM拟合可视化图
        """
        for result in tqdm(gmm_results, desc="Generating GMM plots", unit="cell"):
            file_stem = result['file_stem']
            cell_id = result['cell_id']
                
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                
            # 红色通道
            ax1 = axes[0]
            self._plot_gmm_histogram(ax1, result['red_pixels'], result['red_gmm'],
                                    result['red_selected'], result['background'],
                                    'Red Channel', 'red', 'darkred', is_red=True)
                
            # 绿色通道
            ax2 = axes[1]
            self._plot_gmm_histogram(ax2, result['green_pixels'], result['green_gmm'],
                                    result['green_selected'], result['background'],
                                    'Green Channel', 'green', 'darkgreen', is_red=False)
                
            plt.suptitle(f'{file_stem} - Cell {cell_id}', fontsize=14, fontweight='bold')
            plt.tight_layout()
                
            # 保存
            fig_path = output_dir / f'{file_stem}_cell_{cell_id}_gmm.png'
            fig.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        
    def _plot_gmm_histogram(self, ax, pixels: np.ndarray, gmm_result: Dict,
                           selected_value: float, background: float,
                           title: str, color: str, edge_color: str, is_red: bool):
        """
        绘制单个通道的直方图和GMM拟合曲线
        """
        # 绘制直方图
        counts, bins, _ = ax.hist(pixels, bins=100, density=True, alpha=0.6, 
                                  color=color, edgecolor=edge_color, label='Data')
            
        # 绘制GMM拟合曲线
        x = np.linspace(pixels.min(), pixels.max(), 500)
        gmm = gmm_result['gmm']
            
        # 混合密度
        log_prob = gmm.score_samples(x.reshape(-1, 1))
        pdf = np.exp(log_prob)
        ax.plot(x, pdf, 'k-', linewidth=2, label='GMM fit')
            
        # 单独绘制每个分量
        means = gmm_result['means']
        stds = gmm_result['stds']
        weights = gmm_result['weights']
            
        colors_comp = ['blue', 'orange']
        for i in range(2):
            # 单个高斯分量
            comp_pdf = weights[i] * stats.norm.pdf(x, means[i], stds[i])
                
            # 确定这个分量是主峰还是次峰
            if i == gmm_result['major_peak_idx']:
                label = f'Major peak (μ={means[i]:.1f}, w={weights[i]:.2f})'
                linestyle = '-'
            else:
                label = f'Minor peak (μ={means[i]:.1f}, w={weights[i]:.2f})'
                linestyle = '--'
                
            ax.plot(x, comp_pdf, color=colors_comp[i], linestyle=linestyle, 
                   linewidth=1.5, label=label)
            
        # 标记选中的峰值
        ax.axvline(x=selected_value, color='magenta', linestyle=':', linewidth=2,
                  label=f'Selected: {selected_value:.1f}')
            
        # 标记本底
        ax.axvline(x=background, color='gray', linestyle='--', linewidth=1.5,
                  label=f'Background: {background:.0f}')
            
        # 添加说明：红色取次峰，绿色取主峰
        if is_red:
            selection_note = '(Minor peak = Signal)'
        else:
            selection_note = '(Major peak = Main signal)'
            
        ax.set_xlabel('Intensity', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{title} {selection_note}', fontsize=12)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    def _get_intensity_over_time(self, analysis: FileData, cell_id: int, channel: str) -> Tuple[np.ndarray, np.ndarray]:
        """Helper to get intensity over time (similar to old TimeSeriesAnalysis method)"""
        if cell_id not in analysis.cells:
            raise ValueError(f"Cell {cell_id} not found")
        time_points = []
        intensities = []
        for cell_data in analysis.cells[cell_id]:
            if cell_data.time_point >= analysis.skip_initial_frames:  # 跳过初始帧
                time_points.append(cell_data.time_point)
                if channel == 'channel1':
                    intensities.append(np.mean(cell_data.intensity1))
                elif channel == 'channel2':
                    intensities.append(np.mean(cell_data.intensity2))
                else:
                    raise ValueError(f"Unknown channel: {channel}")
        return np.array(time_points), np.array(intensities)
