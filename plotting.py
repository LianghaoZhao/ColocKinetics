import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
from multiprocessing import Pool
from analyzer import FileData
from reaction_kinetics import first_order_reaction, delayed_first_order_reaction, ReactionFitter # Import fitting functions/models for plotting
from coloc_metrics import CoLocalizationMetrics # Import for getting correlation data
from typing import Tuple

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
