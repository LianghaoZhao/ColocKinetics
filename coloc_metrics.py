# coloc_metrics.py
import numpy as np
from scipy.stats import pearsonr
from typing import Dict, List, Tuple
from data_structures import FileData


class CoLocalizationMetrics:
    """计算共定位相关指标的类"""

    @staticmethod
    def get_correlation_over_time_of_a_cell(analysis: FileData, cell_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """获取指定细胞的相关系数随时间的变化 (from CellData)"""
        if cell_id not in analysis.cells:
            raise ValueError(f"Cell {cell_id} not found")
        time_points = []
        correlations = []
        p_values = []
        for cell_data in analysis.cells[cell_id]:
            if cell_data.time_point >= analysis.skip_initial_frames:  # 跳过初始帧
                time_points.append(cell_data.time_point)
                corr, p_val = CoLocalizationMetrics._calculate_single_timepoint_correlation(
                    cell_data.intensity1, cell_data.intensity2
                )
                correlations.append(corr)
                p_values.append(p_val)
        return np.array(time_points), np.array(correlations), np.array(p_values)

    @staticmethod
    def _calculate_single_timepoint_correlation(intensity1: np.ndarray, intensity2: np.ndarray) -> Tuple[float, float]:
        """计算单个时间点的皮尔逊相关系数和p值"""
        return pearsonr(intensity1.flatten(), intensity2.flatten())


    @staticmethod
    def calculate_metrics_for_file(analysis: FileData) -> Dict[int, Dict[str, np.ndarray]]:
        """
        计算单个文件内所有细胞的共定位指标。
        Parameters:
        - analysis: TimeSeriesAnalysis 对象
        Returns:
        - Dict: {cell_id: {'time_points': [...], 'correlations': [...], 'p_values': [...], 'intensity1': [...], 'intensity2': [...]}}
        """
        results = {}
        for cell_id, cell_list in analysis.cells.items():
            time_points = []
            correlations = []
            p_values = []
            ch1_values = []
            ch2_values = []

            for cell_data in cell_list:
                if cell_data.time_point >= analysis.skip_initial_frames:
                    # Calculate correlation for this specific time point
                    corr, p_val = CoLocalizationMetrics._calculate_single_timepoint_correlation(
                        cell_data.intensity1, cell_data.intensity2
                    )
                    correlations.append(corr)
                    p_values.append(p_val)
                    ch1_values.append(np.mean(cell_data.intensity1))
                    ch2_values.append(np.mean(cell_data.intensity2))
                    time_points.append(cell_data.time_point)

            results[cell_id] = {
                'time_points': np.array(time_points),
                'correlations': np.array(correlations),
                'p_values': np.array(p_values),
                'intensity1': np.array(ch1_values),
                'intensity2': np.array(ch2_values)
            }
        return results

    @classmethod
    def calculate_all_metrics(cls, analyses: List[FileData]) -> Dict[str, Dict[int, Dict[str, np.ndarray]]]:
        """
        批量处理所有文件的共定位计算。
        Parameters:
        - analyses: List of TimeSeriesAnalysis objects
        Returns:
        - Dict: {file_path: {cell_id: {...}}}
        """
        all_results = {}
        for analysis in analyses:
            file_metrics = cls.calculate_metrics_for_file(analysis)
            all_results[analysis.file_path] = file_metrics
        return all_results
