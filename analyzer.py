import os
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from data_structures import FileData
from io_utils import MaskFileMatcher, process_single_file_io
from coloc_metrics import CoLocalizationMetrics
from multiprocessing import Pool
from typing import Optional
from reaction_kinetics import KineticsAnalyzer
import warnings

class MainAnalyzer:
    """荧光共聚焦图像分析器"""
    def __init__(self):
        self.all_files: List[FileData] = []

    def process_files_with_masks(self, image_files: List[str], mask_pattern: Optional[str] = None, skip_initial_frames: int = 0, nd2_search_dirs: Optional[List[str]] = None) -> List[FileData]:
        """处理多个图像文件，自动匹配蒙版（顺序处理）
        
        Parameters:
        - image_files: 图像文件列表
        - mask_pattern: mask文件匹配模式
        - skip_initial_frames: 跳过的初始帧数
        - nd2_search_dirs: 搜索原始ND2文件的目录列表
        """
        # 如果未提供nd2_search_dirs,使用默认搜索路径
        if nd2_search_dirs is None:
            nd2_search_dirs = []
            # 添加图像文件所在的目录
            for img_file in image_files:
                img_dir = str(Path(img_file).parent)
                if img_dir not in nd2_search_dirs:
                    nd2_search_dirs.append(img_dir)
            # 添加当前工作目录
            if '.' not in nd2_search_dirs:
                nd2_search_dirs.append('.')
        
        # 首先匹配所有图像文件和蒙版
        matches = MaskFileMatcher.match_image_with_masks(image_files, mask_pattern)

        for image_file in image_files:
            mask_path = matches[image_file]
            io_result = process_single_file_io((image_file, mask_path, skip_initial_frames, nd2_search_dirs))
            if io_result is not None:
                self.all_files.append(io_result)
        return self.all_files


    def get_summary_dataframe(self) -> pd.DataFrame:
        """获取所有细胞的汇总数据 (based on CellData attributes)"""
        data = []
        for file in self.all_files:
            # 预先计算每个 cell_id 的相关系数（避免重复计算）
            cell_correlations = {}
            for cell_id in file.cells.keys():
                time_points, correlations, p_values = CoLocalizationMetrics.get_correlation_over_time_of_a_cell(file, cell_id)
                # 构建 time_point -> (corr, p_val) 的映射
                cell_correlations[cell_id] = {t: (c, p) for t, c, p in zip(time_points, correlations, p_values)}
            
            for cell in file.all_cells:
                if cell.time_point >= file.skip_initial_frames:
                    # 直接查找预计算的结果
                    corr_map = cell_correlations.get(cell.cell_id, {})
                    corr_val, p_val = corr_map.get(cell.time_point, (np.nan, np.nan))

                    data.append({
                        'file_path': cell.file_path,
                        'cell_id': cell.cell_id,
                        'time_point': cell.time_point,
                        'pearson_corr': corr_val,
                        'p_value': p_val,
                        'n_pixels': cell.n_pixels,
                        'mean_ch1': np.mean(cell.intensity1),
                        'mean_ch2': np.mean(cell.intensity2),
                        'std_ch1': np.std(cell.intensity1),
                        'std_ch2': np.std(cell.intensity2)
                    })
        return pd.DataFrame(data)

    def run_full_analysis(self, fit_model: str = 'first_order') -> pd.DataFrame:
        """执行完整的分析流程：共定位计算 -> 拟合"""
        # Step 1: Calculate Co-localization Metrics
        print("Calculating co-localization metrics for all files...")
        coloc_results = CoLocalizationMetrics.calculate_all_metrics(self.all_files)

        # Step 2: Perform Kinetic Fitting
        print("Performing kinetic fitting...")
        fitting_results = KineticsAnalyzer.fit_all_kinetics(coloc_results, fit_model=fit_model)

        return fitting_results