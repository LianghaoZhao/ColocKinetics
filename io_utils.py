import nd2
import numpy as np
from skimage import io
from pathlib import Path
import glob
import re
from natsort import natsorted
from typing import Dict, List, Tuple, Optional
from data_structures import FileData, CellData # Import data structures


class ImageReader:
    """图像读取器类，支持多种格式"""
    @staticmethod
    def read_image(file_path: str) -> np.ndarray:
        """
        读取图像文件，支持ND2和TIF格式
        Parameters:
        - file_path: 图像文件路径
        Returns:
        - numpy数组格式的图像数据
        """
        file_path = str(file_path)
        file_ext = Path(file_path).suffix.lower()
        if file_ext == '.nd2':
            with nd2.ND2File(file_path) as nd2_file:
                img_array = nd2_file.asarray()
        elif file_ext in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']:
            img_array = io.imread(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        return img_array


def load_mask(mask_path: str) -> np.ndarray:
    """
    加载蒙版并确保其为整数类型
    Parameters:
    - mask_path: 蒙版文件路径
    Returns:
    - 整数类型的蒙版数组
    """
    if mask_path.endswith(('.npy', '.npz')):
        mask = np.load(mask_path, allow_pickle=True)
    else:
        # 使用skimage读取图像
        mask = io.imread(mask_path)
        # 检查是否为整数类型
        if mask.dtype.kind not in ['u', 'i']:  # 不是无符号整数或有符号整数
            raise ValueError(f"Mask file {mask_path} is not integer type. Got dtype: {mask.dtype}")

    # 确保是整数类型
    if mask.dtype.kind not in ['u', 'i']:  # 不是无符号整数或有符号整数
        raise ValueError(f"Mask file {mask_path} is not integer type. Got dtype: {mask.dtype}")
    return mask


class MaskFileMatcher:
    """文件匹配器类"""
    @staticmethod
    def calculate_filename_similarity(nd2_stem: str, mask_stem: str) -> float:
        """
        计算两个文件名的相似度
        返回0-1之间的分数，1表示完全匹配
        """
        # 清理文件名，移除常见后缀和标识
        def clean_filename(name):
            # 移除常见的蒙版标识
            name = re.sub(r'_mask.*$', '', name, flags=re.IGNORECASE)
            name = re.sub(r'_seg.*$', '', name, flags=re.IGNORECASE)
            name = re.sub(r'_segmentation.*$', '', name, flags=re.IGNORECASE)
            name = re.sub(r'_channel.*$', '', name, flags=re.IGNORECASE)
            name = re.sub(r'_cp_masks.*$', '', name, flags=re.IGNORECASE)
            name = name.lower()
            return name

        nd2_clean = clean_filename(nd2_stem)
        mask_clean = clean_filename(mask_stem)

        # 完全匹配
        if nd2_clean == mask_clean:
            return 1.0

        # 计算编辑距离相似度
        def levenshtein_similarity(s1, s2):
            if len(s1) == 0 or len(s2) == 0:
                return 0.0
            # 使用编辑距离计算相似度
            import difflib
            return difflib.SequenceMatcher(None, s1, s2).ratio()

        return levenshtein_similarity(nd2_clean, mask_clean)

    @staticmethod
    def match_image_with_masks(image_files: List[str], mask_pattern: Optional[str] = None) -> Dict[str, Optional[str]]:
        """
        为所有图像文件匹配蒙版
        Returns:
        - 字典，键为图像文件路径，值为对应的蒙版路径（或None）
        """
        matches = {}
        # 收集所有蒙版文件
        if mask_pattern:
            # 如果提供了特定的mask_pattern，只使用该模式匹配的蒙版
            all_mask_files = list(glob.glob(mask_pattern))
        else:
            # 否则收集所有可能的蒙版文件
            parent_dir = Path(image_files[0]).parent if image_files else Path('.')
            mask_extensions = ['.npy', '.tif', '.tiff', '.png', '.jpg', '.jpeg']
            all_mask_files = []
            for ext in mask_extensions:
                all_mask_files.extend(glob.glob(str(parent_dir / f"*{ext}")))

        # 为每个图像文件找到最佳匹配的蒙版
        for image_file in image_files:
            image_path = Path(image_file)
            image_stem = image_path.stem
            best_match = None
            best_score = -1

            for mask_file in all_mask_files:
                mask_path = Path(mask_file)
                mask_stem = mask_path.stem
                score = MaskFileMatcher.calculate_filename_similarity(image_stem, mask_stem)
                if score > best_score:
                    best_score = score
                    best_match = mask_file

            # 简化：直接返回最高分匹配，不再设置阈值
            matches[image_file] = best_match

        return matches


def process_single_file_io(args):
    """
    处理单个文件的IO部分 (加载图像、蒙版，提取细胞数据，创建TimeSeriesAnalysis对象)
    Parameters:
    - args: (image_file, mask_path, skip_initial_frames)
    Returns:
    - TimeSeriesAnalysis object or None
    """
    image_file, mask_path, skip_initial_frames = args
    if mask_path is None:
        return None
    # 读取蒙版
    try:
        mask = load_mask(mask_path)
    except Exception as e:
        print(f"Error reading mask {mask_path}: {e}")
        return None
    # 读取图像数据
    try:
        img_array = ImageReader.read_image(image_file)
    except Exception as e:
        print(f"Error reading image {image_file}: {e}")
        return None

    # 获取图像信息
    shape = img_array.shape
    if len(shape) == 4:
        time_points, channels, height, width = shape
    elif len(shape) == 3:
        # 判断是否为多通道单时间点或单通道多时间点
        if channels := shape[0] if 'c' in ['c'] else None:  # 假设第一个维度是通道数
            if channels > 1:
                channels, height, width = shape
                time_points = 1
            else:
                # 假设是时间维度
                time_points, channels, height, width = shape[0], 1, shape[1], shape[2]
        else:
            # 默认处理方式
            channels, height, width = shape
            time_points = 1
    elif len(shape) == 2:
        # 单通道单时间点
        height, width = shape
        channels = 1
        time_points = 1
    else:
        raise ValueError(f"Unexpected image shape: {shape}")

    # 验证图像和蒙版尺寸匹配
    if (height, width) != mask.shape:
        raise ValueError(f"Image shape {(height, width)} doesn't match mask shape {mask.shape}")

    # 创建分析对象
    analysis = FileData(
        file_path=image_file,
        time_points=time_points,
        skip_initial_frames=skip_initial_frames
    )

    # 获取细胞ID - 确保是整数类型
    unique_cells = np.unique(mask)
    unique_cells = unique_cells[unique_cells > 0]  # 排除背景（0）
    # 确保所有cell_id都是整数
    unique_cells = unique_cells.astype(int)

    # 处理每个时间点
    for t in range(time_points):
        # 获取当前时间点的图像
        if time_points > 1:
            current_img = img_array[t]  # shape: (channels, height, width) or (height, width) if single channel
        else:
            current_img = img_array  # 如果只有一个时间点，可能是 (channels, height, width) 或 (height, width)

        # 如果只有一个通道，扩展维度以保持一致性
        if len(current_img.shape) == 2:
            current_img = current_img[np.newaxis, :, :]  # 添加通道维度

        # 验证通道数
        if current_img.shape[0] < 2:
            continue

        # 处理每个细胞
        for cell_id in unique_cells:
            # 获取细胞在图像中的坐标
            cell_mask = (mask == cell_id)
            y_coords, x_coords = np.where(cell_mask)
            if len(y_coords) == 0:
                continue

            # 提取通道数据
            channel1 = current_img[0][cell_mask]  # 第一个通道
            channel2 = current_img[1][cell_mask]  # 第二个通道

            # 检查是否有NaN值
            if np.any(np.isnan(channel1)) or np.any(np.isnan(channel2)):
                pass # Optionally handle NaNs differently

            # 创建CellData对象
            cell_data = CellData(
                cell_id=int(cell_id),  # 确保是整数
                channel1=channel1,
                channel2=channel2,
                file_path=image_file,
                time_point=t,
                x_coords=x_coords,
                y_coords=y_coords
            )
            analysis.add_cell_data(cell_data)

    return analysis
