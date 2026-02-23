import nd2
import numpy as np
from skimage import io
from pathlib import Path
import glob
import re
from natsort import natsorted
from typing import Dict, List, Tuple, Optional
from data_structures import FileData, CellData # Import data structures
from nd2_metadata import get_nd2_timestamps, extract_position_from_filename, find_original_nd2_for_split_file


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
    加载蒸版并确保其为整数类型
    Parameters:
    - mask_path: 蒸版文件路径
    Returns:
    - 整数类型的蒸版数组
    """
    if mask_path.endswith(('.npy', '.npz')):
        data = np.load(mask_path, allow_pickle=True)
        
        # Cellpose _seg.npy 格式：字典，包含 'masks' 键
        if isinstance(data, np.ndarray) and data.dtype == object:
            # 尝试从字典中提取 masks
            if data.shape == ():
                data_dict = data.item()
                if isinstance(data_dict, dict):
                    if 'masks' in data_dict:
                        mask = data_dict['masks']
                    elif 'outlines' in data_dict:
                        # 如果没有 masks 但有 outlines，报错
                        raise ValueError(f"Mask file {mask_path} contains outlines but no masks")
                    else:
                        raise ValueError(f"Mask file {mask_path} is a dict but has no 'masks' key. Keys: {list(data_dict.keys())}")
                else:
                    raise ValueError(f"Mask file {mask_path} has unexpected format: {type(data_dict)}")
            else:
                raise ValueError(f"Mask file {mask_path} has unexpected shape: {data.shape}")
        else:
            mask = data
    else:
        # 使用skimage读取图像
        mask = io.imread(mask_path)
    
    # 确保是整数类型
    if mask.dtype.kind not in ['u', 'i']:  # 不是无符号整数或有符号整数
        # 尝试转换为整数
        try:
            mask = mask.astype(np.int32)
        except:
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


# 通道元数据读取与映射工具函数

def get_nd2_channel_info(file_path: str) -> List[Dict]:
    """读取 ND2 文件的通道元数据信息，返回标准化的通道列表。"""
    info_list: List[Dict] = []
    try:
        with nd2.ND2File(str(file_path)) as f:
            if hasattr(f, "metadata") and f.metadata is not None:
                md = f.metadata
                if hasattr(md, "channels") and md.channels is not None:
                    for idx, ch in enumerate(md.channels):
                        name = None
                        ex_wl = None
                        em_wl = None
                        try:
                            ch_obj = getattr(ch, "channel", ch)
                            if hasattr(ch_obj, "name") and ch_obj.name is not None:
                                name = str(ch_obj.name)
                            # 尝试获取激发/发射波长（不同 nd2 版本字段可能不同）
                            if hasattr(ch_obj, "excitation"):
                                ex_wl = float(ch_obj.excitation)
                            elif hasattr(ch_obj, "excitationLambda"):
                                ex_wl = float(ch_obj.excitationLambda)
                            if hasattr(ch_obj, "emission"):
                                em_wl = float(ch_obj.emission)
                            elif hasattr(ch_obj, "emissionLambda"):
                                em_wl = float(ch_obj.emissionLambda)
                        except Exception:
                            pass
                        info_list.append({
                            "index": idx,
                            "name": name,
                            "ex_wavelength": ex_wl,
                            "em_wavelength": em_wl,
                        })
    except Exception as e:
        print(f"Warning: failed to read ND2 channel metadata from {file_path}: {e}")
    return info_list


def find_closest_channel_index(channel_infos: List[Dict], target_wavelength: float) -> Optional[int]:
    """根据目标波长在通道列表中找到最匹配的通道 index。"""
    if not channel_infos:
        return None

    # 优先按名称中的数字匹配（例如 "561"、"488"）
    target_str = str(int(round(target_wavelength)))
    name_matches = []
    for info in channel_infos:
        name = str(info.get("name") or "")
        if target_str in name:
            name_matches.append(info.get("index"))
    if len(name_matches) == 1:
        return name_matches[0]

    # 退回到数值波长匹配（激发/发射波长）
    best_idx: Optional[int] = None
    best_delta: Optional[float] = None
    for info in channel_infos:
        for key in ("ex_wavelength", "em_wavelength"):
            wl = info.get(key)
            if wl is None:
                continue
            delta = abs(wl - target_wavelength)
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_idx = info.get("index")
    return best_idx


def auto_map_channels_from_metadata(channel_infos: List[Dict]) -> Optional[Tuple[int, int]]:
    """在未显式指定 --channels 时，尝试从 ND2 元数据自动映射 561/488 通道。"""
    if not channel_infos:
        return None
    default_targets = (561.0, 488.0)
    indices: List[int] = []
    for wl in default_targets:
        idx = find_closest_channel_index(channel_infos, wl)
        if idx is None:
            return None
        indices.append(idx)
    return indices[0], indices[1]


def resolve_analysis_channel_indices(image_file: str, analysis_wavelengths: Optional[Tuple[float, float]], channels_count: int, original_nd2_path: Optional[str] = None) -> Tuple[int, int]:
    """决定用于分析的两个通道索引（channel1/channel2）。
    
    Parameters:
    - image_file: 当前图像文件路径（可能是TIF或ND2）
    - analysis_wavelengths: 分析用波长配置 (w1, w2)
    - channels_count: 图像通道数
    - original_nd2_path: 原始ND2文件路径（用于从拆分的TIF文件读取通道元数据）
    """
    # 默认使用前两个通道
    ch1_idx, ch2_idx = 0, 1 if channels_count > 1 else 0

    # 优先从原始ND2文件读取通道元数据
    channel_infos: List[Dict] = []
    if original_nd2_path and Path(original_nd2_path).suffix.lower() == ".nd2":
        channel_infos = get_nd2_channel_info(original_nd2_path)
    elif Path(image_file).suffix.lower() == ".nd2":
        channel_infos = get_nd2_channel_info(image_file)

    # 显式指定波长：--channels 561,488
    if analysis_wavelengths is not None:
        if channel_infos:
            target1, target2 = analysis_wavelengths
            idx1 = find_closest_channel_index(channel_infos, target1)
            idx2 = find_closest_channel_index(channel_infos, target2)
            if idx1 is None or idx2 is None:
                print(f"Warning: failed to map --channels {analysis_wavelengths} for file {image_file}, falling back to indices 0 and 1.")
            else:
                ch1_idx, ch2_idx = idx1, idx2
        else:
            # 有波长配置但没有 ND2 元数据，退回到索引 0/1
            print(f"Warning: --channels specified but file {image_file} has no ND2 channel metadata; using indices 0 and 1.")
    else:
        # 未显式指定时，尝试自动从 ND2 元数据推断（默认 561/488）
        if channel_infos:
            auto = auto_map_channels_from_metadata(channel_infos)
            if auto is not None:
                ch1_idx, ch2_idx = auto

    # 校验索引有效性
    if channels_count <= 1:
        return 0, 0
    max_idx = max(ch1_idx, ch2_idx)
    if max_idx >= channels_count:
        print(f"Warning: mapped channel indices ({ch1_idx}, {ch2_idx}) exceed available channels ({channels_count}) for file {image_file}; using 0 and 1 instead.")
        ch1_idx, ch2_idx = (0, 1) if channels_count > 1 else (0, 0)

    return ch1_idx, ch2_idx


def process_single_file_io(args):
    """
    处理单个文件的IO部分 (加载图像、蒙版，提取细胞数据，创建TimeSeriesAnalysis对象)
    Parameters:
    - args: (image_file, mask_path, skip_initial_frames, nd2_search_dirs[, analysis_channels])
    Returns:
    - TimeSeriesAnalysis object or None
    """
    # 支持向后兼容的参数解包
    if len(args) == 4:
        image_file, mask_path, skip_initial_frames, nd2_search_dirs = args
        analysis_channels = None
    else:
        image_file, mask_path, skip_initial_frames, nd2_search_dirs, analysis_channels = args
    if mask_path is None:
        return None
    
    # 检测是否是从ND2拆分的文件,并读取时间戳
    timestamps = None
    original_nd2_path = None
    position_idx = None
    
    position_idx = extract_position_from_filename(image_file)
    if position_idx is not None:
        # 这是拆分后的文件
        original_nd2_path = find_original_nd2_for_split_file(image_file, nd2_search_dirs)
        if original_nd2_path:
            timestamps = get_nd2_timestamps(original_nd2_path, position_idx)
            if timestamps is not None:
                print(f"Loaded {len(timestamps)} timestamps for position {position_idx} from {Path(original_nd2_path).name}")
    elif image_file.lower().endswith('.nd2'):
        # 直接是ND2文件
        original_nd2_path = image_file
        timestamps = get_nd2_timestamps(image_file)
        if timestamps is not None:
            print(f"Loaded {len(timestamps)} timestamps from {Path(image_file).name}")
    
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
    if len(shape) == 5:
        # 5D数组: (P, T, C, H, W) - 多视野ND2文件，需要先拆分
        raise ValueError(
            f"Unexpected 5D image shape: {shape}. "
            f"This appears to be a multi-position ND2 file (P={shape[0]}, T={shape[1]}, C={shape[2]}). "
            f"Please run motion correction first to split positions, or use --motioncor-dir to specify existing split files."
        )
    elif len(shape) == 4:
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

    # 决定用于分析的两个通道索引（优先从原始ND2读取元数据）
    ch1_idx, ch2_idx = resolve_analysis_channel_indices(image_file, analysis_channels, channels, original_nd2_path)

    # 创建分析对象
    analysis = FileData(
        file_path=image_file,
        time_points=time_points,
        skip_initial_frames=skip_initial_frames,
        original_nd2_path=original_nd2_path,
        position_index=position_idx
    )
    
    # 调试：输出跳过帧数设置
    if skip_initial_frames > 0:
        print(f"  Skip initial frames: {skip_initial_frames} (will exclude first {skip_initial_frames} time points from fitting)")

    # 获取细胞ID - 确保是整数类型
    unique_cells = np.unique(mask)
    unique_cells = unique_cells[unique_cells > 0]  # 排除背景（0）
    # 确保所有cell_id都是整数
    unique_cells = unique_cells.astype(int)

    # 处理每个时间点
    for t in range(time_points):
        # 获取真实时间
        if timestamps is not None and t < len(timestamps):
            actual_time = timestamps[t]
        else:
            actual_time = float(t)  # fallback到索引
        
        # 获取当前时间点的图像
        if time_points > 1:
            current_img = img_array[t]  # shape: (channels, height, width) or (height, width) if single channel
        else:
            current_img = img_array  # 如果只有一个时间点，可能是 (channels, height, width) 或 (height, width)

        # 如果只有一个通道，扩展维度以保持一致性
        if len(current_img.shape) == 2:
            current_img = current_img[np.newaxis, :, :]  # 添加通道维度

        # 验证通道数
        if current_img.shape[0] < 2 or current_img.shape[0] <= max(ch1_idx, ch2_idx):
            continue

        # 处理每个细胞
        for cell_id in unique_cells:
            # 获取细胞在图像中的坐标
            cell_mask = (mask == cell_id)
            y_coords, x_coords = np.where(cell_mask)
            if len(y_coords) == 0:
                continue

            # 提取通道数据
            channel1 = current_img[ch1_idx][cell_mask]  # 第一个通道
            channel2 = current_img[ch2_idx][cell_mask]  # 第二个通道

            # 检查是否有NaN值
            if np.any(np.isnan(channel1)) or np.any(np.isnan(channel2)):
                pass # Optionally handle NaNs differently

            # 创建CellData对象
            cell_data = CellData(
                cell_id=int(cell_id),  # 确保是整数
                channel1=channel1,
                channel2=channel2,
                file_path=image_file,
                time_point=actual_time,  # 使用真实时间
                x_coords=x_coords,
                y_coords=y_coords
            )
            analysis.add_cell_data(cell_data)

    return analysis
