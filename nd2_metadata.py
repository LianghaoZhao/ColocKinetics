"""
ND2元数据读取工具模块
用于从ND2文件中提取真实的拍摄时间戳
"""
import nd2
import numpy as np
import re
from pathlib import Path
from typing import Optional, List


def get_nd2_timestamps(nd2_path: str, position_index: Optional[int] = None) -> Optional[np.ndarray]:
    """
    从ND2文件读取时间戳
    
    Parameters:
    - nd2_path: ND2文件路径
    - position_index: 可选的position索引(对应P维度),如果为None则读取所有位置
    
    Returns:
    - 时间数组(秒为单位),如果读取失败返回None
    """
    try:
        with nd2.ND2File(nd2_path) as f:
            sizes = f.sizes
            num_timepoints = sizes.get('T', 1)
            num_positions = sizes.get('P', 1)
            
            timestamps = []
            
            # 如果指定了position_index,只读取该位置的时间戳
            if position_index is not None:
                if position_index >= num_positions:
                    print(f"Warning: position_index {position_index} >= num_positions {num_positions}")
                    return None
                
                # 遍历该位置的所有时间点
                for t in range(num_timepoints):
                    # 计算frame索引: 根据维度顺序 (T, P, C, Y, X)
                    frame_idx = t * num_positions + position_index
                    
                    try:
                        fm = f.frame_metadata(frame_idx)
                        if hasattr(fm, 'channels') and len(fm.channels) > 0:
                            ch_meta = fm.channels[0]
                            if hasattr(ch_meta, 'time') and hasattr(ch_meta.time, 'relativeTimeMs'):
                                relative_ms = ch_meta.time.relativeTimeMs
                                timestamps.append(relative_ms / 1000.0)  # 转换为秒
                            else:
                                timestamps.append(float(t))  # fallback
                        else:
                            timestamps.append(float(t))  # fallback
                    except Exception as e:
                        print(f"Warning: Failed to read timestamp for frame {frame_idx}: {e}")
                        timestamps.append(float(t))  # fallback
            else:
                # 读取第一个位置(或所有位置如果没有P维度)
                for t in range(num_timepoints):
                    frame_idx = t * num_positions  # 第一个position
                    
                    try:
                        fm = f.frame_metadata(frame_idx)
                        if hasattr(fm, 'channels') and len(fm.channels) > 0:
                            ch_meta = fm.channels[0]
                            if hasattr(ch_meta, 'time') and hasattr(ch_meta.time, 'relativeTimeMs'):
                                relative_ms = ch_meta.time.relativeTimeMs
                                timestamps.append(relative_ms / 1000.0)
                            else:
                                timestamps.append(float(t))
                        else:
                            timestamps.append(float(t))
                    except Exception as e:
                        print(f"Warning: Failed to read timestamp for frame {frame_idx}: {e}")
                        timestamps.append(float(t))
            
            return np.array(timestamps)
    
    except Exception as e:
        print(f"Error reading timestamps from {nd2_path}: {e}")
        return None


def extract_position_from_filename(filename: str) -> Optional[int]:
    """
    从拆分后的文件名提取P索引
    
    Parameters:
    - filename: 文件名或路径,例如 "xxx_P0_corrected.ome.tif"
    
    Returns:
    - Position索引(整数),如果未找到返回None
    """
    filename = Path(filename).name
    match = re.search(r'_P(\d+)', filename)
    if match:
        return int(match.group(1))
    return None


def find_original_nd2_for_split_file(split_file_path: str, search_dirs: Optional[List[str]] = None) -> Optional[str]:
    """
    查找拆分文件对应的原始ND2文件
    
    Parameters:
    - split_file_path: 拆分后的文件路径,例如 "xxx_P0_corrected.ome.tif"
    - search_dirs: 搜索目录列表,如果为None则搜索当前文件所在目录
    
    Returns:
    - 原始ND2文件的完整路径,如果未找到返回None
    """
    split_path = Path(split_file_path)
    filename = split_path.name
    
    # 提取基础文件名(去除_P*和其他后缀)
    # 例如: "20260214  LH387 LT50 timelapse 3-31 10uM rapa - 006_P0_corrected.ome.tif"
    # -> "20260214  LH387 LT50 timelapse 3-31 10uM rapa - 006"
    base_name = re.sub(r'_P\d+.*$', '', filename)
    
    # 如果没有提供搜索目录,使用文件所在目录
    if search_dirs is None:
        search_dirs = [str(split_path.parent)]
    
    # 在搜索目录中查找匹配的ND2文件
    for search_dir in search_dirs:
        search_path = Path(search_dir)
        if not search_path.exists():
            continue
        
        # 查找匹配的ND2文件
        nd2_pattern = f"{base_name}.nd2"
        nd2_files = list(search_path.glob(nd2_pattern))
        
        if nd2_files:
            return str(nd2_files[0])
        
        # 也尝试递归搜索(限制深度为2)
        nd2_files = list(search_path.glob(f"*/{nd2_pattern}"))
        if nd2_files:
            return str(nd2_files[0])
    
    # 未找到
    print(f"Warning: Could not find original ND2 file for {filename} (base: {base_name})")
    return None
