"""Cellpose mask generation utilities"""
import subprocess
import os
from pathlib import Path
from typing import List, Optional
import glob
import numpy as np


def run_cellpose_on_files(
    image_files: List[str],
    output_dir: str,
    diameter: int = 380,
    gpu_device: int = 0,
    use_gpu: bool = True,
    save_png: bool = True,
    verbose: bool = True,
    niter: Optional[int] = None,
    cp_channel_wavelength: Optional[float] = None
) -> List[str]:
    """
    对图像文件列表运行 Cellpose 生成 mask (使用 CLI 命令行)
    
    Parameters:
    - image_files: 图像文件路径列表
    - output_dir: mask 输出目录
    - diameter: 细胞直径（像素）
    - gpu_device: GPU 设备 ID
    - use_gpu: 是否使用 GPU
    - save_png: 是否保存 PNG 可视化
    - verbose: 是否显示详细输出
    - niter: Cellpose dynamics 迭代次数（None 则使用 Cellpose 默认值）
    - cp_channel_wavelength: Cellpose 分割用通道波长（None 则使用全部通道）
    
    Returns:
    - 生成的 mask 文件路径列表
    """
    # 使用绝对路径确保 cellpose 正确保存到指定目录
    output_dir = str(Path(output_dir).resolve())
    os.makedirs(output_dir, exist_ok=True)
    
    mask_files = []
    
    for image_file in image_files:
        image_path = Path(image_file).resolve()  # 使用绝对路径
        
        # 构建 cellpose 命令
        cmd = [
            "cellpose",
            "--image_path", str(image_path),
            "--diameter", str(diameter),
            "--savedir", output_dir,
            # Cellpose 4.x 默认保存 npy 文件
        ]
        
        if use_gpu:
            cmd.extend(["--use_gpu", "--gpu_device", str(gpu_device)])
        
        if save_png:
            cmd.append("--save_png")
        
        if niter is not None:
            cmd.extend(["--niter", str(niter)])
        
        if verbose:
            cmd.append("--verbose")
            print(f"Running Cellpose on: {image_path.name}")
            print(f"  Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            if verbose:
                if result.stdout:
                    print(result.stdout)
            
            # 查找生成的 mask 文件
            # Cellpose 4.x 输出格式: {filename}_seg.npy 或 {filename}_cp_masks.npy/tif
            possible_patterns = [
                f"{image_path.stem}_seg.npy",
                f"{image_path.stem}_cp_masks.npy",
                f"{image_path.stem}_cp_masks.tif",
                f"{image_path.stem}*_seg.npy",
                f"{image_path.stem}*_masks.npy",
            ]
            
            found_mask = None
            for pattern in possible_patterns:
                matches = list(Path(output_dir).glob(pattern))
                if matches:
                    found_mask = str(matches[0])
                    break
            
            if found_mask:
                mask_files.append(found_mask)
                if verbose:
                    print(f"  -> Mask saved: {Path(found_mask).name}")
            else:
                # 列出目录中的所有文件帮助调试
                all_files = list(Path(output_dir).glob(f"{image_path.stem}*"))
                print(f"  Warning: Mask file not found for {image_path.name}")
                print(f"  Files in {output_dir}: {[f.name for f in all_files]}")
                    
        except subprocess.CalledProcessError as e:
            print(f"  Error running Cellpose on {image_path.name}: {e}")
            if e.stderr:
                print(f"  stderr: {e.stderr}")
        except FileNotFoundError:
            print("Error: Cellpose not found. Please install it with: pip install cellpose")
            raise
    
    return mask_files


def find_existing_masks(image_files: List[str], mask_dir: str) -> dict:
    """
    查找已存在的 mask 文件
    
    Parameters:
    - image_files: 图像文件路径列表
    - mask_dir: mask 目录
    
    Returns:
    - 字典: {image_file: mask_file or None}
    """
    mask_dir = Path(mask_dir)
    matches = {}
    
    if not mask_dir.exists():
        return {f: None for f in image_files}
    
    for image_file in image_files:
        image_stem = Path(image_file).stem
        # 如果是校正后的文件，去掉 _corrected.ome 后缀
        base_stem = image_stem.replace('_corrected.ome', '').replace('_corrected', '')
        
        # 查找可能的 mask 文件 (Cellpose 4.x 使用 _seg.npy 或 _cp_masks.npy/.tif)
        possible_patterns = [
            # _seg.npy 格式（Cellpose 4.x 默认）
            f"{image_stem}_frame0_seg.npy",
            f"{base_stem}_corrected.ome_frame0_seg.npy",
            f"{image_stem}_seg.npy",
            f"{base_stem}_seg.npy",
            # _cp_masks 格式
            f"{image_stem}_frame0_cp_masks.npy",
            f"{image_stem}_frame0_cp_masks.tif",
            f"{base_stem}_corrected.ome_frame0_cp_masks.npy",
            f"{base_stem}_corrected.ome_frame0_cp_masks.tif",
            f"{image_stem}_cp_masks.npy",
            f"{image_stem}_cp_masks.tif",
            # 通配符
            f"{image_stem}*_seg.npy",
            f"{image_stem}*_masks.npy",
            f"{base_stem}*_seg.npy",
            f"{base_stem}*_masks.npy",
        ]
        
        found_mask = None
        for pattern in possible_patterns:
            matches_found = list(mask_dir.glob(pattern))
            if matches_found:
                found_mask = str(matches_found[0])
                break
        
        matches[image_file] = found_mask
    
    return matches


def extract_first_frame_from_nd2(image_files: List[str], output_dir: str, channel: Optional[int] = None, cp_channel_wavelength: Optional[float] = None, gamma: float = 0.5) -> List[str]:
    """
    从 ND2/TIF 文件中提取第一帧用于 Cellpose 分割
    
    Parameters:
    - image_files: 图像文件路径列表 (ND2 或 TIF)
    - output_dir: 输出目录
    - channel: 要提取的通道索引(0-based),如果为None则使用所有通道 (deprecated, 优先使用 cp_channel_wavelength)
    - cp_channel_wavelength: Cellpose 分割用通道波长（None 则使用所有通道）
    - gamma: Gamma 校正值（默认 0.5，1.0 表示不校正）
    
    Returns:
    - 提取的 TIFF 文件路径列表
    """
    import tifffile
    from io_utils import get_nd2_channel_info, find_closest_channel_index
    
    os.makedirs(output_dir, exist_ok=True)
    
    extracted_files = []
    
    for image_file in image_files:
        image_path = Path(image_file)
        output_path = Path(output_dir) / f"{image_path.stem}_frame0.tif"
        
        try:
            ext = image_path.suffix.lower()
            
            if ext == '.nd2':
                # ND2 文件
                import nd2
                with nd2.ND2File(image_file) as f:
                    data = f.asarray()
            elif ext in ['.tif', '.tiff']:
                # TIF/TIFF 文件
                data = tifffile.imread(str(image_path))
            else:
                print(f"Unsupported format: {ext}, skipping {image_path.name}")
                continue
            
            # 提取第一帧
            if data.ndim == 4:  # (T, C, H, W)
                first_frame = data[0]  # (C, H, W)
            elif data.ndim == 3:  # (C, H, W) or (T, H, W)
                # 假设第一个维度是时间或通道
                if data.shape[0] <= 4:  # 可能是 (C, H, W)
                    first_frame = data
                else:  # 可能是 (T, H, W)
                    first_frame = data[0]
            else:
                first_frame = data
            
            # 如果指定了 Cellpose 分割通道波长，按波长选择通道
            if cp_channel_wavelength is not None and ext == '.nd2':
                channel_infos = get_nd2_channel_info(image_file)
                if channel_infos:
                    ch_idx = find_closest_channel_index(channel_infos, cp_channel_wavelength)
                    if ch_idx is not None and first_frame.ndim >= 2:
                        # 如果 first_frame 是多通道 (C, H, W)
                        if first_frame.ndim == 3 and ch_idx < first_frame.shape[0]:
                            first_frame = first_frame[ch_idx]  # 提取单通道 (H, W)
                            print(f"  Cellpose: using channel index {ch_idx} (wavelength ~{cp_channel_wavelength}nm) for {image_path.name}")
                        else:
                            print(f"  Warning: failed to extract channel {ch_idx} from {image_path.name}, using all channels")
                    else:
                        print(f"  Warning: failed to map wavelength {cp_channel_wavelength}nm for {image_path.name}, using all channels")
            
            # 应用 Gamma 校正
            if gamma != 1.0:
                import numpy as np
                # 归一化到 [0, 1]
                first_frame = first_frame.astype(np.float32)
                min_val = first_frame.min()
                max_val = first_frame.max()
                if max_val > min_val:
                    first_frame = (first_frame - min_val) / (max_val - min_val)
                    # 应用 gamma 校正
                    first_frame = np.power(first_frame, gamma)
                    # 恢复到原始范围
                    first_frame = first_frame * (max_val - min_val) + min_val
                    first_frame = first_frame.astype(data.dtype)
            
            # 保存为 TIFF
            tifffile.imwrite(str(output_path), first_frame)
            extracted_files.append(str(output_path))
            print(f"Extracted first frame: {output_path.name}")
            
        except Exception as e:
            print(f"Error extracting frame from {image_file}: {e}")
    
    return extracted_files
