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
    cp_channel_wavelength: Optional[float] = None,
    batch_size: int = 10
) -> List[str]:
    """
    对图像文件列表运行 Cellpose 生成 mask (使用 Python API，模型只加载一次，分批读取避免爆内存)
    
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
    - batch_size: 每批次处理的图像数量（默认 10，避免内存爆炸）
    
    Returns:
    - 生成的 mask 文件路径列表
    """
    from cellpose import models, io
    
    # 使用绝对路径确保正确保存到指定目录
    output_dir = str(Path(output_dir).resolve())
    os.makedirs(output_dir, exist_ok=True)
    
    if not image_files:
        return []
    
    if verbose:
        print(f"Initializing Cellpose model (GPU={use_gpu}, device={gpu_device})...")
    
    # 1. 仅在这里加载一次模型到 GPU/CPU
    # Cellpose 4.x 使用 CellposeModel，默认模型为 cpsam (Cellpose-SAM)
    # device 参数需要 torch.device 对象，不能直接传整数
    import torch
    if use_gpu and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_device}')
    else:
        device = torch.device('cpu')
    model = models.CellposeModel(gpu=use_gpu, device=device, pretrained_model='cpsam')
    
    total_files = len(image_files)
    all_mask_files = []
    
    if verbose:
        print(f"Processing {total_files} images in batches of {batch_size}...")
    
    # 2. 分批处理：每批读取 -> 预测 -> 保存 -> 释放内存
    for batch_idx in range(0, total_files, batch_size):
        batch_end = min(batch_idx + batch_size, total_files)
        batch_files = image_files[batch_idx:batch_end]
        
        if verbose:
            print(f"\nBatch {batch_idx//batch_size + 1}: processing {len(batch_files)} images ({batch_idx}-{batch_end-1}/{total_files})")
        
        # 2.1 读取当前批次图片
        imgs = []
        valid_files = []
        
        for image_file in batch_files:
            try:
                img = io.imread(image_file)
                imgs.append(img)
                valid_files.append(image_file)
                if verbose:
                    print(f"  Loaded: {Path(image_file).name} (shape: {img.shape})")
            except Exception as e:
                print(f"  Error loading {Path(image_file).name}: {e}")
        
        if not imgs:
            print("  Warning: No images loaded in this batch")
            continue
        
        # 2.2 构建 eval 参数字典
        eval_kwargs = {
            'diameter': diameter,
            'channels': [0, 0],  # 默认使用所有通道
        }
        
        # 添加 niter 参数（如果指定）
        if niter is not None:
            eval_kwargs['niter'] = niter
        
        # 2.3 批量运行预测
        if verbose:
            print(f"  Running Cellpose evaluation on {len(imgs)} images...")
        
        masks, flows, styles, diams = model.eval(imgs, **eval_kwargs)
        
        # 2.4 批量保存结果
        if verbose:
            print(f"  Saving {len(masks)} masks...")
        
        for i, (image_file, mask) in enumerate(zip(valid_files, masks)):
            image_path = Path(image_file)
            
            # 生成输出文件名（与 CLI 版本兼容的命名格式）
            output_name = f"{image_path.stem}_seg.npy"
            output_path = Path(output_dir) / output_name
            
            # 保存 mask 为 npy 文件
            np.save(str(output_path), mask)
            all_mask_files.append(str(output_path))
            
            if verbose:
                print(f"    Saved mask: {output_path.name}")
            
            # 可选：保存 PNG 可视化
            if save_png:
                try:
                    png_path = Path(output_dir) / f"{image_path.stem}_seg.png"
                    # 创建 RGB 可视化
                    from matplotlib import pyplot as plt
                    plt.figure(figsize=(10, 10))
                    plt.imshow(mask > 0, cmap='gray')
                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(str(png_path), dpi=150, bbox_inches='tight')
                    plt.close()
                    if verbose:
                        print(f"    Saved PNG: {png_path.name}")
                except Exception as e:
                    if verbose:
                        print(f"    Warning: Failed to save PNG visualization: {e}")
        
        # 2.5 显式释放当前批次数据（帮助垃圾回收）
        del imgs, masks, flows, styles, diams
        if use_gpu:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    if verbose:
        print(f"\nCellpose completed. Generated {len(all_mask_files)} masks in total.")
    
    return all_mask_files


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
                        print(f"  Debug: first_frame.shape={first_frame.shape}, ch_idx={ch_idx}")
                        if first_frame.ndim == 3 and ch_idx < first_frame.shape[0]:
                            first_frame = first_frame[ch_idx]  # 提取单通道 (H, W)
                            print(f"  Cellpose: using channel index {ch_idx} (wavelength ~{cp_channel_wavelength}nm) for {image_path.name}")
                        else:
                            print(f"  Warning: failed to extract channel {ch_idx} from {image_path.name}, using all channels")
                            print(f"    Reason: ndim={first_frame.ndim}, shape={first_frame.shape}")
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
