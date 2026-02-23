import argparse
import os
import numpy as np
from scipy import ndimage
from skimage.registration import phase_cross_correlation
import matplotlib
matplotlib.use('Agg')  # 无GUI后端，必须在导入pyplot之前设置
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import nd2
from pathlib import Path
import json
from datetime import datetime
import tifffile  # 需要安装：pip install tifffile

# 尝试导入cupy和cucim
try:
    import cupy as cp
    from cucim.skimage.registration import phase_cross_correlation as phase_cross_correlation_gpu
    CUCIM_AVAILABLE = True
    print("CUDA acceleration enabled with CuPy + cuCIM")
except ImportError:
    CUCIM_AVAILABLE = False
    print("CuPy/cuCIM not available, using CPU only")

# 当前使用的 GPU 设备 ID
_current_gpu_device = 0

def set_gpu_device(device_id: int):
    """设置 CuPy 使用的 GPU 设备"""
    global _current_gpu_device
    if CUCIM_AVAILABLE and device_id >= 0:
        try:
            cp.cuda.Device(device_id).use()
            _current_gpu_device = device_id
            print(f"Motioncor: using GPU device {device_id}")
        except Exception as e:
            print(f"Warning: failed to set GPU device {device_id}: {e}")

def calculate_drift_correlation_gpu(image1, image2, upsample_factor=10):
    """使用cuCIM在GPU上计算相位相关漂移"""
    if not CUCIM_AVAILABLE:
        return calculate_drift_correlation_cpu(image1, image2, upsample_factor)
    
    try:
        # 将数据移到GPU
        img1_gpu = cp.asarray(image1, dtype=cp.float64)
        img2_gpu = cp.asarray(image2, dtype=cp.float64)
        
        # 使用cuCIM的GPU相位相关
        shift, error, diffphase = phase_cross_correlation_gpu(
            img1_gpu, img2_gpu, upsample_factor=upsample_factor
        )
        # shift可能是cupy数组，需要转换为标量
        shift_y = float(shift[0])
        shift_x = float(shift[1])
        return (-shift_y, -shift_x)
    except Exception as e:
        print(f"GPU calculation failed, falling back to CPU: {e}")
        return calculate_drift_correlation_cpu(image1, image2, upsample_factor)

def calculate_drift_correlation_cpu(image1, image2, upsample_factor=10):
    """使用CPU计算相位相关漂移"""
    shift, error, diffphase = phase_cross_correlation(
        image1, image2, upsample_factor=upsample_factor
    )
    return (-shift[0], -shift[1])

def calculate_drift_correlation(image1, image2, upsample_factor=10, use_gpu=True):
    """使用相位相关计算两帧之间的漂移，优先使用GPU"""
    if use_gpu and CUCIM_AVAILABLE:
        return calculate_drift_correlation_gpu(image1, image2, upsample_factor)
    else:
        return calculate_drift_correlation_cpu(image1, image2, upsample_factor)

def batch_phase_cross_correlation(images_batch1, images_batch2, upsample_factor=10, use_gpu=True):
    """
    批量计算相位相关漂移
    
    Parameters:
    - images_batch1: 批次图像1 (N, H, W)
    - images_batch2: 批次图像2 (N, H, W)
    - upsample_factor: 上采样因子
    - use_gpu: 是否使用GPU
    """
    if use_gpu and CUCIM_AVAILABLE:
        return batch_phase_cross_correlation_gpu(images_batch1, images_batch2, upsample_factor)
    else:
        return batch_phase_cross_correlation_cpu(images_batch1, images_batch2, upsample_factor)

def batch_phase_cross_correlation_cpu(images_batch1, images_batch2, upsample_factor=10):
    """CPU版本的批量相位相关计算"""
    shifts = []
    for img1, img2 in zip(images_batch1, images_batch2):
        shift, error, diffphase = phase_cross_correlation(
            img1, img2, upsample_factor=upsample_factor
        )
        shifts.append((-shift[0], -shift[1]))
    return shifts


def batch_phase_cross_correlation_gpu(images_batch1, images_batch2, upsample_factor=10):
    """
    GPU批量相位相关计算 - 使用cuCIM并保持数据在GPU上
    数据一次性上传到GPU，均在GPU上计算，最后一次性返回结果
    """
    if not CUCIM_AVAILABLE:
        return batch_phase_cross_correlation_cpu(images_batch1, images_batch2, upsample_factor)
    
    try:
        N = len(images_batch1)
        if N == 0:
            return []
        
        # 将批次数据一次性移到GPU，使用float64保证精度
        batch1_gpu = cp.asarray(images_batch1, dtype=cp.float64)
        batch2_gpu = cp.asarray(images_batch2, dtype=cp.float64)
        
        # 使用cuCIM的经过验证的phase_cross_correlation
        # 数据已在GPU上，无需额外传输
        shifts = []
        for i in range(N):
            shift, error, diffphase = phase_cross_correlation_gpu(
                batch1_gpu[i], batch2_gpu[i], 
                upsample_factor=upsample_factor
            )
            # shift可能是cupy数组，需要转换为标量
            shift_y = float(shift[0])
            shift_x = float(shift[1])
            shifts.append((-shift_y, -shift_x))
        
        return shifts
        
    except Exception as e:
        print(f"GPU batch calculation failed, falling back to CPU: {e}")
        import traceback
        traceback.print_exc()
        return batch_phase_cross_correlation_cpu(images_batch1, images_batch2, upsample_factor)

def detect_focus_loss(frames, threshold=0.7, background=100.0):
    """
    检测丢焦（信号强度突然持续下降）
    
    Parameters:
    - frames: 图像序列 (T, H, W) 或 (T, C, H, W)
    - threshold: 帧间强度比值阈值，低于此值判定为丢焦
    - background: 背景信号值，计算强度时扣除
    
    Returns:
    - focus_lost: bool, 是否检测到丢焦
    - focus_loss_frame: int or None, 丢焦开始的帧索引
    - intensity_ratios: list, 每帧相对前一帧的强度比值（扣除背景后）
    - frame_intensities: list, 每帧的平均强度（扣除背景后）
    """
    # 计算每帧的平均强度（扣除背景）
    if frames.ndim == 4:  # (T, C, H, W)
        frame_intensities = [max(0, np.mean(frames[t]) - background) for t in range(len(frames))]
    else:  # (T, H, W)
        frame_intensities = [max(0, np.mean(frames[t]) - background) for t in range(len(frames))]
    
    # 计算帧间强度比值
    intensity_ratios = [1.0]  # 第一帧比值设为1
    focus_lost = False
    focus_loss_frame = None
    
    for t in range(1, len(frame_intensities)):
        if frame_intensities[t-1] > 0:
            ratio = frame_intensities[t] / frame_intensities[t-1]
        else:
            ratio = 1.0
        intensity_ratios.append(ratio)
        
        # 检测丢焦
        if not focus_lost and ratio < threshold:
            focus_lost = True
            focus_loss_frame = t
    
    return focus_lost, focus_loss_frame, intensity_ratios, frame_intensities


def crop_to_valid_region(frames, cumulative_shifts, border=0):
    """
    根据漂移信息裁剪图像，确保所有帧都保持在原始视野内

    Parameters:
    - frames: 原始图像序列 (T, H, W)
    - cumulative_shifts: 累积漂移 (T, 2) - [x, y]
    - border: 额外保留的边框像素数
    """
    H, W = frames.shape[1], frames.shape[2]

    # 计算所有帧的漂移范围
    x_shifts = cumulative_shifts[:, 0]
    y_shifts = cumulative_shifts[:, 1]

    # 计算需要裁剪的边界 - 修正方向
    min_x_shift, max_x_shift = np.min(x_shifts), np.max(x_shifts)
    min_y_shift, max_y_shift = np.min(y_shifts), np.max(y_shifts)

    # 计算裁剪边界（考虑漂移后的有效区域）
    # 修正：left_crop应该考虑最大负偏移，right_crop考虑最大正偏移
    left_crop = int(np.ceil(max(0, -min_x_shift + border)))
    right_crop = int(np.floor(W - max(0, max_x_shift + border)))
    top_crop = int(np.ceil(max(0, -min_y_shift + border)))
    bottom_crop = int(np.floor(H - max(0, max_y_shift + border)))

    # 确保裁剪边界有效
    left_crop = max(0, min(left_crop, W-1))
    right_crop = max(left_crop + 1, min(right_crop, W))
    top_crop = max(0, min(top_crop, H-1))
    bottom_crop = max(top_crop + 1, min(bottom_crop, H))

    # 裁剪图像
    cropped_frames = frames[:, top_crop:bottom_crop, left_crop:right_crop]

    # 更新漂移信息（相对于新坐标系）
    adjusted_shifts = cumulative_shifts.copy()
    adjusted_shifts[:, 0] -= left_crop
    adjusted_shifts[:, 1] -= top_crop

    print(f"Cropped region: [{left_crop}:{right_crop}, {top_crop}:{bottom_crop}]")
    print(f"Original size: {W}x{H}, Cropped size: {right_crop-left_crop}x{bottom_crop-top_crop}")

    return cropped_frames, adjusted_shifts, (left_crop, right_crop, top_crop, bottom_crop)

def apply_shifts_batch(frames, shifts, batch_size=30):
    """
    批量应用位移校正
    
    Parameters:
    - frames: 原始图像序列 (T, C, H, W)
    - shifts: 位移列表 [(x, y), ...]
    - batch_size: 批处理大小
    """
    T, C, H, W = frames.shape
    corrected_frames = np.zeros_like(frames, dtype=frames.dtype)
    
    # 分批处理
    for start_idx in range(0, T, batch_size):
        end_idx = min(start_idx + batch_size, T)
        batch_frames = frames[start_idx:end_idx]
        batch_shifts = shifts[start_idx:end_idx]
        
        # 对当前批次的每一帧进行校正
        for i, (shift_x, shift_y) in enumerate(batch_shifts):
            for c in range(C):  # 遍历所有通道
                corrected_frame = ndimage.shift(
                    batch_frames[i, c],
                    (-shift_y, -shift_x),  # 负号确保正确的校正方向
                    order=1,
                    mode='nearest',
                    cval=0
                )
                corrected_frames[start_idx + i, c] = corrected_frame
    
    return corrected_frames

def iterative_drift_correction(frames, max_iterations=10, threshold=0.5, 
                             upsample_factor=50, batch_size=100, use_gpu=True):
    """
    迭代进行漂移校正，支持批处理和GPU加速

    Parameters:
    - frames: 输入图像序列 (T, H, W) - 用于计算漂移的图像（如合并通道）
    - max_iterations: 最大迭代次数
    - threshold: 停止迭代的阈值（像素）
    - upsample_factor: 相位相关算法的上采样因子
    - batch_size: 批处理大小
    - use_gpu: 是否使用GPU
    """
    T, H, W = frames.shape
    original_frames = frames.copy()  # 保存原始用于计算漂移的图像，每次迭代都从这里开始

    # 初始化累积漂移
    cumulative_shifts = np.zeros((T, 2))  # [x, y] for each frame

    print(f"Starting iterative drift correction (max {max_iterations} iterations, threshold {threshold} pixels, batch_size {batch_size}, GPU: {use_gpu and CUCIM_AVAILABLE})")

    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}:")

        # 使用当前累积漂移校正原始图像（批处理）
        corrected_frames = np.zeros_like(original_frames, dtype=original_frames.dtype)
        corrected_frames[0] = original_frames[0]  # 第一帧不变

        # 批量校正
        for start_idx in range(1, T, batch_size):  # 从第1帧开始（第0帧不变）
            end_idx = min(start_idx + batch_size, T)
            batch_shifts = []
            
            for i in range(start_idx, end_idx):
                shift_x, shift_y = cumulative_shifts[i]
                corrected_frame = ndimage.shift(
                    original_frames[i],
                    (-shift_y, -shift_x),  # 负号确保正确的校正方向
                    order=1,
                    mode='nearest',
                    cval=0
                )
                corrected_frames[i] = corrected_frame

        # 计算当前校正后帧序列的漂移（批处理）
        shifts = [(0, 0)]  # 第一帧作为参考，偏移为(0,0)

        # 批量计算漂移
        for start_idx in range(1, T, batch_size):
            end_idx = min(start_idx + batch_size, T)
            
            # 准备批次数据
            batch_prev = corrected_frames[start_idx-1:end_idx-1]  # 前一帧
            batch_curr = corrected_frames[start_idx:end_idx]      # 当前帧
            
            if len(batch_prev) > 0:
                # 批量计算相位相关
                batch_shifts = batch_phase_cross_correlation(
                    batch_prev, batch_curr, upsample_factor, use_gpu
                )
                shifts.extend(batch_shifts)

        # 将新的漂移添加到累积漂移中
        new_cumulative_shifts = np.zeros((T, 2))
        cum_x, cum_y = 0.0, 0.0
        for idx, shift in enumerate(shifts):
            if idx > 0:  # 从第二帧开始累加
                cum_y += shift[0]  # 注意：相位相关返回的是 (row, col) -> (y, x)
                cum_x += shift[1]
            new_cumulative_shifts[idx] = [cum_x, cum_y]

        # 检查是否收敛
        max_drift = max(np.max(np.abs(new_cumulative_shifts[:, 0])), np.max(np.abs(new_cumulative_shifts[:, 1])))
        print(f"  Max drift: X={np.max(np.abs(new_cumulative_shifts[:, 0])):.3f}, Y={np.max(np.abs(new_cumulative_shifts[:, 1])):.3f}, Total={max_drift:.3f}")

        if max_drift < threshold:
            print(f"  Converged at iteration {iteration + 1} (max drift {max_drift:.3f} < threshold {threshold})")
            cumulative_shifts += new_cumulative_shifts
            break

        # 更新累积漂移
        cumulative_shifts += new_cumulative_shifts

    else:
        print(f"Reached maximum iterations ({max_iterations})")

    return cumulative_shifts  # 只返回累积漂移，不返回校正后的图像

def process_image_sequence(input_path, output_dir, channel_selection='all',
                          sample_interval=1, save_visualization=True, auto_crop=True, 
                          border=0, max_iterations=10, threshold=0.5, 
                          batch_size=100, use_gpu=True, gpu_device=0,
                          focus_loss_threshold=0.7, skip_focus_loss=True,
                          focus_loss_background=100.0):
    """
    处理图像序列并进行漂移校正

    Parameters:
    - input_path: 输入文件路径 (ND2或TIF)
    - output_dir: 输出目录
    - channel_selection: 通道选择 ('all', 'first', 'last', or specific index)
    - sample_interval: 可视化采样间隔
    - save_visualization: 是否保存可视化
    - auto_crop: 是否在漂移校正后自动裁剪
    - border: 裁剪时保留的边框像素数
    - max_iterations: 最大迭代次数
    - threshold: 停止迭代的阈值（像素）
    - batch_size: 批处理大小
    - use_gpu: 是否使用GPU
    - gpu_device: GPU 设备 ID（默认 0）
    - focus_loss_threshold: 丢焦检测阈值（帧间强度比值，默认0.7）
    - skip_focus_loss: 是否跳过丢焦序列（默认True）
    - focus_loss_background: 丢焦检测时扣除的背景值（默认100.0）
    
    Returns:
    - cumulative_shifts: 累积漂移数组，如果跳过则为 None
    - corrected_tiff_path: 校正后的文件路径，如果跳过则为 None
    - focus_loss_info: 丢焦检测信息字典
    """
    
    # 设置 GPU 设备
    if use_gpu and CUCIM_AVAILABLE:
        set_gpu_device(gpu_device)

    input_path = Path(input_path)
    print(f"Processing: {input_path}")

    # 使用原始文件名创建输出路径，避免截断问题
    base_name = input_path.stem  # 保持完整的文件名
    output_path = Path(output_dir) / f"{base_name}_corrected"

    # 读取图像数据
    if input_path.suffix.lower() == '.nd2':
        try:
            with nd2.ND2File(input_path) as ndfile:
                # 获取基本信息
                shape = dict(ndfile.sizes)
                if 'T' not in shape or 'C' not in shape or 'Y' not in shape or 'X' not in shape:
                    print(f"Error: Required dimensions not found in {input_path}")
                    print(f"Available dimensions: {list(shape.keys())}")
                    return None, None

                T, C, Y, X = shape['T'], shape['C'], shape['Y'], shape['X']

                print(f"File shape: T={T}, C={C}, Y={Y}, X={X}")

                # 确定处理的通道
                if channel_selection == 'all':
                    channels_to_process = list(range(C))
                elif channel_selection == 'first':
                    channels_to_process = [0]
                elif channel_selection == 'last':
                    channels_to_process = [C-1]
                else:
                    try:
                        ch_idx = int(channel_selection)
                        if 0 <= ch_idx < C:
                            channels_to_process = [ch_idx]
                        else:
                            print(f"Invalid channel index {ch_idx}, using all channels")
                            channels_to_process = list(range(C))
                    except ValueError:
                        print(f"Invalid channel selection {channel_selection}, using all channels")
                        channels_to_process = list(range(C))

                print(f"Processing channels: {channels_to_process}")

                # 读取所有帧
                all_frames = ndfile.asarray()  # Shape: (T, C, Y, X)
                print(f"Read frames with shape: {all_frames.shape}")

                # 用于计算漂移的图像 - 合并选定的通道
                if len(channels_to_process) == 1:
                    drift_frames = all_frames[:, channels_to_process[0], :, :]
                else:
                    # 合并多个通道 - 使用最大值投影
                    selected_frames = all_frames[:, channels_to_process, :, :]
                    drift_frames = np.max(selected_frames, axis=1)  # 沿通道轴取最大值

                # 用于校正的原始图像 - 保留所有通道
                original_frames = all_frames  # Shape: (T, C, Y, X)
                channels_to_save = channels_to_process  # 记录要保存的通道

        except Exception as e:
            print(f"Error reading ND2 file {input_path}: {str(e)}")
            return None, None, None

    elif input_path.suffix.lower() in ['.tif', '.tiff']:
        # 读取TIF文件
        try:
            with tifffile.TiffFile(str(input_path)) as tif:
                all_frames = tif.asarray()
                print(f"Read TIF frames with shape: {all_frames.shape}")

                # 处理不同维度的TIF文件
                if all_frames.ndim == 3:  # (T, H, W)
                    T, Y, X = all_frames.shape
                    C = 1
                    drift_frames = all_frames
                    original_frames = all_frames  # Shape: (T, 1, Y, X) - 保持一致的维度
                    if all_frames.ndim == 3:
                        original_frames = all_frames[:, np.newaxis, :, :]  # 添加通道维度
                    channels_to_save = [0]
                    print(f"TIF file shape: T={T}, Y={Y}, X={X} (single channel)")
                elif all_frames.ndim == 4:  # (T, C, H, W) or (T, H, W, C)
                    if all_frames.shape[1] < all_frames.shape[-1]:  # Assume (T, C, H, W)
                        T, C, Y, X = all_frames.shape
                        if channel_selection == 'all':
                            channels_to_process = list(range(C))
                        elif channel_selection == 'first':
                            channels_to_process = [0]
                        elif channel_selection == 'last':
                            channels_to_process = [C-1]
                        else:
                            try:
                                ch_idx = int(channel_selection)
                                if 0 <= ch_idx < C:
                                    channels_to_process = [ch_idx]
                                else:
                                    print(f"Invalid channel index {ch_idx}, using first channel")
                                    channels_to_process = [0]
                            except ValueError:
                                print(f"Invalid channel selection {channel_selection}, using first channel")
                                channels_to_process = [0]

                        # 用于计算漂移的图像 - 合并选定的通道
                        if len(channels_to_process) == 1:
                            drift_frames = all_frames[:, channels_to_process[0], :, :]
                        else:
                            selected_frames = all_frames[:, channels_to_process, :, :]
                            drift_frames = np.max(selected_frames, axis=1)  # 沿通道轴取最大值

                        original_frames = all_frames  # Shape: (T, C, Y, X)
                        channels_to_save = channels_to_process
                        print(f"TIF file shape: T={T}, C={C}, Y={Y}, X={X}")
                        print(f"Processing channels: {channels_to_process}")
                    else:  # Assume (T, H, W, C)
                        T, Y, X, C = all_frames.shape
                        print(f"TIF file shape: T={T}, Y={Y}, X={X}, C={C}")
                        if channel_selection == 'all':
                            channels_to_process = list(range(C))
                        elif channel_selection == 'first':
                            channels_to_process = [0]
                        elif channel_selection == 'last':
                            channels_to_process = [C-1]
                        else:
                            try:
                                ch_idx = int(channel_selection)
                                if 0 <= ch_idx < C:
                                    channels_to_process = [ch_idx]
                                else:
                                    print(f"Invalid channel index {ch_idx}, using first channel")
                                    channels_to_process = [0]
                            except ValueError:
                                print(f"Invalid channel selection {channel_selection}, using first channel")
                                channels_to_process = [0]

                        # 用于计算漂移的图像 - 合并选定的通道
                        if len(channels_to_process) == 1:
                            drift_frames = all_frames[:, :, :, channels_to_process[0]]
                        else:
                            selected_frames = all_frames[:, :, :, channels_to_process]
                            drift_frames = np.max(selected_frames, axis=-1)  # 沿通道轴取最大值

                        # 转换为 (T, C, Y, X) 格式
                        original_frames = np.transpose(all_frames, (0, 3, 1, 2))  # (T, C, Y, X)
                        channels_to_save = channels_to_process
                else:
                    raise ValueError(f"Unsupported TIF file dimensions: {all_frames.ndim}")
        except Exception as e:
            print(f"Error reading TIF file {input_path}: {str(e)}")
            return None, None, None
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")

    # 验证数据
    if original_frames is None or original_frames.size == 0:
        print(f"Error: No data read from {input_path}")
        return None, None, None
    
    # 丢焦检测
    focus_lost, focus_loss_frame, intensity_ratios, frame_intensities = detect_focus_loss(
        drift_frames, threshold=focus_loss_threshold, background=focus_loss_background
    )
    
    focus_loss_info = {
        'file': str(input_path),
        'focus_lost': focus_lost,
        'focus_loss_frame': focus_loss_frame,
        'total_frames': len(drift_frames),
        'intensity_ratios': intensity_ratios,
        'frame_intensities': frame_intensities,
        'threshold': focus_loss_threshold
    }
    
    if focus_lost:
        if focus_loss_frame is not None:
            ratio_at_loss = intensity_ratios[focus_loss_frame]
            print(f"WARNING: Focus loss detected at frame {focus_loss_frame} "
                  f"(intensity ratio: {ratio_at_loss:.3f} < {focus_loss_threshold})")
        
        if skip_focus_loss:
            print(f"Skipping sequence due to focus loss: {input_path}")
            return None, None, focus_loss_info

    # 执行迭代漂移校正 - 只计算漂移，不应用
    try:
        cumulative_shifts = iterative_drift_correction(
            drift_frames, max_iterations=max_iterations, threshold=threshold,
            batch_size=batch_size, use_gpu=use_gpu
        )
    except Exception as e:
        print(f"Error in drift correction for {input_path}: {str(e)}")
        return None, None, focus_loss_info

    # 应用漂移到原始的每个通道（批处理）
    print("Applying calculated drift to original frames...")
    try:
        # 使用批处理应用校正
        corrected_frames = apply_shifts_batch(original_frames, 
                                            [(shift[0], shift[1]) for shift in cumulative_shifts], 
                                            batch_size)
    except Exception as e:
        print(f"Error applying drift correction: {str(e)}")
        return None, None, focus_loss_info

    # 如果启用自动裁剪，需要对多通道数据进行裁剪
    if auto_crop:
        print("Applying automatic cropping to keep all frames in field of view...")
        original_shape = corrected_frames.shape
        # 对每个通道应用相同的裁剪
        H, W = corrected_frames.shape[2], corrected_frames.shape[3]

        # 计算裁剪边界
        x_shifts = cumulative_shifts[:, 0]
        y_shifts = cumulative_shifts[:, 1]

        min_x_shift, max_x_shift = np.min(x_shifts), np.max(x_shifts)
        min_y_shift, max_y_shift = np.min(y_shifts), np.max(y_shifts)

        left_crop = int(np.ceil(max(0, -min_x_shift + border)))
        right_crop = int(np.floor(W - max(0, max_x_shift + border)))
        top_crop = int(np.ceil(max(0, -min_y_shift + border)))
        bottom_crop = int(np.floor(H - max(0, max_y_shift + border)))

        # 确保裁剪边界有效
        left_crop = max(0, min(left_crop, W-1))
        right_crop = max(left_crop + 1, min(right_crop, W))
        top_crop = max(0, min(top_crop, H-1))
        bottom_crop = max(top_crop + 1, min(bottom_crop, H))

        # 裁剪多通道图像
        corrected_frames = corrected_frames[:, :, top_crop:bottom_crop, left_crop:right_crop]

        # 更新漂移信息（相对于新坐标系）
        cumulative_shifts[:, 0] -= left_crop
        cumulative_shifts[:, 1] -= top_crop

        print(f"Auto-cropped from {original_shape} to {corrected_frames.shape}")

    # 创建输出目录
    output_path = Path(output_dir) / f"{base_name}_corrected"

    # 保存漂移信息
    drift_info = {
        'file': str(input_path),
        'original_shape': drift_frames.shape,
        'corrected_shape': corrected_frames.shape,
        'channel_selection': channel_selection,
        'cumulative_shifts': [list(s) for s in cumulative_shifts],
        'processing_time': datetime.now().isoformat(),
        'sample_interval': sample_interval,
        'total_frames': len(original_frames),
        'max_x_drift': float(np.max(np.abs(cumulative_shifts[:, 0]))),
        'max_y_drift': float(np.max(np.abs(cumulative_shifts[:, 1]))),
        'final_x_drift': float(cumulative_shifts[-1, 0]),
        'final_y_drift': float(cumulative_shifts[-1, 1]),
        'auto_crop_enabled': auto_crop,
        'border_pixels': border,
        'max_iterations': max_iterations,
        'threshold': threshold,
        'batch_size': batch_size,
        'use_gpu': use_gpu and CUCIM_AVAILABLE,
        'gpu_available': CUCIM_AVAILABLE,
        'channels_saved': channels_to_save
    }

    if auto_crop:
        drift_info['crop_info'] = (int(left_crop), int(right_crop), int(top_crop), int(bottom_crop))

    drift_json_path = output_path.with_name(f"{base_name}_corrected_drift_info.json")
    try:
        with open(drift_json_path, 'w') as f:
            json.dump(drift_info, f, indent=2, ensure_ascii=False)
        print(f"Drift info saved to: {drift_json_path}")
    except Exception as e:
        print(f"Error saving drift info: {str(e)}")

    # 创建可视化
    if save_visualization:
        try:
            create_drift_visualization(
                cumulative_shifts,
                output_dir,
                base_name,
                sample_interval
            )
        except Exception as e:
            print(f"Error creating visualization: {str(e)}")

    # 保存校正后的数据
    corrected_tiff_path = output_path.with_name(f"{base_name}_corrected.ome.tif")

    # 如果只保存选定的通道
    if len(channels_to_save) < corrected_frames.shape[1]:
        corrected_frames_to_save = corrected_frames[:, channels_to_save, :, :]
    else:
        corrected_frames_to_save = corrected_frames

    # 保存校正后的数据
    try:
        tifffile.imwrite(
            str(corrected_tiff_path),
            corrected_frames_to_save,
            ome=True,
            metadata={'axes': 'TCYX'}  # 指定轴顺序
        )
        print(f"Corrected data saved to: {corrected_tiff_path}")
    except Exception as e:
        print(f"Error saving corrected data: {str(e)}")
        return cumulative_shifts, None, focus_loss_info

    return cumulative_shifts, corrected_tiff_path, focus_loss_info

def create_drift_visualization(cumulative_shifts, output_dir, base_name, sample_interval=1):
    """创建漂移轨迹的可视化"""

    # 准备数据
    time_points = np.arange(len(cumulative_shifts))
    x_shifts = cumulative_shifts[:, 0]
    y_shifts = cumulative_shifts[:, 1]

    # 如果采样间隔大于1，对数据进行采样
    if sample_interval > 1:
        indices = np.arange(0, len(cumulative_shifts), sample_interval)
        time_points = time_points[indices]
        x_shifts = x_shifts[indices]
        y_shifts = y_shifts[indices]

    # 确保坐标轴范围至少为10个像素
    x_range = np.max(np.abs(x_shifts)) - np.min(np.abs(x_shifts))
    y_range = np.max(np.abs(y_shifts)) - np.min(np.abs(y_shifts))

    if x_range < 10:
        x_center = (np.max(x_shifts) + np.min(x_shifts)) / 2
        x_min = x_center - 5
        x_max = x_center + 5
    else:
        x_min = np.min(x_shifts) - 1
        x_max = np.max(x_shifts) + 1

    if y_range < 10:
        y_center = (np.max(y_shifts) + np.min(y_shifts)) / 2
        y_min = y_center - 5
        y_max = y_center + 5
    else:
        y_min = np.min(y_shifts) - 1
        y_max = np.max(y_shifts) + 1

    # 创建综合可视化图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Drift Correction Summary - {base_name}', fontsize=16, fontweight='bold')

    # 1. 轨迹图 (XY平面)
    scatter = axes[0, 0].scatter(x_shifts, y_shifts, c=time_points,
                                cmap='viridis', s=20, alpha=0.7, edgecolors='none')
    axes[0, 0].plot(x_shifts, y_shifts, 'k-', alpha=0.3, linewidth=0.5)
    axes[0, 0].set_xlabel('X Shift (pixels)')
    axes[0, 0].set_ylabel('Y Shift (pixels)')
    axes[0, 0].set_title('Drift Trajectory (XY Plane)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(x_min, x_max)
    axes[0, 0].set_ylim(y_min, y_max)
    plt.colorbar(scatter, ax=axes[0, 0], label='Frame Index')

    # 2. X方向时间序列
    axes[0, 1].plot(time_points, x_shifts, 'b-', linewidth=1.5, alpha=0.8)
    axes[0, 1].scatter(time_points, x_shifts, c=time_points, cmap='viridis', s=20, alpha=0.7)
    axes[0, 1].set_xlabel('Frame Index')
    axes[0, 1].set_ylabel('X Shift (pixels)')
    axes[0, 1].set_title('X Drift vs Time')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(x_min, x_max)

    # 3. Y方向时间序列
    axes[1, 0].plot(time_points, y_shifts, 'r-', linewidth=1.5, alpha=0.8)
    axes[1, 0].scatter(time_points, y_shifts, c=time_points, cmap='viridis', s=20, alpha=0.7)
    axes[1, 0].set_xlabel('Frame Index')
    axes[1, 0].set_ylabel('Y Shift (pixels)')
    axes[1, 0].set_title('Y Drift vs Time')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(y_min, y_max)

    # 4. 2D直方图显示密度
    im = axes[1, 1].hexbin(x_shifts, y_shifts, gridsize=30, cmap='Blues', mincnt=1)
    axes[1, 1].set_xlabel('X Shift (pixels)')
    axes[1, 1].set_ylabel('Y Shift (pixels)')
    axes[1, 1].set_title('Drift Density Distribution')
    axes[1, 1].set_xlim(x_min, x_max)
    axes[1, 1].set_ylim(y_min, y_max)
    plt.colorbar(im, ax=axes[1, 1], label='Point Density')

    plt.tight_layout()

    # 保存可视化
    viz_path = Path(output_dir) / f"{base_name}_drift_visualization.png"
    try:
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to: {viz_path}")
    except Exception as e:
        print(f"Error saving visualization: {str(e)}")
        plt.close()

def create_summary_plot(all_shifts_data, output_dir):
    """创建所有文件的漂移轨迹总结图"""
    if not all_shifts_data:
        print("No data to create summary plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Summary of All Files Drift Trajectories', fontsize=16, fontweight='bold')

    # 收集所有文件的x、y漂移范围
    all_x_min, all_x_max = float('inf'), float('-inf')
    all_y_min, all_y_max = float('inf'), float('-inf')

    # 绘制所有文件的轨迹
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_shifts_data)))

    for idx, (filename, shifts) in enumerate(all_shifts_data):
        x_shifts = shifts[:, 0]
        y_shifts = shifts[:, 1]
        time_points = np.arange(len(shifts))

        # 更新全局范围
        current_x_min, current_x_max = np.min(x_shifts), np.max(x_shifts)
        current_y_min, current_y_max = np.min(y_shifts), np.max(y_shifts)

        all_x_min = min(all_x_min, current_x_min)
        all_x_max = max(all_x_max, current_x_max)
        all_y_min = min(all_y_min, current_y_min)
        all_y_max = max(all_y_max, current_y_max)

        # 1. 轨迹图 (XY平面)
        axes[0, 0].plot(x_shifts, y_shifts, color=colors[idx], alpha=0.7, linewidth=1.5,
                       label=Path(filename).stem)
        axes[0, 0].scatter(x_shifts[0], y_shifts[0], color=colors[idx], s=50, marker='o',
                          edgecolors='black', zorder=5)  # 标记起点
        axes[0, 0].scatter(x_shifts[-1], y_shifts[-1], color=colors[idx], s=50, marker='s',
                          edgecolors='black', zorder=5)  # 标记终点

    # 确保坐标轴范围至少为10个像素
    x_range = all_x_max - all_x_min
    y_range = all_y_max - all_y_min

    if x_range < 10:
        x_center = (all_x_max + all_x_min) / 2
        all_x_min = x_center - 5
        all_x_max = x_center + 5
    else:
        all_x_min -= 1
        all_x_max += 1

    if y_range < 10:
        y_center = (all_y_max + all_y_min) / 2
        all_y_min = y_center - 5
        all_y_max = y_center + 5
    else:
        all_y_min -= 1
        all_y_max += 1

    axes[0, 0].set_xlabel('X Shift (pixels)')
    axes[0, 0].set_ylabel('Y Shift (pixels)')
    axes[0, 0].set_title('All Files - Drift Trajectory (XY Plane)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(all_x_min, all_x_max)
    axes[0, 0].set_ylim(all_y_min, all_y_max)
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # 2. X方向时间序列
    for idx, (filename, shifts) in enumerate(all_shifts_data):
        x_shifts = shifts[:, 0]
        time_points = np.arange(len(shifts))
        axes[0, 1].plot(time_points, x_shifts, color=colors[idx], alpha=0.7, linewidth=1.5,
                       label=Path(filename).stem)

    axes[0, 1].set_xlabel('Frame Index')
    axes[0, 1].set_ylabel('X Shift (pixels)')
    axes[0, 1].set_title('All Files - X Drift vs Time')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(all_x_min, all_x_max)
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # 3. Y方向时间序列
    for idx, (filename, shifts) in enumerate(all_shifts_data):
        y_shifts = shifts[:, 1]
        time_points = np.arange(len(shifts))
        axes[1, 0].plot(time_points, y_shifts, color=colors[idx], alpha=0.7, linewidth=1.5,
                       label=Path(filename).stem)

    axes[1, 0].set_xlabel('Frame Index')
    axes[1, 0].set_ylabel('Y Shift (pixels)')
    axes[1, 0].set_title('All Files - Y Drift vs Time')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(all_y_min, all_y_max)
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # 4. 统计信息
    axes[1, 1].axis('off')
    stats_text = "Summary Statistics:\n\n"
    for idx, (filename, shifts) in enumerate(all_shifts_data):
        x_shifts = shifts[:, 0]
        y_shifts = shifts[:, 1]

        max_x_drift = np.max(np.abs(x_shifts))
        max_y_drift = np.max(np.abs(y_shifts))
        final_x_drift = x_shifts[-1]
        final_y_drift = y_shifts[-1]

        stats_text += f"{Path(filename).stem}:\n"
        stats_text += f"  Max X drift: {max_x_drift:.2f}px\n"
        stats_text += f"  Max Y drift: {max_y_drift:.2f}px\n"
        stats_text += f"  Final X drift: {final_x_drift:.2f}px\n"
        stats_text += f"  Final Y drift: {final_y_drift:.2f}px\n\n"

    axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                    verticalalignment='top', fontsize=10, fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

    plt.tight_layout()

    # 保存总结图
    summary_path = Path(output_dir) / "all_files_drift_summary.png"
    try:
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Summary plot saved to: {summary_path}")
    except Exception as e:
        print(f"Error saving summary plot: {str(e)}")
        plt.close()

def split_nd2_by_position(image_file, output_dir):
    """如果 ND2 文件包含多个位置 (P 维度 > 1)，将其按视野拆分为多个 TIF 文件。
    返回用于后续 motioncor 处理的文件列表。
    """
    image_path = Path(image_file)
    if image_path.suffix.lower() != '.nd2':
        return [image_file]

    try:
        with nd2.ND2File(str(image_path)) as f:
            sizes = f.sizes
            if 'P' not in sizes or sizes['P'] <= 1:
                return [image_file]

            # nd2 库的 sizes 是有序字典，键的顺序就是轴顺序
            axes = ''.join(sizes.keys())
            if not all(ax in axes for ax in ['T', 'C', 'Y', 'X']):
                print(f"Multi-position ND2 with unsupported axes {axes}, fallback to original")
                return [image_file]

            print(f"Detected multi-position ND2: {image_path.name}, P={sizes['P']}")
            data = f.asarray()
            p_axis = axes.index('P')
            base_name = image_path.stem

            split_files = []
            for p in range(sizes['P']):
                selector = [slice(None)] * data.ndim
                selector[p_axis] = p
                pos_data = data[tuple(selector)]

                # 删除 P 轴后，当前实现只支持 TCYX 顺序
                pos_axes = axes.replace('P', '')
                if pos_axes != 'TCYX':
                    print(f"Multi-position ND2 axes {axes} (after removing P: {pos_axes}) not supported for splitting, fallback to original")
                    return [image_file]

                out_name = f"{base_name}_P{p}.tif"
                out_path = Path(output_dir) / out_name

                tifffile.imwrite(
                    str(out_path),
                    pos_data,
                    ome=True,
                    metadata={'axes': 'TCYX'}
                )
                split_files.append(str(out_path))

            print(f"Split {image_path.name} into {len(split_files)} positions")
            return split_files
    except Exception as e:
        print(f"Error splitting ND2 file {image_path.name}: {e}")
        return [image_file]



def main():
    parser = argparse.ArgumentParser(description='ND2/TIF Motion Correction Tool with GPU acceleration')
    parser.add_argument('input_dir', help='Input directory containing ND2/TIF files')
    parser.add_argument('--output_dir', '-o', help='Output directory for corrected files (default: MotionCor subdirectory)')
    parser.add_argument('--channel', '-c', default='all',
                       help='Channel to use for drift calculation (all, first, last, or specific index)')
    parser.add_argument('--sample_interval', '-s', type=int, default=5,
                       help='Sampling interval for visualization (default: 5)')
    parser.add_argument('--no_visualization', action='store_true',
                       help='Skip visualization creation')
    parser.add_argument('--no_auto_crop', action='store_true',
                       help='Disable automatic cropping to keep all frames in field of view')
    parser.add_argument('--border', type=int, default=0,
                       help='Additional border pixels to keep when auto-cropping (default: 0)')
    parser.add_argument('--max_iterations', type=int, default=10,
                       help='Maximum number of iterative corrections (default: 10)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for stopping iteration (default: 0.5 pixels)')
    parser.add_argument('--batch_size', type=int, default=100,
                       help='Batch size for GPU processing (default: 100, recommended 50-200)')
    parser.add_argument('--no_gpu', action='store_true',
                       help='Disable GPU acceleration')
    parser.add_argument('--focus_loss_threshold', type=float, default=0.7,
                       help='Threshold for focus loss detection (frame intensity ratio, default: 0.7)')
    parser.add_argument('--no_skip_focus_loss', action='store_true',
                       help='Do not skip sequences with focus loss (process anyway)')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return

    # 设置输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_dir / 'MotionCor'

    output_dir.mkdir(exist_ok=True)

    # 查找所有ND2和TIF文件
    nd2_files = list(input_dir.glob('*.nd2'))
    tif_files = list(input_dir.glob('*.tif')) + list(input_dir.glob('*.tiff'))

    n_nd2 = len(nd2_files)
    n_tif = len(tif_files)

    # 对 ND2 文件进行多视野拆分（如果存在 P 维度）
    all_files = []
    for nd2_file in nd2_files:
        split_files = split_nd2_by_position(str(nd2_file), output_dir)
        all_files.extend([Path(f) for f in split_files])

    # 直接加入已有的 TIF 文件
    all_files.extend(tif_files)

    if not all_files:
        print(f"No ND2 or TIF files found in {input_dir}")
        return

    print(f"Found {len(all_files)} sequences to process ({n_nd2} ND2 files, {n_tif} TIF files)")

    # 检查GPU可用性
    use_gpu = CUCIM_AVAILABLE and not args.no_gpu
    print(f"GPU acceleration: {'Enabled' if use_gpu else 'Disabled'} (cuCIM available: {CUCIM_AVAILABLE})")

    # 存储所有文件的漂移数据
    all_shifts_data = []
    
    # 存储丢焦信息
    focus_loss_records = []  # 记录丢焦的序列
    processed_records = []   # 记录成功处理的序列

    for file_path in all_files:
        try:
            print(f"\nProcessing file: {file_path.name}")
            shifts, output_path, focus_loss_info = process_image_sequence(
                str(file_path),
                str(output_dir),
                channel_selection=args.channel,
                sample_interval=args.sample_interval,
                save_visualization=not args.no_visualization,
                auto_crop=not args.no_auto_crop,
                border=args.border,
                max_iterations=args.max_iterations,
                threshold=args.threshold,
                batch_size=args.batch_size,
                use_gpu=use_gpu,
                focus_loss_threshold=args.focus_loss_threshold,
                skip_focus_loss=not args.no_skip_focus_loss
            )
            
            # 记录丢焦信息
            if focus_loss_info and focus_loss_info.get('focus_lost', False):
                focus_loss_records.append(focus_loss_info)

            if shifts is not None and output_path is not None:
                # 添加到总结数据中
                all_shifts_data.append((str(file_path), shifts))
                
                # 记录成功处理
                processed_records.append({
                    'file': str(file_path),
                    'output': str(output_path),
                    'max_x_drift': float(np.max(np.abs(shifts[:, 0]))),
                    'max_y_drift': float(np.max(np.abs(shifts[:, 1]))),
                    'final_x_drift': float(shifts[-1, 0]),
                    'final_y_drift': float(shifts[-1, 1])
                })

                # 打印统计信息
                total_x_drift = np.abs(shifts[-1, 0])
                total_y_drift = np.abs(shifts[-1, 1])
                max_x_drift = np.max(np.abs(shifts[:, 0]))
                max_y_drift = np.max(np.abs(shifts[:, 1]))

                print(f"File: {file_path.name}")
                print(f"  Final drift: X={total_x_drift:.2f}, Y={total_y_drift:.2f} pixels")
                print(f"  Max drift: X={max_x_drift:.2f}, Y={max_y_drift:.2f} pixels")
                print(f"  Max single-step drift: X={np.max(np.abs(np.diff(shifts[:, 0])))}, Y={np.max(np.abs(np.diff(shifts[:, 1])))} pixels")
                print(f"  Correction completed: {output_path}")
            else:
                print(f"Failed to process: {file_path.name}")

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # 创建总结图
    if all_shifts_data and not args.no_visualization:
        create_summary_plot(all_shifts_data, output_dir)
        print(f"Created summary plot for {len(all_shifts_data)} files")
    else:
        print(f"No successful files processed for summary plot. Total attempted: {len(all_files)}, Successful: {len(all_shifts_data)}")
    
    # 生成汇总报告
    create_processing_report(
        output_dir, 
        all_files, 
        processed_records, 
        focus_loss_records,
        args
    )


def create_processing_report(output_dir, all_files, processed_records, focus_loss_records, args):
    """生成处理汇总报告"""
    report = {
        'processing_time': datetime.now().isoformat(),
        'total_files': len(all_files),
        'processed_count': len(processed_records),
        'focus_loss_count': len(focus_loss_records),
        'parameters': {
            'channel': args.channel,
            'max_iterations': args.max_iterations,
            'threshold': args.threshold,
            'batch_size': args.batch_size,
            'auto_crop': not args.no_auto_crop,
            'border': args.border,
            'focus_loss_threshold': args.focus_loss_threshold,
            'skip_focus_loss': not args.no_skip_focus_loss
        },
        'processed_files': processed_records,
        'focus_loss_files': []
    }
    
    # 添加丢焦文件的详细信息
    for info in focus_loss_records:
        report['focus_loss_files'].append({
            'file': info['file'],
            'focus_loss_frame': info['focus_loss_frame'],
            'total_frames': info['total_frames'],
            'intensity_ratio_at_loss': info['intensity_ratios'][info['focus_loss_frame']] if info['focus_loss_frame'] else None
        })
    
    # 保存报告
    report_path = Path(output_dir) / 'processing_report.json'
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nProcessing report saved to: {report_path}")
    except Exception as e:
        print(f"Error saving report: {e}")
    
    # 打印汇总信息
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total files:      {len(all_files)}")
    print(f"Processed:        {len(processed_records)}")
    print(f"Focus loss:       {len(focus_loss_records)}")
    
    if focus_loss_records:
        print("\nFocus loss files:")
        for info in focus_loss_records:
            frame = info['focus_loss_frame']
            total = info['total_frames']
            ratio = info['intensity_ratios'][frame] if frame else 0
            print(f"  - {Path(info['file']).name}")
            print(f"    Frame {frame}/{total}, intensity ratio: {ratio:.3f}")
    
    print("="*60)

if __name__ == "__main__":
    main()