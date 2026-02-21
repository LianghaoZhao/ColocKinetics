"""
ColocKinetics: ND2/TIF Co-localization Analysis Pipeline

使用方式:
=========

1. 完整流程（自动检测已有的 motioncor 结果，增量处理）:
   python main.py "*.nd2" --output-dir results

2. 单独运行耗时的 motioncor，之后再继续后续分析:
   # 步骤1：单独运行 motioncor
   python motioncor.py input_dir -o results/motioncor
   
   # 步骤2：继续后续处理（自动检测已有校正结果）
   python main.py "*.nd2" --output-dir results

3. 指定外部 motioncor 目录:
   python main.py "*.nd2" --motioncor-dir /path/to/motioncor_results

4. 跳过 motioncor，直接使用原始文件:
   python main.py "*.nd2" --skip-motioncor

5. 使用已有的 mask 文件:
   python main.py "*.nd2" --mask-pattern "masks/*.npy" --skip-cellpose

更多参数请使用 python main.py --help 查看
"""

import os
import glob
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from multiprocessing import Pool

# 模块导入
from analyzer import MainAnalyzer
from io_utils import MaskFileMatcher
from plotting import Visualizer
from cellpose_utils import run_cellpose_on_files, extract_first_frame_from_nd2, find_existing_masks
from motioncor import process_image_sequence
import nd2
import tifffile


def find_corrected_files(motioncor_dir, original_files):
    """
    从 motioncor 目录中查找对应的校正文件
    
    Parameters:
    - motioncor_dir: motioncor 输出目录
    - original_files: 原始文件列表
    
    Returns:
    - 校正后的文件路径列表
    """
    motioncor_dir = Path(motioncor_dir)
    corrected = []
    
    for orig in original_files:
        base_name = Path(orig).stem
        corrected_path = motioncor_dir / f"{base_name}_corrected.ome.tif"
        
        if corrected_path.exists():
            corrected.append(str(corrected_path))
            print(f"Found: {corrected_path.name}")
        else:
            print(f"Warning: No corrected file for {Path(orig).name}, using original")
            corrected.append(orig)
    
    return corrected


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

            axes = f.axes
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


def run_motion_correction(image_files, output_dir, max_iterations=10, threshold=0.5, batch_size=100, use_gpu=True):
    """
    运行漂移校正（支持增量处理，自动检测已有的校正结果）
    
    Returns:
    - 校正后的文件路径列表
    """
    motioncor_dir = Path(output_dir) / 'motioncor'
    motioncor_dir.mkdir(parents=True, exist_ok=True)
    
    corrected_files = []
    files_to_process = []
    
    # 检测已有的校正结果
    for image_file in image_files:
        base_name = Path(image_file).stem
        corrected_path = motioncor_dir / f"{base_name}_corrected.ome.tif"
        
        if corrected_path.exists():
            print(f"Found existing corrected file: {corrected_path.name}")
            corrected_files.append(str(corrected_path))
        else:
            files_to_process.append(image_file)
    
    if corrected_files:
        print(f"\nSkipping {len(corrected_files)} files with existing corrections")
    
    if not files_to_process:
        print("All files already have motion correction results")
        return corrected_files
    
    print(f"\nProcessing {len(files_to_process)} files...")
    
    # 只处理未校正的文件
    for image_file in files_to_process:
        print(f"\nMotion correction: {Path(image_file).name}")
        try:
            # 如果是多视野 ND2，则先按 P 维度拆分为多个视野文件
            split_files = split_nd2_by_position(image_file, motioncor_dir)

            for split_file in split_files:
                print(f"  -> Processing field: {Path(split_file).name}")
                shifts, corrected_path = process_image_sequence(
                    split_file,
                    str(motioncor_dir),
                    channel_selection='all',
                    save_visualization=True,
                    auto_crop=True,
                    max_iterations=max_iterations,
                    threshold=threshold,
                    batch_size=batch_size,
                    use_gpu=use_gpu
                )
                if corrected_path:
                    corrected_files.append(corrected_path)
                    print(f"     -> Corrected: {Path(corrected_path).name}")
                else:
                    print(f"     -> Motion correction failed, using original {Path(split_file).name}")
                    corrected_files.append(split_file)
        except Exception as e:
            print(f"  -> Error: {e}, using original")
            corrected_files.append(image_file)
    
    return corrected_files


def run_cellpose_segmentation(image_files, output_dir, diameter=380, gpu_device=0, use_gpu=True):
    """
    运行 Cellpose 分割
    
    Returns:
    - mask 文件路径字典 {image_file: mask_file}
    """
    # 统一使用 mask 目录，提取帧和生成mask都在这里
    mask_dir = Path(output_dir) / 'mask'
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查是否已有 mask
    existing_masks = find_existing_masks(image_files, str(mask_dir))
    files_need_mask = [f for f, m in existing_masks.items() if m is None]
    
    if files_need_mask:
        print(f"\nNeed to generate masks for {len(files_need_mask)} files")
        
        # 提取第一帧用于 Cellpose（保存到同一目录）
        print("Extracting first frames for Cellpose...")
        extracted_frames = extract_first_frame_from_nd2(files_need_mask, str(mask_dir))
        
        if extracted_frames:
            # 运行 Cellpose（输出到同一目录）
            print(f"\nRunning Cellpose (diameter={diameter}, gpu_device={gpu_device})...")
            run_cellpose_on_files(
                extracted_frames,
                str(mask_dir),
                diameter=diameter,
                gpu_device=gpu_device,
                use_gpu=use_gpu
            )
        
        # 重新查找 mask
        existing_masks = find_existing_masks(image_files, str(mask_dir))
    else:
        print(f"\nAll masks already exist in {mask_dir}")
    
    return existing_masks


def main():
    parser = argparse.ArgumentParser(description='ColocKinetics: ND2/TIF Co-localization Analysis Pipeline')
    
    # 输入输出
    parser.add_argument('image_pattern', type=str,
                       help='Pattern for image files (e.g., "*.nd2", "data/*.tif")')
    parser.add_argument('--output-dir', type=str, default='coloc_result',
                       help='Output directory for results (default: coloc_result)')
    
    # 流程控制
    parser.add_argument('--skip-motioncor', action='store_true',
                       help='Skip motion correction step')
    parser.add_argument('--motioncor-dir', type=str, default=None,
                       help='Path to existing motioncor output directory (use pre-computed results)')
    parser.add_argument('--skip-cellpose', action='store_true',
                       help='Skip Cellpose segmentation (use existing masks)')
    parser.add_argument('--mask-pattern', type=str, default=None,
                       help='Pattern for existing mask files (e.g., "masks/*.npy")')
    
    # Motion correction 参数
    parser.add_argument('--mc-max-iterations', type=int, default=10,
                       help='Motion correction: max iterations (default: 10)')
    parser.add_argument('--mc-threshold', type=float, default=0.5,
                       help='Motion correction: convergence threshold in pixels (default: 0.5)')
    parser.add_argument('--mc-batch-size', type=int, default=100,
                       help='Motion correction: GPU batch size (default: 100, recommended 50-200)')
    parser.add_argument('--mc-no-gpu', action='store_true',
                       help='Motion correction: disable GPU acceleration')
    
    # Cellpose 参数
    parser.add_argument('--cp-diameter', type=int, default=380,
                       help='Cellpose: cell diameter in pixels (default: 380)')
    parser.add_argument('--cp-gpu-device', type=int, default=0,
                       help='Cellpose: GPU device ID (default: 0)')
    parser.add_argument('--cp-no-gpu', action='store_true',
                       help='Cellpose: disable GPU')
    
    # 分析参数
    parser.add_argument('--skip-initial-frames', type=int, default=0,
                       help='Number of initial frames to skip in analysis')
    parser.add_argument('--fit-model', type=str, default='first_order',
                       choices=['first_order', 'delayed_first_order'],
                       help='Fitting model (default: first_order)')
    
    # 输出控制
    parser.add_argument('--save-results', action='store_true',
                       help='Save results to CSV files')
    parser.add_argument('--include-scatter', action='store_true',
                       help='Include scatter plot in figures')
    parser.add_argument('--include-individual-plots', action='store_true',
                       help='Include individual cell plots')
    parser.add_argument('--background', type=float, default=100.0,
                       help='Background signal to subtract for ratio calculation (default: 100)')

    args = parser.parse_args()

    # Step 0: 查找图像文件
    print("=" * 60)
    print("ColocKinetics Pipeline")
    print("=" * 60)
    
    image_files = glob.glob(args.image_pattern)
    if not image_files:
        print(f"No image files found matching: {args.image_pattern}")
        return
    
    print(f"\nFound {len(image_files)} image files:")
    for f in image_files:
        print(f"  - {Path(f).name}")
    
    # 设置输出目录
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Motion Correction (漂移校正)
    if args.motioncor_dir:
        # 使用指定目录中的已有校正结果
        print("\n" + "=" * 60)
        print("Step 1: Motion Correction (using existing results)")
        print("=" * 60)
        print(f"Using existing motioncor results from: {args.motioncor_dir}")
        analysis_files = find_corrected_files(args.motioncor_dir, image_files)
        print(f"\nFound {len(analysis_files)} corrected files")
    elif not args.skip_motioncor:
        # 正常运行（自动检测 + 增量处理）
        print("\n" + "=" * 60)
        print("Step 1: Motion Correction")
        print("=" * 60)
        
        corrected_files = run_motion_correction(
            image_files,
            output_dir,
            max_iterations=args.mc_max_iterations,
            threshold=args.mc_threshold,
            batch_size=args.mc_batch_size,
            use_gpu=not args.mc_no_gpu
        )
        # 使用校正后的文件进行后续分析
        analysis_files = corrected_files
        print(f"\nMotion correction completed: {len(corrected_files)} files")
    else:
        print("\nSkipping motion correction (--skip-motioncor)")
        analysis_files = image_files
    
    # Step 2: Cellpose Segmentation (细胞分割)
    if not args.skip_cellpose and not args.mask_pattern:
        print("\n" + "=" * 60)
        print("Step 2: Cellpose Segmentation")
        print("=" * 60)
        
        mask_matches = run_cellpose_segmentation(
            analysis_files,
            output_dir,
            diameter=args.cp_diameter,
            gpu_device=args.cp_gpu_device,
            use_gpu=not args.cp_no_gpu
        )
        
        # 检查是否所有文件都有 mask
        missing_masks = [f for f, m in mask_matches.items() if m is None]
        if missing_masks:
            print(f"\nWarning: {len(missing_masks)} files missing masks:")
            for f in missing_masks:
                print(f"  - {Path(f).name}")
        
        # 设置 mask pattern 为 mask 目录
        mask_dir = Path(output_dir) / 'mask'
        mask_pattern = str(mask_dir / '*.npy')
    else:
        if args.mask_pattern:
            print(f"\nUsing existing masks: {args.mask_pattern}")
            mask_pattern = args.mask_pattern
        else:
            print("\nSkipping Cellpose (--skip-cellpose)")
            mask_pattern = None
    
    # Step 3: Co-localization Analysis (共定位分析)
    print("\n" + "=" * 60)
    print("Step 3: Co-localization Analysis")
    print("=" * 60)

    # 创建分析器
    analyzer = MainAnalyzer()

    # 加载数据并匹配 mask
    print("\nLoading image and mask data...")
    analyses = analyzer.process_files_with_masks(analysis_files, mask_pattern, args.skip_initial_frames)

    if not analyses:
        print("No files were successfully processed (no matching masks found)")
        return

    # 获取汇总数据 (共定位指标)
    print("Calculating co-localization metrics...")
    summary_df = analyzer.get_summary_dataframe()
    print(f"\nProcessed {len(summary_df)} cell-time points")
    print("First few rows (showing key information):")
    display_cols = ['file_path', 'cell_id', 'time_point', 'pearson_corr', 'p_value', 'n_pixels']
    display_df = summary_df[display_cols].copy()
    display_df['file_path'] = display_df['file_path'].apply(lambda x: Path(x).name)  # 只显示文件名
    print(display_df.head())

    # 执行完整分析 (共定位计算 -> 拟合)
    print(f"\nPerforming {args.fit_model} reaction fitting (parallel - file level)...")
    reaction_df = analyzer.run_full_analysis(fit_model=args.fit_model)

    print(f"Reaction fitting results for {len(reaction_df)} cells:")
    print("First few rows of reaction fitting:")
    reaction_display = reaction_df[['file_path', 'cell_id', 'correlation_k', 'correlation_t50', 'correlation_t90', 'correlation_r_squared']].copy()
    reaction_display['file_path'] = reaction_display['file_path'].apply(lambda x: Path(x).name)
    print(reaction_display.head())

    # 保存结果
    if args.save_results:
        # 保存基础汇总数据
        output_file = os.path.join(output_dir, 'correlation_analysis_results.csv')
        summary_df.to_csv(output_file, index=False)
        print(f"Basic co-localization results saved to: {output_file}")
        # 保存反应拟合结果
        reaction_output_file = os.path.join(output_dir, 'reaction_fitting_results.csv')
        reaction_df.to_csv(reaction_output_file, index=False)
        print(f"Reaction fitting results saved to: {reaction_output_file}")

    # 可视化 (调用 plotting 模块)
    print("\nGenerating visualizations...")
    visualizer = Visualizer(output_dir)
    visualizer.generate_visualizations(
        analyses=analyzer.all_files,
        fit_model=args.fit_model,
        include_individual_plots=args.include_individual_plots,
        include_scatter=args.include_scatter
    )
    
    # 红绿比值与T50关系分析
    visualizer.generate_ratio_vs_t50_analysis(
        analyses=analyzer.all_files,
        reaction_df=reaction_df,
        fit_model=args.fit_model,
        background=args.background
    )
    
    print("\n" + "=" * 60)
    print("Pipeline completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()