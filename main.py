import os
import glob
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from multiprocessing import Pool

# 假设这些模块已按方案拆分
from analyzer import MainAnalyzer
from io_utils import MaskFileMatcher
from plotting import Visualizer


def main():
    parser = argparse.ArgumentParser(description='Analyze fluorescence co-localization in ND2/TIF files')
    parser.add_argument('image_pattern', type=str,
                       help='Pattern for image files (e.g., "data/*.nd2", "data/*.tif", or "data/**/*.nd2")')
    parser.add_argument('--mask-pattern', type=str, default=None,
                       help='Pattern for mask files (e.g., "*.npy", "*_mask.npy"). If not provided, will try to auto-match based on filename similarity.')
    parser.add_argument('--output-dir', type=str, default='coloc_result',
                       help='Output directory for results. Default is "coloc_result".')
    parser.add_argument('--save-results', action='store_true',
                       help='Save results to CSV file')
    parser.add_argument('--skip-initial-frames', type=int, default=0,
                       help='Number of initial frames to skip (not used for fitting)')
    parser.add_argument('--include-scatter', action='store_true', # Note: This affects plotting, handled in plotting module
                       help='Include scatter plot in cell analysis figures')
    parser.add_argument('--fit-model', type=str, default='first_order',
                       choices=['first_order', 'delayed_first_order'],
                       help='Fitting model to use: first_order or delayed_first_order (for handling platform differences)')
    parser.add_argument('--include-individual-plots', action='store_true', # Note: This affects plotting, handled in plotting module
                       help='Include individual cell plots (default: False)')

    args = parser.parse_args()

    # 使用glob查找所有匹配的图像文件
    image_files = glob.glob(args.image_pattern)

    if not image_files:
        print(f"No image files found matching pattern: {args.image_pattern}")
        return
    else:
        print(image_files)
    # 设置输出目录
    output_dir = args.output_dir
    # 自动创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 创建分析器
    analyzer = MainAnalyzer()

    # 处理文件（加载数据，IO相关）
    print("Loading image and mask data...")
    analyses = analyzer.process_files_with_masks(image_files, args.mask_pattern, args.skip_initial_frames)

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

    # 可视化示例 (调用 plotting 模块)
    print("\nGenerating visualizations...")
    visualizer = Visualizer(output_dir)
    visualizer.generate_visualizations(
        analyses=analyzer.all_files, # Pass the loaded analyses
        fit_model=args.fit_model,
        include_individual_plots=args.include_individual_plots,
        include_scatter=args.include_scatter
    )


if __name__ == "__main__":
    main()