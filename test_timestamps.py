"""
测试ND2时间戳读取功能
验证从多P ND2文件读取时间戳是否正确
"""
from nd2_metadata import get_nd2_timestamps, extract_position_from_filename, find_original_nd2_for_split_file
from pathlib import Path
import numpy as np

def test_timestamp_reading():
    """测试时间戳读取功能"""
    
    # 使用实际的ND2文件路径
    nd2_file = r"C:\cygwin\home\zhao\confocal\20260214\20260214  LH387 LT50 timelapse 3-31 10uM rapa - 006.nd2"
    
    if not Path(nd2_file).exists():
        print(f"测试文件不存在: {nd2_file}")
        print("请修改test_timestamps.py中的nd2_file变量指向一个实际的ND2文件")
        return
    
    print("=" * 80)
    print("测试1: 读取ND2文件的时间戳")
    print("=" * 80)
    print(f"文件: {nd2_file}\n")
    
    # 测试读取position 0的时间戳
    print("读取 Position 0 的时间戳:")
    timestamps_p0 = get_nd2_timestamps(nd2_file, position_index=0)
    if timestamps_p0 is not None:
        print(f"  成功读取 {len(timestamps_p0)} 个时间点")
        print(f"  前5个时间戳(秒): {timestamps_p0[:5]}")
        print(f"  时间间隔(秒): {np.diff(timestamps_p0[:5])}")
    else:
        print("  读取失败")
    
    print()
    
    # 测试读取position 1的时间戳
    print("读取 Position 1 的时间戳:")
    timestamps_p1 = get_nd2_timestamps(nd2_file, position_index=1)
    if timestamps_p1 is not None:
        print(f"  成功读取 {len(timestamps_p1)} 个时间点")
        print(f"  前5个时间戳(秒): {timestamps_p1[:5]}")
    else:
        print("  读取失败")
    
    print()
    print("=" * 80)
    print("测试2: 从文件名提取Position索引")
    print("=" * 80)
    
    test_filenames = [
        "20260214  LH387 LT50 timelapse 3-31 10uM rapa - 006_P0_corrected.ome.tif",
        "20260214  LH387 LT50 timelapse 3-31 10uM rapa - 006_P13_corrected.ome.tif",
        "some_file_without_position.tif"
    ]
    
    for filename in test_filenames:
        position = extract_position_from_filename(filename)
        print(f"  {filename}")
        print(f"    -> Position索引: {position}")
    
    print()
    print("=" * 80)
    print("测试3: 查找原始ND2文件")
    print("=" * 80)
    
    # 模拟拆分后的文件
    split_file = "20260214  LH387 LT50 timelapse 3-31 10uM rapa - 006_P0_corrected.ome.tif"
    search_dirs = [r"C:\cygwin\home\zhao\confocal\20260214"]
    
    print(f"拆分文件: {split_file}")
    print(f"搜索目录: {search_dirs}")
    
    original_nd2 = find_original_nd2_for_split_file(split_file, search_dirs)
    if original_nd2:
        print(f"  找到原始ND2: {original_nd2}")
        
        # 验证能否读取时间戳
        position = extract_position_from_filename(split_file)
        timestamps = get_nd2_timestamps(original_nd2, position)
        if timestamps is not None:
            print(f"  成功从原始ND2读取 {len(timestamps)} 个时间戳")
        else:
            print("  无法从原始ND2读取时间戳")
    else:
        print("  未找到原始ND2文件")
    
    print()
    print("=" * 80)
    print("测试完成!")
    print("=" * 80)


if __name__ == "__main__":
    test_timestamp_reading()
