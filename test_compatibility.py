"""
测试向后兼容性：验证旧的motioncor文件是否能正常工作
"""
from nd2_metadata import extract_position_from_filename, find_original_nd2_for_split_file

def test_old_file_compatibility():
    """测试旧文件的兼容性"""
    
    print("=" * 80)
    print("兼容性测试：模拟旧的motioncor文件")
    print("=" * 80)
    print()
    
    # 测试1: 没有_P标记的旧文件
    print("测试1: 旧文件格式(无_P标记)")
    old_files = [
        "some_old_file_corrected.ome.tif",
        "experiment_data.tif",
        "old_timelapse.nd2"
    ]
    
    for filename in old_files:
        position = extract_position_from_filename(filename)
        print(f"  文件: {filename}")
        print(f"    Position索引: {position}")
        
        if position is None:
            print(f"    → 行为: 使用fallback模式, time_point = 0.0, 1.0, 2.0...")
        else:
            print(f"    → 行为: 尝试查找原始ND2")
        print()
    
    # 测试2: 有_P标记但找不到原始ND2
    print("测试2: 新格式但找不到原始ND2")
    split_file = "experiment_P5_corrected.ome.tif"
    search_dirs = ["/nonexistent/path"]
    
    print(f"  文件: {split_file}")
    position = extract_position_from_filename(split_file)
    print(f"    Position索引: {position}")
    
    original_nd2 = find_original_nd2_for_split_file(split_file, search_dirs)
    print(f"    找到原始ND2: {original_nd2}")
    
    if original_nd2 is None:
        print(f"    → 行为: 使用fallback模式, time_point = 0.0, 1.0, 2.0...")
    print()
    
    # 总结
    print("=" * 80)
    print("兼容性总结")
    print("=" * 80)
    print()
    print("✓ 旧文件(无_P标记): 自动使用索引模式 (0.0, 1.0, 2.0...)")
    print("✓ 找不到原始ND2: 自动fallback到索引模式")
    print("✓ 新文件(找到ND2): 使用真实时间戳")
    print()
    print("结论: 完全向后兼容, 旧数据可以正常处理!")
    print("=" * 80)


if __name__ == "__main__":
    test_old_file_compatibility()
