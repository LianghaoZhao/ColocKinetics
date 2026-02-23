"""
临时脚本：读取ND2文件的metadata信息，特别是每张照片的拍摄时间
"""
import nd2
from pathlib import Path
from datetime import datetime

nd2_file_path = r"C:\cygwin\home\zhao\confocal\20260214\20260214  LH387 LT50 timelapse 3-31 10uM rapa - 006.nd2"

print(f"读取文件: {nd2_file_path}\n")

with nd2.ND2File(nd2_file_path) as f:
    # 基本信息
    print("=" * 80)
    print("基本信息")
    print("=" * 80)
    print(f"形状 (shape): {f.shape}")
    print(f"维度 (axes): {f.sizes}")
    print(f"数据类型: {f.dtype}")
    print()
    
    # 详细的维度信息
    print("=" * 80)
    print("维度详情")
    print("=" * 80)
    for axis, size in f.sizes.items():
        print(f"{axis}: {size}")
    print()
    
    # Metadata
    print("=" * 80)
    print("Metadata")
    print("=" * 80)
    if hasattr(f, 'metadata'):
        metadata = f.metadata
        print(f"Metadata类型: {type(metadata)}")
        if hasattr(metadata, 'channels'):
            print(f"通道数: {len(metadata.channels)}")
            for i, ch in enumerate(metadata.channels):
                print(f"  通道 {i}: {ch.channel.name if hasattr(ch.channel, 'name') else 'N/A'}")
        print()
    
    # 时间戳信息
    print("=" * 80)
    print("时间戳信息")
    print("=" * 80)
    
    # 检查是否有时间戳
    if hasattr(f, 'frame_metadata'):
        print("包含frame_metadata")
        # 获取总帧数
        total_frames = f.shape[0] if len(f.shape) > 0 else 0
        print(f"总帧数: {total_frames}")
        print()
        
        # 显示前几帧的详细信息
        print("前10帧的时间戳:")
        print("-" * 80)
        for i in range(min(10, total_frames)):
            try:
                fm = f.frame_metadata(i)
                # 尝试获取时间戳
                if hasattr(fm, 'channels'):
                    for ch_idx, ch_meta in enumerate(fm.channels):
                        if hasattr(ch_meta, 'time'):
                            timestamp = ch_meta.time
                            # 转换时间戳
                            if hasattr(timestamp, 'absoluteJulianDayNumber'):
                                # Julian day格式
                                print(f"帧 {i}, 通道 {ch_idx}: Julian Day = {timestamp.absoluteJulianDayNumber}, MS = {timestamp.msOfDay}")
                            elif isinstance(timestamp, (int, float)):
                                # Unix时间戳
                                dt = datetime.fromtimestamp(timestamp)
                                print(f"帧 {i}, 通道 {ch_idx}: {dt.strftime('%Y-%m-%d %H:%M:%S.%f')}")
                            else:
                                print(f"帧 {i}, 通道 {ch_idx}: {timestamp}")
                        break  # 只显示第一个通道的时间
                elif hasattr(fm, 'time'):
                    timestamp = fm.time
                    if isinstance(timestamp, (int, float)):
                        dt = datetime.fromtimestamp(timestamp)
                        print(f"帧 {i}: {dt.strftime('%Y-%m-%d %H:%M:%S.%f')}")
                    else:
                        print(f"帧 {i}: {timestamp}")
                else:
                    print(f"帧 {i}: 无时间信息")
            except Exception as e:
                print(f"帧 {i}: 读取错误 - {e}")
    
    # 尝试获取实验设置中的时间信息
    if hasattr(f, 'experiment'):
        print("\n" + "=" * 80)
        print("实验信息")
        print("=" * 80)
        exp = f.experiment
        if exp:
            for loop in exp:
                print(f"循环类型: {loop.type}")
                print(f"参数: {loop.parameters}")
    
    # 尝试获取text_info
    if hasattr(f, 'text_info'):
        print("\n" + "=" * 80)
        print("文本信息 (text_info)")
        print("=" * 80)
        for key, value in f.text_info.items():
            if 'time' in key.lower() or 'date' in key.lower():
                print(f"{key}: {value}")
    
    # 尝试获取自定义数据
    if hasattr(f, 'custom_data'):
        print("\n" + "=" * 80)
        print("自定义数据")
        print("=" * 80)
        print(f.custom_data)
    
    # 显示P维度的详细信息
    if 'P' in f.sizes and f.sizes['P'] > 1:
        print("\n" + "=" * 80)
        print(f"多视野信息 (P维度 = {f.sizes['P']})")
        print("=" * 80)
        
        # 尝试获取每个position的信息
        for p in range(min(f.sizes['P'], 5)):  # 只显示前5个position
            print(f"\n视野 (Position) {p}:")
            # 尝试获取该position第一帧的metadata
            if 'T' in f.sizes:
                for t in range(min(f.sizes['T'], 3)):  # 每个position显示前3个时间点
                    try:
                        # 计算frame索引 (假设是 P*T 顺序)
                        if len(f.sizes) == 4 and 'C' in f.sizes:  # PTCYX
                            frame_idx = p * f.sizes['T'] + t
                        else:
                            frame_idx = p * f.sizes['T'] + t
                        
                        fm = f.frame_metadata(frame_idx)
                        if hasattr(fm, 'channels') and len(fm.channels) > 0:
                            ch_meta = fm.channels[0]
                            if hasattr(ch_meta, 'time'):
                                timestamp = ch_meta.time
                                print(f"  时间点 {t}: {timestamp}")
                    except Exception as e:
                        print(f"  时间点 {t}: 无法读取 - {e}")

print("\n完成!")
