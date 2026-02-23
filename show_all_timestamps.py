"""
显示ND2文件中所有视野和时间点的拍摄时间
"""
import nd2
from datetime import datetime, timedelta

nd2_file_path = r"C:\cygwin\home\zhao\confocal\20260214\20260214  LH387 LT50 timelapse 3-31 10uM rapa - 006.nd2"

def julian_to_datetime(jd):
    """将Julian Day Number转换为datetime对象"""
    # Julian Day 2440588 = 1970-01-01 00:00:00 UTC (Unix epoch)
    unix_epoch_jd = 2440587.5
    unix_timestamp = (jd - unix_epoch_jd) * 86400.0
    return datetime.utcfromtimestamp(unix_timestamp)

print(f"读取文件: {nd2_file_path}\n")
print("=" * 100)

with nd2.ND2File(nd2_file_path) as f:
    print(f"文件信息:")
    print(f"  维度: {dict(f.sizes)}")
    print(f"  形状: {f.shape}")
    print(f"  通道: {[ch.channel.name for ch in f.metadata.channels]}")
    print()
    
    # 获取文件创建日期
    if hasattr(f, 'text_info') and 'date' in f.text_info:
        print(f"文件创建时间: {f.text_info['date']}")
    print()
    
    num_positions = f.sizes.get('P', 1)
    num_timepoints = f.sizes.get('T', 1)
    num_channels = f.sizes.get('C', 1)
    
    print("=" * 100)
    print(f"共有 {num_positions} 个视野 (Position), 每个视野有 {num_timepoints} 个时间点")
    print("=" * 100)
    print()
    
    # 遍历所有视野和时间点
    for p in range(num_positions):
        print(f"【视野 (Position) {p}】")
        print("-" * 100)
        
        for t in range(num_timepoints):
            # 计算当前帧的索引
            # 根据维度顺序 (T, P, C, Y, X) 计算索引
            frame_idx = t * num_positions + p
            
            try:
                fm = f.frame_metadata(frame_idx)
                
                # 获取时间戳
                if hasattr(fm, 'channels') and len(fm.channels) > 0:
                    ch_meta = fm.channels[0]  # 使用第一个通道的时间
                    if hasattr(ch_meta, 'time'):
                        timestamp = ch_meta.time
                        
                        # 转换时间戳
                        jd = timestamp.absoluteJulianDayNumber
                        relative_ms = timestamp.relativeTimeMs
                        
                        # 转换为可读时间
                        dt = julian_to_datetime(jd)
                        
                        # 相对时间转换为 分:秒
                        relative_sec = relative_ms / 1000.0
                        relative_min = int(relative_sec // 60)
                        relative_sec_remainder = relative_sec % 60
                        
                        print(f"  时间点 {t:2d}: {dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} " 
                              f"(相对时间: {relative_min:3d}分{relative_sec_remainder:06.3f}秒 = {relative_ms:.2f}ms)")
                
            except Exception as e:
                print(f"  时间点 {t:2d}: 无法读取 - {e}")
        
        print()

print("=" * 100)
print("完成!")
