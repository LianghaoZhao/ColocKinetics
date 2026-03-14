"""
调试脚本：检查T50和T90分析是否使用了相同的数据
"""
import csv
import statistics

# 读取数据
csv_path = r'C:\cygwin\home\zhao\confocal\kinetics\20260215_res\k_app_analysis_data.csv'

with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"=== 数据统计 ===")
print(f"总行数: {len(rows)}")

# 提取T50和T90数据
t50_values = [float(row['t50']) for row in rows]
t90_values = [float(row['t90']) for row in rows]
k50_values = [float(row['k_app_T50']) for row in rows]
k90_values = [float(row['k_app_T90']) for row in rows]

print(f"\nT50统计:")
print(f"  Mean: {statistics.mean(t50_values):.2f}")
print(f"  Median: {statistics.median(t50_values):.2f}")
print(f"  Min: {min(t50_values):.2f}")
print(f"  Max: {max(t50_values):.2f}")

print(f"\nT90统计:")
print(f"  Mean: {statistics.mean(t90_values):.2f}")
print(f"  Median: {statistics.median(t90_values):.2f}")
print(f"  Min: {min(t90_values):.2f}")
print(f"  Max: {max(t90_values):.2f}")

print(f"\nk_app_T50统计:")
print(f"  Mean: {statistics.mean(k50_values):.6f}")
print(f"  Median: {statistics.median(k50_values):.6f}")
print(f"  Min: {min(k50_values):.6f}")
print(f"  Max: {max(k50_values):.6f}")

print(f"\nk_app_T90统计:")
print(f"  Mean: {statistics.mean(k90_values):.6f}")
print(f"  Median: {statistics.median(k90_values):.6f}")
print(f"  Min: {min(k90_values):.6f}")
print(f"  Max: {max(k90_values):.6f}")

# 检查是否完全相同
print(f"\n=== 数据一致性检查 ===")
all_same_t = all(t50 == t90 for t50, t90 in zip(t50_values, t90_values))
all_same_k = all(k50 == k90 for k50, k90 in zip(k50_values, k90_values))
print(f"T50 == T90所有值: {all_same_t}")
print(f"k_app_T50 == k_app_T90所有值: {all_same_k}")

# 检查比值
ratios = [t90 / t50 for t50, t90 in zip(t50_values, t90_values)]
print(f"\n=== T90/T50比值统计 ===")
print(f"  Mean: {statistics.mean(ratios):.4f}")
print(f"  Median: {statistics.median(ratios):.4f}")
print(f"  Min: {min(ratios):.4f}")
print(f"  Max: {max(ratios):.4f}")

# 检查前5行的实际值
print(f"\n=== 前5行数据样本 ===")
for i in range(min(5, len(rows))):
    print(f"Row {i+1}:")
    print(f"  T50={t50_values[i]:.2f}, T90={t90_values[i]:.2f}, T90/T50={t90_values[i]/t50_values[i]:.2f}")
    print(f"  k_T50={k50_values[i]:.6f}, k_T90={k90_values[i]:.6f}")

print(f"\n结论: T50和T90的数据{'相同' if all_same_t else '不同'}")
