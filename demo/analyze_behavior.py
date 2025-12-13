import pandas as pd

# 读取CSV数据
df = pd.read_csv('primordial_core_v0_1_output.csv')

# 提取行为序列
actions = df['action'].tolist()

# 统计连续相同行为的长度
sequences = []
current_action = actions[0]
current_length = 1

for action in actions[1:]:
    if action == current_action:
        current_length += 1
    else:
        sequences.append((current_action, current_length))
        current_action = action
        current_length = 1
# 添加最后一个序列
sequences.append((current_action, current_length))

# 统计交替情况
print("=== 行为序列分析 ===")
print(f"总步数: {len(actions)}")
print(f"总序列数: {len(sequences)}")

# 计算交替频率
alternations = len(sequences) - 1
print(f"交替次数: {alternations}")
print(f"平均每步交替概率: {alternations / len(actions):.4f}")

# 统计explore和rest的总时长
explore_total = sum(length for action, length in sequences if action == 'explore')
rest_total = sum(length for action, length in sequences if action == 'rest')

print(f"\nexplore总时长: {explore_total} 步 ({explore_total / len(actions) * 100:.2f}%)")
print(f"rest总时长: {rest_total} 步 ({rest_total / len(actions) * 100:.2f}%)")

# 分析explore序列长度
explore_sequences = [length for action, length in sequences if action == 'explore']
print(f"\nexplore序列分析:")
print(f"  次数: {len(explore_sequences)}")
print(f"  平均长度: {sum(explore_sequences) / len(explore_sequences):.2f} 步")
print(f"  最长: {max(explore_sequences)} 步")
print(f"  最短: {min(explore_sequences)} 步")

# 分析rest序列长度
rest_sequences = [length for action, length in sequences if action == 'rest']
print(f"\nrest序列分析:")
print(f"  次数: {len(rest_sequences)}")
print(f"  平均长度: {sum(rest_sequences) / len(rest_sequences):.2f} 步")
print(f"  最长: {max(rest_sequences)} 步")
print(f"  最短: {min(rest_sequences)} 步")

# 查看前20个序列模式
print(f"\n前20个序列模式:")
for i, (action, length) in enumerate(sequences[:20]):
    print(f"  {i+1}. {action} × {length}")

# 检查是否有明显的周期性
print(f"\n=== 周期性分析 ===")
print("查看序列长度分布:")
explore_length_counts = {}
for length in explore_sequences:
    explore_length_counts[length] = explore_length_counts.get(length, 0) + 1
print(f"explore长度分布: {dict(sorted(explore_length_counts.items()))}")

rest_length_counts = {}
for length in rest_sequences:
    rest_length_counts[length] = rest_length_counts.get(length, 0) + 1
print(f"rest长度分布: {dict(sorted(rest_length_counts.items()))}")
