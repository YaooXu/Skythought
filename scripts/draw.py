import matplotlib.pyplot as plt
import numpy as np

# 数据
data1 = {
    "k=128": 66.667,
    "k=64": 58.611,
    "k=32": 50.492,
    "k=16": 42.687,
    "k=8": 35.459,
    "k=4": 28.5,
    "k=2": 22.232,
    "k=1": 16.641
}

data2 = {
    "k=128": 66.667,
    "k=64": 59.916,
    "k=32": 52.458,
    "k=16": 45.061,
    "k=8": 37.916,
    "k=4": 30.66,
    "k=2": 23.74,
    "k=1": 17.552
  }

# 提取x轴和y轴数据
k_values = [1, 2, 4, 8, 16, 32, 64, 128]
y1 = [data1[f"k={k}"] for k in k_values]
y2 = [data2[f"k={k}"] for k in k_values]

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制两条线
plt.plot(k_values, y1, marker='o', label='Group 1')
plt.plot(k_values, y2, marker='s', label='Group 2')

# 设置x轴为对数刻度
plt.xscale('log', base=2)
plt.xticks(k_values, labels=k_values)

# 添加标题和标签
plt.title('Pass@k Comparison')
plt.xlabel('k (log scale)')
plt.ylabel('Pass Rate (%)')
plt.grid(True, which="both", ls="--")
plt.legend()

# 显示图形
plt.tight_layout()
plt.show()