import pickle
import torch

# 打开并加载 pickle 文件
with open("grad.pickle", "rb") as f:
    loaded_data = pickle.load(f)

# 打印加载的数据
print("加载的字典内容：", loaded_data)

# 如果需要将 NumPy 数组转换回 PyTorch 张量

tensor_data = {key: torch.from_numpy(value) for key, value in loaded_data.items()}
