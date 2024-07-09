import torch

# .ptファイルのパス
file_path = "./data/train_Y.pt"

# ファイルを読み込む
data = torch.load(file_path)

# dataの内容を確認する（例えばテンソルの形状を表示する場合）
if isinstance(data, torch.Tensor):
    print("Tensor shape:", data.shape)
    print(data[0])
elif isinstance(data, dict):
    for key, value in data.items():
        print(key, type(value))
else:
    print("Unknown data type")