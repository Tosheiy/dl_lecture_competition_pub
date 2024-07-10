import torch

if torch.cuda.is_available():
    print("CUDAは利用可能です。PyTorchはGPUを使用できます。")
    print(f"使用可能なGPU数: {torch.cuda.device_count()}")
    print(f"GPU名: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDAは利用できません。PyTorchはGPUを使用できません。")