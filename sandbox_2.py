import torch

x = torch.load("caching/Qwen3VL_8B_cls/ImageNet_1K/0/pos.pt")
y = torch.load("caching/Qwen3VL_8B_cls/ImageNet_1K/0/qry.pt")

print(x)

print('===============================')

print(y)