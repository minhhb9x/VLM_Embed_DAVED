import torch

x = torch.load("caching/B3_Qwen2_2B_cls/ImageNet_1K/10/pos.pt")
y = torch.load("caching/B3_Qwen2_2B_cls/ImageNet_1K/10/qry.pt")

print(x)

print('===============================')

print(y)