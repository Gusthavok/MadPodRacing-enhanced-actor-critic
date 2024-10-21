import torch
a = torch.randn(2, 4)
b = torch.randn(2, 8)
print(a, b, torch.cat((a,b), dim=1))