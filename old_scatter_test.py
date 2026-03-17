import torch
# import torch_scatter
from torch_scatter import scatter_sum, scatter_logsumexp


nul = torch.zeros((2,5), dtype=torch.float32)

# indices = torch.ones(2, dtype=torch.long).unsqueeze(1).expand(-1, 5)
indices = torch.ones(2, dtype=torch.long)
src = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=torch.float32, requires_grad=True)

print(src.grad)
print(src.requires_grad)
res = scatter_logsumexp(src, indices, dim=0, dim_size=2)

print(res)
res.sum().backward()
print(src.grad)