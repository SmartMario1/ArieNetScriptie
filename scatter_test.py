import torch

nul = torch.zeros((2,5), dtype=torch.float32)

indices = torch.ones(2, dtype=torch.long).unsqueeze(1).expand(-1, 5)
src = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=torch.float32, requires_grad=True)

print(src.grad)
print(src.requires_grad)
# res = torch.log(nul.scatter_reduce(0, indices, torch.exp(src), reduce="sum"))

res = nul.scatter_reduce(0, indices, src, reduce="sum")
res = torch.logsumexp(res)
print(res)

# x = torch.rand(2, 5, requires_grad=True)
# out = torch.ones(3, 5).scatter_reduce(0, torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]]), x, reduce="sum")
# out.sum().backward()
# print(x.grad)

res.sum().backward()
print(src.grad)