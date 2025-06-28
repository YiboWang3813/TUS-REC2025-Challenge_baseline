import torch 

x = torch.zeros((10, 3, 10, 10)) 
y = torch.randint_like(torch.ones(5, 3), 0, 10, dtype=torch.long) 

z = x[y[:, 0], :, y[:, 1], y[:, 2]]

print(x.shape, y.shape, z.shape)