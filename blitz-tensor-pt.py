# https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py
# 2018-12-19
# wuyi

from __future__ import print_function
import torch

x = torch.empty(5, 3)
print(x)
x = torch.rand(5, 3)
print(x)
x = torch.zeros(5, 3, dtype=torch.long)
print(x)
x = torch.tensor([5.5, 3])
print(x)
x = x.new_ones(5, 3, dtype=torch.double)    # new_* methods take in sizes
print(x)
x = torch.randn_like(x, dtype=torch.float)  # override dtype!
print(x)                                    # result has the same size
print(x.size())

x = torch.ones(5, 3)
print(x)
y = torch.rand(5, 3)
print(x + y)
print(torch.add(x, y))

result = torch.empty(6, 4)
torch.add(x, y, out=result)
print(result)

y.add_(x)
print(y)
print(y[:, 1])      # show the 2sd column

x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)   # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

x = torch.randn(1)
print(x)
print(x.item())

a = torch.ones(5)
print("a:",end='')
print(a)
b = a.numpy()
print(b)
a.add_(1)
print(a)
print(b)

# changing the np array changed the Torch Tensor automatically
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a) # a tensor
np.add(a, 1, out=a)
print(a)
print(b)

# let us run this cell only if CUDA is available
# We will use "torch.device" objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")   # a CUDA device object
    y = torch.ones_like(x, device=device) # directly create a tensor on GPU
    x = x.to(device)    # or just use strings ".to("cuda")"
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))    # ".to" can also change dtype together!