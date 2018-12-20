# https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html

import torch

x = torch.ones(2, 2, requires_grad=True)
print(x)
