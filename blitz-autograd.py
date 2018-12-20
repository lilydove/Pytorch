# https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html

import torch

x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2
print(y)
print(y.grad_fn)

z = y * y * 3
out = z.mean()
print("z:   ", z, "\nout: ", out)

a = torch.randn(2, 2)
a = ((a * 3))/(a - 1)
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

out.backward()
print(x.grad)

# 3 dimentions
x = torch.randn(3, requires_grad=True)
print(x)
y = x * 2
while y.data.norm() < 800:
    y = y * 2
print(y)

# x1=.2 x2=1.0 x3=.00001 they are inserting values.
v = torch.tensor([.2, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
print(x.grad)

print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)