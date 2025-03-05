#!/usr/bin/env python3


"""
Some code snippets from the following tutorial

https://www.youtube.com/watch?v=x9JiIFvlUwk&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=2

with my personal comments
"""

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

### Part 1: Init a tensor
my_tensor = torch.tensor(
    [[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=device, requires_grad=True
)

print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)

# other common initialization methods

x = torch.empty(size=(3,3))
print(x)
x = torch.zeros((3,3))
print(x)
x = torch.rand((3,3)) # sampling from uniform distribution
print(x)
x= torch.ones((3,3))
print(x)
x = torch.eye((3,3))
print(x)
x = torch.arange(start=0,end=5, step=1)
print(x)
x = torch.linspace(start=0.1, end=1, step=10)
print(x)
x = empty(size=(1,5)).normal_(mean=0,std=1)
print(x)
x = empty(size=(1,5)).uniform_(0,1)
print(x)
x = torch.diag(torch.ones(3))
print(x)

# how to init and convert tensors to other types(int, float, double)

tensor = torch.arange(4)
print(tensor.bool()) # convert to boolean all > 1 is True (working in cpu and gpu)
print(tensor.short()) # convert all to int16
print(tensor.long().dtype) # convert all to int64
print(tensor.half().dtype) # convert to float16
print(tensor.float().dtype) # convert to float32
print(tensor.double().dtype) # converto to float64

# Arry to Tensor convertion and vice-versa
import numpy as np
np_array = np.zeros((5,5))
tensor = tensor.from_numpy()
np_array_back = tensor.numpy() # potentiall numeric rounding errors

### Part 2: Tensor Math and Comparison Operations

x = torch.tensor([1,2,3])
y = torch.tensor([9,8,7])

# addition
z1 = torch.empty(3)
torch.add(x,y, out=z1)
print(z1)

z2 = torch.add(x,y)

z = x + y

# substraction
z = x - y

# Division
z = torch.true_divide(x,y)

# Implece operations
# implace operation, mutate the tensor, not create any copy
t = torch.zeros(3) # inplace
t.add_(x)

t += x # inplace operation, mutate the tensor, not create any copy

# Exponentiation

z = x.pow(2) # elem wise exponential
z = x**2

# Single comparison
z = x > 0
z = x < 0

# Matrix Multiplication
x1 = torch.rand((2,5))
x2 = torch.rand((5,3))
x3 = torch.mm(x1,x2) # 2x3

x3 = x1.mm(x2)

# matrix exponentiation
matrix_exp = torch.rand(5,5)
print(matrix_exp.matrix_power(3))

# element wise matrix multi
z = x * y
print(z)

# Dot product
z = torch.dot(x,y)
print(z)

# Batch Matrix Multiplication
batch = 32
n = 10
m = 20
p = 30
tensor1 = torch.rand(batch,n,m)
tensor2 = torch.rand(batch,m,p)

out_bmm = torch.bmm(tensor1,tensor2)
print(out_bmm.shape)

# Example of Broadcasting
x1 = torch.rand(5,5)
x2 = torch.rand(1,5)

z = x1 - x2 # x2 will be expanded to match x1
print(z)


z = x1 ** x2 # x2 will be expanded to match x1
print(z)

# Other useful tensor operations

sum_z = torch.sum(x, dim=0)
print(sum_z)

values, indices = torch.max(x, dim=0)
values, indices = torch.min(x, dim=0)
print(values, indices)
abs_x = torch.abs(x)

z = torch.argmax(x,dim=0)
print(z)

z = torch.argmin(x, dim=0)

mean_x = torch.mean(x.float(), dim=0)
z = torch.eq(x,y) # wich elements are equal element wise
print(z)

sorted_tensor = torch.sort(y, dim=0, descending=False)
print(sorted_tensor)


z = torch.clamp(x, min=0) # all element that are minor that 0 will be set to 0
z = torch.clamp(x, min=0, max=10) # all element that are minor that 0 will be set to 0, and all value greater to 10 will be set to 10

# Boolean
x = torch.tensor([1,0,1,1,1], dtype= torch.bool)

z = torch.any(x)
print(z)
z = torch.all(x)
print(z)


### Part 3: Tensor Indexing

batch_size = 10
features = 25
x = torch.rand(batch_size, features)

print(x[0].shape) # x[0,:]

print(x[:,0].shape)

print(x[2,0:10])

x[0,0] = 100

# Fancy

x = torch.arange(10)

indices = [2,5,8]

print(x[indices])

x = torch.rand(3,5)
rows = torch.tensor([1,0])
cols = torch.tensor([4,0])
print(x[rows, cols].shape)

# More advanced indexing
x = torch.arange(10)
print(x[(x < 2) & (x > 8)])
print(
    x[x.remainder(2) == 0]
)

# Useful operations
print(torch.where(x > 5, x, x*2))

print(torch.tensor([0,0,1,2,2,3,4]).unique())
print(x.ndimension())

print(x.numel())

### Part 4: Tensor reshaping

x = torch.arange(9)

x_3x3 = x.view(3,3) # contigus tensors
print(x_3x3.shape)

x_3x3_ = x.reshape(3,3) # contiguos and not contiguos tensors

print(x_3x3_.shape)


y = x_3x3.T
print(y)


# print(y.view(9))

# RuntimeError: view size is not compatible with input tensor's size
# and stride (at least one dimension spans across two contiguous
# subspaces). Use .reshape(...) instead.

x1 = torch.rand((2,5))
x2 = torch.rand((2,5))

print(torch.cat((x1,x2),dim=0).shape)


z = x1.view(-1)
print(z.shape)


batch = 64
x = torch.rand(batch, 2, 5)


z = x.view(batch, -1)
print(z.shape)


# Permute
z = x.permute(0,2,1)
print(z.shape)

x = torch.arange(10)

print(x.unsqueeze(0).shape)
print(x.unsqueeze(1).shape)
print(x.unsqueeze(0).unsqueeze(1).shape)
