import torch
from torch.autograd import Variable

x = Variable(torch.zeros(2, 2), requires_grad=False)
y = Variable(torch.ones(2, 2), requires_grad=False)
z = Variable(torch.randn(2,2), requires_grad=False)

print(x.data.numpy(), "\n")
print(y.data.numpy(), "\n")
print(z.data.numpy())