import torch
import numpy as np

print("\n<<< Scalar >>>")
x = torch.rand(1)
print(x.type())
print(x.item())
print(x.size())

print("\n<<< Vector >>>")
x = torch.FloatTensor([23, 24, 24.5, 26, 27.2, 23.0])
print(x.type())
print(x.data)
print(x.size())

print("\n<<< Matrix >>>")
np.random.seed(0)
data = np.random.randn(3,3)
x = torch.from_numpy(data)
print(x.type())
print(x.data)
print(x.size())

print("\n<<< Some Operations >>>")
torch.manual_seed(0)
a = torch.rand(2, 2)
b = torch.rand(2, 2)
print("a : ", a.data)
print("b : ", b)

c = torch.add(a, b)
print("a+b : ", c)
d = torch.add(a,-b)
print("a-b : ", d)
e = torch.matmul(a,b)
print("axb : ", e)

print("\n<<< Variables >>>")
from torch.autograd import Variable
x = Variable(torch.ones(2,2), requires_grad=True)
y = x.mean()
y.backward()
print("x : ", x.data)
print("x_grad : ", x.grad)
print("\n>>> 학습 시작")
for epoch in range(10):
    x.data = x.data-0.01*x.grad
    y = x.mean()
    y.backward()
    print("\n=== Epoch : {} ===".format(epoch+1))
    print("x : ", x.data)
    print("x.grad : ", x.grad)
    print("y : ", y.item())

