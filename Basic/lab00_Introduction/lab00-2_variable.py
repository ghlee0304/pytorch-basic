import torch

print("\n<<< Variables >>>")
from torch.autograd import Variable
x = Variable(torch.ones(2, 2), requires_grad=True)
y = x.mean()
y.backward()

print("\n====== 초깃값 ======")
print("x : ", x.data)
print("y : ", y.item())

print("\n>>> 학습 시작")
for epoch in range(10):
    x.data = x.data-0.01*x.grad
    y = x.mean()
    y.backward()
    print("\n======= Epoch : {} =======".format(epoch+1))
    print("x : ", x.data)
    print("x.grad : ", x.grad)
    print("y : ", y.item())