import torch
from torch.autograd import Variable

x = Variable(torch.ones(2, 2), requires_grad=True)
target = x.mean()

print("\n====== 초깃값 ======")
print("x : \n", x.data.numpy())
print("y : ", target.item())

print("\n>>> 학습 시작")
for epoch in range(5):
    target.backward()
    x.data = x.data - 0.1 * x.grad
    target = x.mean()
    print("\n======= Epoch : {} =======".format(epoch + 1))
    print("x : \n", x.data.numpy())
    print("x.grad : \n", x.grad.numpy())
    print("y : ", target.item())

    if not x.grad is None:
        x.grad.data.zero_()




