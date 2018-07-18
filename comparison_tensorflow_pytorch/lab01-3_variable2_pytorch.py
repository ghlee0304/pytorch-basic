import torch
from torch.autograd import Variable

x = Variable(torch.ones(2, 2), requires_grad=True)
y = x.mean()

print("\n====== 초깃값 ======")
print("x : \n", x.data.numpy())
print("y : ", y.item())

print("\n>>> 학습 시작")
for epoch in range(5):
    if not x.grad is None:
        x.grad.data.zero_()
    y.backward()
    x.data = x.data - 0.1 * x.grad
    y_pred = x.mean()
    print("\n======= Epoch : {} =======".format(epoch + 1))
    print("x : \n", x.data.numpy())
    print("x.grad : \n", x.grad.numpy())
    print("y : ", y_pred.item())
