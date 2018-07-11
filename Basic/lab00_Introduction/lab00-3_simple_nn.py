import torch
from torch.autograd import Variable

x_data = Variable(data=torch.FloatTensor([[1., 5.], [2., 10.], [3., 15.], [4., 20.]]), requires_grad=False)
y_data = Variable(data=torch.FloatTensor([[1.1, 4.9], [2.3, 10.2], [2.9, 14.9], [3.8, 18.]]), requires_grad=False)

#like tensorflow
torch.manual_seed(0)
ReLU = torch.nn.ReLU()

W1 = Variable(torch.randn(2, 4), requires_grad=True)
b1 = Variable(torch.zeros(4), requires_grad=True)
h1 = ReLU(torch.matmul(x_data, W1)+b1)

W2 = Variable(torch.randn(4, 2), requires_grad=True)
b2 = Variable(torch.zeros(2), requires_grad=True)

parameters = [W1, b1, W2, b2]

for epoch in range(100):
    y_pred = torch.matmul(h1, W2) + b2

    loss = (y_pred-y_data).pow(2).sum()
    if (epoch+1) % 10 == 0:
        print("Epoch : {}, Loss : {}".format(epoch+1, loss.item()))

    loss.backward(retain_graph=True)

    for param in [W1, b1, W2, b2]:
        param.data -= 0.0001 * param.grad.data

    for param in [W1, b1, W2, b2]:
        if not param.grad is None: param.grad.data.zero_()

y_pred = torch.matmul(h1, W2) + b2
print("\n<<< 최종 예측 결과 >>>")
print(y_pred.data.numpy())

print("\n<<< 실제 값 >>>")
print(y_data.data.numpy())