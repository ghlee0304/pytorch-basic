import torch
print(torch.__version__)

import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable


#책에서 local value와 global value의 경계가 모호하여 코드가 엄밀하게 작성되지 않아서 다시 작성하였다.
def get_data():
    x_train = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
    y_train = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
    dtype = torch.FloatTensor
    X = Variable(torch.from_numpy(x_train).type(dtype),requires_grad=False).view(17,1)
    Y = Variable(torch.from_numpy(y_train).type(dtype),requires_grad=False)
    return X, Y, x_train, y_train


X, Y, x_train, y_train = get_data()
learning_rate = 0.0001

torch.manual_seed(0)
W = Variable(torch.randn(1),requires_grad=True)
b = Variable(torch.randn(1),requires_grad=True)

for epoch in range(100):
    y_pred = torch.matmul(X, W)+b
    loss = (y_pred-Y).pow(2).sum()
    if epoch % 10 == 0:
        print(loss.item())

    for param in [W, b]:
        if not param.grad is None: param.grad.data.zero_()
    loss.backward()

    W.data -= learning_rate * W.grad.data
    b.data -= learning_rate * b.grad.data

plt.scatter(x_train, y_train)
plt.plot(x_train, x_train*W.item()+b.item(), c='r')
plt.show()