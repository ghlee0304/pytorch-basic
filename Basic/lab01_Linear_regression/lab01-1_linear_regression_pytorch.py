import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import load_data

NPOINTS = 1000
TOTAL_EPOCH = 100

dataX, dataY = load_data.generate_data_for_linear_regression(NPOINTS)
#dataY = np.reshape(dataY, [-1])

dtype = torch.FloatTensor
x_train = Variable(torch.from_numpy(dataX).type(dtype),requires_grad=False)
y_train = Variable(torch.from_numpy(dataY).type(dtype),requires_grad=False)


class Model(torch.nn.Module):
    def __init__(self):
        torch.manual_seed(0)
        super(Model, self).__init__()
        self.layer = Linear(1,1)

    def forward(self, x):
        out = self.layer(x)
        return out


m = Model()
loss = torch.nn.MSELoss()
optimizer = torch.optim.SGD(m.parameters(), lr=0.1)


#for graph
arg_ = 0
plt.figure(num=None, figsize=(8, 14), dpi=60, facecolor='w', edgecolor='k')
plt.subplots_adjust(hspace=0.4, top=0.9, bottom=0.05)

for epoch in range(TOTAL_EPOCH):
    y_pred = m(x_train)
    #y_pred = m.forward(x_data)
    l = loss(y_pred, y_train)
    if (epoch+1) % 10 == 0:
        print("Epoch : {}, loss : {}".format(epoch+1, l.item()))

        # for graph
        arg_ += 1
        plt.subplot(5, 2, arg_)
        plt.scatter(x_train.data, y_train.data, marker='.')
        plt.plot(x_train.view(-1).data.numpy(), m(x_train).view(-1).data.numpy(), c='r')
        plt.title('Epoch {}'.format(epoch + 1))
        plt.grid()
        plt.xlim(-2, 2)

    optimizer.zero_grad()
    l.backward()
    optimizer.step()

plt.suptitle('LinearRegression', fontsize=20)
plt.savefig('./image/lab01-1_linear_regression.jpg')
plt.show()