import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import visdom

#load data
x_data = [[1., 5., 9., 27.], [2., 10., 14., 42.],
          [3., 15., 19., 57.], [4., 20., 24., 72.]]
y_data = [[1.1, 4.9], [2.3, 10.2], [2.9, 14.9], [3.8, 18.]]
x_train = Variable(data=torch.FloatTensor(x_data), requires_grad=False)
y_train = Variable(data=torch.FloatTensor(y_data), requires_grad=False)

#parameter setting
TOTAL_EPOCH = 2500


def weight_init(layer):
    layer.weight.data.normal_(0.0, 0.02).clamp_(min=-2.0, max=2.0)
    layer.bias.data.fill_(0)

          
class Model(torch.nn.Module):
    def __init__(self):
        torch.manual_seed(0)
        super(Model, self).__init__()
        self.layer1 = torch.nn.Linear(4, 3)
        self.layer2 = torch.nn.Linear(3, 2)
        weight_init(self.layer1)
        weight_init(self.layer2)

          
    def forward(self, x):
        h1 = self.layer1(x)
        h2 = F.relu(h1)
        h3 = self.layer2(h2)
        return h3


class Solver(object):
    def __init__(self):
        self.m = Model()
        self.vis = visdom.Visdom()
        assert self.vis.check_connection()
        self.loss_plot = \
            self.vis.line(Y=np.array([0]), X=np.array([0]),
                          opts=dict(title="During Training", xlabel='Epoch', ylabel='Loss'))

                    
    def fit(self, x_train, y_train):
        losses = []
        for epoch in range(TOTAL_EPOCH):
            y_pred = self.m(x_train)
            loss = self.loss(y_pred, y_train)
            if (epoch + 1) % 100 == 0:
                print("Epoch : {}, loss : {}".format(epoch + 1, loss.item()))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if epoch>10:
                losses.append([loss.item(), epoch+1])
                tmp_loss = np.array(losses)
                self.vis.line(Y=tmp_loss[:,0], X=tmp_loss[:,1],
                              win=self.loss_plot, update='insert')

        losses = np.array(losses)
        self.vis.line(Y=losses[:,0], X=losses[:,1],
                      opts=dict(title="After Training", xlabel='Epoch', ylabel='Loss'))

          
    def predict(self, x_test):
        return self.m(x_test)


    @property
    def loss(self):
        return torch.nn.MSELoss()


    @property
    def optimizer(self):
        return torch.optim.SGD(self.m.parameters(), lr=0.001)


def main():
    solver = Solver()
    solver.fit(x_train, y_train)
    y_pred = solver.predict(x_train)

    print("\n<<< 최종 예측 결과 >>>")
    print(y_pred.data.numpy())

    print("\n<<< 실제 값 >>>")
    print(y_train.data.numpy())


if __name__ == "__main__":
    main()
