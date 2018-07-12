import torch
import torch.nn.functional as F
from torch.autograd import Variable
import pdb

#load data
x_data = [[1., 5., 9., 27.], [2., 10., 14., 42.],
          [3., 15., 19., 57.], [4., 20., 24., 72.]]
y_data = [[1.1, 4.9], [2.3, 10.2], [2.9, 14.9], [3.8, 18.]]
x_train = Variable(data=torch.FloatTensor(x_data), requires_grad=False)
y_train = Variable(data=torch.FloatTensor(y_data), requires_grad=False)

#parameter setting
TOTAL_EPOCH = 1000


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


    def fit(self, x_train, y_train):
        for epoch in range(TOTAL_EPOCH):
            y_pred = self.m(x_train)
            loss = self.loss(y_pred, y_train)
            pdb.set_trace()
            if (epoch + 1) % 100 == 0:
                print("Epoch : {}, loss : {}".format(epoch + 1, loss.item()))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

                    
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
