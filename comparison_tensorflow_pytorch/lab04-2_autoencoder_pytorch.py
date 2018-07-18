import torch
from load_mnist import save_and_load_mnist
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import os
import visdom
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
torch.manual_seed(0)

#parameter setting
TOTAL_EPOCH = 10
BATCH_SIZE = 32


#load data
class MnistDataset(Dataset):
    def __init__(self, data_name):
        dataset = save_and_load_mnist("./data/mnist/")
        x = dataset[data_name]
        self.len = len(x)
        self.x_data = torch.from_numpy(x)

    def __getitem__(self, index):
        return self.x_data[index]

    def __len__(self):
        return self.len


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(784, 256),
            torch.nn.Sigmoid())

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(256, 784),
            torch.nn.Sigmoid())


    def forward(self, x):
        h1 = self.encoder(x)
        h2 = self.decoder(h1)
        return h2


class Solver(object):
    def __init__(self):
        self.m = Model()
        self.train_dataset = MnistDataset('train_data')
        self.test_dataset = MnistDataset('test_data')
        self.vis = visdom.Visdom()
        assert self.vis.check_connection()
        self.loss_plot = \
            self.vis.line(Y=np.array([0]), X=np.array([0]),
                          opts=dict(title="During Training", xlabel='Epoch', ylabel='Loss'))


    def fit(self):
        print(">>> Start Train ")
        losses = []
        for epoch in range(TOTAL_EPOCH):
            loss_per_epoch = 0
            total_step = 0
            for i, data in enumerate(self.train_loader, 0):

                total_step += 1
                inputs = Variable(data)
                x_pred = self.m(inputs)
                c = self.loss(x_pred, inputs)
                loss_per_epoch += c

                self.optimizer.zero_grad()
                c.backward()
                self.optimizer.step()

            loss_per_epoch /= (total_step)
            losses.append([loss_per_epoch.item(), epoch + 1])
            tmp_loss = np.array(losses)
            self.vis.line(Y=tmp_loss[:, 0], X=tmp_loss[:, 1], win=self.loss_plot, update='insert')
            test_loader = DataLoader(dataset=self.test_dataset, batch_size=25, shuffle=False, num_workers=0)

            for d in test_loader:
                sample_data = d
                x_pred = self.predict(sample_data)
                break

            if epoch == 0:
                real_image = plot_mnist(sample_data.data.numpy(), 25, 1)
                self.vis.image(real_image, opts=dict(title='Real Image'))

            pred_image = plot_mnist(x_pred.data.numpy(), 25, 2)
            self.vis.image(pred_image, opts=dict(title='Epoch {} Gen Image'.format(epoch+1)))

            print("Epoch : [{:4d}/{:4d}], cost : {:.6f}".format(epoch + 1, TOTAL_EPOCH, loss_per_epoch))


    def predict(self, x_test):
        return self.m(x_test)


    @property
    def train_loader(self):
        return DataLoader(dataset=self.train_dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=0)


    @property
    def loss(self):
        return torch.nn.MSELoss()


    @property
    def optimizer(self):
        return torch.optim.Adam(self.m.parameters(), lr=0.01)


def plot_mnist(images, n_images, fig, seed=0):
    images = np.reshape(images, [len(images), 28, 28])
    plt.figure()
    plt.gca().set_axis_off()
    h_num = int(np.sqrt(n_images))
    v_num = int(np.sqrt(n_images))
    v_list = []
    np.random.seed(seed)
    mask = np.random.permutation(len(images))
    count = 0
    for j in range(v_num):
        h_list = []
        for i in range(h_num):
            h_list.append(images[mask[count]])
            count+=1
        tmp = np.hstack(h_list)
        v_list.append(tmp)
    im = np.vstack(v_list)

    return im


def main():
    solver = Solver()
    solver.fit()


if __name__ == "__main__":
    main()
