import numpy as np
import visdom
from torchvision import datasets, transforms
train_dataset = datasets.MNIST(root="./data/MNIST",
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)
np.random.seed(0)

def plot_mnist(images, n_images, seed=0):
    images = np.reshape(images, [len(images), 28, 28])
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

images = plot_mnist(train_dataset.train_data, 25)


vis = visdom.Visdom()

x_data = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
y_data = x_data**2

# plot
vis.line(Y=y_data, X=x_data,
         opts=dict(title="y = x^2", xlabel='X', ylabel='Y'))

# image
vis.image(images, opts=dict(title='MNIST'))

# scatter1
vis.scatter(X=np.random.rand(100, 2),Y=(np.random.rand(100)[np.random.rand(100) > 0] + 1.5).astype(int),
            opts=dict(legend=['Men', 'Women'], markersize=5))

# scatter2
'''
vis.scatter(X=np.random.rand(100,2))

# bar
vis.bar(X=np.random.rand(20))
'''

# histogram
vis.histogram(X=np.random.randn(1000), opts=dict(numbins=20))


# contour
x = np.tile(np.arange(1, 101), (100, 1))
y = x.transpose()
X = np.exp((((x - 50) ** 2) + ((y - 50) ** 2)) / -(20.0 ** 2))
vis.contour(X=X, opts=dict(colormap='Viridis'))

# heatmap
vis.heatmap(
    X=np.outer(np.arange(1, 6), np.arange(1, 11)),
    opts=dict(
        columnnames=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
        rownames=['y1', 'y2', 'y3', 'y4', 'y5'],
        colormap='Electric',
    )
)