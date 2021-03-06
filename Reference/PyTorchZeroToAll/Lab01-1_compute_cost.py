import torch
import numpy as np
import matplotlib.pyplot as plt

print(torch.__version__)

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0


def forward(x, weight):
    return x * weight


def loss(x, y, weight):
    y_pred = forward(x, weight)
    return (y_pred - y) * (y_pred - y)


w_list = []
mse_list = []
for w in np.arange(0.0, 4.1, 0.1):
    print("w =", w)
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val, w)
        l = loss(x_val, y_val, w)
        l_sum += l
        print("\t", x_val, y_val, y_pred_val, l)
    print("MSE = ", l_sum/3)
    w_list.append(w)
    mse_list.append(l_sum/3)

plt.plot(w_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()