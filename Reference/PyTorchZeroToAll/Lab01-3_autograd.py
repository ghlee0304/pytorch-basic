import torch
from torch.autograd import Variable

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = Variable(torch.Tensor([1.0]), requires_grad=True)


def forward(x, weight):
    return x * weight


def loss(x, y, weight):
    y_pred = forward(x, weight)
    return (y_pred - y) * (y_pred - y)


learning_rate = 0.01
print("predict (before training) {}: {}".format(4, forward(4, w).data[0]))

for epoch in range(10):
    print("<<< Epoch : {} >>>".format(epoch))
    for x_val, y_val in zip(x_data, y_data):
        print(x_val)
        l = loss(x_val, y_val, w)
        l.backward()
        w.data = w.data - learning_rate * w.grad.data
        print("\tx_val: {}".format(x_val))
        print("\ty_val: {}".format(y_val))
        print("\tgrad: {}".format(w.grad.data[0]))
        w.grad.data.zero_()
    print("Epoch: {}, loss: {}".format(epoch, l.item()))
    print()

print("predict (after training) {}: {}".format(4, forward(4, w).data[0]))

'''
predict (before training) 4: 4.0
<<< Epoch : 0 >>>
	x_val: 1.0
	y_val: 2.0
	grad: -2.0
	x_val: 2.0
	y_val: 4.0
	grad: -7.840000152587891
	x_val: 3.0
	y_val: 6.0
	grad: -16.228801727294922
Epoch: 0, loss: 7.315943717956543

<<< Epoch : 1 >>>
	x_val: 1.0
	y_val: 2.0
	grad: -1.478623867034912
	x_val: 2.0
	y_val: 4.0
	grad: -5.796205520629883
	x_val: 3.0
	y_val: 6.0
	grad: -11.998146057128906
Epoch: 1, loss: 3.9987640380859375

<<< Epoch : 2 >>>
	x_val: 1.0
	y_val: 2.0
	grad: -1.0931644439697266
	x_val: 2.0
	y_val: 4.0
	grad: -4.285204887390137
	x_val: 3.0
	y_val: 6.0
	grad: -8.870372772216797
Epoch: 2, loss: 2.1856532096862793

<<< Epoch : 3 >>>
	x_val: 1.0
	y_val: 2.0
	grad: -0.8081896305084229
	x_val: 2.0
	y_val: 4.0
	grad: -3.1681032180786133
	x_val: 3.0
	y_val: 6.0
	grad: -6.557973861694336
Epoch: 3, loss: 1.1946394443511963

<<< Epoch : 4 >>>
	x_val: 1.0
	y_val: 2.0
	grad: -0.5975041389465332
	x_val: 2.0
	y_val: 4.0
	grad: -2.3422164916992188
	x_val: 3.0
	y_val: 6.0
	grad: -4.848389625549316
Epoch: 4, loss: 0.6529689431190491

<<< Epoch : 5 >>>
	x_val: 1.0
	y_val: 2.0
	grad: -0.4417421817779541
	x_val: 2.0
	y_val: 4.0
	grad: -1.7316293716430664
	x_val: 3.0
	y_val: 6.0
	grad: -3.58447265625
Epoch: 5, loss: 0.35690122842788696

<<< Epoch : 6 >>>
	x_val: 1.0
	y_val: 2.0
	grad: -0.3265852928161621
	x_val: 2.0
	y_val: 4.0
	grad: -1.2802143096923828
	x_val: 3.0
	y_val: 6.0
	grad: -2.650045394897461
Epoch: 6, loss: 0.195076122879982

<<< Epoch : 7 >>>
	x_val: 1.0
	y_val: 2.0
	grad: -0.24144840240478516
	x_val: 2.0
	y_val: 4.0
	grad: -0.9464778900146484
	x_val: 3.0
	y_val: 6.0
	grad: -1.9592113494873047
Epoch: 7, loss: 0.10662525147199631

<<< Epoch : 8 >>>
	x_val: 1.0
	y_val: 2.0
	grad: -0.17850565910339355
	x_val: 2.0
	y_val: 4.0
	grad: -0.699742317199707
	x_val: 3.0
	y_val: 6.0
	grad: -1.4484672546386719
Epoch: 8, loss: 0.0582793727517128

<<< Epoch : 9 >>>
	x_val: 1.0
	y_val: 2.0
	grad: -0.1319713592529297
	x_val: 2.0
	y_val: 4.0
	grad: -0.5173273086547852
	x_val: 3.0
	y_val: 6.0
	grad: -1.070866584777832
Epoch: 9, loss: 0.03185431286692619

predict (before training) 4: 7.804864406585693
'''