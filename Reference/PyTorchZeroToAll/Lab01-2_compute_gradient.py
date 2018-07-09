x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


def forward(x, weight):
    return x * weight


def loss(x, y, weight):
    y_pred = forward(x, weight)
    return (y_pred - y) * (y_pred - y)


def gradient(x, y, weight):
    return 2 * x *(x * weight - y)


init_w = 1.0
learning_rate = 0.01
print("predict (before training) {}: {}".format(4, forward(4, init_w)))

w = init_w
for epoch in range(10):
    print("<<< Epoch : {} >>>".format(epoch))
    for x_val, y_val in zip(x_data, y_data):
        grad = gradient(x_val, y_val, w)
        w = w - learning_rate * grad
        print("\tx_val: {}".format(x_val))
        print("\ty_val: {}".format(y_val))
        print("\tgrad: {}".format(grad))
        l = loss(x_val, y_val, w)
    print("Epoch: {}, loss: {}".format(epoch, l))
    print()

print("predict (before training) {}: {}".format(4, forward(4, w)))


'''
predict (before training) 4: 4.0
<<< Epoch : 0 >>>
	x_val: 1.0
	y_val: 2.0
	grad: -2.0
	x_val: 2.0
	y_val: 4.0
	grad: -7.84
	x_val: 3.0
	y_val: 6.0
	grad: -16.2288
Epoch: 0, loss: 4.919240100095999

<<< Epoch : 1 >>>
	x_val: 1.0
	y_val: 2.0
	grad: -1.478624
	x_val: 2.0
	y_val: 4.0
	grad: -5.796206079999999
	x_val: 3.0
	y_val: 6.0
	grad: -11.998146585599997
Epoch: 1, loss: 2.688769240265834

<<< Epoch : 2 >>>
	x_val: 1.0
	y_val: 2.0
	grad: -1.093164466688
	x_val: 2.0
	y_val: 4.0
	grad: -4.285204709416961
	x_val: 3.0
	y_val: 6.0
	grad: -8.87037374849311
Epoch: 2, loss: 1.4696334962911515

<<< Epoch : 3 >>>
	x_val: 1.0
	y_val: 2.0
	grad: -0.8081896081960389
	x_val: 2.0
	y_val: 4.0
	grad: -3.1681032641284723
	x_val: 3.0
	y_val: 6.0
	grad: -6.557973756745939
Epoch: 3, loss: 0.8032755585999681

<<< Epoch : 4 >>>
	x_val: 1.0
	y_val: 2.0
	grad: -0.59750427561463
	x_val: 2.0
	y_val: 4.0
	grad: -2.3422167604093502
	x_val: 3.0
	y_val: 6.0
	grad: -4.848388694047353
Epoch: 4, loss: 0.43905614881022015

<<< Epoch : 5 >>>
	x_val: 1.0
	y_val: 2.0
	grad: -0.44174208101320334
	x_val: 2.0
	y_val: 4.0
	grad: -1.7316289575717576
	x_val: 3.0
	y_val: 6.0
	grad: -3.584471942173538
Epoch: 5, loss: 0.2399802903801062

<<< Epoch : 6 >>>
	x_val: 1.0
	y_val: 2.0
	grad: -0.3265852213980338
	x_val: 2.0
	y_val: 4.0
	grad: -1.2802140678802925
	x_val: 3.0
	y_val: 6.0
	grad: -2.650043120512205
Epoch: 6, loss: 0.1311689630744999

<<< Epoch : 7 >>>
	x_val: 1.0
	y_val: 2.0
	grad: -0.241448373202223
	x_val: 2.0
	y_val: 4.0
	grad: -0.946477622952715
	x_val: 3.0
	y_val: 6.0
	grad: -1.9592086795121197
Epoch: 7, loss: 0.07169462478267678

<<< Epoch : 8 >>>
	x_val: 1.0
	y_val: 2.0
	grad: -0.17850567968888198
	x_val: 2.0
	y_val: 4.0
	grad: -0.6997422643804168
	x_val: 3.0
	y_val: 6.0
	grad: -1.4484664872674653
Epoch: 8, loss: 0.03918700813247573

<<< Epoch : 9 >>>
	x_val: 1.0
	y_val: 2.0
	grad: -0.13197139106214673
	x_val: 2.0
	y_val: 4.0
	grad: -0.5173278529636143
	x_val: 3.0
	y_val: 6.0
	grad: -1.0708686556346834
Epoch: 9, loss: 0.021418922423117836

predict (before training) 4: 7.804863933862125
'''