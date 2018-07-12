import tensorflow as tf

x = tf.get_variable(name='x', shape=[2, 2], trainable=True,
                    initializer=tf.ones_initializer())
y = tf.reduce_mean(x)
grad = tf.gradients(y, x)[0]
apply_gradients = x.assign(x-0.1*grad)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    x_val = sess.run(x)
    x_grad = sess.run(grad)
    target = sess.run(y)

    print("\n====== 초깃값 ======")
    print("x : \n", x_val)
    print("y : ", target)

    for epoch in range(5):
        sess.run(apply_gradients)
        x_val = sess.run(x)
        x_grad = sess.run(grad)
        target = sess.run(y)
        print("\n======= Epoch : {} =======".format(epoch + 1))
        print("x : \n", x_val)
        print("x.grad : \n", x_grad)
        print("y : ", target)


