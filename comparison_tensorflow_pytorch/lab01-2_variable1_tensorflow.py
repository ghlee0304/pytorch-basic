import tensorflow as tf

x = tf.get_variable(name='x', shape=[2, 2], trainable=False,
                    initializer=tf.zeros_initializer())
y = tf.get_variable(name='y', shape=[2, 2], trainable=False,
                    initializer=tf.ones_initializer())
z = tf.get_variable(name='z', shape=[2,2], trainable=False,
                    initializer=tf.random_normal_initializer())

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(x), "\n")
    print(sess.run(y), "\n")
    print(sess.run(z))


