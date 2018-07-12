import tensorflow as tf
import numpy as np

#load data
x_data = [[1., 5., 9., 27.], [2., 10., 14., 42.],
          [3., 15., 19., 57.], [4., 20., 24., 72.]]
y_data = [[1.1, 4.9], [2.3, 10.2], [2.9, 14.9], [3.8, 18.]]

x_train = np.array(x_data)
y_train = np.array(y_data)

#parameter setting
TOTAL_EPOCH = 1000


def Linear(x, output_size, name):
    input_size=x.get_shape().as_list()
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[input_size[-1], output_size],
                            initializer=
                            tf.truncated_normal_initializer(mean=0.0, stddev=0.02))
        b = tf.get_variable(name='b', shape=[output_size],
                            initializer=tf.zeros_initializer())
        h = tf.nn.bias_add(tf.matmul(x, W), b)
    return h


def ReLU(x, output_size, name):
    input_size = x.get_shape().as_list()
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[input_size[-1], output_size],
                            initializer=
                            tf.truncated_normal_initializer(mean=0.0, stddev=0.02))
        b = tf.get_variable(name='b', shape=[output_size],
                            initializer=tf.zeros_initializer())
        h = tf.nn.bias_add(tf.matmul(x, W), b)
    return tf.nn.relu(h)


class Model(object):
    def __init__(self, sess):
        tf.set_random_seed(0)
        self.build_net()
        self.sess = sess


    def build_net(self):
        self.X = tf.placeholder(tf.float32, [None, 4], name='X')
        self.Y = tf.placeholder(tf.float32, [None, 2], name='Y')

        self.layer1 = ReLU(self.X, 3, "layer1")
        self.layer2 = Linear(self.layer1, 2, "layer2")
        self.optim = self.optimizer


    def fit(self, x_train, y_train):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        for epoch in range(TOTAL_EPOCH):
            l, _ = self.sess.run([self.loss, self.optim],
                                 feed_dict={self.X: x_train, self.Y: y_train})
            if (epoch + 1) % 100 == 0:
                print("Epoch : {}, loss : {}".format(epoch + 1, (l)))

          
    def predict(self, x_test):
        return self.sess.run(self.layer2, feed_dict={self.X: x_test})


    @property
    def optimizer(self):
        return tf.train.GradientDescentOptimizer(0.001).minimize(self.loss)


    @property
    def loss(self):
        return tf.reduce_mean(tf.square(self.Y-self.layer2))


def main():
    sess = tf.Session()
    m = Model(sess)
    m.fit(x_train, y_train)
    y_pred = m.predict(x_train)

    print("\n<<< 최종 예측 결과 >>>")
    print(y_pred)

    print("\n<<< 실제 값 >>>")
    print(y_train)


if __name__ == "__main__":
    main()
