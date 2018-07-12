import tensorflow as tf
import numpy as np
import shutil

#load data
x_data = [[1., 5., 9., 27.], [2., 10., 14., 42.],
          [3., 15., 19., 57.], [4., 20., 24., 72.]]
y_data = [[1.1, 4.9], [2.3, 10.2], [2.9, 14.9], [3.8, 18.]]

x_train = np.array(x_data)
y_train = np.array(y_data)

#parameter setting
TOTAL_EPOCH = 1000
BOARD_PATH = "./board"
shutil.rmtree(BOARD_PATH)


def Linear(x, output_size, name):
    input_size=x.get_shape().as_list()
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[input_size[-1], output_size],
                            initializer=
                            tf.truncated_normal_initializer(mean=0.0, stddev=0.02))
        b = tf.get_variable(name='b', shape=[output_size],
                            initializer=tf.zeros_initializer())
        h = tf.nn.bias_add(tf.matmul(x, W), b)
    return h, W, b


def ReLU(x, output_size, name):
    input_size = x.get_shape().as_list()
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[input_size[-1], output_size],
                            initializer=
                            tf.truncated_normal_initializer(mean=0.0, stddev=0.02))
        b = tf.get_variable(name='b', shape=[output_size],
                            initializer=tf.zeros_initializer())
        h = tf.nn.bias_add(tf.matmul(x, W), b)
    return tf.nn.relu(h), W, b


class Model(object):
    def __init__(self, sess):
        tf.set_random_seed(0)
        self.build_net()
        self.sess = sess


    def build_net(self):
        self.X = tf.placeholder(tf.float32, [None, 4], name='X')
        self.Y = tf.placeholder(tf.float32, [None, 2], name='Y')

        self.layer1, W1, b1 = ReLU(self.X, 3, "layer1")
        self.layer2, W2, b2 = Linear(self.layer1, 2, "layer2")
        self.optim = self.optimizer

        with tf.variable_scope("Histogram"):
            W1_hist = tf.summary.histogram('W1_hist', W1)
            b2_hist = tf.summary.histogram('W2_hist', W2)
            b1_hist = tf.summary.histogram('b1_hist', b1)
            b2_hist = tf.summary.histogram('b2_hist', b2)

        with tf.variable_scope("Scalars"):
            loss_scalar = tf.summary.scalar('loss', self.loss)

        self.merged = tf.summary.merge_all()


    def fit(self, x_train, y_train):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer.add_graph(self.sess.graph)

        for epoch in range(TOTAL_EPOCH):
            l, m, _ = self.sess.run([self.loss, self.merged, self.optim],
                                    feed_dict={self.X: x_train, self.Y: y_train})
            self.writer.add_summary(m, global_step=epoch)
            if (epoch + 1) % 100 == 0:
                print("Epoch : {}, loss : {}".format(epoch + 1, (l)))


    def predict(self, x_test):
        return self.sess.run(self.layer2, feed_dict={self.X: x_test})


    @property
    def optimizer(self):
        return tf.train.GradientDescentOptimizer(0.001).minimize(self.loss)


    @property
    def loss(self):
        with tf.variable_scope("Loss"):
            _loss = tf.reduce_mean(tf.square(self.Y-self.layer2), name='_loss')
        return _loss


    @property
    def writer(self):
        return tf.summary.FileWriter(BOARD_PATH)


def main():
    sess = tf.Session()
    m = Model(sess)
    m.fit(x_train, y_train)
    y_pred = m.predict(x_train)

    print("\n<<< 최종 예측 결과 >>>")
    print(y_pred)

    print("\n<<< 실제 값 >>>")
    print(y_train)

          
main()if __name__ == "__main__":
    main()
