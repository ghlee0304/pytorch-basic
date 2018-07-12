import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#define tensor op
scalar = tf.constant(value=1, shape=[])
vector = tf.constant(value=[1, 2, 3], shape=[3])
matrix = tf.constant(value=[[1, 2], [3, 4]], shape=[2, 2])

with tf.Session() as sess:
    s = sess.run(scalar)
    v = sess.run(vector)
    m = sess.run(matrix)

    print("\n<<< Scalar >>>\n", s)
    print("\n<<< Vector >>>\n", v)
    print("\n<<< Matrix >>>\n", m)


import torch

scalar = torch.FloatTensor([1])
print("\n<<< Scalar >>>")
print("value\t: ", scalar.item())
print("type\t: ", scalar.type())
print("size\t: ", scalar.size())

vector = torch.FloatTensor([1, 2, 3])
print("\n<<< Vector >>>")
print("value\t: ", vector.data.numpy())
print("type\t: ", vector.type())
print("size\t: ", vector.size())

matrix = torch.FloatTensor([[1, 2], [3, 4]])
print("\n<<< Matrix >>>")
print("value\t: \n", matrix.data.numpy())
print("type\t: ", matrix.type())
print("size\t: ", matrix.size())
