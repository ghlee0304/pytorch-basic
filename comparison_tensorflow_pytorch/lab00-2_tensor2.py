import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

tf.set_random_seed(0)

a = tf.constant(1)
b = tf.constant(2)
c = tf.add(a, b)
d = tf.random_normal(shape=[], mean=0.0, stddev=0.01)

with tf.Session() as sess:
    print("<<< tensorflow >>>")
    print("a\t:\t", sess.run(a))
    print("b\t:\t", sess.run(b))
    print("a+b\t:\t", sess.run(c))
    print("d\t:\t", sess.run(d))

print()


import torch

torch.manual_seed(0)

a = torch.IntTensor([1])
b = torch.IntTensor([2])
c = torch.add(a, b)
d = torch.FloatTensor(1).data.normal_(mean=0.0, std=0.01)

print("<<< pytorch >>>")
print("a\t:\t", a.item())
print("b\t:\t", b.item())
print("a+b\t:\t", c.item())
print("d\t:\t", d.item())


