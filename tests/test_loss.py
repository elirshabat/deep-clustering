import tensorflow as tf
import numpy as np
import deepclustering.model.loss.mis_loss as mis_loss

image = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 1.2, 0],
                  [0, 0.1, 0.97],
                  [0.02, 0.89, -0.5],
                  [1.2, 0.03, 0.2],
                  [0.03, 0, 0.98],
                  [1, 0, 1],
                  [0, 1, 0]])

labels = np.array([[0],
                   [1],
                   [1],
                   [2],
                   [1],
                   [0],
                   [2],
                   [2],
                   [1]])

x = tf.placeholder(tf.float32, [None, 3])
y = tf.placeholder(tf.int8, [None, 1])
loss_node = mis_loss.loss(x, y, 0.5, 1.0, var_weight=1.0, dist_weight=1.0, regularization_weight=1.0)

with tf.Session() as sess:
    eval_x, eval_y, eval_loss = sess.run([x, y, loss_node], feed_dict={x: image, y: labels})

    print("\nx:\n", eval_x)
    print("\ny:\n", eval_y)
    print("\nloss:\n", eval_loss)
