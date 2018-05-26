import tensorflow as tf
import numpy as np
from deepclustering.model.loss.mis_loss import regularization

# Configuration:
d = 3

# Data placeholders:
x = tf.placeholder(tf.float32, shape=[None, d])
y = tf.placeholder(tf.int8, shape=[None, 1])

# List of classes
u, idx = tf.unique(tf.reshape(y, [-1]))
n_clusters = tf.size(u)

# For each label, a vector with the length of the number of points, with True where the point has the current label.
# Row i correspond to label i in the unique vector u.
masks = tf.equal(tf.reshape(y, [-1]), tf.expand_dims(u, axis=1))

# Tiled masks:
tiled_masks = tf.tile(tf.expand_dims(masks, axis=2), [1, 1, d])
tiled_masks = tf.cast(tiled_masks, tf.float32)

# Tiled x:
tiled_x = tf.tile(tf.expand_dims(x, axis=0), [n_clusters, 1, 1])

# Cluster sums:
cluster_sums = tf.reduce_sum(tf.multiply(tiled_masks, tiled_x), axis=1)

# Cluster sizes:
c_size = tf.reduce_sum(tf.cast(masks, tf.int32), axis=1)

# Cluster centers:
tiled_c_size = tf.tile(tf.expand_dims(c_size, axis=1), [1, d])
c_centers = tf.divide(cluster_sums, tf.cast(tiled_c_size, tf.float32))

# Regularization:
local_loss = tf.divide(tf.reduce_sum(tf.norm(c_centers, axis=1)), tf.cast(n_clusters, tf.float32))


# Data:
image = np.array([[0, 0, 0],
                  [1, 1, 1],
                  [2, 2, 2],
                  [3, 3, 3],
                  [4, 4, 4]])
labels = np.array([[0], [1], [3], [1], [2]], dtype=np.int8)


with tf.Session() as sess:
    eval_x, eval_y, eval_c_centers, eval_loss = \
        sess.run([x, y, c_centers, local_loss],
                 feed_dict={x: image, y: labels})

    print("\nx:\n", eval_x)
    print("\ny:\n", eval_y)
    print("\ncenters:\n", eval_c_centers)
    print("\nregularization loss:\n", eval_loss)


regularization_graph = regularization(x, y)

with tf.Session() as sess:
    reg_value = sess.run(regularization_graph, feed_dict={x: image, y: labels})
    print("\nRegularization using the graph:\n", reg_value)
