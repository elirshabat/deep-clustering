import tensorflow as tf
import numpy as np
from deepclustering.model.loss.mis_loss import distance_loss

# Configuration:
d = 3
delta_dist = 0.5

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

# Tiled centers:
tiled_centers_1 = tf.tile(tf.expand_dims(c_centers, axis=2), [1, 1, n_clusters])
tiled_centers_2 = tf.transpose(tiled_centers_1, perm=[2, 1, 0])

# Distance matrix:
distance_matrix = tf.norm(tiled_centers_1 - tiled_centers_2, axis=1)

# Un-normalized loss:
local_loss = tf.divide(tf.reduce_sum(tf.square(tf.nn.relu(2*delta_dist - distance_matrix))),
                       tf.cast(n_clusters*(n_clusters - 1), tf.float32))


# Data:
image = np.array([[0, 0, 0],
                  [1, 1, 1],
                  [2, 2, 2],
                  [3, 3, 3],
                  [4, 4, 4]])
labels = np.array([[0], [1], [3], [1], [2]], dtype=np.int8)


with tf.Session() as sess:
    eval_x, eval_y, eval_c_centers, eval_tiled_centers1, eval_tiled_centers2, eval_distance_matrix, eval_local_loss = \
        sess.run([x, y, c_centers, tiled_centers_1, tiled_centers_2, distance_matrix, local_loss],
                 feed_dict={x: image, y: labels})

    print("\nx:\n", eval_x)
    print("\ny:\n", eval_y)
    print("\ncenters:\n", eval_c_centers)
    print("\ntiled centers 1:\n", eval_tiled_centers1)
    print("\ntiled centers 2:\n", eval_tiled_centers2)
    print("\ndistance matrix:\n", eval_distance_matrix)
    print("\ndistance matrix shape:\n", eval_distance_matrix.shape)
    print("\nlocal loss:\n", eval_local_loss)


distance_loss_graph = distance_loss(x, y, hinge_value=2.0)

with tf.Session() as sess:
    loss_value = sess.run(distance_loss_graph, feed_dict={x: image, y: labels})
    print("\nDistance loss value using loss graph:\n", loss_value)