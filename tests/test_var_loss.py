import numpy as np
import tensorflow as tf
from deepclustering.model.loss.mis_loss import var_loss

# Configurations
d_input = 3
delta_variance = 1.0

# Placeholders for data and labels - row represent a point
x = tf.placeholder(tf.float32, shape=(None, d_input))
y = tf.placeholder(tf.int8, shape=(None, 1))

# List of classes
u, idx = tf.unique(tf.reshape(y, [-1]))
n_clusters = tf.size(u)

# For each label, a vector with the length of the number of points, with True where the point has the current label.
# Row i correspond to label i in the unique vector u.
masks = tf.equal(tf.reshape(y, [-1]), tf.expand_dims(u, axis=1))

# Tiled masks:
tiled_masks = tf.tile(tf.expand_dims(masks, axis=2), [1, 1, d_input])
tiled_masks = tf.cast(tiled_masks, tf.float32)

# Tiled x:
tiled_x = tf.tile(tf.expand_dims(x, axis=0), [n_clusters, 1, 1])

# Cluster sums:
cluster_sums = tf.reduce_sum(tf.multiply(tiled_masks, tiled_x), axis=1)

# Cluster sizes:
c_size = tf.reduce_sum(tf.cast(masks, tf.int32), axis=1)

# Cluster centers:
tiled_c_size = tf.tile(tf.expand_dims(c_size, axis=1), [1, d_input])
c_centers = tf.divide(cluster_sums, tf.cast(tiled_c_size, tf.float32))

# Center point for each input point:
indices = tf.expand_dims(idx, axis=1)
x_centers = tf.gather_nd(c_centers, indices)

# Cluster size for each input point:
x_cluster_size = tf.cast(tf.gather_nd(c_size, indices), tf.float32)

# Distance from centers:
x_distance = tf.norm(x - x_centers, axis=1)

# Un-normalized loss per output point
x_unnormalized_loss = tf.square(tf.nn.relu(x_distance - delta_variance))

# Normalized loss per output point
x_loss = tf.divide(x_unnormalized_loss, x_cluster_size)

# Variance loss:
local_var_loss = tf.reduce_sum(x_loss) / tf.cast(n_clusters, tf.float32)

# Data:
image = np.array([[0, 0, 0],
                  [1, 1, 1],
                  [2, 2, 2],
                  [3, 3, 3],
                  [4, 4, 4]])
labels = np.array([[0], [1], [3], [1], [2]], dtype=np.int8)

with tf.Session() as sess:

    eval_x, eval_y, eval_u, eval_n_clusters, eval_masks, eval_tiled_masks, eval_tiled_x, eval_cluster_sums, \
    eval_c_size, eval_tiled_c_size, eval_c_centers, eval_x_centers, eval_x_cluster_size, eval_x_distance, \
    eval_x_unnormalized_loss, eval_x_loss, eval_var_loss = \
        sess.run([x,y,u, n_clusters, masks, tiled_masks, tiled_x, cluster_sums, c_size, tiled_c_size,
                  c_centers, x_centers, x_cluster_size, x_distance, x_unnormalized_loss, x_loss, local_var_loss],
                 feed_dict={x: image, y: labels})

    print("\nx:\n", eval_x)
    print("\nx.shape:\n", eval_x.shape)
    print("\ny:\n", eval_y)
    print("\nunique:\n", eval_u)
    print("\nnum clusters:\n", eval_n_clusters)
    print("\nmasks:\n", eval_masks)
    print("\ntiled masks:\n", eval_tiled_masks)
    print("\ntiled masks shape:\n", eval_tiled_masks.shape)
    print("\ntiled x:\n", eval_tiled_x)
    print("\ntiled x shape:\n", eval_tiled_x.shape)
    print("\ncluster sums:\n", eval_cluster_sums)
    print("\ncluster sums shape:\n", eval_cluster_sums.shape)
    print("\ncluster sizes:\n", eval_c_size)
    print("\ntiled cluster sizes:\n", eval_tiled_c_size)
    print("\ncluster centers:\n", eval_c_centers)
    print("\ncenter of each input point:\n", eval_x_centers)
    print("\ncluster size per data point:\n", eval_x_cluster_size)
    print("\ndistance from centers (l2 norm):\n", eval_x_distance)
    print("\nun-normalized loss per output point:\n", eval_x_unnormalized_loss)
    print("\nnormalized loss per output point:\n", eval_x_loss)
    print("\nVariance loss:\n", eval_var_loss)


var_loss_graph = var_loss(x, y, hinge_value=0.5)

with tf.Session() as sess:
    loss_value = sess.run(var_loss_graph, feed_dict={x: image, y: labels})
    print("\nLoss value using loss graph:\n", loss_value)
