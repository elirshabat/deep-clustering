import tensorflow as tf


def var_loss(x, y, hinge_value=1.0):
    """
    Compute the variance loss.
    The variance loss gives high error for output points that are distance from the center of their clusters.
    :param x: Matrix of data points in it's rows.
    :param y: Column vector of labels.
    :param hinge_value: Points that closer to the centers of their clusters then this value, will have zero error.
    :return: The variance loss.
    """
    # x data dimension
    d_input = x.shape[1]

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
    x_unnormalized_loss = tf.square(tf.nn.relu(x_distance - hinge_value))

    # Normalized loss per output point
    x_loss = tf.divide(x_unnormalized_loss, x_cluster_size)

    # Variance loss:
    return tf.reduce_sum(x_loss) / tf.cast(n_clusters, tf.float32)


def distance_loss(x, y, hinge_value=1.0):
    """
    Compute the distance loss.
    The distance loss gives high error for two close cluster centers
    :param x: Matrix of data points in it's rows.
    :param y: Column vector of labels.
    :param hinge_value: Centers that are distant in more then this value, will have zero error.
    :return: The distance loss.
    """
    # x data dimension
    d = x.shape[1]

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
    return tf.divide(tf.reduce_sum(tf.square(tf.nn.relu(2*hinge_value - distance_matrix))),
                     tf.cast(n_clusters * (n_clusters - 1), tf.float32))


def regularization(x, y):
    """
    Compute the regularization term.
    The regularization term try to minimize the distance of cluster centers from the origin.
    :param x: Matrix of data points in it's rows.
    :param y: Column vector of labels.
    :return: The regularization.
    """
    # x data dimension
    d = x.shape[1]

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
    return tf.divide(tf.reduce_sum(tf.norm(c_centers, axis=1)), tf.cast(n_clusters, tf.float32))


def loss(x, y, var_hinge, dist_hinge, var_weight=1.0, dist_weight=1.0, regularization_weight=1.0):
    """
    Multi-instance-segmentation loss.
    A loss function that is meant to minimize intra-cluster distance, maximize inter-cluster distance and keep
    cluster centers close to the origin.
    :param x: Matrix of data points in it's rows.
    :param y: Column vector of labels.
    :param var_hinge: Points that closer to the centers of their clusters then this value, will have zero error.
    :param dist_hinge: Centers that are distant in more then this value, will have zero error.
    :param var_weight: Weight of the variance term.
    :param dist_weight: Weight of the distance term.
    :param regularization_weight: Wight of the regularization term.
    :return: MIS loss.
    """
    var_term = var_loss(x, y, hinge_value=var_hinge)
    distance_term = distance_loss(x, y, hinge_value=dist_hinge)
    regularization_term = regularization(x, y)

    return var_weight*var_term + dist_weight*distance_term + regularization_weight*regularization_term
