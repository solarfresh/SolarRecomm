import tensorflow as tf


def factorization_machine(x, w, v):
    """
    Besides a linear (order-1) interactions
    among features, FM models pairwise (order-2) feature interactions
    as inner product of respective feature latent vectors

    :param x: a d-dimensional vector and its elements have m-fields involves
              pairs of users and items, which is a (m, d) matrix
    :param w: a d-dimensional vector which is the weight in the model
    :param v: a d-dimensional vector and its elements are k-dimensional vectors
              which are latency of features, which is a (k, d) matrix
    :return: m-dimensional vector
    """

    y_1 = tf.matmul(x, w)
    # uncertainty of (v * x) with the shape (k, m)
    y_2 = tf.pow(tf.matmul(v, x, transpose_b=True), 2) - tf.matmul(tf.pow(v, 2), tf.pow(x, 2), transpose_b=True)
    # Sum over k
    y_2 = tf.reduce_sum(y_2, 1)
    return y_1 + tf.multiply(0.5, y_2)