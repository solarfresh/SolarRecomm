import tensorflow as tf


def factorization_machine(x, w, v):
    # This function should be simplized the input as an outer product between features.
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


def deep_neural_net(features, hidden_units=[150, 150, 150]):
    """

    :param features: a d-dimensional vectors and its elements have m-fields involves
                     pairs of users and items, which is a (m, d) matrix
    :param hidden_units: a list with number of nuerons of hidden layers
    :return:
    """

    return True


def neural_net(features, label_size, dtype=tf.float32, name=None):
    """
    A basic connect used to construct a neural network

    :param features: a d-dimensional vectors
    :param label_size: a k-dimenional vectors
    :param dtype: to clarify the type of values inside the matrix
    :param name: to index the neuron
    :return: a k-dimenional vectors
    """

    #  To guarantee the size is integer
    feature_size = int(features.shape[1].value)
    label_size = int(label_size)
    w = tf.Variable(tf.random_normal([feature_size, label_size]), name="neural_net_weight" + name)
    b = tf.Variable(tf.random_normal([label_size]), name="neural_net_bias" + name)
    return tf.add(tf.matmul(features, w), b)