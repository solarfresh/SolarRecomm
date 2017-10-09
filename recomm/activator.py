import tensorflow as tf


def relu(kwargs):
    """
    the python code in tensorflow has to interleave C++ code, which also uses indirect dependencies.
    Computes rectified linear: max(features, 0)

    :param features: A Tensor. Must be one of the following types: float32, float64, int32,
                     int64, uint8, int16, int8, uint16, half
    :param name: A name for the operation (optional)
    :return: A Tensor. Has the same type as features
    """
    return tf.nn.relu(**kwargs)