import tensorflow as tf


def cross_entropy(sample_label, estimated_label, activation="softmax"):
    """
    To describe the entropy of the samples satisfying the binary distribution

    :param sample_label: labels obtaining from samples
    :param estimated_label: labels estimated from neural network
    :return: cross entropy
    """

    entropy = {
        "softmax": tf.nn.softmax_cross_entropy_with_logits,
        "sigmoid": tf.nn.sigmoid_cross_entropy_with_logits,
    }

    return tf.reduce_mean(entropy[activation](labels=sample_label,
                                              logits=estimated_label))