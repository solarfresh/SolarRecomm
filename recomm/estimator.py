import numpy as np
import tensorflow as tf
from recomm.layer import neural_net
from recomm.objective import cross_entropy


class EstimatorNN(object):
    def __init__(self, features, labels, dtype=tf.float32):
        self.features = tf.convert_to_tensor(features, dtype=dtype)
        self.labels = tf.convert_to_tensor(labels, dtype=dtype)
        self.sample_features = tf.placeholder(tf.float32,
                                              shape=self.features.shape.as_list(),
                                              name="sample_features")
        self.sample_labels = tf.placeholder(tf.float32,
                                            shape=self.labels.shape.as_list(),
                                            name="smaple_labels")
        self.estimated_labels = neural_net(self.features,
                                           self.labels,
                                           name="estimated_labels")
        self.loss = []
        self.objective = cross_entropy(self.sample_labels, self.estimated_labels)
        self.solver = None

        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._sample_num = self.features.shape[0].value
        self._features = self.features
        self._labels = self.labels

    def estimate(self, batch_size=100, iter_max=1e4):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for _ in range(int(iter_max)):
                sample_features, sample_labels = self._next_batch(batch_size)
                _, loss = sess.run([self.solver],
                                   feed_dict={self.sample_features: sample_features,
                                              self.sample_labels: sample_labels})
                self.loss.append(loss)
        return self

    def get_loss(self):
        return self.loss

    def optimize(self, learning_rate):
        self.solver = tf\
            .train.AdamOptimizer(learning_rate=learning_rate)\
            .minimize(self.objective)
        return self

    def _next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._sample_num)
            np.random.shuffle(perm0)
            self._features = self.features[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._sample_num:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._sample_num - start
            features_rest_part = self._features[start:self._sample_num]
            labels_rest_part = self._labels[start:self._sample_num]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            features_new_part = self._features[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((features_rest_part, features_new_part), axis=0),\
                   np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._features[start:end], self._labels[start:end]
