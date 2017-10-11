import numpy as np
import tensorflow as tf
from recomm.layer import neural_net
from recomm.objective import cross_entropy


class ClassifierBase(object):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.features_size = self.features.shape[1]
        try:
            self.labels_size = self.labels.shape[1]
        except:
            self.labels_size = 1
            self.labels.shape = [self.labels.shape[0], 1]
        self.sample_features = tf.placeholder(tf.float32,
                                              shape=[None, self.features_size],
                                              name="sample_features")
        self.sample_labels = tf.placeholder(tf.float32,
                                            shape=[None, self.labels_size],
                                            name="smaple_labels")
        self.estimated_labels = None
        self.objective = None
        self.solver = None
        self.loss = []

        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._sample_num = self.features.shape[0]
        self._features = self.features
        self._labels = self.labels
        self._sess = tf.Session()

    def optimize(self, learning_rate):
        self.solver = tf\
            .train.AdamOptimizer(learning_rate=learning_rate)\
            .minimize(self.objective)
        return self

    def estimate(self, batch_size=100, iter_max=1e4):
        self._sess.run(tf.global_variables_initializer())
        for _ in range(int(iter_max)):
            sample_features, sample_labels = self._next_batch(batch_size)
            _, loss = self._sess.run([self.solver, self.objective],
                                     feed_dict={self.sample_features: sample_features,
                                                self.sample_labels: sample_labels})
            self.loss.append(loss)
        return self

    def predict(self, features):
        predicted_label = self._sess.run([self.estimated_labels],
                                         feed_dict={self.sample_features: features})
        return predicted_label

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


class ClassifierDNN(ClassifierBase):
    def __init__(self, *args, **kwargs):
        ClassifierBase.__init__(self, *args, **kwargs)
        self.hidden_neurons = []
        self.estimated_labels = None

    def build_network(self, hidden_layers=None):
        #  Build hidden layers
        prev_neuron_size = self.features_size
        self.hidden_neurons.append(self.sample_features)
        for idx, layer_size in enumerate(hidden_layers):
            self.hidden_neurons.append(neural_net(self.hidden_neurons[idx],
                                                  layer_size,
                                                  name="_estimated_neurons_{}".format(idx)))
        #  Connect to labels
        self.estimated_labels = neural_net(self.hidden_neurons[len(hidden_layers)],
                                           self.labels_size,
                                           name="_estimated_labels")
        return self


class ClassifierNN(ClassifierBase):
    def __init__(self, *args, **kwargs):
        ClassifierBase.__init__(self, *args, **kwargs)

    def build_network(self):
        self.estimated_labels = neural_net(self.sample_features,
                                           self.labels_size,
                                           name="_estimated_labels")
        return self

    def set_objective(self):
        # It is meaningless if softmax is used because entroy will be 0 or 1 always and then
        # the objective will be 0 only.
        self.objective = cross_entropy(self.sample_labels, self.estimated_labels, activation="sigmoid")
        return self
