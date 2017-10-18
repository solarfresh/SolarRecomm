import numpy as np
import tensorflow as tf
from recomm.layer import neural_net
from recomm.objective import cross_entropy, l2_loss


class ClassifierBase(object):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        try:
            self.features_size = self.features.shape[1]
        except:
            self.features_size = 1
            self.features.shape = self.features.shape + (1,)
        try:
            self.labels_size = self.labels.shape[1]
        except:
            self.labels_size = 1
            self.labels.shape = self.labels.shape + (1,)
        self.sample_features = tf.placeholder(tf.float32,
                                              shape=[None, self.features_size],
                                              name="sample_features")
        self.sample_labels = tf.placeholder(tf.float32,
                                            shape=[None, self.labels_size],
                                            name="smaple_labels")
        self.estimated_labels = None
        self.objective = None
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.solver = None
        self.loss = []
        self.accuracy = None
        self.predicted_label = None

        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._sample_num = self.features.shape[0]
        self._features = self.features
        self._labels = self.labels
        self._sess = tf.Session()

    def activate_label(self):
        activated_labels = []
        for data in self.predicted_label[0]:
            activated_labels.append(np.where(data < data.max(), 0, 1))
        activated_labels = np.array(activated_labels)
        self.predicted_label = activated_labels
        return self

    def estimate(self,
                 batch_size=100,
                 iter_max=1e4,
                 learning_rate=1e-2,
                 init=True):
        if init:
            self._sess.run(tf.global_variables_initializer())
        for _ in range(int(iter_max)):
            sample_features, sample_labels = self._next_batch(batch_size)
            _, loss = self._sess.run([self.solver, self.objective],
                                     feed_dict={self.sample_features: sample_features,
                                                self.sample_labels: sample_labels,
                                                self.learning_rate: learning_rate})
            self.loss.append(loss)
        return self

    def get_accuracy(self, test_labels):
        error = np.linalg.norm((test_labels - self.predicted_label) / np.sqrt(2), axis=1).sum()
        error /= test_labels.shape[0]
        self.accuracy = 1.0 - error
        return self

    def optimize(self):
        self.solver = tf\
            .train.AdamOptimizer(learning_rate=self.learning_rate)\
            .minimize(self.objective)
        return self

    def predict(self, features):
        self.predicted_label = self._sess.run([self.estimated_labels],
                                              feed_dict={self.sample_features: features})
        return self

    def set_objective(self, method="entropy"):
        # It is meaningless if softmax is used because entroy will be 0 or 1 always and then
        # the objective will be 0 only.
        self.objective = {
            "entropy": cross_entropy(self.sample_labels, self.estimated_labels, activation="sigmoid"),
            "l2_loss": l2_loss(self.sample_labels, self.estimated_labels),
        }[method]
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


class ClassifierDNN(ClassifierBase):
    def __init__(self, *args, **kwargs):
        ClassifierBase.__init__(self, *args, **kwargs)
        self.hidden_neurons = []
        self.estimated_labels = None

    def build_network(self, hidden_layers=None, activate=None):
        #  Build hidden layers
        prev_neuron_size = self.features_size
        self.hidden_neurons.append(self.sample_features)
        for idx, layer_size in enumerate(hidden_layers):
            estimated_labels = neural_net(self.hidden_neurons[idx],
                                          layer_size,
                                          name="_estimated_neurons_{}".format(idx))
            if activate:
                self.hidden_neurons.append(tf.sigmoid(estimated_labels,
                                                      name="activated_neurons_{}".format(idx)))
            else:
                self.hidden_neurons.append(estimated_labels)
        #  Connect to labels
        estimated_labels = neural_net(self.hidden_neurons[len(hidden_layers)],
                                      self.labels_size,
                                      name="_estimated_labels")
        if activate:
            self.estimated_labels = tf.sigmoid(estimated_labels, name="activated_neurons")
        else:
            self.estimated_labels = estimated_labels
        return self


class ClassifierNN(ClassifierBase):
    def __init__(self, *args, **kwargs):
        ClassifierBase.__init__(self, *args, **kwargs)

    def build_network(self, activate=None):
        estimated_labels = neural_net(self.sample_features,
                                      self.labels_size,
                                      name="_estimated_labels")
        if activate:
            self.estimated_labels = tf.sigmoid(estimated_labels, name="activated_neurons")
        else:
            self.estimated_labels = estimated_labels
        return self
