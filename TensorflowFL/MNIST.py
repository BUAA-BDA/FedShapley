import tensorflow.compat.v1 as tf
import tensorflow_federated as tff
import collections
import numpy

# tf.compat.v1.enable_v2_behavior()
# tf.compat.v1.enable_eager_execution()

NUM_EPOCHS = 10
BATCH_SIZE = 20
SHUFFLE_BUFFER = 500
NUM_CLIENTS = 3

MnistVariables = collections.namedtuple('MnistVariables', 'weights bias num_examples loss_sum accuracy_sum')

def preprocess(dataset):
    def element_fn(element):
        return collections.OrderedDict([
            ('x', tf.reshape(element['pixels'], [-1])),
            ('y', tf.reshape(element['label'], [1])),
        ])

    return dataset.repeat(NUM_EPOCHS).map(element_fn).shuffle(
        SHUFFLE_BUFFER).batch(BATCH_SIZE)

def preprocess4test(dataset):
    def element_fn(element):
        return collections.OrderedDict([
            ('x', tf.reshape(element['pixels'], [-1])),
            ('y', tf.reshape(element['label'], [1])),
        ])
    return dataset.map(element_fn)

def make_federated_data(client_data, client_ids):
    return [preprocess(client_data.create_tf_dataset_for_client(x))
            for x in client_ids]


'''def create_compiled_keras_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(
            10, activation=tf.nn.softmax, kernel_initializer='zeros', input_shape=(784,))])
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.02),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    return model


def model_fn():
    keras_model = create_compiled_keras_model()
    return tff.learning.from_compiled_keras_model(keras_model, sample_batch)'''


def create_mnist_variables():
    return MnistVariables(
        weights=tf.Variable(
            lambda: tf.zeros(dtype=tf.float32, shape=(784, 10)),
            name='weights',
            trainable=True),
        bias=tf.Variable(
            lambda: tf.zeros(dtype=tf.float32, shape=(10)),
            name='bias',
            trainable=True),
        num_examples=tf.Variable(0.0, name='num_examples', trainable=False),
        loss_sum=tf.Variable(0.0, name='loss_sum', trainable=False),
        accuracy_sum=tf.Variable(0.0, name='accuracy_sum', trainable=False))


def mnist_forward_pass(variables, batch):
    y = tf.nn.softmax(tf.matmul(batch['x'], variables.weights) + variables.bias)
    predictions = tf.cast(tf.argmax(y, 1), tf.int32)

    flat_labels = tf.reshape(batch['y'], [-1])
    loss = -tf.reduce_mean(tf.reduce_sum(tf.one_hot(flat_labels, 10) * tf.log(y), reduction_indices=[1]))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, flat_labels), tf.float32))

    num_examples = tf.cast(tf.size(batch['y']), tf.float32)

    variables.num_examples.assign_add(num_examples)
    variables.loss_sum.assign_add(loss * num_examples)
    variables.accuracy_sum.assign_add(accuracy * num_examples)

    return loss, predictions


def get_local_mnist_metrics(variables):
    return collections.OrderedDict([
        ('num_examples', variables.num_examples),
        ('loss', variables.loss_sum / variables.num_examples),
        ('accuracy', variables.accuracy_sum / variables.num_examples)
    ])


@tff.federated_computation
def aggregate_mnist_metrics_across_clients(metrics):
    return {
        'num_examples': tff.federated_sum(metrics.num_examples),
        'loss': tff.federated_mean(metrics.loss, metrics.num_examples),
        'accuracy': tff.federated_mean(metrics.accuracy, metrics.num_examples)
    }


class MnistModel(tff.learning.Model):

    def __init__(self):
        self._variables = create_mnist_variables()

    @property
    def trainable_variables(self):
        return [self._variables.weights, self._variables.bias]

    @property
    def non_trainable_variables(self):
        return []

    @property
    def local_variables(self):
        return [
            self._variables.num_examples, self._variables.loss_sum,
            self._variables.accuracy_sum
        ]

    @property
    def input_spec(self):
        return collections.OrderedDict([('x', tf.TensorSpec([None, 784],
                                                            tf.float32)),
                                        ('y', tf.TensorSpec([None, 1], tf.int32))])

    @tf.function
    def forward_pass(self, batch, training=True):
        del training
        loss, predictions = mnist_forward_pass(self._variables, batch)
        return tff.learning.BatchOutput(loss=loss, predictions=predictions)

    @tf.function
    def report_local_outputs(self):
        return get_local_mnist_metrics(self._variables)

    @property
    def federated_output_computation(self):
        return aggregate_mnist_metrics_across_clients


class MnistTrainableModel(MnistModel, tff.learning.TrainableModel):

    @tf.function
    def train_on_batch(self, batch):
        output = self.forward_pass(batch)
        optimizer = tf.train.GradientDescentOptimizer(0.02)
        optimizer.minimize(output.loss, var_list=self.trainable_variables)
        return output

def readTestImagesFromFile():
    ret = []
    f = open(os.path.join(os.path.dirname(__file__), "test_images1_.txt"), encoding="utf-8")
    lines = f.readlines()
    for line in lines:
        tem_ret = []
        p = line.replace("[", "").replace("]", "").replace("\n", "").split("\t")
        for i in p:
            if i != "":
                tem_ret.append(float(i))
        ret.append(tem_ret)
    return numpy.asarray(ret)

def readTestLabelsFromFile():
    ret = []
    f = open(os.path.join(os.path.dirname(__file__), "test_labels_.txt"), encoding="utf-8")
    lines = f.readlines()
    for line in lines:
        tem_ret = []
        p = line.replace("[", "").replace("]", "").replace("\n", "").split(" ")
        for i in p:
            if i!="":
                tem_ret.append(float(i))
        ret.append(tem_ret)
    return numpy.asarray(ret)


if __name__ == "__main__":
    emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
    example_dataset = emnist_train.create_tf_dataset_for_client(emnist_train.client_ids[0])

    pre = preprocess(example_dataset)
    sample_batch = tf.nest.map_structure(lambda x: x.numpy(), next(iter(pre)))

    sample_clients = emnist_train.client_ids[0:10]
    #all_clients = emnist_train.client_ids[0:len(emnist_train.client_ids)]

    federated_train_data = make_federated_data(emnist_train, sample_clients)

    #iterative_process = tff.learning.build_federated_averaging_process(model_fn)
    iterative_process = tff.learning.build_federated_averaging_process(MnistTrainableModel)
    state = iterative_process.initialize()
    new_state = None

    for round_num in range(0, 10):
        state, metrics = iterative_process.next(state, federated_train_data)
        new_state = state
        print('round {:2d}, metrics={}'.format(round_num, metrics))

    '''federated_test_data = make_federated_data(emnist_test, sample_clients)
    evaluation = tff.learning.build_federated_evaluation(MnistModel)
    train_metrics = evaluation(state.model, federated_test_data)
    print(train_metrics)'''


    test_images = readTestImagesFromFile()
    test_labels_onehot = readTestLabelsFromFile()
    #print(numpy.asarray(state.model.trainable.bias).shape)
    #print(numpy.asarray(state.model.trainable.weights).shape)
    m = numpy.dot(test_images,numpy.asarray(new_state.model.trainable.weights))
    #print(m.shape)
    test_result = m + numpy.asarray(new_state.model.trainable.bias)
    y = tf.nn.softmax(test_result)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.arg_max(test_labels_onehot, 1))
    #print(list(tf.argmax(y, 1).numpy()))
    #print(list(tf.arg_max(test_labels_onehot, 1).numpy()))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(accuracy.numpy())