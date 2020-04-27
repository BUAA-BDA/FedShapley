import tensorflow_federated as tff
import tensorflow.compat.v1 as tf
import numpy as np
import time
import random
np.random.seed(42)
import os

from scipy.special import comb, perm


# tf.compat.v1.enable_v2_behavior()
# tf.compat.v1.enable_eager_execution()

NUM_EXAMPLES_PER_USER = 1000
ROUND_NUM = 50
BATCH_SIZE = 100
NUM_AGENT = 5
NOISE_STEP = 0.1
rand_index = []
rand_label = []


def checkRange(x):
    for i in range(len(x)):
        if x[i] < 0:
            x[i] = 0

        if x[i] > 1:
            x[i] = 1
    return x


def get_data_for_digit(source, digit):
    output_sequence = []
    all_samples = [i for i, d in enumerate(source[1]) if d == digit]
    for i in range(0, len(all_samples), BATCH_SIZE):
        batch_samples = all_samples[i:i + BATCH_SIZE]
        output_sequence.append({
            'x': np.array([source[0][i].flatten() / 255.0 for i in batch_samples],
                          dtype=np.float32),
            'y': np.array([source[1][i] for i in batch_samples], dtype=np.int32)})
    return output_sequence


def get_data_for_digit_test(source, digit):
    output_sequence = []
    all_samples = [i for i, d in enumerate(source[1]) if d == digit]
    for i in range(0, len(all_samples)):
        output_sequence.append({
            'x': np.array(source[0][all_samples[i]].flatten() / 255.0,
                          dtype=np.float32),
            'y': np.array(source[1][all_samples[i]], dtype=np.int32)})
    return output_sequence


def get_data_for_federated_agents(source, num):
    output_sequence = []
    Samples = []
    for digit in range(0, 10):
        samples = [i for i, d in enumerate(source[1]) if d == digit]
        samples = samples[0:5421]
        Samples.append(samples)

    all_samples = []
    for sample in Samples:
        for sample_index in range(int(num * (len(sample) / NUM_AGENT)), int((num + 1) * (len(sample) / NUM_AGENT))):
            all_samples.append(sample[sample_index])

    # all_samples = [i for i in range(int(num*(len(source[1])/NUM_AGENT)), int((num+1)*(len(source[1])/NUM_AGENT)))]

    for i in range(0, len(all_samples), BATCH_SIZE):
        batch_samples = all_samples[i:i + BATCH_SIZE]
        output_sequence.append({
            'x': np.array([source[0][i].flatten() / 255.0 for i in batch_samples],
                          dtype=np.float32),
            'y': np.array([source[1][i] for i in batch_samples], dtype=np.int32)})

    # add noise 0x-0.2x
    ratio = num * 0.05
    sum_agent = int(len(all_samples))
    for i in range(0, sum_agent):
        noiseHere = ratio * np.random.randn(28*28)
        output_sequence[int(i/BATCH_SIZE)]['x'][i % BATCH_SIZE] = checkRange(np.add(
            output_sequence[int(i/BATCH_SIZE)]['x'][i % BATCH_SIZE], noiseHere))

    return output_sequence


BATCH_TYPE = tff.NamedTupleType([
    ('x', tff.TensorType(tf.float32, [None, 784])),
    ('y', tff.TensorType(tf.int32, [None]))])

MODEL_TYPE = tff.NamedTupleType([
    ('weights', tff.TensorType(tf.float32, [784, 10])),
    ('bias', tff.TensorType(tf.float32, [10]))])


@tff.tf_computation(MODEL_TYPE, BATCH_TYPE)
def batch_loss(model, batch):
    predicted_y = tf.nn.softmax(tf.matmul(batch.x, model.weights) + model.bias)
    return -tf.reduce_mean(tf.reduce_sum(
        tf.one_hot(batch.y, 10) * tf.log(predicted_y), axis=[1]))


@tff.tf_computation(MODEL_TYPE, BATCH_TYPE, tf.float32)
def batch_train(initial_model, batch, learning_rate):
    # Define a group of model variables and set them to `initial_model`.
    model_vars = tff.utils.create_variables('v', MODEL_TYPE)
    init_model = tff.utils.assign(model_vars, initial_model)

    # Perform one step of gradient descent using loss from `batch_loss`.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    with tf.control_dependencies([init_model]):
        train_model = optimizer.minimize(batch_loss(model_vars, batch))

    # Return the model vars after performing this gradient descent step.
    with tf.control_dependencies([train_model]):
        return tff.utils.identity(model_vars)


LOCAL_DATA_TYPE = tff.SequenceType(BATCH_TYPE)


@tff.federated_computation(MODEL_TYPE, tf.float32, LOCAL_DATA_TYPE)
def local_train(initial_model, learning_rate, all_batches):
    # Mapping function to apply to each batch.
    @tff.federated_computation(MODEL_TYPE, BATCH_TYPE)
    def batch_fn(model, batch):
        return batch_train(model, batch, learning_rate)

    return tff.sequence_reduce(all_batches, initial_model, batch_fn)


@tff.federated_computation(MODEL_TYPE, LOCAL_DATA_TYPE)
def local_eval(model, all_batches):
    return tff.sequence_sum(
        tff.sequence_map(
            tff.federated_computation(
                lambda b: batch_loss(model, b), BATCH_TYPE),
            all_batches))


SERVER_MODEL_TYPE = tff.FederatedType(MODEL_TYPE, tff.SERVER, all_equal=True)
CLIENT_DATA_TYPE = tff.FederatedType(LOCAL_DATA_TYPE, tff.CLIENTS)


@tff.federated_computation(SERVER_MODEL_TYPE, CLIENT_DATA_TYPE)
def federated_eval(model, data):
    return tff.federated_mean(
        tff.federated_map(local_eval, [tff.federated_broadcast(model), data]))


SERVER_FLOAT_TYPE = tff.FederatedType(tf.float32, tff.SERVER, all_equal=True)


@tff.federated_computation(
    SERVER_MODEL_TYPE, SERVER_FLOAT_TYPE, CLIENT_DATA_TYPE)
def federated_train(model, learning_rate, data):
    return tff.federated_map(
        local_train,
        [tff.federated_broadcast(model),
         tff.federated_broadcast(learning_rate),
         data])


def readTestImagesFromFile(distr_same):
    ret = []
    if distr_same:
        f = open(os.path.join(os.path.dirname(__file__), "test_images1_.txt"), encoding="utf-8")
    else:
        f = open(os.path.join(os.path.dirname(__file__), "test_images1_.txt"), encoding="utf-8")
    lines = f.readlines()
    for line in lines:
        tem_ret = []
        p = line.replace("[", "").replace("]", "").replace(
            "\n", "").split("\t")
        for i in p:
            if i != "":
                tem_ret.append(float(i))
        ret.append(tem_ret)
    return np.asarray(ret)


def readTestLabelsFromFile(distr_same):
    ret = []
    if distr_same:
        f = open(os.path.join(os.path.dirname(__file__), "test_labels_.txt"), encoding="utf-8")
    else:
        f = open(os.path.join(os.path.dirname(__file__), "test_labels_.txt"), encoding="utf-8")
    lines = f.readlines()
    for line in lines:
        tem_ret = []
        p = line.replace("[", "").replace("]", "").replace("\n", "").split(" ")
        for i in p:
            if i != "":
                tem_ret.append(float(i))
        ret.append(tem_ret)
    return np.asarray(ret)


def remove_list_indexed(removed_ele, original_l, ll):
    new_original_l = []
    for i in original_l:
        new_original_l.append(i)
    for i in new_original_l:
        if i == removed_ele:
            new_original_l.remove(i)
    for i in range(len(ll)):
        if set(ll[i]) == set(new_original_l):
            return i
    return -1


def shapley_list_indexed(original_l, ll):
    for i in range(len(ll)):
        if set(ll[i]) == set(original_l):
            return i
    return -1


def PowerSetsBinary(items):
    N = len(items)
    set_all = []
    for i in range(2 ** N):
        combo = []
        for j in range(N):
            if (i >> j) % 2 == 1:
                combo.append(items[j])
        set_all.append(combo)
    return set_all


if __name__ == "__main__":

    start_time = time.time()
    mnist_train, mnist_test = tf.keras.datasets.mnist.load_data()

    # data_num = np.asarray([5923, 6742, 5958, 6131, 5842])
    # agents_weights = np.divide(data_num, data_num.sum())
    # agents_weights = 0.2032

    DISTRIBUTION_TYPE = "SAME"

    federated_train_data_divide = None
    test_images = None
    test_labels_onehot = None
    if DISTRIBUTION_TYPE == "SAME":
        federated_train_data_divide = [get_data_for_federated_agents(
            mnist_train, d) for d in range(NUM_AGENT)]
        test_images = readTestImagesFromFile(False)
        test_labels_onehot = readTestLabelsFromFile(False)

    all_sets = PowerSetsBinary([i for i in range(NUM_AGENT)])
    group_shapley_value = []
    for ss in all_sets:
        federated_train_data = []
        for item in ss:
            federated_train_data.append(federated_train_data_divide[item])

        f_ini_p = open(os.path.join(os.path.dirname(__file__), "initial_model_parameters.txt"), "r")
        para_lines = f_ini_p.readlines()
        w_paras = para_lines[0].split("\t")
        w_paras = [float(i) for i in w_paras]
        b_paras = para_lines[1].split("\t")
        b_paras = [float(i) for i in b_paras]
        w_initial = np.asarray(w_paras, dtype=np.float32).reshape([784, 10])
        b_initial = np.asarray(b_paras, dtype=np.float32).reshape([10])
        f_ini_p.close()

        initial_model = {
            'weights': w_initial,
            'bias': b_initial
        }

        model = initial_model
        learning_rate = 0.1
        for round_num in range(ROUND_NUM):
            local_models = federated_train(
                model, learning_rate, federated_train_data)
            # print(len(local_models))
            print("learning rate: ", learning_rate)

            m_w = np.zeros([784, 10], dtype=np.float32)
            m_b = np.zeros([10], dtype=np.float32)

            for local_model_index in range(len(local_models)):
                m_w = np.add(np.multiply(
                    local_models[local_model_index][0], 1/len(ss)), m_w)
                m_b = np.add(np.multiply(
                    local_models[local_model_index][1], 1/len(ss)), m_b)
                model = {
                    'weights': m_w,
                    'bias': m_b
                }
            learning_rate = learning_rate * 0.9
            loss = federated_eval(model, federated_train_data)
            print('round {}, loss={}'.format(round_num, loss))
            print(time.time() - start_time)
            '''model = federated_train(model, learning_rate, federated_train_data)
            learning_rate = learning_rate * 0.9
            loss = federated_eval(model, federated_train_data)
            print('round {}, loss={}'.format(round_num, loss))'''

        m = np.dot(test_images, np.asarray(model['weights']))
        test_result = m + np.asarray(model['bias'])
        y = tf.nn.softmax(test_result)
        correct_prediction = tf.equal(
            tf.argmax(y, 1), tf.arg_max(test_labels_onehot, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        group_shapley_value.append(accuracy.numpy())
        print("combination finished ", time.time() - start_time)
        print(str(ss) + "\t" +
              str(group_shapley_value[len(group_shapley_value) - 1]))

    agent_shapley = []
    for index in range(NUM_AGENT):
        shapley = 0.0
        for j in all_sets:
            if index in j:
                remove_list_index = remove_list_indexed(index, j, all_sets)
                if remove_list_index != -1:
                    shapley += (group_shapley_value[shapley_list_indexed(j, all_sets)] - group_shapley_value[
                        remove_list_index]) / (comb(NUM_AGENT - 1, len(all_sets[remove_list_index])))
        agent_shapley.append(shapley)
    for ag_s in agent_shapley:
        print(ag_s)
    print("end_time", time.time() - start_time)
