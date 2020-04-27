from __future__ import absolute_import, division, print_function
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

# NUM_EXAMPLES_PER_USER = 1000
BATCH_SIZE = 100
RoundNum = 50
NUM_AGENT = 5
NOISE_STEP = 0.1
# rand_index = []
# rand_label = []


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
    index = 0
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

    l = tff.sequence_reduce(all_batches, initial_model, batch_fn)
    return l


@tff.federated_computation(MODEL_TYPE, LOCAL_DATA_TYPE)
def local_eval(model, all_batches):
    #
    return tff.sequence_sum(
        tff.sequence_map(
            tff.federated_computation(lambda b: batch_loss(model, b), BATCH_TYPE),
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
    l = tff.federated_map(
        local_train,
        [tff.federated_broadcast(model),
         tff.federated_broadcast(learning_rate),
         data])
    return l
    # return tff.federated_mean()


def readTestImagesFromFile(distr_same):
    ret = []
    if distr_same:
        f = open(os.path.join(os.path.dirname(__file__), "test_images1_.txt"), encoding="utf-8")
    else:
        f = open(os.path.join(os.path.dirname(__file__), "test_images1_.txt"), encoding="utf-8")
    lines = f.readlines()
    for line in lines:
        tem_ret = []
        p = line.replace("[", "").replace("]", "").replace("\n", "").split("\t")
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
            if i!="":
                tem_ret.append(float(i))
        ret.append(tem_ret)
    return np.asarray(ret)


def getParmsAndLearningRate(agent_no):
    f = open(os.path.join(os.path.dirname(__file__), "weights_" + str(agent_no) + ".txt"))
    content = f.read()
    g_ = content.split("***\n--------------------------------------------------")
    parm_local = []
    learning_rate_list = []
    for j in range(len(g_) - 1):
        line = g_[j].split("\n")
        if j == 0:
            weights_line = line[0:784]
            learning_rate_list.append(float(line[784].replace("*", "").replace("\n", "")))
        else:
            weights_line = line[1:785]
            learning_rate_list.append(float(line[785].replace("*", "").replace("\n", "")))
        valid_weights_line = []
        for l in weights_line:
            w_list = l.split("\t")
            w_list = w_list[0:len(w_list) - 1]
            w_list = [float(i) for i in w_list]
            valid_weights_line.append(w_list)
        parm_local.append(valid_weights_line)
    f.close()

    f = open(os.path.join(os.path.dirname(__file__), "bias_" + str(agent_no) + ".txt"))
    content = f.read()
    g_ = content.split("***\n--------------------------------------------------")
    bias_local = []
    for j in range(len(g_) - 1):
        line = g_[j].split("\n")
        if j == 0:
            weights_line = line[0]
        else:
            weights_line = line[1]
        b_list = weights_line.split("\t")
        b_list = b_list[0:len(b_list) - 1]
        b_list = [float(i) for i in b_list]
        bias_local.append(b_list)
    f.close()
    ret = {
        'weights': np.asarray(parm_local),
        'bias': np.asarray(bias_local),
        'learning_rate': np.asarray(learning_rate_list)
    }
    return ret


def train_with_gradient_and_valuation(agent_list, grad, bi, lr, distr_type):
    f_ini_p = open(os.path.join(os.path.dirname(__file__), "initial_model_parameters.txt"), "r")
    para_lines = f_ini_p.readlines()
    w_paras = para_lines[0].split("\t")
    w_paras = [float(i) for i in w_paras]
    b_paras = para_lines[1].split("\t")
    b_paras = [float(i) for i in b_paras]
    w_initial_g = np.asarray(w_paras, dtype=np.float32).reshape([784, 10])
    b_initial_g = np.asarray(b_paras, dtype=np.float32).reshape([10])
    f_ini_p.close()
    model_g = {
        'weights': w_initial_g,
        'bias': b_initial_g
    }
    for i in range(len(grad[0])):
        # i->迭代轮数
        gradient_w = np.zeros([784, 10], dtype=np.float32)
        gradient_b = np.zeros([10], dtype=np.float32)
        for j in agent_list:
            gradient_w = np.add(np.multiply(grad[j][i], 1/len(agent_list)), gradient_w)
            gradient_b = np.add(np.multiply(bi[j][i], 1/len(agent_list)), gradient_b)
        model_g['weights'] = np.subtract(model_g['weights'], np.multiply(lr[0][i], gradient_w))
        model_g['bias'] = np.subtract(model_g['bias'], np.multiply(lr[0][i], gradient_b))

    test_images = readTestImagesFromFile(False)
    test_labels_onehot = readTestLabelsFromFile(False)
    m = np.dot(test_images, np.asarray(model_g['weights']))
    test_result = m + np.asarray(model_g['bias'])
    y = tf.nn.softmax(test_result)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.arg_max(test_labels_onehot, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy.numpy()


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

    #data_num = np.asarray([5923,6742,5958,6131,5842])
    #agents_weights = np.divide(data_num, data_num.sum())

    for index in range(NUM_AGENT):
        f = open(os.path.join(os.path.dirname(__file__), "weights_"+str(index)+".txt"), "w")
        f.close()
        f = open(os.path.join(os.path.dirname(__file__), "bias_" + str(index) + ".txt"), "w")
        f.close()
    mnist_train, mnist_test = tf.keras.datasets.mnist.load_data()

    DISTRIBUTION_TYPE = "SAME"

    federated_train_data_divide = None
    federated_train_data = None
    if DISTRIBUTION_TYPE == "SAME":
        federated_train_data_divide = [get_data_for_federated_agents(mnist_train, d) for d in range(NUM_AGENT)]
        federated_train_data = federated_train_data_divide
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
    for round_num in range(RoundNum):
        local_models = federated_train(model, learning_rate, federated_train_data)
        print("learning rate: ", learning_rate)
        # print(local_models[0][0])#第0个agent的weights矩阵
        # print(local_models[0][1])#第0个agent的bias矩阵
        # print(len(local_models))
        for local_index in range(len(local_models)):
            f = open(os.path.join(os.path.dirname(__file__), "weights_"+str(local_index)+".txt"),"a",encoding="utf-8")
            for i in local_models[local_index][0]:
                line = ""
                arr = list(i)
                for j in arr:
                    line += (str(j)+"\t")
                print(line, file=f)
            print("***"+str(learning_rate)+"***",file=f)
            print("-"*50,file=f)
            f.close()
            f = open(os.path.join(os.path.dirname(__file__), "bias_" + str(local_index) + ".txt"), "a", encoding="utf-8")
            line = ""
            for i in local_models[local_index][1]:
                line += (str(i) + "\t")
            print(line, file=f)
            print("***" + str(learning_rate) + "***",file=f)
            print("-"*50,file=f)
            f.close()
        m_w = np.zeros([784, 10], dtype=np.float32)
        m_b = np.zeros([10], dtype=np.float32)
        for local_model_index in range(len(local_models)):
            m_w = np.add(np.multiply(local_models[local_model_index][0], 1/NUM_AGENT), m_w)
            m_b = np.add(np.multiply(local_models[local_model_index][1], 1/NUM_AGENT), m_b)
            model = {
                'weights': m_w,
                'bias': m_b
            }
        learning_rate = learning_rate * 0.9
        loss = federated_eval(model, federated_train_data)
        print('round {}, loss={}'.format(round_num, loss))
        print(time.time()-start_time)

    gradient_weights = []
    gradient_biases = []
    gradient_lrs = []
    for ij in range(NUM_AGENT):
        model_ = getParmsAndLearningRate(ij)
        gradient_weights_local = []
        gradient_biases_local = []
        learning_rate_local = []

        for i in range(len(model_['learning_rate'])):
            if i == 0:
                gradient_weight = np.divide(np.subtract(initial_model['weights'], model_['weights'][i]),
                                            model_['learning_rate'][i])
                gradient_bias = np.divide(np.subtract(initial_model['bias'], model_['bias'][i]),
                                          model_['learning_rate'][i])
            else:
                gradient_weight = np.divide(np.subtract(model_['weights'][i - 1], model_['weights'][i]),
                                            model_['learning_rate'][i])
                gradient_bias = np.divide(np.subtract(model_['bias'][i - 1], model_['bias'][i]),
                                          model_['learning_rate'][i])
            gradient_weights_local.append(gradient_weight)
            gradient_biases_local.append(gradient_bias)
            learning_rate_local.append(model_['learning_rate'][i])

        gradient_weights.append(gradient_weights_local)
        gradient_biases.append(gradient_biases_local)
        gradient_lrs.append(learning_rate_local)

    all_sets = PowerSetsBinary([i for i in range(NUM_AGENT)])
    group_shapley_value = []
    for s in all_sets:
        group_shapley_value.append(
            train_with_gradient_and_valuation(s, gradient_weights, gradient_biases, gradient_lrs, DISTRIBUTION_TYPE))
        print(str(s)+"\t"+str(group_shapley_value[len(group_shapley_value)-1]))

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
    print("end_time", time.time()-start_time)
