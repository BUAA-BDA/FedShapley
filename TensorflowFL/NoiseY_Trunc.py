import tensorflow_federated as tff
import tensorflow.compat.v1 as tf
import numpy as np
import time
import random

import os

# tf.compat.v1.enable_v2_behavior()
# tf.compat.v1.enable_eager_execution()

BATCH_SIZE = 100
NUM_AGENT = 5
NOISE_STEP = 0.05
rand_index = []
rand_label = []

# with open(os.path.join(os.path.dirname(__file__), "random_index.txt"), "w") as randomIndex:
#     randomIndex.write("")
# with open(os.path.join(os.path.dirname(__file__), "random_label.txt"), "w") as randomLabel:
#     randomLabel.write("")




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
    # add noise 0%-40%
    ratio = NOISE_STEP *(num)
    sum_agent = int(len(all_samples))

    #TODO: write random list into file and change randint to a number

    noiseList = rand_index[num][0:int(ratio*sum_agent)]
    noiseLabel = rand_label[num][0:int(ratio*sum_agent)]
    # noiseList = random.sample(range(0, sum_agent), int(ratio*sum_agent))
    # noiseLabel = []
    index = 0
    for i in noiseList:
        # noiseHere = random.randint(1, 9)
        # noiseLabel.append(noiseHere)
        noiseHere = noiseLabel[index]
        index = index + 1
        output_sequence[int(i/BATCH_SIZE)]['y'][i % BATCH_SIZE] = (
            output_sequence[int(i/BATCH_SIZE)]['y'][i % BATCH_SIZE]+noiseHere) % 10

    # with open(os.path.join(os.path.dirname(__file__), "random_index.txt"), "a") as randomIndex:
    #     randomIndex.write(str(noiseList)+"\n")
    #     # lines = randomIndex.read()
    #     # for line in lines:
    #     #     rand_index.append(eval(line))
    # with open(os.path.join(os.path.dirname(__file__), "random_label.txt"), "a") as randomLabel:
    #     randomLabel.write(str(noiseLabel)+"\n")
    #     # lines = randomLabel.read()
    #     # for line in lines:
    #     #     rand_label.append(eval(line))
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

def ModelValuation(train_set_index, distri_type):
    federated_train_data = []
    for index in train_set_index:
        federated_train_data.append(federated_train_data_divide[index])

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
    for round_num in range(50):
        local_models = federated_train(model, learning_rate, federated_train_data)
        # print(len(local_models))
        print("learning rate: ", learning_rate)

        m_w = np.zeros([784, 10], dtype=np.float32)
        m_b = np.zeros([10], dtype=np.float32)

        for local_model_index in range(len(local_models)):
            m_w = np.add(np.multiply(local_models[local_model_index][0], 1 / len(train_set_index)), m_w)
            m_b = np.add(np.multiply(local_models[local_model_index][1], 1 / len(train_set_index)), m_b)
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

    test_images = readTestImagesFromFile(False)
    test_labels_onehot = readTestLabelsFromFile(False)
    m = np.dot(test_images, np.asarray(model['weights']))
    test_result = m + np.asarray(model['bias'])
    y = tf.nn.softmax(test_result)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.arg_max(test_labels_onehot, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy.numpy()

def getFiFromFile():
    f = open(os.path.join(os.path.dirname(__file__), "fi_set.txt"))
    lines = f.readlines()
    ret = []
    for line in lines:
        s = line.replace("\n", "").split("\t")
        ret.append(s)
    f.close()
    return ret

def appendFiInFile(item):
    f = open(os.path.join(os.path.dirname(__file__), "fi_set.txt"), "a", encoding="utf-8")
    s = ""
    for i in item:
        s += str(i) + "\t"
    s=s[0:len(s)-1]
    print(s,file=f)
    f.close()

def clearFi():
    f = open(os.path.join(os.path.dirname(__file__), "fi_set.txt"), "w", encoding="utf-8")
    f.close()

def getPermutationFromFile():
    f = open(os.path.join(os.path.dirname(__file__), "per_set.txt"))
    lines = f.readlines()
    ret = []
    for line in lines:
        s = line.replace("\n", "").split("\t")
        ss = [int(i) for i in s]
        ret.append(ss)
    f.close()
    return ret

def appendPermutationInFile(item):
    f = open(os.path.join(os.path.dirname(__file__), "per_set.txt"), "a", encoding="utf-8")
    s = ""
    for i in item:
        s += str(i) + "\t"
    s=s[0:len(s)-1]
    print(s,file=f)
    f.close()

def clearPermutation():
    f = open(os.path.join(os.path.dirname(__file__), "per_set.txt"), "w", encoding="utf-8")
    f.close()

def ifNotConverge(fi_list, convergence_criteria, convergence_bias):
    if len(fi_list)<=convergence_bias:
        return True
    fi_list_last_index = len(fi_list)-1
    for j in range(fi_list_last_index - 5, fi_list_last_index+1):
        sum = 0.0
        if j - convergence_bias < 0:
            continue
        for i in range(len(fi_list[j])):
            if float(fi_list[j][i]) == 0.0:
                continue
            sum += abs( float(fi_list[j][i]) - float(fi_list[j - convergence_bias][i])) / abs(float(fi_list[j][i]))
        print("______sum=", sum)
        if sum/NUM_AGENT >= convergence_criteria:
            return True
    return False


def readCachedPerformance():
    f = open(os.path.join(os.path.dirname(__file__), "cached_truncated_performance.txt"), "r")
    lines = f.readlines()
    ret = []
    for line in lines:
        agent_part = line.split("\t")[0]
        agent_part = agent_part.split(" ")
        if '' in agent_part and len(agent_part) == 1:
            agent_part = []
        per = float(line.split("\t")[1].replace("\n", ""))
        ret.append((agent_part, per))
    f.close()
    return ret

def writeCachedPerformance(agent_set, performance):
    as_per = readCachedPerformance()
    as_per.append((agent_set, performance))
    f = open(os.path.join(os.path.dirname(__file__), "cached_truncated_performance.txt"), "w")
    f.close()
    f = open(os.path.join(os.path.dirname(__file__), "cached_truncated_performance.txt"), "a")
    
    for i in as_per:
        ss = ""
        for j in i[0]:
            ss += (str(j) + " ")
        ss = ss[0:-1]
        ss+=("\t"+str(i[1]))
        print(ss,file=f)
    f.close()

def getCachedPerformance(agent_set):
    as_per = readCachedPerformance()
    agent_s = [str(i) for i in agent_set]
    for as_p in as_per:
        if set(agent_s) == set(as_p[0]):
            return as_p[1]
    return -1


if __name__ == "__main__":

    # input the index&&noised label

    with open(os.path.join(os.path.dirname(__file__), "random_index.txt"), "r") as randomIndex:
        lines = randomIndex.readlines()
        for line in lines:
            rand_index.append(eval(line))
    with open(os.path.join(os.path.dirname(__file__), "random_label.txt"), "r") as randomLabel:
        lines = randomLabel.readlines()
        for line in lines:
            rand_label.append(eval(line))

    # initialize parms
    start_time = time.time()
    f = open(os.path.join(os.path.dirname(__file__), "fi_set.txt"), "w", encoding="utf-8")
    f.close()
    f = open(os.path.join(os.path.dirname(__file__), "per_set.txt"), "w", encoding="utf-8")
    f.close()
    f = open(os.path.join(os.path.dirname(__file__), "cached_truncated_performance.txt"), "w")
    f.close()
    #data_num = np.asarray([5923, 6742, 5958, 6131, 5842])
    #agents_weights = np.divide(data_num, data_num.sum())

    mnist_train, mnist_test = tf.keras.datasets.mnist.load_data()

    DISTRIBUTION_TYPE = "SAME"

    federated_train_data_divide = None
    if DISTRIBUTION_TYPE == "SAME":
        federated_train_data_divide = [get_data_for_federated_agents(mnist_train, d) for d in range(NUM_AGENT)]

    t = 0
    fi = [0 for i in range(NUM_AGENT)]
    appendFiInFile(fi)
    #fi_set = getFiFromFile()
    v = [0 for i in range(NUM_AGENT+1)]
    initial_permutation = [i for i in range(NUM_AGENT)]
    random_permutation = [i for i in range(NUM_AGENT)]
    appendPermutationInFile(initial_permutation)
    all_in_performance = ModelValuation(initial_permutation, DISTRIBUTION_TYPE)
    empty_performance = ModelValuation([], DISTRIBUTION_TYPE)
    print("all in performance:", all_in_performance)
    print("empty performace:", empty_performance)
    ##pi_0 的初始化？？
    performance_tolerence = 0.01
    convergence_criteria = 0.10
    convergence_bias = int( 2.5 * NUM_AGENT)

    while ifNotConverge(getFiFromFile(), convergence_criteria, convergence_bias):
        t = t + 1
        print("time: ", time.time()-start_time)
        print("permutation-round ", t)
        np.random.shuffle(random_permutation)
        appendPermutationInFile(random_permutation)
        v[0] = empty_performance
        for j in range(1, NUM_AGENT+1):
            l = []
            for i in range(0, j):
                l.append(random_permutation[i])
            if abs( all_in_performance - v[j-1]) < performance_tolerence:
                v[j] = v[j-1]
            else:
                if getCachedPerformance(l) == -1:
                    v[j] = ModelValuation(l, DISTRIBUTION_TYPE)
                    writeCachedPerformance(l, v[j])
                else:
                    v[j] = getCachedPerformance(l)
            sampled_per = getPermutationFromFile()
            fi[random_permutation[j - 1]] = (t - 1) / t * (float(getFiFromFile()[t - 1][random_permutation[j - 1]])) + (1 / t) * (
                        v[j] - v[j - 1])
        appendFiInFile(fi)
        print(fi)

    print("end time: ", time.time()-start_time)
    for f in fi:
        print(f)
    clearFi()
    clearPermutation()