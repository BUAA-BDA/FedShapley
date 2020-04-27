import random
import numpy as np
import tensorflow as tf
import json
import os


#MIX_RATIO =
BATCH_SIZE = 100
NUM_AGENT = 5

def get_data_for_digit_mix(source, digit):
    output_sequence = []
    all_samples = []
    all_samples = [i for i, d in enumerate(source[1]) if d == digit]
    for i in range(0, len(all_samples)):
        output_sequence.append({
            'x': np.array(source[0][all_samples[i]].flatten() / 255.0,
                          dtype=np.float32),
            'y': np.array(source[1][all_samples[i]], dtype=np.int32)})
    return output_sequence

def dumps2file():
    mnist_train, mnist_test = tf.keras.datasets.mnist.load_data()
    federated_train_data_digit = [get_data_for_digit_mix(mnist_train, d) for d in range(10)]
    federated_train_data = federated_train_data_digit[0:NUM_AGENT]

    mix_list_images = []
    mix_list_labels = []
    for i in range(len(federated_train_data)):
        tem_img = []
        tem_lbl = []
        for j in range(len(federated_train_data[i])):
            tem_img.append(federated_train_data[i][j]['x'].tolist())
            tem_lbl.append(np.eye(10)[federated_train_data[i][j]['y']].tolist())
        mix_list_images.append(tem_img)
        mix_list_labels.append(tem_lbl)
    # print(mix_list_labels[0][0])
    for i in range(NUM_AGENT):
        for j in range(len(mix_list_images[i])):
            if random.random() > MIX_RATIO:
                exchange_dest = random.choice([k for k in range(NUM_AGENT) if k != i])
                container_img = mix_list_images[i][j]
                container_lbl = mix_list_labels[i][j]
                exchange_j = int(j / 2)
                mix_list_images[i][j] = mix_list_images[exchange_dest][exchange_j]
                mix_list_labels[i][j] = mix_list_labels[exchange_dest][exchange_j]
                mix_list_images[exchange_dest][exchange_j] = container_img
                mix_list_labels[exchange_dest][exchange_j] = container_lbl

    ret = {}
    for i in range(NUM_AGENT):
        output_sequence = []
        for j in range(0, len(mix_list_labels[i]), BATCH_SIZE):
            batch_images = mix_list_images[i][j: j + BATCH_SIZE]
            batch_labels = mix_list_labels[i][j: j + BATCH_SIZE]
            output_sequence.append({
                'x': batch_images,
                'y': batch_labels})
        ret[i] = output_sequence

    with open(os.path.join(os.path.dirname(__file__), "mix.json"), "w") as f:
        json.dump(ret, f)
    print("加载入文件完成...")

if __name__ == "__main__":
    data_num = np.asarray([5842, 5842, 5842, 5842, 5842])
    for i,d in enumerate(data_num):
        ff = open(os.path.join(os.path.dirname(__file__), "exchange_"+str(i)+".txt"),"w")
        ff.close()
    for i,d in enumerate(data_num):
        ff = open(os.path.join(os.path.dirname(__file__), "exchange_"+str(i)+".txt"),"a")
        for j in range(d):
            if random.random()>MIX_RATIO:
                print(j,file=ff)
        ff.close()