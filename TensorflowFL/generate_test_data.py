import numpy as np
import tensorflow as tf
import os


NUM_AGENT = 5

def get_data_for_digit_test(source, digit):
    output_sequence = []
    all_samples = [i for i, d in enumerate(source[1]) if d == digit]
    #all_samples = all_samples[0:892]
    for i in range(0, len(all_samples)):
        output_sequence.append({
            'x': np.array(source[0][all_samples[i]].flatten() / 255.0,
                          dtype=np.float32),
            'y': np.array(source[1][all_samples[i]], dtype=np.int32)})
    return output_sequence

if __name__ == "__main__":
    mnist_train, mnist_test = tf.keras.datasets.mnist.load_data()
    federated_test_data_divide = [get_data_for_digit_test(mnist_test, d) for d in range(10)]
    federated_test_data = federated_test_data_divide

    f = open(os.path.join(os.path.dirname(__file__), "test_images1_.txt"), "w")
    f.close()
    f = open(os.path.join(os.path.dirname(__file__), "test_labels_.txt"), "w")
    f.close()

    f = open(os.path.join(os.path.dirname(__file__), "test_images1_.txt"), "a")
    ff = open(os.path.join(os.path.dirname(__file__), "test_labels_.txt"), "a")
    for i in range(len(federated_test_data)):
        for j in range(len(federated_test_data[i])):
            string = ""
            for k in range(len(federated_test_data[i][j]['x'])):
                string += (str(federated_test_data[i][j]['x'][k]) + "\t")
            string = "[" + string + "]"
            print(string, file=f)
            print(np.eye(10)[federated_test_data[i][j]['y']], file=ff)
    f.close()
    ff.close()