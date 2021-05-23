import numpy as np


def split_data(raw_data, test_size=0.1, num_of_param=1):
    raw_data = np.array(raw_data)
    np.random.shuffle(raw_data)
    data = np.array([])
    for i in raw_data:
        data = np.append(data, np.array(i[0]))
    data = np.reshape(data, (-1, num_of_param))
    train_length = int(len(data) * (1 - test_size))
    train_data_in = data[:train_length]
    train_labels_in = raw_data[:train_length, 1]
    eval_data_in = data[train_length:]
    eval_labels_in = raw_data[train_length:, 1]
    return np.array(train_data_in).reshape(-1, num_of_param), np.array(train_labels_in, dtype=float), \
        np.array(eval_data_in).reshape(-1, num_of_param), np.array(eval_labels_in, dtype=float)
