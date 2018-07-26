import os
import glob
import urllib
import numpy as np


def download_data(data_dir, labels_file, base_url):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    with open(labels_file, 'r') as f:
        labels = f.readlines()

    labels = [label.replace('\n', '').replace(' ', '_') for label in labels]

    for label in labels:
        lbl_url = label.replace('_', '%20')
        url = base_url + lbl_url + '.npy'
        urllib.request.urlretrieve(url, os.path.join(data_dir, label + '.npy'))
        print('Done: %s' % url)


def load_data(data_dir, test_ratio=0.2, items_limit_per_label=None):
    all_files = glob.glob(os.path.join(data_dir, '*.npy'))

    # initialize variables
    X = np.empty([0, 784])
    Y = np.empty([0])
    label_names = []

    # load each data file
    for idx, file in enumerate(all_files):
        data = np.load(file)
        if items_limit_per_label is not None:
            np.random.shuffle(data)
            data = data[:items_limit_per_label]
        labels = np.full(data.shape[0], idx)

        X = np.concatenate((X, data), axis=0)
        Y = np.append(Y, labels)

        label_name, ext = os.path.splitext(os.path.basename(file))
        label_names.append(label_name)

    # let gc work
    data = None
    labels = None

    # shuffle the dataset
    random_idx = np.random.permutation(Y.shape[0])
    X = X[random_idx]
    Y = Y[random_idx]

    # separate into training and testing dataset
    test_size = int(Y.shape[0] * test_ratio)

    X_test = X[:test_size]
    Y_test = Y[:test_size]

    X_train = X[test_size:]
    Y_train = Y[test_size:]

    return X_train, Y_train, X_test, Y_test, label_names

