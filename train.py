import numpy as np
import tensorflow as tf

from model import Model
from utils import load_data

# for data loading
TEST_RATIO = 0.1
ITEMS_LIMIT_PER_LABEL = 60000

# for training
EPOCHS = 10
VERBOSE = 2
BATCH_SIZE = 256
VALIDATION_SPLIT = 0.1

def preprocess_data(X, img_size):
    X = X.reshape(X.shape[0], img_size, img_size, 1)
    X /= 255.0
    return X.astype(np.float32)

def convert_label(Y, num_labels):
    Y = tf.keras.utils.to_categorical(Y, num_labels)
    return Y

def train_model(data_dir, save_path, classes_file):
    X_train, Y_train, X_test, Y_test, label_names = load_data(data_dir, 
        test_ratio=TEST_RATIO, items_limit_per_label=ITEMS_LIMIT_PER_LABEL)

    img_size = int(np.sqrt(X_train.shape[1]))
    num_labels = len(label_names)

    X_train = preprocess_data(X_train, img_size)
    Y_train = convert_label(Y_train, num_labels)

    X_test = preprocess_data(X_test, img_size)
    Y_test = convert_label(Y_test, num_labels)

    model = Model(input_shape=X_train.shape[1:], output_labels_size=num_labels)
    model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, 
        validation_split=VALIDATION_SPLIT, verbose=VERBOSE)

    model.save(save_path)

    with open(classes_file, 'w') as f:
        for item in label_names:
            f.write('%s\n' % item)

    train_score = model.score(X_train, Y_train)
    test_score = model.score(X_test, Y_test)

    print('Train accuracy: %.2f%%' % (train_score[1] * 100))
    print('Test accuracy: %.2f%%' % (test_score[1] * 100))

    return model
