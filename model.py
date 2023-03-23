import keras
import tensorflow as tf

from keras import layers
from keras.models import load_model

class Model(object):
    def __init__(self, input_shape, output_labels_size):
        model = keras.Sequential()

        model.add(layers.Convolution2D(16, (3, 3), padding='same', input_shape=input_shape, activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Convolution2D(32, (3, 3), padding='same', activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Convolution2D(64, (3, 3), padding='same', activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(output_labels_size, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.train.AdamOptimizer(),
                      metrics=['top_k_categorical_accuracy'])

        self.model = model
        print(model.summary())

    def fit(self, X, Y, batch_size=256, epochs=5, validation_split=0.1, verbose=2):
        self.model.fit(x=X, y=Y, batch_size=batch_size, epochs=epochs, 
            validation_split=validation_split, verbose=verbose)

    def score(self, X, Y):
        return self.model.evaluate(X, Y, verbose=0)

    def predict_one(self, x):
        x = tf.expand_dims(x, axis=0)
        pred = self.model.predict(x)[0]
        return pred

    def predict(self, X):
        return self.model.predict(X)

    def save(self, file):
        self.model.save(file)

    def load(file):
        return load_model(file)
