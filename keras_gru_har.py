# GRU model for HAR with Keras
# https://machinelearningmastery.com/how-to-develop-rnn-models-for-human-activity-recognition-time-series-classification/
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import GRU
from keras.utils.vis_utils import plot_model
import tensorflow as tf
import matplotlib.pyplot as plt

from uci_data_loader import *  # load UCI dataset

import netron

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 0 : GPU
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 0, 50, 32
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(GRU(120, input_shape=(n_timesteps, n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(120, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    adam = tf.keras.optimizers.Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    # fit network
    history = model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)

    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return accuracy, history, model


# run an experiment
def run_experiment(data_dir=""):
    # load data
    trainX, trainy, testX, testy = load_dataset(data_dir)
    # experiment
    predict_acc, history, model = evaluate_model(trainX, trainy, testX, testy)

    # model information
    model.summary()  # parameters
    plot_model(model, to_file='./keras_gru_har_model.png', show_shapes=True)  # model structure
    model.save('keras_gru_har_model.h5')  # model
    netron.start('keras_gru_har_model.h5')  # model structure

    # summarize history for accuracy
    plt.title("test acc : " + str(np.round(predict_acc, 2)))
    plt.plot(history.history['loss'], color="blue")
    plt.ylabel('train loss')
    plt.twinx()
    plt.plot(history.history['accuracy'], color="orange")
    plt.ylabel('train accuracy')
    plt.xlabel('epoch')
    plt.savefig("keras_gru_har_model_epoch" + '.png')
    # plt.show()
    plt.close()
    plt.clf()

data_dir = r"G:\HAR\HAR"
if __name__ == '__main__':
    # run the experiment
    run_experiment(data_dir)

