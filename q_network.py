import numpy as np
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.optimizers import Adam


def q_network(observation_space, action_space, LR=0.00001):
    """
    The neural network used for predicting Q values
    layer 1: fnn with 64 nodes
    layer 2: fnn with 32 nodes
    output layer 

    """
    model = Sequential()
    model.add(Dense(64, input_shape=(observation_space, ), activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(action_space, activation="linear"))
    model.compile(loss="mse", optimizer=Adam(lr=LR))
    return model
