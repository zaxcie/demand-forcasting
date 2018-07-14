from keras.models import Sequential
from keras.layers import LSTM, Dense

from demfor.utils.metrics import keras_SMAPE


def get_LSTM(X):
    model = Sequential()
    model.add(LSTM(50, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(90))

    return model
