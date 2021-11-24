import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model

class Model1(Model):
    def __init__(self, time_size=10):
        super().__init__()
        #self.lstm1 = layers.LSTM(100, return_sequences=True, return_state=False, input_shape=(time_size, 5))
        self.lstm2 = layers.LSTM(100, activation="relu")
        self.dropout1 = layers.Dropout(0.2)
        #self.dense1 = layers.Dense(100, activation="relu")
        self.dense2 = layers.Dense(1)
    
    def call(self, x):
        #x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.dropout1(x)
        #x = self.dense1(x)
        return self.dense2(x)