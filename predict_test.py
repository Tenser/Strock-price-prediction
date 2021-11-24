from lstm import Model1
from data_manager import DataManager
import numpy as np

class TestPredictManager:
    def __init__(self, code, time_size):
        dm = DataManager(code)
        (self.x_train, self.t_train), (self.x_test, self.t_test), (self.x_max, self.x_min)= dm.getData2(time_size)
        
        self.model = Model1(time_size)
        self.model.compile(optimizer='adam',
              loss='mse')

    def reset(self, time_size):
        self.model = Model1(time_size)
        self.model.compile(optimizer='adam',
              loss='mse')

    def fit(self, epochs=20, batch_size=20):
        print("optimize start")
        self.model.fit(self.x_train, self.t_train, validation_data=(self.x_test, self.t_test), epochs=epochs, batch_size=batch_size)
        print("optimize finish")

    def evaluate(self):
        print("evaluate start")
        self.model.evaluate(self.x_test, self.t_test)
        y_test = self.model(self.x_test)
        y_test = y_test * (self.x_max - self.x_min) + self.x_min
        t_test = self.t_test * (self.x_max - self.x_min) + self.x_min
        t_test = t_test.reshape((t_test.shape[0], -1))
        res = np.mean(np.abs(y_test - t_test))
        return res
