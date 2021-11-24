import sys, os
sys.path.append(os.pardir)
import tensorflow as tf
from lstm import Model1
from data_manager import DataManager

class PredictManager:
    def __init__(self, code, time_size, name=None):
        self.checkpoint_path = "weights/" + name + ".ckpt"
        checkpoint_dir = os.path.dirname(self.checkpoint_path)
        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
        self.name = name
        self.time_size = time_size
        self.dm = DataManager(code)
        self.x_train = None
        self.t_train = None
        #self.x_today = None
        #self.x_max = None
        #self.x_min = None
        
        self.model = Model1(time_size)
        self.model.compile(optimizer='adam',
              loss='mse')

    def reset(self, time_size):
        self.model = Model1(time_size)
        self.model.compile(optimizer='adam',
              loss='mse')
        self.time_size = time_size

    def dataload(self):
        self.x_train, self.t_train = self.dm.getData3(self.time_size)

    def fit(self, epochs=20, batch_size=10):
        print("optimize start")
        self.model.fit(self.x_train, self.t_train, epochs=epochs, batch_size=batch_size, callbacks=[self.cp_callback])
        print("optimize finish")

    def loadWeights(self):
        self.model.load_weights(self.checkpoint_path)

    def predict_tomorrow(self):
        x_today = self.dm.getToday(self.time_size)
        x = x_today[0,x_today.shape[1]-1,0]
        y = self.model(x_today)
        if self.name is None:
            print("{x} -> {y}".format(x=x, y=y))
        else:
            print("{name}: {x} -> {y}".format(name=self.name, x=x, y=y))
        return x, y
    

