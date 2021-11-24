from predict import PredictManager

class PredictManagers:
    def __init__(self, codes, time_size, names=None):
        self.code_num = len(codes)
        self.pms = []
        if names is None:
            for i in range(self.code_num):
                self.pms.append(PredictManager(codes[i], time_size))
        else:
            for i in range(self.code_num):
                self.pms.append(PredictManager(codes[i], time_size, name=names[i]))

    def reset(self, time_size):
        for pm in self.pms:
            pm.reset(time_size)

    def run(self, epochs=20, batch_size=20):
        for pm in self.pms:
            pm.fit(epochs=epochs, batch_size=batch_size)
        for pm in self.pms:
            pm.predict_tomorrow()