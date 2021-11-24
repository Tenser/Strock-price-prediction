from predict import PredictManager
from lstm import Model1

def predict(code):
    pm = PredictManager(code, 10, name="test")
    pm.dataload()
    pm.fit()
    x, y = pm.predict_tomorrow()
    return y

pm = PredictManager("081000", 10, name="test")
#pm.dataload()
#pm.fit()
pm.loadWeights()
pm.predict_tomorrow()

