import sys, os
sys.path.append(os.pardir)
from predict import PredictManager
from predict_test import TestPredictManager
from predict_many import PredictManagers

code = "086060"
time_size = 10
pt = TestPredictManager(code, time_size)
pt.fit(epochs=60, batch_size=10)
print(pt.evaluate())


