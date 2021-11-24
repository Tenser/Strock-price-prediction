import sys, os
sys.path.append(os.pardir)
from predict import PredictManager
from predict_test import TestPredictManager
from predict_many import PredictManagers

"""
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0
print(np.max(train_labels), np.min(train_labels))

model = Sequential([
    layers.Conv2D(8, 3, padding="same", activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D(2),
    layers.Conv2D(16, 3, padding="same", activation='relu'),
    layers.MaxPooling2D(2),
    layers.Conv2D(32, 3, padding="same", activation='relu'),
    layers.MaxPooling2D(2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=15, batch_size=50)

model.evaluate(test_images, test_labels)

code = "039490"
time_size = 10
pt = TestPredictManager(code, time_size)
print(pt.evaluate())

"""
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

time_size = 10
codes = ["081000", "220630", "086060", "161390"]
names = ["일진다이아", "해마로푸드서비스", "진바이오텍", "한국타이어앤테크놀로지"]
pms = PredictManagers(codes, time_size, names=names)
pms.run(epochs=50, batch_size=10)
