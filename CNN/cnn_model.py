import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import itertools
import warnings
from cnn_data_processing import data_prep

import sys
sys.path.append( '.' )
from MLP.confusion_matrix import plot_confusion_matrix

warnings.simplefilter(action='ignore', category=FutureWarning)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)


DATAPATH = r'C:/Users/55519/Desktop/Cursos/keras_deeplizard/CNN/data'
train_data, val_data, test_data = data_prep(DATAPATH)

# CNN MODEL
model = Sequential([
    Conv2D(filters=32, kernel_size = (3, 3), activation='relu',
            padding='same', input_shape=(224,224, 3)),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=64, kernel_size = (3, 3), activation='relu',
            padding='same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(units=2, activation='softmax'),
])

print(model.summary())

opt = Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])

# Training Model
model.fit(x=train_data, validation_data=val_data, epochs=10, verbose=2)

# Predict
predictions = model.predict(x=test_data)

cm_class = ['cat','dog']
plot_confusion_matrix(y=test_data.classes, yp=np.argmax(predictions, axis=-1),classes=cm_class)