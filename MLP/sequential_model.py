import numpy as np
import os.path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

from data_processing_NN import generate_norm_data
from confusion_matrix import plot_confusion_matrix

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# getting data
X_train, y_train, X_test, y_test = generate_norm_data()

# Building the model
model = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax'),
])

print(model.summary())
opt = Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x=X_train, y=y_train, validation_split=0.1, batch_size=10, epochs=30, 
            shuffle=True, verbose=2)

predictions = model.predict(x=X_test, batch_size=10, verbose=0)

"""for i in predictions:
    print(i)"""

rounded_predictions = np.argmax(predictions, axis=-1) 

"""for i in rounded_predictions:
    print(i)
"""
cm_plot_labels = ['no side effects', 'had side effects']
plot_confusion_matrix(y=y_test,yp=rounded_predictions,classes=cm_plot_labels)

# Saving a Model
"""
        Checks first to see if file exists already
        If not, the model is saved to disk.
"""
if os.path.isfile('models/medical_trial_model.h5') is False:
   pass

