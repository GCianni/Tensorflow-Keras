import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import sys
sys.path.append( '.' )
from MLP.confusion_matrix import plot_confusion_matrix
from CNN.cnn_data_processing import data_prep


DATAPATH = r'C:/Users/55519/Desktop/Cursos/keras_deeplizard/CNN/data'
train_data, val_data, test_data = data_prep(DATAPATH)

vgg16_model = tf.keras.applications.vgg16.VGG16()
print(vgg16_model.summary())

def count_params(model):
    non_trainable_params = np.sum([np.prod(v.get_shape().as_list()) for v in model.non_trainable_weights])
    trainable_params = np.sum([np.prod(v.get_shape().as_list())for v in model.trainable_weights])
    return {'non_trainable_params':non_trainable_params, 'trainable_params':trainable_params}

params = count_params(vgg16_model)
print(params)

model = Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)
for layer in model.layers:
    layer.trainable = False

model.add(Dense(units=2, activation='softmax'))
print(model.summary())
params = count_params(model)
print(params)

opt = Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])

# Training Model
model.fit(x=train_data, validation_data=val_data, epochs=3, verbose=2)

# Predict
predictions = model.predict(x=test_data)

cm_class = ['cat','dog']
plot_confusion_matrix(y=test_data.classes, yp=np.argmax(predictions, axis=-1),classes=cm_class)