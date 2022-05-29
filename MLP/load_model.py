import tensorflow as tf
from tensorflow.keras.models import load_model, model_from_json, Sequential
from tensorflow.keras.layers import Dense

def load_ht5(filepath):
    new_model = load_model(filepath)
    return new_model


def load_json(filepath):
    with open(filepath) as f:
        json_string = f.read()
        print(json_string)
    new_model = model_from_json(json_string)
    return new_model

def load_weights(model, filepath):
    model.load_weights(filepath)
    return model

ht5_model = load_ht5('MLP/models/medical_trial_model.h5')
print(ht5_model.summary())
print(ht5_model.get_weights())
#print(ht5_model.optimizer())

json_model = load_json('MLP/models/medical_trial_json_model.txt')
print(json_model.summary())


model = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax'),
])

model = load_weights(model, 'MLP/models/medical_trial_weights.h5')
print(model.get_weights())