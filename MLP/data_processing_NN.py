# Data Processing for Neural Network Training
# Create venv cmd: pythom -m venv <venv_name>
# To activate the vevn: project_folder\<venv_name>\Scripts\activate.bat

import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler


def generate_norm_data ():

    train_samples = []
    train_labels = []
    test_samples = []
    test_labels = []

    """ Exemple Data:
        - An Experiemental drug was tested on individuals from ages 13 to 100 in clinical trial
        - The trial had 2100 participants. Half ware nder 65 Y.O, half ware 65 year or older
        - Around 95% of patientes 65 or older experienced side effects
        - Around 95% of patients under 65 experienced no side effects
    """

    for i in range(50):
        # The ~5% of younger individuals who did experience side effects
        random_younger = randint(13,64)
        train_samples.append(random_younger)
        train_labels.append(1)

        # The ~5% of older individuals who did not experience side effects
        random_older = randint(65,100)
        train_samples.append(random_older)
        train_labels.append(0)

    for i in range(1000):
        # The ~95% of younger individuals who did not experience side effects
        random_younger = randint(13,64)
        train_samples.append(random_younger)
        train_labels.append(0)
        # The ~95% of younger individuals who did experience side effects
        random_older = randint(65,100)
        train_samples.append(random_older)
        train_labels.append(1)
    
    # Test Set
    for i in range(10):
        # The ~5% of younger individuals who did experience side effects
        random_younger = randint(13,64)
        test_samples.append(random_younger)
        test_labels.append(1)

        # The ~5% of older individuals who did not experience side effects
        random_older = randint(65,100)
        test_samples.append(random_older)
        test_labels.append(0)

    for i in range(200):
        # The ~95% of younger individuals who did not experience side effects
        random_younger = randint(13,64)
        test_samples.append(random_younger)
        test_labels.append(0)
        # The ~95% of younger individuals who did experience side effects
        random_older = randint(65,100)
        test_samples.append(random_older)
        test_labels.append(1)

    """
    for i in train_samples:
        print(i)
    """
    train_labels = np.array(train_labels)
    train_samples = np.array(train_samples)
    train_samples, train_labels = shuffle(train_samples, train_labels)

    test_labels = np.array(test_labels)
    test_samples = np.array(test_samples)
    test_samples, test_labels = shuffle(test_samples, test_labels)


    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))
    scaled_test_samples = scaler.fit_transform(test_samples.reshape(-1,1))
    """
    for i in scaled_train_samples:
        print(i)]
    """
    return scaled_train_samples, train_labels, scaled_test_samples, test_labels
