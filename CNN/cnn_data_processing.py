
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

"""physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)"""


def plotImages (images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def data_prep(datapath):
    os.chdir(datapath)
    if os.path.isdir('train/dog') is False:
        os.makedirs('train/dog')
        os.makedirs('train/cat')
        os.makedirs('valid/dog')
        os.makedirs('valid/cat')
        os.makedirs('test/dog')
        os.makedirs('test/cat')
    
        #downsampling images
        for c in random.sample(glob.glob('cat*'), 500):
            shutil.move(c,'train/cat')
        for c in random.sample(glob.glob('dog*'), 500):
            shutil.move(c,'train/dog')
        for c in random.sample(glob.glob('cat*'), 100):
            shutil.move(c,'valid/cat')
        for c in random.sample(glob.glob('dog*'), 100):
            shutil.move(c,'valid/dog')
        for c in random.sample(glob.glob('cat*'), 50):
            shutil.move(c,'test/cat')
        for c in random.sample(glob.glob('dog*'), 50):
            shutil.move(c,'test/dog')
    
    train_path = datapath + '/train'
    valid_path = datapath + '/valid'
    test_path = datapath + '/test'

    #train_batches = ImageDataGenerator(preprocessing_function=)
    train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=train_path, target_size=(224,224), classes=['cat', 'dog'], batch_size=10)
    valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=valid_path, target_size=(224,224), classes=['cat', 'dog'], batch_size=10)
    test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=test_path, target_size=(224,224), classes=['cat', 'dog'], batch_size=10, shuffle=False)

    imgs, labels = next(train_batches)
    plotImages(imgs)
    print(labels)
    return train_batches, valid_batches, test_batches

