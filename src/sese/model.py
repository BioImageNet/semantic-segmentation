
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.ndimage import binary_fill_holes

from keras.models import Model

from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

from keras.optimizers import SGD

import keras.callbacks

from sys import getsizeof


from skimage import io, img_as_float


import os

import numpy as np

import img.basic

def create_trainset(imagepath, groundtruthpath, window):


    filenames = [x for x in os.listdir(imagepath)]

    X = []
    Y = []

    for file in filenames[:2]:
        if file[0] != '.':

            groundtruth = img.basic.loadfile(groundtruthpath + file)
            groundtruth = np.invert(groundtruth)
            groundtruth = binary_fill_holes(groundtruth, structure=np.ones((3, 3))).astype(int)

            image = img_as_float(img.basic.loadfile(imagepath + file))
            border = window / 2

            l = len(image)
            big_image= np.zeros([l + border * 2, l + border * 2])
            big_image[border:border + l, border:border + l] = image

            for x in xrange(len(image)):
                for y in xrange(len(image)):
                    X.append(img.basic.neighbors2(big_image, border + x, border + y , window).reshape([25*25]))
                    Y.append(int(groundtruth[x,y]))


    return X, Y


def create_model():

        model = Sequential()


        model.add(Dense(10, input_dim=25))
        model.add(Dense(1))



        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='binary_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])


        return model


def run(X, Y):
        seed = 7
        np.random.seed(seed)
        # evaluate model with standardized dataset

        model = create_model()

        print "model OK"
        batch_size = 32
        nb_epoch = 1

        X = X[:1000]

        print "sample size: " + str(len(X))

        model.fit(X, Y,
                  batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  validation_data=(X, Y),
                  shuffle=True, verbose=2)
