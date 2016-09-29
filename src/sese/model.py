
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.ndimage import binary_fill_holes
from sklearn.cross_validation import KFold

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

    X = np.empty([320000 * len(filenames), 1, 25, 25])
    Y = np.empty([320000 * len(filenames), 2])

    for file in filenames:
        if file[0] != '.':

            print file

            groundtruth = img.basic.loadfile(groundtruthpath + file)
            groundtruth = np.invert(groundtruth)
            groundtruth = binary_fill_holes(groundtruth, structure=np.ones((3, 3))).astype(int)

            image = img_as_float(img.basic.loadfile(imagepath + file))
            border = window / 2

            l = len(image)
            big_image= np.zeros([l + border * 2, l + border * 2])
            big_image[border:border + l, border:border + l] = image

            i = 0
            for x in xrange(len(image)):
                for y in xrange(len(image)):
                    x_ = img.basic.neighbors2(big_image, border + x, border + y , window)

                    X[i, 0] = x_
                    y_ = int(groundtruth[x, y])

                    Y[i, 0] = 0
                    Y[i, 1] = 0

                    Y[i, y_] = 1

                    i = i + 1




    return X, Y


def create_model():

        model = Sequential()

        model.add(Convolution2D(32, 3, 3, border_mode='same',
                                input_shape=(1, 25,25)))
        model.add(Activation('relu'))
        model.add(Convolution2D(16, 3, 3))
        model.add(Activation('relu'))

        model.add(Flatten())
        model.add(Dense(20))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2))
        model.add(Activation('softmax'))



        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])


        return model


def run(X, Y):
        seed = 7
        np.random.seed(seed)
        # evaluate model with standardized dataset

        model = create_model()

        print "Model: OK"

        estimator = KerasClassifier(build_fn=create_model, nb_epoch=10, batch_size=256, verbose=1)
        kfold = KFold(n=len(X), n_folds=5, shuffle=True, random_state=seed)

        results = cross_val_score(estimator, X, Y, cv=kfold)
        print("Results: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

        model.save_weights("model.hd5")
