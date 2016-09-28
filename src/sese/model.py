
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.ndimage import binary_fill_holes

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
            image = img.basic.loadfile(imagepath + file)
            border = window / 2
            big_image = img.basic.addborder(image, border)

            for x in xrange(100):
                print str(x) + "/" + str(len(image))
                for y in xrange(len(image)):
                    X.append(img.basic.neighbors(big_image, border + x, border + y , window))
                    Y.append(groundtruth[x,y])

    print "size" +str(getsizeof(X))


def create_model():

        model = Sequential()
        model.add(Dense(60, input_dim=(400,400), init='normal', activation='relu'))
        model.add(Dense(1, init='normal', activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model


def run(X, Y):
        seed = 7
        np.random.seed(seed)
        # evaluate model with standardized dataset
        estimator = KerasClassifier(build_fn=create_model, nb_epoch=100, batch_size=5, verbose=0)
        kfold = StratifiedKFold(y=Y, n_folds=10, shuffle=True, random_state=seed)
        results = cross_val_score(estimator, X, Y, cv=kfold)
        print("Results: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))