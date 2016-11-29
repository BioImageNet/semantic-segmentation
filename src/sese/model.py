from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers import Dense,  Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from skimage import io, img_as_float
import os
import numpy as np
import img.basic
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


def create_trainsetimages(imagepath, groundtruthpath, window):
    filenames = [x for x in os.listdir(imagepath)]

    for file in filenames:
        if file[0] != '.':
            groundtruth = img.basic.loadfile(groundtruthpath + file)

            groundtruth = np.invert(groundtruth)
            groundtruth[groundtruth == 255] = 1

            image = img_as_float(img.basic.loadfile(imagepath + file))
            border = window / 2
            l = len(image)
            big_image = np.zeros([l + border * 2, l + border * 2])
            big_image[border:border + l, border:border + l] = image

            i = 0
            for x in xrange(len(image)):
                for y in xrange(len(image)):
                    x_ = img.basic.neighbors2(big_image, border + x, border + y, window)

                    i = i + 1
                    newfilename = file[:-4]
                    #ext = file[-3:]

                    ext = 'png'

                    prefix = '../data/train'

                    if (np.random.random() < .2):
                        prefix = '../data/test'



                    if (groundtruth[x, y] == 1):
                        img.basic.savefile(x_, prefix + "/cell/" + newfilename + "_" + str(x) + "_" + str(y) + "." + ext)
                    else:
                        img.basic.savefile(x_, prefix + "/nocell/" + newfilename + "_" + str(x) + "_" + str(y) + "." + ext)

                    print "% - " + str((i) / float(len(image) * 2))


def create_trainset(imagepath, groundtruthpath, window):
    filenames = [x for x in os.listdir(imagepath)]

    X = np.empty([320000 * len(filenames), 1, 25, 25])
    Y = np.empty([320000 * len(filenames), 2])

    for file in filenames:
        if file[0] != '.':

            print file

            groundtruth = img.basic.loadfile(groundtruthpath + file)
            groundtruth = np.invert(groundtruth)

            #groundtruth = binary_fill_holes(groundtruth, structure=np.ones((3, 3))).astype(int)

            groundtruth[groundtruth == 255] = 1


            #plt.imshow(groundtruth, cmap='Greys_r')
            #plt.show()
            #return

            image = img_as_float(img.basic.loadfile(imagepath + file))
            border = window / 2

            l = len(image)
            big_image= np.zeros([l + border * 2, l + border * 2])
            big_image[border:border + l, border:border + l] = image

            i = 0
            for x in xrange(len(image)):
                for y in xrange(len(image)):
                    x_ = img.basic.neighbors2(big_image, border + x, border + y , window)

                    X[i] = x_
                    newfilename = file[:-4]

                    #ext = file[-3:]
                    ext = 'png'

                    img.basic.savefile(x_, "../temp/"+newfilename+"_"+str(x)+"_"+str(y)+"."+ext)


                    y_ = int(groundtruth[x, y])

                    Y[i, 0] = 0
                    Y[i, 1] = 0

                    Y[i, y_] = 1

                    i = i + 1

                    print "% - " + str( (x+y) / float(len(image)*2))




    return X, Y


def create_model():
        model = Sequential()
        model.add(Convolution2D(32, 3, 3, input_shape=(25,25,3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(32, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(64, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
        return model

def runfromimage():
    seed = 7
    np.random.seed(seed)
    # evaluate model with standardized dataset

    model = create_model()

    train_datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        '../data/train',
        target_size=(25, 25),
        batch_size=32,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        '../data/test',
        target_size=(25, 25),
        batch_size=32,
        class_mode='binary')

    model.fit_generator(
        train_generator,
        samples_per_epoch=2000,
        nb_epoch=50,
        validation_data=validation_generator,
        nb_val_samples = 50)

def run(X, Y):
        seed = 7
        np.random.seed(seed)
        # evaluate model with standardized dataset

        model = create_model()

        print "Model: OK"

        estimator = KerasClassifier(build_fn=create_model, nb_epoch=1, batch_size=256, verbose=1)
        kfold = KFold(n=len(X), n_folds=5, shuffle=True, random_state=seed)

        results = cross_val_score(estimator, X, Y, cv=kfold)
        print("Results: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

        model.save_weights("model.hd5")

        return model

def testimage(model, imagepath, window):



    image = img_as_float(img.basic.loadfile(imagepath))

    result = np.zeros(image.shape)


    border = window / 2

    l = len(image)
    big_image = np.zeros([l + border * 2, l + border * 2])
    big_image[border:border + l, border:border + l] = image

    X = np.empty([1, 1, 25, 25])

    i = 0


    for x in xrange(len(image)):
        for y in xrange(len(image)):
            x_ = img.basic.neighbors2(big_image, border + x, border + y, window)
            X[0,0] = x_
            out = model.predict(X,batch_size=1, verbose=0)



            if out[0,0] > out[0,1]:
                result[x,y] = 0
            else:
                result[x, y] = 1


    #plt.imshow(result, cmap='Greys_r')
    #plt.show()
    img.basic.savefile(result, "output.tif")


