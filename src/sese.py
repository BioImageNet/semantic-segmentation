
import argparse
import sys
import types
import os

import img.basic
import sese.model
import sys, getopt




def main(argv):
    image_directory = "../image/test_img/"
    outline_directory = "../image/BBBC007_v1_outlines/"


    try:
        opts, args = getopt.getopt(argv, "gti:o:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print 'test.py -i <inputfile> -o <outputfile>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-g':
            print 'Generate images...'
            sese.model.create_trainsetimages(image_directory, outline_directory, 25)
            sys.exit()
        if opt == '-t':
            print 'Train images...'
            sese.model.runfromimage()
            sys.exit()



    #model = sese.model.run(X, Y)

    #model = sese.model.create_model()

    #model.load_weights("model.hd5")

    #sese.model.testimage(model,"../image/20P1_POS0005_D_1UL.tif",25)


if __name__ == "__main__":
    main(sys.argv[1:])



