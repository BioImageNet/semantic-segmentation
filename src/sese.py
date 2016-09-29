
import argparse
import sys
import types
import os

import img.basic
import sese.model


if __name__ == '__main__':

    image_directory = "../image/test_img/"
    outline_directory = "../image/BBBC007_v1_outlines/"

    X, Y = sese.model.create_trainset(image_directory,outline_directory, 25)

    sese.model.run(X, Y)

