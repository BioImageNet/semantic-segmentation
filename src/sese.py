
import argparse
import sys
import types
import os

import img.basic
import sese.model


if __name__ == '__main__':

    image_directory = "../image/BBBC007_v1_images/"
    outline_directory = "../image/BBBC007_v1_outlines/"

    sese.model.create_trainset(image_directory,outline_directory, 25)