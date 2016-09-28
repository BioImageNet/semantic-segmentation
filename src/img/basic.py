from skimage import io, img_as_float
import glob
from math import sqrt, radians
import math
from numpy.core.multiarray import ndarray
from scipy.ndimage import distance_transform_edt
from scipy.stats import entropy
from skimage import io, img_as_float
from skimage.exposure import rescale_intensity, histogram
from skimage.feature import peak_local_max
from skimage.measure import regionprops,  structural_similarity
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.morphology import remove_small_objects, medial_axis, watershed, binary_opening, disk
from skimage.transform import estimate_transform

from scipy.ndimage import binary_fill_holes

import numpy


def neighbors(arr,x,y,n=25):
    arr=np.roll(np.roll(arr,shift=-x+1,axis=0),shift=-y+1,axis=1)
    return arr[:n,:n]

def neighbors2(arr,x,y,n=25):
    h = n / 2
    return arr[x-(h):x-(h)+n,y-(h):y-(h)+n]


def loadfile(filepath):
    seg = io.imread(filepath, as_grey = True)
    return seg


def addborder(image, size):
    l = len(image)
    big = np.zeros([l+size*2, l+size*2])
    big[size:size+l,size:size+l] = image
    return big


def test():

    p = np.ones([3,3])
    print p

    p = addborder(p, 2)

    print p

    print neighbors2(p,3,3,5)















