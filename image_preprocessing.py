import cv2
import numpy as np
import matplotlib.pyplot
from skimage import exposure, feature
from scipy import ndimage
from PIL import Image, ImageOps
# File that will contain image preprocessing code for before feature detection


def histogram_equalize(img):
    # Equalize the given image's histogram
    eq_image = exposure.equalize_hist(img)
    return eq_image


def gaussian_filter(img, sigma):
    # Apply a gaussian filter to the given image with the given sigma
    filtered_image = ndimage.gaussian_filter(img, sigma)
    return filtered_image


def canny_det(img, sigma):
    # Apply a Canny Edge Detector to the given image with the given sigma
    edges = feature.canny(img, sigma)
    return edges


def binary_threshold(img, thresh):
    # Apply a binary threshold of the given value to the given image
    ret, threshold_image = cv2.threshold(img, thresh, 1, cv2.THRESH_BINARY)
    return threshold_image


def gaussian_filter_from_project(image, m, n, sigma):
    m = int(m) // 2
    n = int(n) // 2
    x, y = np.mgrid[-m:m, -n:n]
    normal = 1 / (2.0 * np.pi * sigma ** 2)
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
    blurred_image = ndimage.filters.convolve(image, g)
    return blurred_image


def sobel_filters(img):
    sobel_horizontal = ndimage.filters.sobel(img, 0)
    sobel_vertical = ndimage.filters.sobel(img, 1)
    grad = np.hypot(sobel_horizontal, sobel_vertical)
    grad = grad/grad.max() * 255
    angle = np.arctan2(sobel_vertical, sobel_horizontal)
    return [grad.astype(np.uint8), angle.astype(np.uint8)]


def quadImage(img):
    width,height = img.size
    q1 = img.crop((0,0,width/2,height/2))
    q2 = img.crop((width/2,0,width,height/2))
    q3 = img.crop((0,height/2,width/2,height))
    q4 = img.crop((width/2,height/2,width,height))
    return q1,q2,q3,q4


def sixteenImage(img):
    """
    takes in an image and returns list of 16 sections
    1  2  ||  5  6
    3  4  ||  7  8
    ===============
    9 10  ||  13 14
    11 12 ||  15 16
    """
    width,height = img.size
    q1,q2,q3,q4 = quadImage(img)
    s1,s2,s3,s4 = quadImage(q1)
    s5,s6,s7,s8 = quadImage(q2)
    s9,s10,s11,s12 = quadImage(q3)
    s13,s14,s15,s16 = quadImage(q4)
    ret = [s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16]
    return ret
