import cv2
import numpy as np
import matplotlib.pyplot
from skimage import exposure, feature
from scipy import ndimage
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
