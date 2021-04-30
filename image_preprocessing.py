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
    img = img.resize((992, 992))
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


def differential_mask(array, target):
    difference = array - target
    mask = np.ma.less_equal(difference, -1)

    if np.all(mask):
        c = np.abs(difference).argmin()
        return c
    masked_difference = np.ma.masked_array(difference, mask)
    return masked_difference.argmin()


def match_histograms(original_image, specified_histogram):
    original_shape = original_image.shape
    original_image = original_image.ravel()

    s_values, bin_index, s_counts = np.unique(original_image, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(specified_histogram, return_counts=True)

    s = np.cumsum(s_counts).astype(np.float32)
    s /= s[-1]

    t = np.cumsum(t_counts).astype(np.float32)
    t /= t[-1]

    s_out = np.around(s*255)
    t_out = np.around(t*255)
    output = []
    for data in s_out[:]:
        output.append(differential_mask(t_out, data))
    output = np.array(output, dtype=np.uint8)
    return output[bin_index].reshape(original_shape)
