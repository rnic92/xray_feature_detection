import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
import image_preprocessing as pre


# Main file which will contain a driver that will load the images and run the respective algorithm.


def main():
    image = plt.imread("person100_bacteria_475.jpeg")
    plt.imshow(image, cmap="gray")
    plt.show()
    eq_image = pre.histogram_equalize(image)
    plt.imshow(eq_image, cmap="gray")
    plt.show()
    gauss_filtered = pre.gaussian_filter(eq_image, 3)
    plt.imshow(gauss_filtered, cmap="gray")
    plt.show()
    threshed = pre.binary_threshold(eq_image, .5)
    plt.imshow(threshed, cmap="gray")
    plt.show()
    edges = pre.canny_det(threshed, 5)
    plt.imshow(edges, cmap="gray")
    plt.show()
    selected = edges+eq_image  # Just for fun to see what it looks like
    selected = pre.histogram_equalize(selected)
    plt.imshow(selected, cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()
