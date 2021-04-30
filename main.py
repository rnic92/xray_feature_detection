import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
import image_preprocessing as pre
from PIL import Image, ImageOps


# Main file which will contain a driver that will load the images and run the respective algorithm.


def main():
    new_img = Image.open("person100_bacteria_475.jpeg")
    #new_img.show()
    quadrants = pre.sixteenImage(new_img)
    histogram_list = []
    for i in range(16):
        np_quadrant = np.array(quadrants[i])
        filtered_quadrant = pre.gaussian_filter(np_quadrant, 5)
        quadrant_gradient, quadrant_angle = pre.sobel_filters(filtered_quadrant)
        equalized_quadrant = pre.histogram_equalize(quadrant_gradient)
        quadrant_histogram = np.histogram(equalized_quadrant)
        match = pre.match_histograms(np_quadrant, quadrant_histogram[1])
        histogram_list.append(match)

    q1 = [histogram_list[0], histogram_list[1], histogram_list[4], histogram_list[5]]
    q2 = [histogram_list[2], histogram_list[3], histogram_list[6], histogram_list[7]]
    q3 = [histogram_list[8], histogram_list[9], histogram_list[12], histogram_list[13]]
    q4 = [histogram_list[10], histogram_list[11], histogram_list[14], histogram_list[15]]
    final_img = [q1, q2, q3, q4]
    final_hist = np.block(final_img)
    #final_hist = final_hist.flatten()
    print(final_hist.shape)
    #plt.hist(final_hist)
    #plt.show()

    #matched = pre.match_histograms(new_img, final_hist)
    plt.imshow(final_hist, cmap="gray")
    plt.show()
    #plt.hist(matched)
    #plt.show()

if __name__ == "__main__":
    main()


"""image = plt.imread("person100_bacteria_475.jpeg")
    plt.imshow(image, cmap="gray")
    plt.show()
    eq_image = pre.histogram_equalize(image)
    plt.imshow(eq_image, cmap="gray")
    plt.show()
    gauss_filtered = pre.gaussian_filter(eq_image, 5)
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

    sobel_gradient, sobel_angle = pre.sobel_filters(gauss_filtered)
    equalized_gradient = pre.histogram_equalize(sobel_gradient)
    plt.imshow(equalized_gradient)
    plt.title('Equalized Histogram of the Gradient')

    plt.show()
    plt.hist(equalized_gradient)
    plt.title("Histogram of equalized gradient")
    plt.show()
"""