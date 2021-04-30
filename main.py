import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
import image_preprocessing as pre
from PIL import Image, ImageOps


# Main file which will contain a driver that will load the images and run the respective algorithm.
def naive_image_smooth(image):
    output = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output[i, j] = image[i, j]
    return output


def main():
    new_img = Image.open("person100_bacteria_475.jpeg")
    quadrants = pre.sixteenImage(new_img)
    histogram_list = []
    image_list = []
    for i in range(16):
        np_quadrant = np.array(quadrants[i])
        filtered_quadrant = pre.gaussian_filter(np_quadrant, 5)
        quadrant_gradient, quadrant_angle = pre.sobel_filters(filtered_quadrant)
        equalized_quadrant = pre.histogram_equalize(quadrant_gradient)
        quadrant_histogram = np.histogram(equalized_quadrant)
        match = pre.match_histograms(np_quadrant, quadrant_histogram[1])
        histogram_list.append(quadrant_histogram[1])
        image_list.append(match)

    fig, axs = plt.subplots(4, 4)
    axs[0, 0].hist(histogram_list[0])
    axs[0, 0].set_title("")
    axs[0, 1].hist(histogram_list[1])
    axs[0, 1].set_title("")
    axs[0, 2].hist(histogram_list[4])
    axs[0, 2].set_title("")
    axs[0, 3].hist(histogram_list[5])
    axs[0, 3].set_title("")
    axs[1, 0].hist(histogram_list[2])
    axs[1, 0].set_title("")
    axs[1, 1].hist(histogram_list[3])
    axs[1, 1].set_title("")
    axs[1, 2].hist(histogram_list[6])
    axs[1, 2].set_title("")
    axs[1, 3].hist(histogram_list[7])
    axs[1, 3].set_title("")
    axs[2, 0].hist(histogram_list[8])
    axs[2, 0].set_title("")
    axs[2, 1].hist(histogram_list[9])
    axs[2, 1].set_title("")
    axs[2, 2].hist(histogram_list[12])
    axs[2, 2].set_title("")
    axs[2, 3].hist(histogram_list[13])
    axs[2, 3].set_title("")
    axs[3, 0].hist(histogram_list[10])
    axs[3, 0].set_title("")
    axs[3, 1].hist(histogram_list[11])
    axs[3, 1].set_title("")
    axs[3, 2].hist(histogram_list[14])
    axs[3, 2].set_title("")
    axs[3, 3].hist(histogram_list[15])
    axs[3, 3].set_title("")
    fig.tight_layout()
    plt.show()

    plt.stairs(histogram_list[0])
    plt.show()
    print(histogram_list[0])

    q1 = [image_list[0], image_list[1], image_list[4], image_list[5]]
    q2 = [image_list[2], image_list[3], image_list[6], image_list[7]]
    q3 = [image_list[8], image_list[9], image_list[12], image_list[13]]
    q4 = [image_list[10], image_list[11], image_list[14], image_list[15]]
    final_img = [q1, q2, q3, q4]
    final_img = np.block(final_img)
    final_img = naive_image_smooth(final_img)
    plt.imshow(final_img, cmap="gray")
    plt.show()


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