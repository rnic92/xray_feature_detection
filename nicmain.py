import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
import image_preprocessing as pre
import os
from PIL import Image, ImageOps
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk, diamond
from skimage.util import img_as_ubyte
from sklearn.preprocessing import normalize

def naive_image_smooth(image):
    output = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output[i, j] = image[i, j]
    return output

def plot_comparison(original, filtered, filter_name):

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(filter_name)
    ax2.axis('off')


def main():
    new_img = Image.open("person100_bacteria_475.jpeg")
    quadrants = pre.sixteenImage(new_img)
    histogram_list = []
    image_list = []
    oimage_list = []
    selem = diamond(6)
    for i in range(16):
        np_quadrant = np.array(quadrants[i])
        filtered_quadrant = pre.gaussian_filter(np_quadrant, 5)
        quadrant_gradient, quadrant_angle = pre.sobel_filters(filtered_quadrant)
        equalized_quadrant = pre.histogram_equalize(quadrant_gradient)
        quadrant_histogram = np.histogram(equalized_quadrant)
        match = pre.match_histograms(np_quadrant, quadrant_histogram[1])
        bt = black_tophat(match, selem)
        selem = disk(3)
        bt = dilation(bt, selem)
        image_list.append(bt)
        oimage_list.append(np_quadrant)
    q1 = [image_list[0], image_list[1], image_list[4], image_list[5]]
    q2 = [image_list[2], image_list[3], image_list[6], image_list[7]]
    q3 = [image_list[8], image_list[9], image_list[12], image_list[13]]
    q4 = [image_list[10], image_list[11], image_list[14], image_list[15]]
    r1 = [oimage_list[0], oimage_list[1], oimage_list[4], oimage_list[5]]
    r2 = [oimage_list[2], oimage_list[3], oimage_list[6], oimage_list[7]]
    r3 = [oimage_list[8], oimage_list[9], oimage_list[12], oimage_list[13]]
    r4 = [oimage_list[10], oimage_list[11], oimage_list[14], oimage_list[15]]
    final_img = [q1, q2, q3, q4]
    final_img = np.block(final_img)
    final_img = naive_image_smooth(final_img)
    final_img = pre.threshold(final_img,1.2)
    orig_img = [r1,r2,r3,r4]
    orig_img = np.block(orig_img)
    orig_img = naive_image_smooth(orig_img)
    plot_comparison(orig_img, final_img, 'black tophat with dilation')
    plt.show()


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


def readall(path):
    for filename in os.listdir(path):
        img = Image.open(path+filename)
        #### do stuff ####
if __name__ == '__main__':
    main()
