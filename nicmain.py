import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
import image_preprocessing as pre
import os
from PIL import Image, ImageOps

def main():
    #read in all files
    width= 1000
    height = 1000
    path = "/Users/nic/Desktop/Classnotes/DIP/finalProj/chest_xray/test/PNEUMONIA/"
    #  ## reading each file
    #
    filename = "person3_virus_15.jpeg"
    img = Image.open(path+filename)
    img = img.resize((width, height))
    q1,q2,q3,q4 = quadImage(img)
    q1arry = np.array(q1)
    print(q1arry)

    sixteen = sixteenimage(img)
    histeq = ImageOps.equalize(q1)
    histeq.show()
    grad1, angle = pre.sobel_filters(histeq)
    plt.imshow(grad1)
    plt.show()
    plt.imshow(angle)
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
