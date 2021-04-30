import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
import image_preprocessing as pre
import os
from PIL import Image

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




def quadimage(img):
    q1 = img.crop((0,0,width/2,height/2))
    q2 = img.crop((width/2,0,width,height/2))
    q3 = img.crop((0,height/2,width/2,height))
    q4 = img.crop((width/2,height/2,width,height))

    return q1,q2,q3,q4

def readall(path):
    for filename in os.listdir(path):
        img = Image.open(path+filename)
        #### do stuff ####
if __name__ == '__main__':
    main()
