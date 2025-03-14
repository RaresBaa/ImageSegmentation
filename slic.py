import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import random


def slic_segm(image_name, num_superpixels=300, compactness=10, show=True):
    image = np.array(Image.open(image_name))
    #convert the image to the CIE Lab color space
    cie_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    #create the SLIC superpixel object
    slic = cv2.ximgproc.createSuperpixelSLIC(cie_image, cv2.ximgproc.SLICO, num_superpixels, compactness)
    #run the segmentation algorithm
    slic.iterate()
    #get the labels and mask for each superpixel
    labels = slic.getLabels()
    mask = slic.getLabelContourMask()
    #convert the mask to a 3 channel image
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    #overlay the mask on the original image and add random colors to each superpixel
    num_segments = np.max(labels) + 1
    segment_colors = {}
    for label in range(num_segments):
        segment_colors[label] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    colored_img = np.zeros_like(image)
    for label in range(num_segments):
        #color the superpixel with the corresponding color
        colored_img[labels == label] = segment_colors[label]
    #blend the original image with the colored segments
    out_img = cv2.addWeighted(image, 0.5, colored_img, 0.5, 0)

    if show:
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(out_img)
        plt.title('Segmented Image')
        plt.axis('off')
        plt.show()

    # return the segmented image along with the name for it and which parameters were used as a string
    params = 'SLIC: Num Superpixels = ' + str(num_superpixels) + ', Compactness = ' + str(compactness)
    return out_img, params
