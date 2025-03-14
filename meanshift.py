import numpy as np
from PIL import Image
from skimage import color
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt


def meanshift_segm(image_name, Quantile=0.2, N_samples=500, show=True):
    image = Image.open(image_name)
    #convert the image from rgb to CIE Lab color space
    cie_image = color.rgb2lab(image)
    #reshape the image to a 2D array where each row represents a pixel
    #and each column represents the color in CIE Lab space
    reshaped_image = np.reshape(cie_image, [-1, 3])
    #estimate the bandwidth for the given samples and quantile
    bandwidth = estimate_bandwidth(reshaped_image, quantile=Quantile, n_samples=N_samples)
    #apply the mean shift transform from the library sklearn
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    #fit the model to the reshaped image
    ms.fit(reshaped_image)
    #reshape the labels to the shape of the image
    segmented_image = np.reshape(ms.labels_, cie_image.shape[:2])

    if show:
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(segmented_image, cmap='nipy_spectral')
        plt.title('Mean Shift Segmented Image')
        plt.axis('off')
        plt.show()

    # return the segmented image along with the name for it and which parameters were used as a string
    params = 'MeanShift: Quantile = ' + str(Quantile) + ', N Samples = ' + str(N_samples)
    return segmented_image, params
