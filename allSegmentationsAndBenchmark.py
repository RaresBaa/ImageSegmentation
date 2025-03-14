from matplotlib import pyplot as plt
from skimage.metrics import adapted_rand_error, variation_of_information
from skimage.color import rgb2gray
import numpy as np

from felzenszwalb import felzenszwalb_segm
from kmeans import kmeans_segm
from regionGrow import regionGrow_segm
from grabcut import grabcut_segm
from meanshift import meanshift_segm
from slic import slic_segm

def runSegmFromList(image_name, listOfParams, show=False):
    # this function will run the segmentation algorithms based on the list of parameters and return the segmented images

    # this list will store the segmented images, along with the string that represents the algorithm used and the parameters used
    # because each algorith returns: segmented image, name and parameters as a string
    segmentedImages = []
    for params in listOfParams:
        if params[0] == 'felzenszwalb':
            segmentedImages.append(felzenszwalb_segm(image_name, sigma=params[1], k=params[2], min_size=params[3], show=show))
        elif params[0] == 'regionGrow':
            segmentedImages.append(regionGrow_segm(image_name, thr=params[1], show=show).return_segmented_image())
        elif params[0] == 'kmeans':
            segmentedImages.append(kmeans_segm(image_name, k=params[1], num_iterations=params[2], boostH=params[3], show=show))
        elif params[0] == 'grabcut':
            segmentedImages.append(grabcut_segm(image_name, iterations=params[1], excludeH=params[2], excludeW=params[3], show=show))
        elif params[0] == 'meanshift':
            segmentedImages.append(meanshift_segm(image_name, Quantile=params[1], N_samples=params[2], show=show))
        elif params[0] == 'slic':
            segmentedImages.append(slic_segm(image_name, num_superpixels=params[1], compactness=params[2], show=show))
    return segmentedImages


def showSegmentedImages(original_image, reference_segmentation,  segmented_images):
    # this function will display the segmented image along with the name of the algorithm and the parameters used
    # for the first image, we will display the original image and for the second the reference segmentation.
    # since the name is too long, we will create a legend that will hold all of the parameters
    # on the first row we will display the original image and the reference segmentation
    # on the second row we will display the segmented images

    segmentedSize = len(segmented_images)
    plt.figure()

    # Plotting the original image
    plt.subplot(2, segmentedSize + 1, 1)
    plt.imshow(original_image)
    plt.title('Original')
    plt.axis('off')

    # Plotting the reference segmentation
    plt.subplot(2, segmentedSize + 1, 2)
    plt.imshow(reference_segmentation)
    plt.title('Reference')
    plt.axis('off')

    # Plotting segmented images
    for index, (segm_img, name) in enumerate(segmented_images, start=1):
        # clip or normalize the image
        if segm_img.dtype == np.float32 or segm_img.dtype == np.float64:
            segm_img = np.clip(segm_img, 0, 1)
        elif segm_img.dtype == np.int32 or segm_img.dtype == np.int64:
            segm_img = np.clip(segm_img, 0, 255).astype(np.uint8)

        # plot the segmented image
        plt.subplot(2, segmentedSize + 1, index + segmentedSize + 1)
        plt.imshow(segm_img)
        plt.title(name.split()[0], fontsize=8)
        plt.axis('off')

    # print the legend with the parameters used
    legend_text = "Legend:\n"
    for _, name in segmented_images:
        legend_text += '\t' + name + '\n'
    print(legend_text)

    plt.tight_layout()
    plt.show()


def segmentation_statistics_single(segmented_image, ground_truth):
    # Ensure both images are of integer type
    segmented_image = segmented_image.astype(int)

    # Convert ground truth to grayscale if segmented image is grayscale
    if len(segmented_image.shape) == 2 and ground_truth.ndim == 3:
        ground_truth = rgb2gray(ground_truth)

    # Ensure the ground truth is also integer labeled
    ground_truth = (ground_truth * 255).astype(int) if ground_truth.dtype == np.float64 else ground_truth.astype(int)

    # Calculate the statistics for the segmentation
    error, precision, recall = adapted_rand_error(ground_truth, segmented_image)
    splits, merges = variation_of_information(ground_truth, segmented_image)

    print("\tError: ", error)
    print("\tPrecision: ", precision)
    print("\tRecall: ", recall)
    print("\tSplits: ", splits)
    print("\tMerges: ", merges)


def segmentation_statistics(segmented_images, ground_truth):
    # this function will calculate the statistics for all the segmented images

    for segm_img, name in segmented_images:
        print(name)
        segmentation_statistics_single(segm_img, ground_truth)