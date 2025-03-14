import matplotlib.pyplot as plt
import numpy as np
import cv2

def grabcut_segm(image_name, iterations=5, excludeW=50, excludeH=50, show=True):
    image = cv2.imread(image_name)

    #create an empty mask
    mask = np.zeros(image.shape[:2], np.uint8)
    #create two arrays with the same shape as the image, but only 1 channel
    background = np.zeros((1, 65), np.float64)
    foreground = np.zeros((1, 65), np.float64)
    #create a rectange that contains what we want to segment, excluding the borders
    rect = (excludeH, excludeW, image.shape[1] - excludeH, image.shape[0] - excludeW)
    #apply the grabcut algorithm from cv2
    cv2.grabCut(image, mask, rect, background, foreground, iterations, cv2.GC_INIT_WITH_RECT)
    #create a combined mask
    comb_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    #apply the mask to the image
    segmented_image = image * comb_mask[:, :, np.newaxis]
    if show:
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
        plt.title('Segmented Image')
        plt.axis('off')
        plt.show()

    # return the segmented image along with the name for it and which parameters were used as a string
    params = 'Grabcut: Iterations = ' + str(iterations) + ', Exclude Width = ' + str(excludeW) + ', Exclude Height = ' + str(excludeH)
    return segmented_image, params
