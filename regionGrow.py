import cv2
import itertools
import numpy as np
import random

from PIL import Image

from utils import Custom_stack, imgDistance
import matplotlib.pyplot as plt


def getNeighbour(x0, y0, width, height):
    return [(x, y)
            # itertools.product generates the following pairs:
            # (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)
            # so we can get all of the 8 neighbours of a pixel
            for i, j in itertools.product((-1, 0, 1), repeat=2)
            #filter out the pixel at (0, 0) and the pixels that are out of the image
            if (i, j) != (0, 0) and 0 <= (x := x0 + i) < height and 0 <= (y := y0 + j) < width]


def reset_region(x0, y0, width, height):
    #reset the region with random values
    x0 = random.randint(x0 - 4, x0 + 4)
    y0 = random.randint(y0 - 4, y0 + 4)
    #clip the values to the image size
    x0 = np.clip(x0, 0, height - 1)
    y0 = np.clip(y0, 0, width - 1)
    return x0, y0


class regionGrow_segm:
    def __init__(self, im_path, thr=15, show=True):
        # set the initial values for the members in the class
        self.image = cv2.imread(im_path, 1).astype('int')
        self.rgbImage = Image.open(im_path)
        self.height, self.width, _ = self.image.shape
        self.displayImage = show
        # array to store the regions passed by the algorithm
        self.passedBy = np.zeros((self.height, self.width), np.double)
        # segmented image
        self.segmented_img = np.zeros((self.height, self.width, 3), dtype='uint8')
        # treshold for the region growing
        self.thr = float(thr)
        self.currentRegion = 0
        self.iterations = 0
        # this stack is used for BFS
        self.stack = Custom_stack()

        for x0 in range(self.height):
            for y0 in range(self.width):
                # for each pixel in the photo

                # if the current pixel hasn't been processed
                if self.passedBy[x0, y0] == 0:
                    # inncrement the region count and add the pixel to the processed ones
                    self.currentRegion += 1
                    self.passedBy[x0, y0] = self.currentRegion
                    # push the pixel into the stack
                    self.stack.push((x0, y0))
                    # reset the count for the previous region
                    self.prev_region_count = 0

                    # process the pixels from the stack
                    while not self.stack.isEmpty():
                        x, y = self.stack.pop()
                        # use BFS for each pixel from the stack
                        self.BFS(x, y)
                        self.iterations += 1

                    # if we have processed all of the pixels, we can stop the algorithm
                    if np.all(self.passedBy > 0):
                        break

                    # if the region is too small, reset it with random values
                    if self.prev_region_count < 8 * 8:
                        self.passedBy[self.passedBy == self.currentRegion] = 0
                        x0, y0 = reset_region(x0, y0, self.width, self.height)
                        self.currentRegion -= 1

        # color the pixels considering the region
        [self.color_pixel(i, j) for i, j in itertools.product(range(self.height), range(self.width))]
        if self.displayImage:
            self.display()

    def return_segmented_image(self):
        # return the segmented image along with the name for it and which parameters were used as a string
        params = 'RegionGrowing: Threshold = ' + str(self.thr)
        return self.segmented_img, params

    def color_pixel(self, i, j):
        val = self.passedBy[i][j]
        # color the pixel based on the region it belongs to
        self.segmented_img[i][j] = (255, 255, 255) if (val == 0) else (val * 35, val * 90, val * 30)

    def display(self):
        plt.subplot(1, 2, 1)
        plt.imshow(self.rgbImage)
        plt.title('Original Image')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(self.segmented_img)
        plt.title('Segmented Image')
        plt.axis('off')
        plt.show()

    def BFS(self, x0, y0):
        #breadth-first searh algorithm

        regionNum = self.passedBy[x0, y0]
        #store the pixel intensities
        elems = [np.mean(self.image[x0, y0])]
        #variance tresold
        var = self.thr
        #get all of the neighbours of the current pixel, without the invalid ones
        neighbours = getNeighbour(x0, y0, self.width, self.height)
        for x, y in neighbours:
            #if the pixel isn't processed and the distance between the pixel and the current one is smaller than the variance
            if self.passedBy[x, y] == 0 and imgDistance(self.image, x, y, x0, y0) < var:
                #if we have processed all of the pixels, we can stop the algorithm
                if np.all(self.passedBy > 0):
                    break
                #mark the pixel as processed
                self.passedBy[x, y] = regionNum
                #and add it to the stack to continue processing
                self.stack.push((x, y))
                #add the pixel intensity to the list
                elems.append(np.mean(self.image[x, y]))
                #calculate the variance
                var = np.var(elems)
                #increase the region count for the current region
                self.prev_region_count += 1
            #update the variance
            var = max(var, self.thr)
