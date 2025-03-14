import numpy as np
from math import sqrt, ceil, exp, pow


def convolution(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    height, width = a.shape
    # create the output image
    output = np.zeros(shape=a.shape, dtype=float)
    for y in range(height):
        for x in range(width):
            # compute the convolution of a and b at (x, y)
            total = float(b[0] * a[y, x])
            for i in range(1, len(b)):
                # add the weighted values of the neighbors, clipping at the edges
                total += b[i] * (a[y, max(x - i, 0)] + a[y, min(x + i, width - 1)])
            output[y, x] = total
    return output


def make_gaussian_filter(sigma):
    sigma = max(sigma, 0.01)
    length = int(ceil(sigma * 4.0)) + 1
    mask = np.zeros(shape=length, dtype=float)
    #we can use exponentials to create the gaussian form
    for i in range(length):
        mask[i] = exp(-0.5 * pow(i / sigma, i / sigma))
    return mask


def smoothen(img, sigma):
    #apply the gaussian filter to the image
    mask = make_gaussian_filter(sigma)
    # normalize the mask
    mask = np.divide(mask, 2 * np.sum(np.absolute(mask)) + abs(mask[0]))
    #apply the convolution twice to smoothen it more
    dst = convolution(convolution(img, mask), mask)
    return dst


def difference(red, green, blue, x1, y1, x2, y2):
    #return the square root of the sum of squared differences
    #between the two pixels for each color channel
    return sqrt((red[y1, x1] - red[y2, x2]) ** 2 + \
                (green[y1, x1] - green[y2, x2]) ** 2 + \
                (blue[y1, x1] - blue[y2, x2]) ** 2)


def imgDistance(image, x1, y1, x2, y2):
    return np.linalg.norm(image[x1, y1] - image[x2, y2])


class Custom_disjoint_set:
    def __init__(self, n_elements):
        self.num = n_elements
        self.elements = np.empty(
            shape=(n_elements, 3),
            dtype=int
        )
        for i in range(n_elements):
            self.elements[i, 0] = 0
            self.elements[i, 1] = 1
            self.elements[i, 2] = i

    def getSize(self, x):
        return self.elements[x, 1]

    def getNumSets(self):
        return self.num

    def find(self, x):
        y = int(x)
        while y != self.elements[y, 2]:
            y = self.elements[y, 2]
        self.elements[x, 2] = y
        return y

    def join(self, x, y):
        if self.elements[x, 0] > self.elements[y, 0]:
            self.elements[y, 2] = x
            self.elements[x, 1] += self.elements[y, 1]
        else:
            self.elements[x, 2] = y
            self.elements[y, 1] += self.elements[x, 1]
            if self.elements[x, 0] == self.elements[y, 0]:
                self.elements[y, 0] += 1
        self.num -= 1


class Custom_stack:
    def __init__(self):
        self.item = []
        self.obj = []

    def push(self, value):
        self.item.append(value)

    def pop(self):
        return self.item.pop()

    def getSize(self):
        return len(self.item)

    def isEmpty(self):
        return self.getSize() == 0

    def clear(self):
        self.item = []
