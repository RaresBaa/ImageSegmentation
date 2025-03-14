import numpy as np
from utils import smoothen, difference, Custom_disjoint_set
from random import randint
import matplotlib.pyplot as plt
from PIL import Image


def segment_graph(nVertices, nEdges, edges, ref):
    #we will use a disjoint set to store the vertices
    vertSet = Custom_disjoint_set(nVertices)
    #sort the edges by weight
    edges[0: nEdges, :] = edges[edges[0: nEdges, 2].argsort()]
    #initialize the threshold array for each vertex
    threshold = np.zeros(shape=nVertices, dtype=float)
    # set initial references for each vertex
    for i in range(nVertices):
        threshold[i] = ref
    #for each edge in the graph
    for i in range(nEdges):
        #get an edge from the sorted array
        semi_edge = edges[i, :]
        #find the sets that represent the vertices of the edge
        a = vertSet.find(semi_edge[0])
        b = vertSet.find(semi_edge[1])
        #if the sets are different
        if a != b:
            #if the edge weight is smaller than the threshold of both vertices
            if (semi_edge[2] <= threshold[a]) and (semi_edge[2] <= threshold[b]):
                #join the vertices
                vertSet.join(a, b)
                a = vertSet.find(a)
                #and update the threshold for the new set
                threshold[a] = semi_edge[2] + (ref / vertSet.getSize(a))
    return vertSet


def segment(in_image, sigma, k, min_size):
    # get image dimensions
    height, width, _ = in_image.shape
    #smooth each color channel individually
    smoothR = smoothen(in_image[:, :, 0], sigma)
    smoothG = smoothen(in_image[:, :, 1], sigma)
    smoothB = smoothen(in_image[:, :, 2], sigma)
    #estimate the maximum number of edges and allocate space for them
    edges_size = width * height * 4
    edges = np.zeros(shape=(edges_size, 3), dtype=object)
    #total edge number
    num = 0
    #for each pixel in the image
    #we will add between pixels edges that are the difference
    #between the the smoothed value and the pixel next to it
    for y in range(height):
        for x in range(width):
            #if we aren't at the right edge of the image
            if x < width - 1:
                #add an edge between the current pixel and the right one
                edges[num, 0] = int(y * width + x)
                edges[num, 1] = int(y * width + (x + 1))
                edges[num, 2] = difference(
                    smoothR, smoothG, smoothB, x, y, x + 1, y
                )
                num += 1
            #if we aren't at the bottom of the image
            if y < height - 1:
                #add an edge between the current pixel and the one below
                edges[num, 0] = int(y * width + x)
                edges[num, 1] = int((y + 1) * width + x)
                edges[num, 2] = difference(
                    smoothR, smoothG, smoothB, x, y, x, y + 1
                )
                num += 1
            #if we aren't at the bottom right of the image
            if (x < width - 1) and (y < height - 2):
                #add an edge between the current pixel and the one below and to the right
                edges[num, 0] = int(y * width + x)
                edges[num, 1] = int((y + 1) * width + (x + 1))
                edges[num, 2] = difference(
                    smoothR, smoothG, smoothB, x, y, x + 1, y + 1
                )
                num += 1
            #if we aren't at the top right of the image
            if (x < width - 1) and (y > 0):
                #add an edge between the current pixel and the one above and to the right
                edges[num, 0] = int(y * width + x)
                edges[num, 1] = int((y - 1) * width + (x + 1))
                edges[num, 2] = difference(
                    smoothR, smoothG, smoothB, x, y, x + 1, y - 1
                )
                num += 1

    segm_graph = segment_graph(width * height, num, edges, k)
    #for each edge in the graph
    for i in range(num):
        #find the sets that represent the vertices of the edge
        a = segm_graph.find(edges[i, 0])
        b = segm_graph.find(edges[i, 1])
        #if the vertices are in different sets and one of them is smaller than the minimum size
        if (a != b) and ((segm_graph.getSize(a) < min_size) or (segm_graph.getSize(b) < min_size)):
            #merge the sets containing the vertices
            segm_graph.join(a, b)

    #create the output image and the color's array
    output = np.zeros(shape=(height, width, 3))
    colors = np.zeros(shape=(height * width, 3))

    for i in range(height * width):
        #assign a random color to each segment
        colors[i, :] = [randint(0, 255), randint(0, 255), randint(0, 255)]
    for y in range(height):
        for x in range(width):
            #assign the color of the segment to each pixel
            output[y, x, :] = colors[segm_graph.find(y * width + x), :]
    return output


def felzenszwalb_segm(image_name, sigma=0.2, k=400, min_size=50, show=True):
    image = np.array(Image.open(image_name))
    segmented_image = segment(image, sigma, k, min_size)
    if show:
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(segmented_image.astype(np.uint8))
        plt.title('Segmented Image')
        plt.axis('off')
        plt.show()
    # return the segmented image along with the name for it and which parameters were used as a string
    params = 'Felzenszwalb: Sigma: ' + str(sigma) + ', K: ' + str(k) + ', Min Size: ' + str(min_size)
    return segmented_image, params

