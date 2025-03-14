import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


def get_feature_space(image):
    #function that extracts the feature space of the image
    height, width = image.shape[0:2]
    #convert the image from rgb to hsv
    hsv_image = colors.rgb_to_hsv(image)
    #take the hue and saturation channels
    im_space = hsv_image[..., 0:2]
    #reshape the hue and saturation channels to a 2D array
    feature_vector = np.reshape(im_space, (height * width, 2)).T
    return feature_vector


def get_segmented_image(clustering_out, clusters, image, boostH):
    #function that returns the segmented image

    height, width = image.shape[0:2]
    height_cl, width_cl = clusters.shape[0:2]
    #empty array to store the cluster features
    cls_features = np.zeros((len(clustering_out), clusters.shape[0])).T
    #for each cluster
    for c in range(width_cl):
        #find the indexes of the points that belong to the cluster
        idxs = np.where(clustering_out == c)
        #for each point in the cluster
        for j in idxs[0]:
            #add the cluster features to the indices
            cls_features[:, j] = clusters[:, c]
    #reshape the cluster features to the original image size
    im_space = np.reshape(cls_features.T, (height, width, height_cl))
    #convert the image from rgb to hsv
    hsv_image = colors.rgb_to_hsv(image)
    #boost the hue and saturation channels to see better the segments
    hsv_image[..., 0:2] = im_space * boostH
    #normalize the values
    hsv_image[..., 2] /= np.max(hsv_image[..., 2])
    #convert the image from hsv to rgb
    return colors.hsv_to_rgb(hsv_image)


def kmeans_alg(image, k, num_iterations, boostH):
    height, width = image.shape[0:2]
    #total number of pixels
    num_points = height * width
    #get the feature space of the image
    feature_space = get_feature_space(image)
    #make the initial indices random for the clusters
    idxs = np.round(num_points * np.random.rand(k))
    #if the indices are bigger than the number of pixels, decrease them
    idxs[np.where(idxs >= height * width)] -= 1
    #array to store the cluster centers
    initial_centers = np.zeros((2, k))
    for i in range(k):
        #initialize the cluster centers with the feature space values
        initial_centers[:, i] = feature_space[:, int(idxs[i])]
    #initialize the cluster centers
    clusters_centers = initial_centers
    #empty array to store the distances
    distance = np.zeros((k, 1))
    #empty array to store the cluster points
    cluster_points = np.zeros((num_points, 1))

    for it in range(num_iterations):
        #for each point
        for point in range(num_points):
            for h in range(k):
                #calculate the distance between the point and the cluster centers
                distance[h] = np.sqrt(np.sum((feature_space[:, point] - clusters_centers[:, h]) ** 2))
            #assign the point to the cluster with the minimum distance
            cluster_points[point] = np.argmin(distance)
        for c in range(k):
            #find the indexes of the points that belong to the cluster
            idxs = np.where(cluster_points == c)
            points = feature_space[:, idxs[0]]

            if points.size > 0:
                #if the cluster has points, update the cluster center
                clusters_centers[:, c] = np.mean(points, 1)
            else:
                #if the cluster has no points, assign a random point to the cluster
                idx = np.round(num_points * np.random.rand())
                clusters_centers[:, c] = feature_space[:, int(idx)]
        #segment the image
        return get_segmented_image(cluster_points, clusters_centers, image, boostH)


def kmeans_segm(image_name, k=5, num_iterations=10, boostH=1, show=True):
    image = plt.imread(image_name)
    segmented_image = kmeans_alg(image, k, num_iterations, boostH)
    if show:
        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.axis('off')
        plt.imshow(image)
        plt.subplot(1, 2, 2)
        plt.title('Segmented Image')
        plt.axis('off')
        plt.imshow(segmented_image)
        plt.show()

    # return the segmented image along with the name for it and which parameters were used as a string
    # boostH is not returned because it is used only to change the visualization, not how the image is segmented
    params = 'KMeans: K = ' + str(k) + ', Iterations = ' + str(num_iterations)
    return segmented_image, params

