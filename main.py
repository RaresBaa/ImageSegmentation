import cv2
from allSegmentationsAndBenchmark import *

#image to segment
image_name = ''
#reference segmentation to benchmark against
reference_segmentation = ''

#which algorithms to run
segmAlgs = [
    ['meanshift', 0.2, 500],
    ['slic', 300, 10],
    ['felzenszwalb', 0.2, 400, 50],
    ['regionGrow', 15],
    ['grabcut', 5, 50, 50],
    #['kmeans', 5, 10, 1.5]
]

#can use any combination of algorithms
exampleAlgs = [
    ['slic', 300, 10],
    ['slic', 500, 20],
    ['slic', 100, 5],
    ['meanshift', 0.2, 500],
    ['meanshift', 0.5, 1000],
    ['meanshift', 0.1, 100],
]

original_image = cv2.imread(image_name)
ground_truth = cv2.imread(reference_segmentation)

#runs every segmentation from the list, it is very compute heavy
segmented_images = runSegmFromList(image_name, segmAlgs)

#show segmented images
showSegmentedImages(original_image, ground_truth, segmented_images)
#print segmentation benchmarks
segmentation_statistics(segmented_images, ground_truth)

