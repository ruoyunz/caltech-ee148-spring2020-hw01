import os
import numpy as np
import json
from PIL import Image

import cv2

# set the path to the downloaded data: 
data_path = '../data/RedLights2011_Medium'

# set a path for saving predictions: 
preds_path = '../data/hw01_preds' 

# Helper function to calculate the amount of overlap between two 
# boxes of the same size given the top left coordinates
def areaOverlap(r, c, temp_r, temp_c, box_height, box_width):
    if (abs(r - temp_r) > box_height) or (abs(c - temp_c) > box_width):
        return 0
    if r > temp_r:
        h = (box_height + temp_r - r)
    else:
        h = (box_height - temp_r + r)
    if c > temp_c:
        w = (box_width + c - temp_c)
    else:
        w = (box_width - c + temp_c)
    return w * h / (box_width * box_height)

# Helper function to smooth the kernel by averaging the value of each value of the
# kernel with a bit of it's neighbor's values (weighted average with 1/4 weight
# per neighbor)
#
# Didn't really help algorithm.
def smoothKernel(k, n_channels, box_height, box_width):
    avg_ngbr = np.copy(k)
    neighbor = [-1, 0, 1] # neighboring values to check. can choose more neighbors
    for ch in range(n_channels):
        for i in range(box_height):
            for j in range(box_width):
                n = 0
                for m in neighbor:
                    for l in neighbor:
                        if (box_height > (i + m) >= 0) and (box_width > (j + l) >= 0):
                            if not (m == 0 and l == 0):
                                avg_ngbr[i, j, ch] += 0.25 * k[i+m, j+l, ch]
                            n += 1
                avg_ngbr[i, j, ch] = avg_ngbr[i, j, ch] / ((n-1) * 0.25 + 1)
    return avg_ngbr

# Convolution: apply kernel over entire image and save bounding 
# boxes in the areas where dot product is greater than threshold
def convolution(I, k, box_height, box_width):
    (n_rows,n_cols,n_channels) = np.shape(I)
    all_boxes = []

    for r in range(int(0.5 * (n_rows - box_height))):
        for c in range(n_cols - box_width):
            x = (I[r:(r + box_height), c:(c + box_width), :]).flatten()
            x = x / np.linalg.norm(x)
            R = np.dot(k, x)

            # Ensure that the red light (most important feature) is taken care of
            # by only checking for pixels where the top light area has a high R 
            # channel value
            if R > 0.9 and I[int((r + box_height/5)), int((c + box_width/2)), 0] > 200:
                all_boxes.append([r, c, R])
    return all_boxes

# Consolidate bounding boxes--get overlapping bounding boxes and 
# only save the one with the highest threshold. This is done by using
# non maximum suppression--all boxes that overlap by more than 50% will
# be consolidated.
def consolidation(I, all_boxes, box_height, box_width):
    temp_boxes = [all_boxes[0]]

    for i in range(1, len(all_boxes)):
        (r, c, R) = all_boxes[i]
        new_max = False
        overlap = False
        j = 0
        while j < len(temp_boxes):
            (temp_r, temp_c, temp_R) = temp_boxes[j]
            a = areaOverlap(r, c, temp_r, temp_c, box_height, box_width)
            if a > 0.5:
                overlap = True
                if R > temp_R:
                    new_max = True
                    temp_boxes.pop(j)
                    j -= 1
            j += 1
        if (overlap and new_max) or (not overlap):
            temp_boxes.append([r, c, R])

    return temp_boxes


def detect_red_light(I):
    '''
    This function takes a numpy array <I> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the 
    image. Each element of <bounding_boxes> should itself be a list, containing 
    four integers that specify a bounding box: the row and column index of the 
    top left corner and the row and column index of the bottom right corner (in
    that order). See the code below for an example.
    
    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''
    
    # Get the traffic light image (kernel)
    k = Image.open(os.path.join(data_path, 
        "../../caltech-ee148-spring2020-hw01/truth.jpg"))
    (box_height, box_width, n_channels) = np.shape(k)

    k = np.asarray(k)
    smoothKernel(k, n_channels, box_height, box_width)

    k = k.flatten()
    k = k / np.linalg.norm(k)

    bounding_boxes = [] # This should be a list of lists, each of length 4. See format example below. 

    all_boxes = convolution(I, k, box_height, box_width)
    if len(all_boxes) == 0:
        return bounding_boxes
    
    temp_boxes = consolidation(I, all_boxes, box_height, box_width)

    for i in range(len(temp_boxes)):
        (r, c, R) = temp_boxes[i]
        bounding_boxes.append([r, c, r + box_height, c + box_width])
    
    for i in range(len(bounding_boxes)):
        assert len(bounding_boxes[i]) == 4
    
    return bounding_boxes


# Interactive way to check if detection algorithm is running as it should be. 
# Change the image ID as needed. Relies on OpenCV but purely for image 
# displaying efforts.
def main():
    I = Image.open(os.path.join(data_path,"RL-001.jpg"))
    I = np.asarray(I)
    img = cv2.imread(os.path.join(data_path,"RL-001.jpg"),cv2.IMREAD_COLOR)
    cv2.imshow('image',img)
    cv2.waitKey(0)

    boxes = detect_red_light(I)

    for i in range(len(boxes)):
        (r, c, r2, c2) = boxes[i]
        cv2.rectangle(img, (c, r), (c2, r2), (0, 255, 0), 1)
        cv2.imshow('image',img)
        ch = cv2.waitKey(0)
        if ch == 27:
            return bounding_boxes


if __name__ == "__main__":
    main()
