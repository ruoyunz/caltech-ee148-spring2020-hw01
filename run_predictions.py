import os
import numpy as np
import json
from PIL import Image

import cv2
from scipy import ndimage

def detect_red_light(I, img):
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
    
    # get the traffic light image (kernel)
    k = Image.open(os.path.join(data_path, "../truth.jpg"))
    (box_height, box_width, n_channels) = np.shape(k)

    k = np.asarray(k)
    avg_ngbr = np.copy(k)
    for ch in range(n_channels):
        for i in range(box_height):
            for j in range(box_width):
                n = 0
                for m in range(-1, 1):
                    for l in range(-1, 1):
                        if (box_height > (i + m) >= 0) and (box_width > (j + l) >= 0):
                            avg_ngbr[i, j, ch] += k[i+m, j+l, ch]
                            n += 1
                avg_ngbr[i, j, ch] = avg_ngbr[i, j, ch] / n

    #k = avg_ngbr

    k = k.flatten()
    k = k / np.linalg.norm(k)

    bounding_boxes = [] # This should be a list of lists, each of length 4. See format example below. 
    
    '''
    BEGIN YOUR CODE
    '''
    
    '''
    As an example, here's code that generates between 1 and 5 random boxes
    of fixed size and returns the results in the proper format.
    '''

    # Convolution: apply kernel over entire image and save bounding 
    # boxes in the areas where dot product is greater than threshold
    (n_rows,n_cols,n_channels) = np.shape(I)

    for r in range(int(0.5 * (n_rows - box_height))):
        for c in range(n_cols - box_width):
            x = (I[r:(r + box_height), c:(c + box_width), :]).flatten()
            x = x / np.linalg.norm(x)

            # Dot product with traffic light
            R = np.dot(k, x)
            if r == 324 and c == 147:
                print(R)

            # If this is > T, yes. draw box.
            if R > 0.9:
                bounding_boxes.append([r, c, r + box_height, c + box_width])
                cv2.rectangle(img, (c, r), (c + box_width, r + box_height), (0,0,255), 2)
                cv2.imshow('image',img)
                ch = cv2.waitKey(0)
                if ch == 27:
                    return bounding_boxes


    # Consolidate bounding boxes--get overlapping bounding boxes and 
    # only save the one with the highest threshold.
    
    '''
    END YOUR CODE
    '''
    
    for i in range(len(bounding_boxes)):
        assert len(bounding_boxes[i]) == 4
    
    return bounding_boxes

# set the path to the downloaded data: 
data_path = '../RedLights2011_Medium'

# set a path for saving predictions: 
preds_path = '../data/hw01_preds' 
os.makedirs(preds_path,exist_ok=True) # create directory if needed 

# get sorted list of files: 
file_names = sorted(os.listdir(data_path)) 

# remove any non-JPEG files: 
file_names = [f for f in file_names if '.jpg' in f] 

preds = {}

I = Image.open(os.path.join(data_path,"RL-015.jpg"))
I = np.asarray(I)

img = cv2.imread(os.path.join(data_path,"RL-015.jpg"),cv2.IMREAD_COLOR)
cv2.imshow('image',img)
cv2.waitKey(0)

detect_red_light(I, img)

# for i in range(len(file_names)):
    
#     # read image using PIL:
#     I = Image.open(os.path.join(data_path,file_names[i]))
    
#     # convert to numpy array:
#     I = np.asarray(I)
    
#     preds[file_names[i]] = detect_red_light(I)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds.json'),'w') as f:
    json.dump(preds,f)
