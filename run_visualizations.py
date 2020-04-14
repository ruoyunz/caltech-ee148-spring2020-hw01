import os
import numpy as np
import json
from PIL import Image, ImageDraw

import cv2 

from run_predictions import detect_red_light


# set the path to the downloaded data: 
data_path = '../data/RedLights2011_Medium'

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

preds = detect_red_light(I, img)
print(preds)

img1 = Image.fromarray(I)
draw = ImageDraw.Draw(img1)
for i in preds:
	draw.rectangle((i[1], i[0], i[3], i[2]), outline=(0, 255, 0))

img1.save(os.path.join(data_path,"../pred.jpg"))


# for i in range(len(file_names)):
#     # read image using PIL:
#     I = Image.open(os.path.join(data_path,file_names[i]))
    
#     # convert to numpy array:
#     I = np.asarray(I)

#     img = Image.fromarray(I)
    
#     preds = detect_red_light(I)
#     for i in preds:
#     	img.rectangle((preds[0], preds[1], preds[2], preds[3]), outline=(0, 0, 255))

#     img.save(os.path.join(data_path,"../pred.jpg"))

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds.json'),'w') as f:
    json.dump(preds,f)