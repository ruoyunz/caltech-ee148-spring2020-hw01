import os
import numpy as np
import json
from PIL import Image

from run_detection import detect_red_light

# set the path to the downloaded data: 
data_path = '../data/RedLights2011_Medium'

# set a path for saving predictions: 
preds_path = '../data/hw01_preds' 
os.makedirs(preds_path,exist_ok=True) # create directory if needed 

def main():
    os.makedirs(preds_path,exist_ok=True) # create directory if needed 

    # get sorted list of files: 
    file_names = sorted(os.listdir(data_path)) 

    # remove any non-JPEG files: 
    file_names = [f for f in file_names if '.jpg' in f] 

    preds = {}

    for i in range(len(file_names)):
        
        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names[i]))
        
        # convert to numpy array:
        I = np.asarray(I)
        
        preds[file_names[i]] = detect_red_light(I)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds.json'),'w') as f:
        json.dump(preds,f)

if __name__ == "__main__":
    main()







