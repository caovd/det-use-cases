'''
Process datasets and save to a local path.
'''

import os
# import pandas as pd
import ast
from PIL import Image
import shutil as sh
from pathlib import Path
import json
# from sklearn.model_selection import train_test_split

def config_json(username, key, json_path):
    # Download dataset from Kaggle
    Kaggle_dict = {"username":username,"key":key}
    
    # Serializing json 
    json_object = json.dumps(Kaggle_dict, indent = 4)
    
    # Writing to path
    with open(os.path.join(json_path,"kaggle.json"), "w") as outfile:
        outfile.write(json_object)


if __name__=='__main__':
    json_path = "./data"
    try:
        os.makedirs(json_path)
    except FileExistsError:
        pass
    
    username = "caovudung"
    key = "5c2632c0d5e76ae5f221c63f70af5c90"
    
    config_json(json_path)