import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import scipy.io as sio

## download dataset from https://www.kaggle.com/c/leaf-classification/data
## and unzip them to ./dataset/


## the images will be processed as https://www.kaggle.com/abhmul/keras-convnet-lb-0-0052-w-visualization
## to make them 96*96 and save them in ./dataset/leaves/ folder
## the folder ./dataset/leaves/ will be arranged as
'''
- train
    - Acer_Capillipes
        - 0201.jpg
        - 0227.jpg
        - ...
    - Acer_Circinatum
        - ****.jpg
- test
    - 000
        0004.jpg

## You can download the processed file in
'http://pan.baidu.com/s/1qYqGYRU'

''' 



data_path = './dataset'



def load_training_att():
    data = pd.read_csv(os.path.join(data_path,'train.csv'))
    ID =data.pop('id')
    y = data.pop('species')
    y = LabelEncoder().fit(y).transform(y)

    x = StandardScaler().fit(data).transform(data)
    att = {}
    for i, ids in enumerate(ID, 0):
        name = '%04d.jpg'%ids
        att[name] = x[i,:].astype(np.float32)
    return att
        

    # return ID, x, y

def load_test_att(att_train):
    data = pd.read_csv(os.path.join(data_path,'test.csv'))
    ID =data.pop('id')
    x = StandardScaler().fit(data).transform(data)
    for i, ids in enumerate(ID, 0):
        name = '%04d.jpg'%ids
        att_train[name] = x[i,:].astype(np.float32)
    return att

    # return ID, x, y
if __name__ == '__main__':
    att = load_training_att()
    att = load_test_att(att)
    print len(att)
    sio.savemat('dataset/att.mat', att)
    
    