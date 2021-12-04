import pandas as pd
import numpy as np
import os
import cv2
import keras
from skimage.transform import resize
import tensorflow as tf

def load_labels(file="archive/dataset-master/dataset-master/labels.csv"):
    labels = pd.read_csv(file)
    labels = labels["Category"]
    return labels
    
def load_data(file_loc):
    X,Y = [],[]
    for i in os.listdir(file_loc):
        if not i.startswith('.'):
            if i in ['NEUTROPHIL']: y = 1
            elif i in ['EOSINOPHIL']: y = 2
            elif i in ['MONOCYTE']: y = 3
            elif i in ['LYMPHOCYTE']: y = 4
            else: label = 5
            for j in os.listdir(file_loc + i):
                image = cv2.imread(file_loc + i + '/' + j)
                if image is not None:
                    #image = np.array(Image.fromarray(image).resize(size=(60, 80)))
                    X.append(np.asarray(resize(image, (60, 80, 3))))
                    Y.append(y)
    return np.asarray(X), tf.keras.utils.to_categorical(np.asarray(Y),5)