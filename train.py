import os
import model
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
import cv2
import matplotlib.pyplot as plt

from glob import glob
data = glob(r'C:\Users\Susmitha Rachamreddy\Desktop\masters\Dency_project\data/**/*.png',
            recursive=True)
images=[]
labels=[]
for i in data[:15000]:
    if i.endswith('.png'):
        label=i[-5]
        img=cv2.imread(i)
        img_1=cv2.resize(img,(100,100))
        images.append(img_1)
        labels.append(label)
x = np.stack(images)
y = to_categorical(labels)
