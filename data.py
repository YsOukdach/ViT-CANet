import glob
from json import load
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold

def load_data(path_Abnormal, path_Normal):

    Abnormal = glob.glob(path_Abnormal)
    Normal = glob.glob(path_Normal)

    data = []
    labels = []

    for i in Abnormal:
        image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb',
        target_size= (336,336))
        image=np.array(image)
        data.append(image)
        labels.append(0)
    for i in Normal:
        image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb',
        target_size= (336,336))
        image=np.array(image)
        data.append(image)
        labels.append(1)
    return data, labels


