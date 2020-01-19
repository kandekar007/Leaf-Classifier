import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

DATADIR = "/content/dataset/train"

CATEGORIES = os.listdir(DATADIR)

for category in CATEGORIES:  
    path = os.path.join(DATADIR,category)  
    for img in os.listdir(path):  
        img_array = cv2.imread(os.path.join(path,img))  
        plt.imshow(img_array, cmap='gray')  # graph it
        plt.show() 
        break  
    break

training_data = []
IMG_SIZE=224

def create_training_data():
    for category in CATEGORIES:  

        path = os.path.join(DATADIR,category)  
        class_num = CATEGORIES.index(category)  

        for img in tqdm(os.listdir(path)):  
            try:
                img_array = cv2.imread(os.path.join(path,img))  # convert to array
                training_data.append([img_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            except OSError as e:
                print("OSErrroBad img most likely", e, os.path.join(path,img))
            except Exception as e:
                print("general exception", e, os.path.join(path,img))

create_training_data()

print(len(training_data))

DATADIR1 = "/content/dataset/test"

CATEGORIES1 = os.listdir(DATADIR1)

for category in CATEGORIES1: 
    path = os.path.join(DATADIR1,category)  
    for img in os.listdir(path):  
        img_array = cv2.imread(os.path.join(path,img))  # convert to array
        plt.imshow(img_array)  # graph it
        plt.show() 
        break  
    break

test_data = []
IMG_SIZE=224.00

def create_test_data():
    for category in CATEGORIES1:  

        path = os.path.join(DATADIR1,category)  
        class_num = CATEGORIES1.index(category)  

        for img in tqdm(os.listdir(path)):  
            try:
                img_array = cv2.imread(os.path.join(path,img))  
                test_data.append([img_array, class_num])  
            except Exception as e:  
                pass
            except OSError as e:
                print("OSErrroBad img most likely", e, os.path.join(path,img))
            except Exception as e:
                print("general exception", e, os.path.join(path,img))

create_test_data()


import random

random.shuffle(training_data)
random.shuffle(test_data)

import keras
num_classes=185
x_train = []
y_train = []
IMG_SIZE=224

for features,label in training_data:
    x_train.append(features)
    y_train.append(label)



print(np.shape(x_train[0].reshape(-1, IMG_SIZE, IMG_SIZE, 3)))

x_train = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
x_train=np.asarray(x_train)
y_train = keras.utils.to_categorical(y_train, num_classes)

np.shape(y_train)

x_test = []
y_test = []
IMG_SIZE=224

for features,label in test_data:
    x_test.append(features)
    y_test.append(label)
print(np.shape(x_test[0].reshape(-1, IMG_SIZE, IMG_SIZE, 3)))

x_test = np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
x_test=np.asarray(x_test)
y_test = keras.utils.to_categorical(y_test, num_classes)
