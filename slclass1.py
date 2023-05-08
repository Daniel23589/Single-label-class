import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("MAFood121"))

import sys
import cv2
import torch
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn as nn
from torchvision import datasets, transforms
from glob import glob
import matplotlib.pyplot as plt
import torchvision.models as models
from torchvision.models import ResNet50_Weights

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(42) # try and make the results more reproducible
BASE_PATH = 'MAFood121/'

epochs = 35
batch_size = 64
MICRO_DATA = True # very small subset (just 3 groups)
SAMPLE_TRAINING = False # make train set smaller for faster iteration
IMG_SIZE = (384, 384) # Try to change the model to U-net to avoid the resizing

#Classes
f = open(BASE_PATH + '/annotations/dishes.txt', "r")
classes = f.read().split('\n')
f.close()

print("***** classes = dishes.txt: ***** " + str(classes))
print("#######################################################################################")

new_classes = []
for arr in classes:
    arr = arr.split(",")
    new_classes.append(arr)
print(new_classes)

from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()

#train
f = open(BASE_PATH + '/annotations/train.txt', "r")
train_images = f.read().split('\n')
f.close()
f = open(BASE_PATH + '/annotations/train_lbls_d.txt', "r")
train_labels = f.read().split('\n')
f.close()

#val
f = open(BASE_PATH + '/annotations/val.txt', "r")
val_images = f.read().split('\n')
f.close()
f = open(BASE_PATH + '/annotations/val_lbls_d.txt', "r")
val_labels = f.read().split('\n')
f.close()

#test
f = open(BASE_PATH + '/annotations/test.txt', "r")
test_images = f.read().split('\n')
f.close()
f = open(BASE_PATH + '/annotations/test_lbls_d.txt', "r")
test_labels = f.read().split('\n')
f.close()

train_images = ["MAFood121/images/" + s  for s in train_images]
all_img_df = pd.DataFrame({'path': train_images, 'class_id': train_labels})

val_images = ["MAFood121/images/" + s  for s in val_images]
val_img_df = pd.DataFrame({'path': val_images, 'class_id': val_labels})

test_images = ["MAFood121/images/" + s  for s in test_images]
test_img_df = pd.DataFrame({'path': test_images, 'class_id': test_labels})

all_img_df = all_img_df[:-1]
val_img_df = val_img_df[:-1]
test_img_df = test_img_df[:-1]

all_img_df['class_name'] = all_img_df['path'].map(lambda x: os.path.split(os.path.dirname(x))[-1])
print(all_img_df)
print("-------------------------------------------------------------------------------------------------")

val_img_df['class_name'] = val_img_df['path'].map(lambda x: os.path.split(os.path.dirname(x))[-1])
print(val_img_df)
print("-------------------------------------------------------------------------------------------------")

test_img_df['class_name'] = test_img_df['path'].map(lambda x: os.path.split(os.path.dirname(x))[-1])
print(test_img_df)

from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()

#Dataframe for train images

train_classes = []
train_classid = []
with open(BASE_PATH + '/annotations/train_lbls_d.txt') as f1:
    for line in f1:
       idx_classes = []
       classid = int(line)
       train_classid.append(classid)
       for ing in classes[classid].strip().split(","):
           idx_classes.append(str(classes.index(ing)))
       train_classes.append(idx_classes)
df_train = pd.DataFrame(mlb.fit_transform(train_classes),columns=mlb.classes_) #binary encode classes
df_train["path"] = all_img_df['path']
df_train["class_id"] = train_classid 
food_dict_train = df_train
print(df_train)

new_data = []
for index, row in all_img_df.iterrows():
    #get binary encoding ingredients from lookup
    food = row["class_name"]
    path = row["path"]
    class_id = row["class_id"]
    
    binary_encod = food_dict_train.loc[food_dict_train["path"] == path]
    new_data.append(np.array(binary_encod)[0])
    
col_names = list(binary_encod.columns.values)
train_df = pd.DataFrame(new_data, columns = col_names)

#Dataframe for val images

val_classes = []
val_classid = []
with open(BASE_PATH + '/annotations/val_lbls_d.txt') as f1:
    for line in f1:
       idx_classes = []
       classid = int(line)
       val_classid.append(classid)
       for ing in classes[classid].strip().split(","):
           idx_classes.append(str(classes.index(ing)))
       val_classes.append(idx_classes)
df_val = pd.DataFrame(mlb.fit_transform(val_classes),columns=mlb.classes_) #binary encode classes
df_val["path"] = val_img_df['path']
df_val["class_id"] = val_classid 
food_dict_val = df_val
print(df_val)

val_data = []
for index, row in val_img_df.iterrows():
    #get binary encoding ingredients from lookup
    food = row["class_name"]
    path = row["path"]
    class_id = row["class_id"]
    binary_encod = food_dict_val.loc[food_dict_val["path"] == path]
    val_data.append(np.array(binary_encod)[0])

col_names = list(binary_encod.columns.values)
val_df = pd.DataFrame(val_data, columns = col_names)

#Dataframe for test images

test_classes = []
test_classid = []
with open(BASE_PATH + '/annotations/test_lbls_d.txt') as f1:
    for line in f1:
       idx_classes = []
       classid = int(line)
       test_classid.append(classid)
       for ing in classes[classid].strip().split(","):
           idx_classes.append(str(classes.index(ing)))
       test_classes.append(idx_classes)
df_test = pd.DataFrame(mlb.fit_transform(test_classes),columns=mlb.classes_) #binary encode classes
df_test["path"] = test_img_df['path']
df_test["class_id"] = test_classid 
food_dict_test = df_test
print(df_test)

test_data = []
for index, row in test_img_df.iterrows():
    #get binary encoding ingredients from lookup
    food = row["class_name"]
    path = row["path"]
    class_id = row["class_id"]
    binary_encod = food_dict_test.loc[food_dict_test["path"] == path]
    test_data.append(np.array(binary_encod)[0])

col_names = list(binary_encod.columns.values)
test_df = pd.DataFrame(test_data, columns = col_names)

train_df.to_hdf('trainSL_df.h5','df',mode='w',format='table',data_columns=True)
val_df.to_hdf('valSL_df.h5','df',mode='w',format='table',data_columns=True)
test_df.to_hdf('testSL_df.h5','df',mode='w',format='table',data_columns=True)
