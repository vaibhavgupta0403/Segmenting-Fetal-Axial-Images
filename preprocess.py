import cv2
import os
import shutil
import random
import pandas as pd
import numpy as np
from modules import img_crop, img_augumentation

    
# To save preprocess images
train_img_folder = 'C:/Users/asus/Desktop/CSM-Task2/code/Task - Segmentation/preprocessed_images/'
os.makedirs(train_img_folder,exist_ok=True)

# To save preprocess labels
train_label_folder = 'C:/Users/asus/Desktop/CSM-Task2/code/Task - Segmentation/preprocessed_labels/'
os.makedirs(train_label_folder,exist_ok=True)


all_training_set_folder = 'C:/Users/asus/Desktop/CSM-Task2/code/Task - Segmentation/images'
# Croping images and labels in training_set to（512，768）, then divide them to images and labels
dirs = os.listdir(all_training_set_folder)

for i in range(len(dirs)-1):
    print("Dividing images and labels: %d / %d" % (i+1,len(dirs)))
    img_name = dirs[i]
    img_path = os.path.join(all_training_set_folder, img_name)
    # print(img_path)
    img = cv2.imread(img_path,0)
    crop_img = img_crop(img)

    # Save images
    if img_name[-14:] == 'Annotation.png':
        save_path = os.path.join(train_label_folder,img_name)
        cv2.imwrite(save_path, crop_img)
    else:
        save_path = train_img_folder+img_name
        cv2.imwrite(save_path, crop_img)
        

# Fill ellipse to create segmentation mask for model training

dirs = os.listdir(train_label_folder)
for i in range(len(dirs)):
        print('Ellip filling: %d / %d' % (i + 1, len(dirs)))
        img_name = dirs[i]
        img_path = train_label_folder+img_name
        img = cv2.imread(img_path, 0)

        img = 255 - img
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=4)
        ellipse = labels == 2
        ellipse = ellipse * 255
        ellipse = ellipse.astype('uint8')

        # Original label is 3 pixels wide so an expansion process is needed
        kernel = np.ones((3, 3), np.uint8)
        ellipse = cv2.dilate(ellipse, kernel, iterations=1)

        # Save
        save_path = train_label_folder + img_name
        cv2.imwrite(save_path, ellipse)


# Divide data for train and val  with a ratio of 8.5:1.5

val_img_folder = 'C:/Users/asus/Desktop/CSM-Task2/code/Task - Segmentation/val_images'
val_label_folder = 'C:/Users/asus/Desktop/CSM-Task2/code/Task - Segmentation/val_labels'
os.makedirs(val_img_folder,exist_ok=True)
os.makedirs(val_label_folder,exist_ok=True)

dirs = os.listdir(train_img_folder)
l = len(dirs)
val_l= round(0.15*l)
random.shuffle(dirs)

for i in range(val_l):
    print("Dividing the train and val set: %d / %d" % (i+1, val_l))
    img_name = dirs[i]
    label_name = img_name[:-4] + '_Annotation.png'

    img_path = os.path.join(train_img_folder, img_name)
    label_path = os.path.join(train_label_folder, label_name)

    img_save_path = os.path.join(val_img_folder, img_name)
    label_save_path = os.path.join(val_label_folder, label_name)


    img = cv2.imread(img_path, 0)
    label = cv2.imread(label_path, 0)

    cv2.imwrite(img_save_path, img)
    cv2.imwrite(label_save_path, label)

    #Remove validation images and labels from training images and training labels
    os.remove(img_path)
    os.remove(label_path)


# Data augumentation in train set
train_augu_img_folder = 'C:/Users/asus/Desktop/CSM-Task2/code/Task - Segmentation/aug_images'
train_augu_label_folder = 'C:/Users/asus/Desktop/CSM-Task2/code/Task - Segmentation/augu_labels/'
os.makedirs(train_augu_img_folder,exist_ok=True)
os.makedirs(train_augu_label_folder,exist_ok=True)


# Image augumentation
img_augumentation(train_img_folder,train_augu_img_folder,data="image")

# Label augumentation
img_augumentation(train_label_folder, train_augu_label_folder, data="label")

