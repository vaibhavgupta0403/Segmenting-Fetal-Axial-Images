"""
This script is used for extracting edge images from prediction results.
Requirement: predict.py has been executed so prediction results exist.
"""
import os
import cv2
from modules import mcc_edge
import shutil

# Folder of model predictions
input_folder = 'C:/Users/asus/Desktop/CSM-Task2/code/Task - Segmentation/predictions'

#  Folder to save postprocess results
edge_folder = 'C:/Users/asus/Desktop/CSM-Task2/code/Task - Segmentation/predictions_edge/'
os.makedirs(edge_folder,exist_ok=True)

# Extract fetal contour
dirs = os.listdir(input_folder)
for i in range(len(dirs)):
    print('Extracting max connect component edge: Image = %d / %d' % (i + 1, len(dirs)))
    img_name = dirs[i]
    img_path = os.path.join(input_folder,img_name)

    img = cv2.imread(img_path, 0)
    edge_img = mcc_edge(img)

    save_path = os.path.join(edge_folder,img_name)
    cv2.imwrite(save_path, edge_img)

