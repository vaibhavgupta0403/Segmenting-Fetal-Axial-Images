"""
Prediction on validation set.
"""
import torch
import os
import shutil
from modules import CSM, predict

# Validation images folder
input_folder = 'C:/Users/asus/Desktop/CSM-Task2/code/Task - Segmentation/val_images'

# Prediction folder for results saving
predict_folder = 'C:/Users/asus/Desktop/CSM-Task2/code/Task - Segmentation/predictions/'
os.makedirs(predict_folder,exist_ok=True)

# Load the network
net_dict_file = 'C:/Users/asus/Desktop/CSM-Task2/code/test_model.pth'
net = CSM()
net.load_state_dict(torch.load(net_dict_file))

# Predict
predict(net,input_folder,predict_folder,device='cpu')









