
# Train CSM model on training set and save.
from modules import CSM, train_model,Dataset_label
from torch.utils.data import DataLoader

input_folder = 'C:/Users/asus/Desktop/CSM-Task2/code/Task - Segmentation/preprocessed_images'
label_folder = 'C:/Users/asus/Desktop/CSM-Task2/code/Task - Segmentation/preprocessed_labels'

#  Training data
train_data = Dataset_label(input_folder,label_folder,[192,128])
dataloader = DataLoader(dataset=train_data, batch_size=8, shuffle=True,pin_memory=False,num_workers=0,prefetch_factor=2)

# CSM model
net = CSM()

# File to save the trained model
save_model_name='C:/Users/asus/Desktop/CSM-Task2/code/test_model.pth'

# Model training
train_model(model=net,dataloader=dataloader,epoches=20,lr=0.001,device='cpu',save_model_name=save_model_name)









