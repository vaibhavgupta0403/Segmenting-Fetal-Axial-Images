import pandas as pd
import os
import cv2
from modules import ellip_fit
import numpy as np

# Postprocess results folder
edge_folder = 'C:/Users/asus/Desktop/CSM-Task2/code/Task - Segmentation/predictions_edge'

# To save ellipse parameters
results = []
name = ['image_name', 'ofd_1_x', 'ofd_1_y', 'ofd_2_x', 'ofd_2_y', 'bpd_1_x', 'bpd_1_y', 'bpd_2_x', 'bpd_2_y']

# Filename of ellipse parameters file
save_ellip_para_file = 'C:/Users/asus/Desktop/CSM-Task2/code/landmarks.csv'
# os.makedirs(save_ellip_para_file,exist_ok=True)
# upsample factor
u = 16

# Ellipse fitting to obtain parameters
dirs = os.listdir(edge_folder)
for i in range(len(dirs)):
    print('Ellip fitting: Image = %d / %d' % (i + 1, len(dirs)))
    img_name = dirs[i]
    img_path = os.path.join(edge_folder, img_name)

    edge_img = cv2.imread(img_path, 0)
    result = ellip_fit(edge_img)
    if result is None:
        print(f"No edge points detected in {img_name}. Cannot fit an ellipse.")
        continue
    else:
        xc, yc, theta, a, b = result

        # Ensure a is the longer axis (BPD) and b is the shorter axis (OFD)
        if a < b:
            a, b = b, a  # Swap if necessary

        # Restore to original size with the factor 'u'
        xc = (xc + 0.5) * u - 0.5
        yc = (yc + 0.5) * u - 0.5
        bpd = a * u  # Biparietal diameter
        ofd = b * u  # Occipitofrontal diameter

        # Calculate landmark coordinates
        theta_rad = np.deg2rad(theta)  # Convert theta to radians

        # OFD landmarks
        ofd_1_x = xc + ofd * np.cos(theta_rad)
        ofd_1_y = yc + ofd * np.sin(theta_rad)
        ofd_2_x = xc - ofd * np.cos(theta_rad)
        ofd_2_y = yc - ofd * np.sin(theta_rad)

        # BPD landmarks
        bpd_1_x = xc + bpd * np.sin(theta_rad)
        bpd_1_y = yc - bpd * np.cos(theta_rad)
        bpd_2_x = xc - bpd * np.sin(theta_rad)
        bpd_2_y = yc + bpd * np.cos(theta_rad)

        # Round coordinates to nearest integer
        ofd_1_x, ofd_1_y, ofd_2_x, ofd_2_y = round(ofd_1_x), round(ofd_1_y), round(ofd_2_x), round(ofd_2_y)
        bpd_1_x, bpd_1_y, bpd_2_x, bpd_2_y = round(bpd_1_x), round(bpd_1_y), round(bpd_2_x), round(bpd_2_y)

        results.append([img_name, ofd_1_x, ofd_1_y, ofd_2_x, ofd_2_y, bpd_1_x, bpd_1_y, bpd_2_x, bpd_2_y])

# Save
predict_results = pd.DataFrame(columns=name, data=results)
predict_results.to_csv(save_ellip_para_file, index=False)
