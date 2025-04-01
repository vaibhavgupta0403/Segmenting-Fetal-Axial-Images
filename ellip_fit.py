"""
This script is used for ellipse fitting from edge images.
Requirement: postprocess.py has been executed so edge images exist.
"""
import pandas as pd
import os
import cv2
from modules import ellip_fit
import numpy as np

# Postprocess results folder
edge_folder = 'C:/Users/asus/Desktop/CSM-Task2/code/Task - Segmentation/predictions_edge'

# To save ellipse parameters
results = []
name = ['filename', 'center_x(pixel)', 'center_y(pixel)', 'semi_axes_a(pixel)',
        'semi_axes_b(pixel)', 'HC(pixel)', 'angle(rad)']

# Filename of ellipse parameters file
save_ellip_para_file = 'C:/Users/asus/Desktop/CSM-Task2/code/ellip_params.csv'
# os.makedirs(save_ellip_para_file,exist_ok=True)
# upsample factor
u = 16

# Ellipse fitting to obtain parameters
dirs = os.listdir(edge_folder)
for i in range(len(dirs)):
   print('Ellip fitting: Image = %d / %d' % (i + 1, len(dirs)))
   img_name = dirs[i]
   img_path = os.path.join(edge_folder,img_name)

   edge_img = cv2.imread(img_path, 0)
   result = ellip_fit(edge_img)
   if result is None:
        print(f"No edge points detected in {img_name}. Cannot fit an ellipse.")
        continue
   else:
    xc, yc, theta, a, b = result

    # Restore to original size with the factor 'u'
    xc = (xc + 0.5) * u - 0.5
    yc = (yc + 0.5) * u - 0.5
    a = a * u
    b = b * u
    hc = 2 * np.pi * b + 4 * (a - b)  # HC

    results.append([img_name, xc, yc, a, b, hc, theta])

# Save
predict_results = pd.DataFrame(columns=name, data=results)
predict_results.to_csv(save_ellip_para_file, index=False)

