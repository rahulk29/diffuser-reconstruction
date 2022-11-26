import matplotlib.pyplot as plt

import numpy as np
import os, sys, time, glob
import skimage
import scipy.io
import cv2 as cv

sys.path.append('models/')
from utils import *

my_device = 'cpu'

le_admm_u = torch.load('saved_models/model_le_admm_u_custom_final.pt', map_location=my_device)
le_admm_u.admm_model.cuda_device = my_device

img_index = 7

file_path_diffuser = 'sample_images/diffuser/'
file_path_lensed = 'sample_images/lensed/'

files = glob.glob(file_path_diffuser + '/*.npy')


image_np = np.load(file_path_diffuser + files[img_index].split('/')[-1]).transpose((2, 0, 1))
label_np = np.load(file_path_lensed + files[img_index].split('/')[-1]).transpose((2, 0, 1))

image = torch.tensor(image_np).unsqueeze(0)
label = torch.tensor(label_np).unsqueeze(0)

# Plot sample image: 
fig1, ax = plt.subplots(1,2,figsize=(15,5))
ax[0].imshow(preplot(image_np)/np.max(image_np)); 
ax[1].imshow(preplot(label_np)); 
ax[0].set_xticks([]); ax[0].set_yticks([]); 
ax[1].set_xticks([]); ax[1].set_yticks([]); 
ax[0].set_title('Lensless Image');  
ax[1].set_title('Lensed Image');

with torch.no_grad():
    inputs = image.to(my_device)
    
    output_le_admm_u = le_admm_u(inputs)

fig1, ax = plt.subplots(1,1,figsize=(15,5))

ax.imshow(preplot(output_le_admm_u[0].cpu().detach().numpy()))

ax.set_xticks([]); ax.set_yticks([]); 
    
ax.set_title('Le-ADMM-U');

plt.show()
