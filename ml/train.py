import torch
import torch.nn as nn
import torch.optim as optim
# import lpips

import numpy as np
import os, sys, time, glob
import skimage
import scipy.io
import cv2 as cv

sys.path.append('models/')
import admm_model as admm_model_plain
import learned_prox as learned_prox
import unet as unet_model
from ensemble import *
from utils import *

ML_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(ML_DIR, "../data")
CSV_PATH = os.path.join(DATASET_DIR, "data.csv")
DATA_DIR = os.path.join(DATASET_DIR, "diffuser")
LABEL_DIR = os.path.join(DATASET_DIR, "lensed")
MODEL_PATH = os.path.join(ML_DIR, "saved_models/model_le_admm_u_custom.pt")

my_device = 'cpu'

var_options = {'plain_admm': [],
               'mu_and_tau': ['mus', 'tau'],
              }

num_images = 2
trainset = DiffuserDataset_preprocessed_number(CSV_PATH, DATA_DIR, LABEL_DIR, None, num_images, transform=ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 1, shuffle=True, num_workers=0)

learning_options_admm = {'learned_vars': var_options['mu_and_tau']} 

# Load PSF
path_diffuser = 'sample_images/psf.tiff'
psf_diffuser = load_psf_image(path_diffuser, downsample=1, rgb= False)

ds = 4   # Amount of down-sampling.  Must be set to 4 to use dataset images 

print('The shape of the loaded diffuser is:' + str(psf_diffuser.shape))

psf_diffuser = np.sum(psf_diffuser,2)


h = skimage.transform.resize(psf_diffuser, 
                             (psf_diffuser.shape[0]//ds,psf_diffuser.shape[1]//ds), 
                                                          mode='constant', anti_aliasing=True)

le_admm_u_admm = admm_model_plain.ADMM_Net(batch_size = 1, h = h, iterations = 5, 
                           learning_options = learning_options_admm, cuda_device = my_device)

le_admm_u_unet = unet_model.UNet270480((3,270,480))

le_admm_u2 = MyEnsemble(le_admm_u_admm, le_admm_u_unet)

# criterion = lpips.LPIPS()
criterion = torch.nn.MSELoss(size_average=None)

optimizer = optim.Adam(le_admm_u2.parameters(), lr=0.0001)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data['image'], data['label']

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = le_admm_u2(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished training')

torch.save(le_admm_u2, MODEL_PATH)
