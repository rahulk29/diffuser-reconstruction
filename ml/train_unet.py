import torch
import torch.nn as nn
import torch.optim as optim

# import lpips

import numpy as np
import os, sys, time, glob
import skimage
import scipy.io
import cv2 as cv

sys.path.append("models/")
import admm_model as admm_model_plain
import learned_prox as learned_prox
import unet as unet_model
from ensemble import *
from utils import *

print("Starting training")

ML_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(ML_DIR, "../../dataset")
CSV_PATH = os.path.join(DATASET_DIR, "dataset_train.csv")
DATA_DIR = os.path.join(DATASET_DIR, "diffuser_images")
LABEL_DIR = os.path.join(DATASET_DIR, "ground_truth_lensed")

my_device = "cpu"
num_epochs = 2
num_print = 5
num_save = 1000

trainset = DiffuserDataset_preprocessed(
    CSV_PATH, DATA_DIR, LABEL_DIR, None, transform=ToTensor()
)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)

unet = unet_model.UNet270480((3, 270, 480))

# criterion = lpips.LPIPS()
criterion = nn.MSELoss(size_average=None)

optimizer = optim.Adam(unet.parameters(), lr=1e-3)

sys.stdout.flush()

for epoch in range(num_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data["image"].to(my_device), data["label"].to(my_device)

        # forward + backward + optimize
        outputs = unet(inputs)
        loss = criterion(outputs, labels)

        # zero the parameter gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % num_print == num_print - 1:  # print every 2000 mini-batches
            print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / num_print:.3f}")
            running_loss = 0.0
        if i % num_save == num_save - 1:
            print("Saving current model...")
            MODEL_PATH = os.path.join(ML_DIR, f"saved_models/model_unet_custom_{epoch}_{i}.pt")
            torch.save(unet, MODEL_PATH)
        sys.stdout.flush()


print("Finished training, saving final model")

MODEL_PATH = os.path.join(ML_DIR, f"saved_models/model_unet_custom_final.pt")
torch.save(unet, MODEL_PATH)
