import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import os, sys, time, glob
import skimage
import scipy.io
import cv2 as cv
import argparse
# import lpips

sys.path.append('models/')

import admm_model as admm_model_plain
import learned_prox as learned_prox
import unet as unet_model
from ensemble import *
from utils import *

ML_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(ML_DIR, "../../dataset")
CSV_PATH = os.path.join(DATASET_DIR, "dataset_train.csv")
DATA_DIR = os.path.join(DATASET_DIR, "diffuser_images")
LABEL_DIR = os.path.join(DATASET_DIR, "ground_truth_lensed")
PSF_PATH = os.path.join(ML_DIR, "sample_images", "psf.tiff")

PRINT_ITERS = 5
DOWNSAMPLING = 4 # Must be set to 4 to use dataset images 

def load_psf():
    psf_diffuser = load_psf_image(PSF_PATH, downsample=1, rgb=False)
    psf_diffuser = np.sum(psf_diffuser, axis=2)


    h = skimage.transform.resize(psf_diffuser, 
                                 (psf_diffuser.shape[0]//ds,psf_diffuser.shape[1]//ds), 
                                                              mode='constant', anti_aliasing=True)
    return h

def le_admm_u(fill_factor, nx, ny, device):
    le_admm_u_admm = admm_model_plain.ADMM_Net(batch_size = 1, h = h, iterations = 5, 
                               fill_factor = fill_factor, nx = nx, ny = ny, learning_options = learning_options_admm, cuda_device = device)

    le_admm_u_unet = unet_model.UNet270480((3,270,480))

    return MyEnsemble(le_admm_u_admm, le_admm_u_unet)

def unet(fill_factor, nx, ny):
    return unet_model.Unet270480((3, 270, 480))

def get_prefix(args):
    return f"model_custom_{args.model}_ff{args.fill_factor}_nx{args.nx}_ny{args.ny}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="train.py", description="Train models for reconstructing diffuser images"
    )
    parser.add_argument(
        "-m", "--model", choices=["le-admm-u", "unet"], default="le-admm-u"
    )
    parser.add_argument(
        "-d", "--device", choices=["cpu"], default="cpu"
    )
    parser.add_argument("-e", "--epochs", type=int, default=2)
    parser.add_argument("-l", "--learning-rate", type=float, default=1e-3)
    parser.add_argument("-f", "--fill-factor", type=float, default=1, help="Fill factor of sensor array")
    parser.add_argument("--nx", type=int, default=1, help="Width of sensor array")
    parser.add_argument("--ny", type=int, default=1, help="Height of sensor array")
    parser.add_argument("-s", "--save-iters", type=int, default=5000, help="Number of iterations between saving intermediate models")

    args = parser.parse_args()

    prefix = get_prefix(args)

    print(f"Starting training for model {prefix}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Fill factor: {args.fill_factor}")
    print(f"Width of sensor array: {args.nx}")
    print(f"Height of sensor array: {args.ny}")

    var_options = {'plain_admm': [],
                   'mu_and_tau': ['mus', 'tau'],
                  }

    trainset = DiffuserDataset_preprocessed(CSV_PATH, DATA_DIR, LABEL_DIR, None, transform=ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = 1, shuffle=True)

    learning_options_admm = {'learned_vars': var_options['mu_and_tau']} 

    # Load PSF
    h = load_psf()

    if args.model == "le-admm-u":
        model = le_admm_u(args.fill_factor, args.nx, args.ny, args.device)
    else:
        model = unet(args.fill_factor, args.nx, args.ny)

    # criterion = lpips.LPIPS()
    criterion = nn.MSELoss(size_average=None)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    sys.stdout.flush()

    for epoch in range(args.epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data['image'].to(my_device), data['label'].to(my_device)

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % num_print == num_print - 1:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / num_print:.3f}')
                running_loss = 0.0
            if i % num_save == num_save - 1:
                print('Saving current model...')
                MODEL_PATH = os.path.join(ML_DIR, f"saved_models/{prefix}_{i}.pt")
                torch.save(model, MODEL_PATH)
            sys.stdout.flush()


    print('Finished training, saving final model')

    MODEL_PATH = os.path.join(ML_DIR, f"saved_models/{prefix}_final.pt")
    torch.save(model, MODEL_PATH)
