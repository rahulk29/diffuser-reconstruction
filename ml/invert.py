import os
import numpy as np
import torch
import glob
import sys
import utils
import matplotlib.pyplot as plt
sys.path.append('models/')

num_save = 1000
PSF_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inverted_psfs/psf_final.pt')

def invert_psf():
    img_index = 3
    device = 'cpu'

    file_path_diffuser = 'sample_images/diffuser/'
    file_path_lensed = 'sample_images/lensed/'


    files = glob.glob(file_path_diffuser + '/*.npy')

    image_np = np.load(file_path_diffuser + files[img_index].split('/')[-1]).transpose((2, 0, 1))

    img = torch.tensor(image_np).unsqueeze(0)


    #forward = torch.load('saved_models/model_le_admm.pt', map_location=device)
    #forward.cuda_device = device
    forward = torch.load('saved_models/model_unet.pt', map_location=device)

    x = torch.rand_like(img, requires_grad=True)
    x = x.to(device)
    optimizer = torch.optim.Adam([x])
    loss = torch.nn.MSELoss()

    y = torch.zeros_like(img) + 0.005
    _, c, h, w = y.shape
    cy, cx = h//2, w//2
    y[:, :, cy-3:cy+3, cx-3:cx+3] = 1
    y = y.to(device)

    iters = 100000
    with torch.autograd.detect_anomaly():
        for i in range(iters):
            optimizer.zero_grad()
            cost = loss(forward(x), y)
            cost.backward()
            optimizer.step()
            print(f'Completed iteration {i+1}/{iters}')

            if i % num_save == num_save - 1:
                print('Saving current image...')
                intermediate_psf_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'inverted_psfs/psf_{i}.pt')
                torch.save(x, intermediate_psf_file_path)

    torch.save(x, PSF_FILE_PATH)

def plot():
    x = torch.load(PSF_FILE_PATH)
    plt.imshow(utils.preplot(x[0].detach().numpy()))
    plt.show()


if __name__ == "__main__":
    invert_psf()
    plot()
