import os
import numpy as np
import torch
import glob
import sys
import utils
import matplotlib.pyplot as plt
import pathlib
import argparse
from admm_helper_functions_torch import TVnorm_tf

sys.path.append("models/")

# How often to save intermediate PSFs
num_save = 1000


def invert_psf(regularizer, iters, lamb=0.001):
    device = "cpu"

    print(f'Inverting PSF with regularizer {regularizer}, lambda = {lamb:.3e}, iters = {iters}, device = {device}')

    file_path_diffuser = "sample_images/diffuser/"
    file_path_lensed = "sample_images/lensed/"
    save_dir = get_savedir(regularizer, lamb)
    # Only used for referencing expected image sizes
    img_index = 3

    files = glob.glob(file_path_diffuser + "/*.npy")

    image_np = np.load(file_path_diffuser + files[img_index].split("/")[-1]).transpose(
        (2, 0, 1)
    )

    img = torch.tensor(image_np).unsqueeze(0)

    # forward = torch.load('saved_models/model_le_admm.pt', map_location=device)
    # forward.cuda_device = device
    forward = torch.load("saved_models/model_unet.pt", map_location=device)

    x = torch.rand_like(img, requires_grad=True)
    x = x.to(device)
    optimizer = torch.optim.Adam([x])
    loss = torch.nn.MSELoss()

    y = torch.zeros_like(img) + 0.005
    _, c, h, w = y.shape
    cy, cx = h // 2, w // 2
    y[:, :, cy - 3 : cy + 3, cx - 3 : cx + 3] = 1
    y = y.to(device)

    if regularizer is None or regularizer == "NONE":
        reg_fn = lambda x: 0.0
    elif regularizer == "TV":
        reg_fn = TVnorm_tf
    elif regularizer == "L0":
        reg_fn = lambda x: torch.norm(x, p=0)
    elif regularizer == "L1":
        reg_fn = lambda x: torch.norm(x, p=1)
    elif regularizer == "L2":
        reg_fn = lambda x: torch.norm(x, p=2)

    print("Starting iteration")
    sys.stdout.flush()
    with torch.autograd.detect_anomaly():
        for i in range(iters):
            optimizer.zero_grad()
            cost = loss(forward(x), y) + lamb * reg_fn(x)
            cost.backward()
            optimizer.step()
            print(f"Completed iteration {i+1}/{iters}")

            if i % num_save == num_save - 1:
                print("Saving current image...")
                intermediate_psf_file_path = os.path.join(save_dir, f"psf_{i+1}.pt")
                torch.save(x, intermediate_psf_file_path)
            sys.stdout.flush()

    print("Finished iterations, saving final image")
    torch.save(x, PSF_FILE_PATH)


def plot():
    x = torch.load(PSF_FILE_PATH)
    plt.imshow(utils.preplot(x[0].detach().numpy()))
    plt.show()


def test_psf(regularizer, normalize=True, lamb=None, epoch=None, iters=None):
    device = "cpu"
    if iters is None:
        filename = "psf.pt"
    else:
        filename = f"psf_{epoch}_{iters}.pt"

    path = os.path.join(get_savedir(regularizer, lamb), filename)
    x = torch.load(path).to(device)
    forward = torch.load("saved_models/model_unet.pt", map_location=device)

    out = forward(x)[0].detach().numpy()
    x = x[0].detach().numpy()

    if normalize:
        x = x / np.max(x)

    print(x)

    plt.imshow(utils.preplot(out))
    plt.title("Forward model applied to PSF")
    plt.show()
    plt.imshow(utils.preplot(x))
    plt.title("Raw PSF")
    plt.show()


def get_savedir(regularizer, lamb):
    if lamb is None:
        SAVE_DIR = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), f"experiments/inverted_psfs_reg_{regularizer}/"
        )
    else:
        SAVE_DIR = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), f"experiments/inverted_psfs_reg_{regularizer}_{lamb:.3e}/"
        )
    pathlib.Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    return SAVE_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="invert.py", description="Invert PSFs given a machine learning model"
    )
    parser.add_argument(
        "-r", "--regularizer", choices=["NONE", "TV", "L0", "L1", "L2"], default="NONE"
    )
    parser.add_argument("-n", "--num-iters", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lamb", type=float, default=0.001)
    parser.add_argument("-i", "--invert", action="store_true")
    parser.add_argument("-t", "--test", action="store_true")

    args = parser.parse_args()

    if args.invert:
        invert_psf(args.regularizer, args.num_iters, lamb=args.lamb)
    if args.test:
        test_psf(args.regularizer, epoch=args.epochs, iters=args.num_iters)
    if not (args.test or args.invert):
        print(
            "No work done. Specify --invert or --test to invert or test a PSF, respectively."
        )
