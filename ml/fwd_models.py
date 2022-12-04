import numpy as np 
import torch
import skimage
from utils import *
import torch.nn.functional as F
from admm_helper_functions_torch import *
from train import *

def conv_fwd(img, h, H):
    y = crop(Hfor(pad_dim2(None, img), H), h)
    return y

def Hfor(x, H):
    xc = torch.stack((x, torch.zeros_like(x, dtype=torch.float32)), -1)
    #X = torch.fft(batch_ifftshift2d(xc),2)
    X = torch.fft(xc,2)
    HX = complex_multiplication(H,X)
    out = torch.ifft(HX,2)
    out_r, _ = torch.unbind(out,-1)
    return out_r

def pad_dim2(model, inputs):
    return inputs

def crop(x, h):
    dims0, dims1 = h.shape[0], h.shape[1]
    pad0, pad1 = int((dims0)//2), int((dims1)//2)
    
    C01 = pad0; C02 = pad0 + dims0
    C11 = pad1; C12 = pad1 + dims1
    return x[:, :, C01:C02, C11:C12]

def load_psf():
    ds = 1
    psf_diffuser = load_psf_image(PSF_PATH, downsample=1, rgb=False)
    psf_diffuser = np.sum(psf_diffuser, axis=2)


    h = skimage.transform.resize(psf_diffuser, 
                                 (psf_diffuser.shape[0]//ds,psf_diffuser.shape[1]//ds), 
                                                              mode='constant', anti_aliasing=True)
    return h

def pad_zeros_torch(x, h):
    dims0, dims1 = h.shape[0], h.shape[1]
    pad0, pad1 = int((dims0)//2), int((dims1)//2)
    PADDING = (pad1, pad1, pad0, pad0)
    return F.pad(x, PADDING, 'constant', 0)

if __name__ == "__main__":
    device = 'cpu'
    # Load PSF
    h = load_psf()
    h_var = torch.nn.Parameter(torch.tensor(h, dtype=torch.float32, device=device),
                                        requires_grad=False)
    
    dims0, dims1 = h.shape[0], h.shape[1]
    h_zeros = torch.nn.Parameter(torch.zeros(dims0*2, dims1*2, dtype=torch.float32, device=device),
                                      requires_grad=False)

    
    h_complex = torch.stack((pad_zeros_torch(h_var, h), h_zeros),2).unsqueeze(0)
    
    H = torch.fft(batch_ifftshift2d(h_complex).squeeze(), 2)   

    trainset = DiffuserDataset_preprocessed(CSV_PATH, DATA_DIR, LABEL_DIR, None, transform=ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = 1, shuffle=True)

    for i, data in enumerate(trainloader):
        print("hi")
