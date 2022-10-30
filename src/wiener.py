from skimage import color, data, restoration
from utils import load_psf, load_diffuser_image, display_array

img = load_diffuser_image(10)
psf = load_psf()
deconvolved_img = restoration.wiener(img, psf, 0.1)
display_array(deconvolved_img)
