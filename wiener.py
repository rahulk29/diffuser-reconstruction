from skimage import color, data, restoration

img = load_image()
psf = load_psf()
deconvolved_img = restoration.wiener(img, psf, 0.1)
display_image(deconvolved_img)
