# MCam Reconstruction Goals

## Performance

Ideas for improving performance:
* Parallelize across multiple cores or machines
* Block-wise reconstruction (may not be possible)
* Run algorithms in a faster language (C,C++,Rust)

## Calibration

The reconstruction algorithm should consider the relative positions
of cameras in the array. It should not assume that cameras are evenly
spaced, perfectly aligned, etc.

There should be a calibration step, where we test the fully-built MCam
to measure camera positions and orientations. This calibration can be done
by sweeping a point source in front of the MCam.

Calibration should also account for the fact that different cameras in the array
may have different PSFs.

## Iterative Optimization

To enable the MCam to be used interactively, image reconstruction should be
performed iteratively. This allows lower-quality images to be previewed
while a user is positioning the camera, and a final high-quality image
can then be generated offline.

Iterative optimization ideas:
* Image and PSF downsampling
* Iterative optimization algorithms
* Making simplifying assumptions, such as:
  * Assuming all cameras have the same PSF
  * Assuming cameras are grid-aligned and evenly spaced
  * Only reconstructing one color channel

## PSF-Free Reconstruction

Use machine learning methods to train a neural network for reconstructing
diffuser cam images.

Then use another machine learning method to search for an input image
that results in an output image that is approximately a single point.
This input image is the PSF.

This implicit PSF measurement eliminates the need for sensitive, error-prone,
and time consuming manual measurements of the PSF in an optics lab.

We can then use the learned PSF in classical algorithms.

We will apply machine learning and classical methods to MCam.
