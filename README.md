# SC-project
# 1. INTRODUCTION
This project is part of the workflow of the track reconstruction in the GRAIN detector of the SAND calorimeter in the DUNE experiment. Its aim consists in classifying two types of images, observing the state of a single camera: if it is “blinded”, i.e., it has been reached by too many photons, or not.

The GRAIN detector, filled with ~ 1 ton of Liquid Argon, is covered in total by 54 pixel-segmented photodetectors (the so-called cameras). They aim to detect scintillation photons produced by the de-excitation of Ar atoms after interacting with charged particles. Capturing these photons, it should be possible to reconstruct the track of the charged particles in LAr, as is done in classic bubble chambers.

The approach chosen in this detector is the Coded Aperture Imaging technique, which requires a mask in front of each camera. The images formed on the camera are convolutions of images from each hole. So, if the photons are produced before the mask, then we will see a pattern similar to the mask pattern on the camera; otherwise, if the photons are produced between the mask and the camera, we will see an accumulation of photons in a point and the photons count will be much larger than the previous case. These are the so-called "blinded cameras".

For the reconstruction task, the blinded cameras are unuseful, since don't give us information about the passage of the particle. For this reason, this project aims to classify the cameras, distinguishing the good ones from the blinded ones and allowing us to discard the latter ones.

# 2. DATASET
# 2.1 Simulated Data
The data used in this code are simulated since the experiment is still being built. They are in \textit{.drdf} format, created by the researchers of the DUNE group. Data are stored in the "response.drdf" file. They are organized as a list, where each element contains the number of the event and a dictionary, where the keys are the names of the cameras and the values are matrices where each element represents the number of photons arrived in that pixel.
