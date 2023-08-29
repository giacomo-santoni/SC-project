# SC-project
# 1. INTRODUCTION
This project is part of the workflow of the track reconstruction in the GRAIN detector of the SAND calorimeter in the DUNE experiment. Its aim consists in classifying two types of images, observing the state of a single camera: if it is “blinded”, i.e., it has been reached by too many photons, or not.

The GRAIN detector, filled with ~ 1 ton of Liquid Argon, is covered in total by 54 pixel-segmented photodetectors (the so-called cameras). They aim to detect scintillation photons produced by the de-excitation of Ar atoms after interacting with charged particles. Capturing these photons, it should be possible to reconstruct the track of the charged particles in LAr, as is done in classic bubble chambers.

The approach chosen in this detector is the Coded Aperture Imaging technique, which requires a mask in front of each camera. The images formed on the camera are convolutions of images from each hole. So, if the photons are produced before the mask, then we will see a pattern similar to the mask pattern on the camera; otherwise, if the photons are produced between the mask and the camera, we will see an accumulation of photons in a point and the photons count will be much larger than the previous case. These are the so-called "blinded cameras".

For the reconstruction task, the blinded cameras are unuseful, since don't give us information about the passage of the particle. For this reason, this project aims to classify the cameras, through a CNN, distinguishing the good ones from the blinded ones and allowing us to discard the latter ones.

# 2. DATASET
# 2.1 Simulated Data
The data used in this code are simulated since the experiment is still being built. They are in **.drdf** format, created by the researchers of the DUNE group. Data are stored in the _"response.drdf"_ file. They are organized as a list, where each element is composed of 2 objects: the number of the event and a dictionary. The dictionary gives us information on the photons arrived on the camera: the keys are the names of the cameras and the values are matrices where each element represents the number of photons arrived in a pixel. This file contains 1000 events (i.e. charged particle interaction in the detector with photon production), the cameras are 54 and they have 31x31 pixels. The code for the import of these data is in _"import_drdf.py"_ file. 
Below two typical images of the file are reported: the right one is a normal camera and the left one is a blinded camera. The functions for the plotting can be found in the _"functions.py"_ file.

![ev_0_cam_1](https://github.com/giacomo-santoni/SC-project/assets/133137485/25a9b943-60e5-4cca-9ec6-d2557ce180a6)                                                                  ![blindcam](https://github.com/giacomo-santoni/SC-project/assets/133137485/eab6400d-084f-4fa2-915d-9771940680f2)

# 2.2 True Data
Together with the simulated file _"response.drdf"_, there is the file _"sensors.root"_ with the "truth" of data, that will represent the labels for CNN training. It is a **ROOT** file: each camera is a ROOT Tree, and each Tree has some variables, organized in TLeaves. The variable of our interest is only _innerPhotons_, which tells us how many photons are produced between the mask and the camera. 



