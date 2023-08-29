# SOFTWARE & COMPUTING for NUCLEAR and SUBNUCLEAR PHYSICS project
# 1. INTRODUCTION
This project is part of the workflow of the track reconstruction in the GRAIN detector of the SAND calorimeter in the DUNE experiment. Its aim consists in classifying two types of images, observing the state of a single camera: if it is “blinded”, i.e., it has been reached by too many photons, or not.

The GRAIN detector, filled with ~ 1 ton of Liquid Argon, is covered in total by 54 pixel-segmented photodetectors (the so-called cameras). They aim to detect scintillation photons produced by the de-excitation of Ar atoms after interacting with charged particles. Capturing these photons, it should be possible to reconstruct the track of the charged particles in LAr, as is done in classic bubble chambers.

The approach chosen in this detector is the Coded Aperture Imaging technique, which requires a mask in front of each camera. The images formed on the camera are convolutions of images from each hole. So, if the photons are produced before the mask, then we will see a pattern similar to the mask pattern on the camera; otherwise, if the photons are produced between the mask and the camera, we will see an accumulation of photons in a point and the photons count will be much larger than the previous case. These are the so-called "blinded cameras".

For the reconstruction task, the blinded cameras are unuseful, since don't give us information about the passage of the particle. For this reason, this project aims to classify the cameras, distinguishing the good ones from the blinded ones and allowing us to discard the latter ones. To accomplish this task, a CNN written in Python was used. The code has been uploaded in this repo, moduled in 5 different files to be clearer: import_drdf.py, functions.py, scaling.py, root_file.py, cnn_model.py.

# 2. DATASET
# 2.1 Simulated Data
In this code, simulated data are used since the experiment is still being built. They are in _.drdf_ format, created by the researchers of the DUNE group. Data are stored in the _"response.drdf"_ file. They are organized as a list, where each element is composed of 2 objects: the number of the event and a dictionary. The dictionary gives us information on the photons arrived on the camera: the keys are the names of the cameras and the values are matrices where each element represents the number of photons arrived in a pixel. This file contains 1000 events (i.e. charged particle interaction in the detector with photon production), the cameras are 54 and they have 31x31 pixels. The code for the import of these data is in **import_drdf.py**. 
Below two typical images of the file are reported: the right one is a normal camera and the left one is a blinded camera. The functions for the plotting are defined in **functions.py**.

![ev_0_cam_1](https://github.com/giacomo-santoni/SC-project/assets/133137485/25a9b943-60e5-4cca-9ec6-d2557ce180a6)                                                                  ![blindcam](https://github.com/giacomo-santoni/SC-project/assets/133137485/eab6400d-084f-4fa2-915d-9771940680f2)

Thus, these data are stored in a matrix of 54 columns(cameras) x 1000 rows(events). This matrix has been flattened, obtaining an array of 54000 elements, where each element is a matrix representing a single camera 31x31. 

# 2.2 True Data
Together with the simulated file _"response.drdf"_, there is the file _"sensors.root"_ with the "truth" of data, that will represent the labels for CNN training. It is a _ROOT_ file: each camera is a ROOT Tree, and each Tree has some variables, organized in TLeaves. The variable of our interest is only _innerPhotons_, which tells us how many photons are produced between the mask and the camera and if the camera is blinded or not. These data are imported into the code in **root_file.py**. In their original format, they look like this: 

![innerPhotons](https://github.com/giacomo-santoni/SC-project/assets/133137485/710f0478-5db0-4ffb-9b0c-ce0a4574870b)

So, in order to handle these data, they have been reorganized in a matrix of 1000 columns(events) x 54 rows(cameras). In order to be consistent with the simulated dataset, the matrix has been transposed and flattened, obtaining an array of 54000 elements, where each element is the number of inner photons. Lastly, since these data have to represent only the state of the camera, i.e. blinded/not blinded, a 1 was assigned to the element if the number of inner photons is larger than 5, meaning blinded camera; a 0 otherwise. So at the end, an array of 0 and 1 was obtained, that can tell the truth about the state of the camera.

# 3.DATA RESCALING

