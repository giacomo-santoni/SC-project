# MACHINE LEARNING METHOD for DAZZLED CAMERAS RECOGNITION

> N.B. The instructions to download data are explained in _Dataset_ section.

## Abstract
This project is part of the workflow of the simulation chain for the GRAIN detector of the SAND calorimeter in the DUNE experiment. GRAIN, a LAr detector, detects scintillating photons produced inside the Ar volume using cameras, devices formed by a sensor (matrix of SiPMs) and an Hadamard mask. These photons can be produced outside or inside the camera: depending on the number of photons produced inside, the camera will be defined as dazzled or not-dazzled. Since the dazzled cameras can't be used in the current reconstruction algorithm, a classification that separates these two classes of images is needed.
This project tries to accomplish this task through a convolutional neural network. 

In _Introduction_, I will introduce the experiment in which this project is set. In _Dataset_, I explain the organization of data and the rearrangements done. In the _CNN model and Example_, I present the model and an example of a possible output of the project. Last section is the _Conclusions_.

## 1. INTRODUCTION
### 1.1 Experiment overview
The Deep Underground Neutrino Experiment (DUNE) is a long-baseline neutrino oscillation experiment that is being built in the United States. It will consists of two detectors: a Near Detector, close to Fermilab in Illinois and a Far Detector, in South Dakota, 1300 km away. Below a schematic design of the experiment is presented.
<p align="center">
<img width="758" alt="Screenshot 2023-10-19 alle 09 32 51" src="https://github.com/giacomo-santoni/SC-project/assets/133137485/0a0f477e-2ce3-4b4d-aa48-10de63a9adee">
</p>
<p align="center">
  <em>Schematic design of DUNE experiment.</em>
</p>

Basically, the motivation for the design of this experiment is a more precise study of some neutrinos' properties: the mass hierarchy, the determination of CP-violating phase &delta;<sub>CP</sub>, the measurement of the octant &theta;<sub>23</sub> and precise calculations of all the mixing angles.
Moreover, it will contribute in the study of proton lifetime and in Beyond Standard Model physics.

### 1.2 The detector 
As said before, DUNE is provided with two main detectors since it's an neutrino oscillation experiment. <br> The Far Detector is a LArTPC, that combines tracking and calorimetry, allowing us to identify $\nu_{\mu}$ interactions and reconstruct particle's energies. <br> The Near Detector is made of multiple sub-detectors, as it shown in the scheme below. A LArTPC is present, to be consistent with the Far one. In the Phase I, The Muon Spectrometer (TMS), a detector that measures the momentum and charge sign of the muons will be built; in Phase II it will be substituted with ND-GAr, a magnetized high-pressure gaseous argon TPC with a surrounding calorimeter. It allows particle-by-particle charge and momentum reconstruction. These two detectors can move off-axis, so they can study interaction of neutrinos at different energy spectra. 
<p align="center">
<img width="594" alt="Screenshot 2023-10-19 alle 10 28 15" src="https://github.com/giacomo-santoni/SC-project/assets/133137485/2b4a1099-72e3-4a93-b9d8-9773b1b9a63d">
<em></em>
</p>

SAND (System for on-Axis Neutrino Detection) is the third detector and it's fixed on-axis. It has the aim of measuring neutrino beam spectrum and performing tracking and calorimetry measurements. Below, the design it is shown. The idea of this system comes from an existing magnet and ECAL of KLOE detector, used in DAFNE experiment at INFN LNF Laboratory. <br> SAND consists in two subdetectors: a Straw Tube Target tracker (STT) for momentum measurements and the GRAIN (GRanular Argon for Interactions of Neutrinos) detector. 
<p align="center">
<img width="537" alt="Screenshot 2023-10-19 alle 10 42 23" src="https://github.com/giacomo-santoni/SC-project/assets/133137485/8eef6263-4dba-47e3-958c-4a41c12c48a6">
</p>

#### 1.2.1 GRAIN detector
The GRAIN detector, filled with ~ 1 ton of Liquid Argon, is placed upstream in the SAND volume. It will provide inclusive Ar interactions to find systematic uncertainties from nuclear effects located on-axis, cross-calibrating with other detectors.

<p align="center">
<img width="185" alt="image" src="https://github.com/giacomo-santoni/SC-project/assets/133137485/6b086b6d-ca07-4a65-8c1d-39f18c07093f">
</p>

Since the construction of a LAr TPC in the ND is not easy due to numerous events and the pile-up that occur, a new detection technique is developed: the tracking and calorimetry system of GRAIN is based on the exploitation of the LAr scintillation light through imaging. Indeed, charged particles in LAr ionize and excite Ar atoms. Then, with the subsequent de-excitation, a photon emission is induced. 
Matrices of SiPMs are placed as photosensors, each provided with a Hadamard's mask. The combination of mask and sensor is the so-called camera. The approach chosen to study the in this detector is the Coded Aperture Imaging technique. The images formed on the sensors are convolutions of images from each hole. We can have different situations depending on the photon emission. Basically, are three, as it is shown in the figure below:
* the photons are emitted before the camera, hence on the sensor there will be a clear mask pattern
* the photons are emitted inside the camera from a particle that hits straight the camera, then on the sensor there will be a peak of photon in one single point
* the photons are emitted both inside and outside the camera, from an oblique particle: this leads to an unclear pattern on the sensor.

<p align="center">
<img width="523" alt="Screenshot 2023-10-18 alle 11 11 18" src="https://github.com/giacomo-santoni/SC-project/assets/133137485/45bb503d-fc04-4d3a-9f1c-c1094b7fbc52">
</p>

For the reconstruction task, the dazzled cameras can't be used in the current algorithm. For this reason, this project aims to classify the cameras, distinguishing the good ones from the dazzled ones and allowing us to discard the latter ones. Since, as shown above, the classification it's not always clear, a CNN written in Python is used to accomplish this task. Indeed, up to now, this classification is done using the MonteCarlo truth, but when the experiment will be built a classification that relies only on the data will be needed.
The code has been uploaded in this repo, in a VSCode Jupyter Notebook. It is divided in 5 sections: _Simulated Data - Preprocessing_, where the simulated data are loaded and rearranged; _ROOT "True" Data - RootPreprocessing_, where the data from MonteCarlo simulations are loaded a prepared; _Data Rearrangement_, where data are prepared for the training; _CNN Model_ where model is build and data are trained; _Results_ where some results are reported to evaluate the performance of the model.

# 2. DATASET
Since the files in their original format are too heavy to be uploaded in the GitHub repo or Google Drive, the info of interest were taken and saved into numpy files. Then, the files have been uploaded in Google Drive.
To download these files, follow these steps: 
1. Check if you have `gdown`, otherwise install it with `pip install gdown`: it is a package needed to download folders from the web

2. Download from Google Drive the *data* folder that contains two files: *data1* and *data2*, each provided with a simulated-data file and a true-data one. The command you have to run is: 
```
gdown --folder https://drive.google.com/drive/folders/1iAL9C_re_lVVf_Go8OUbI6DmOm9di53S -O /path/to/this/repo/folder

```

# 2.1 Simulated Data
In this project, simulated data are used since the experiment is still being built. They are in _.drdf_ format, created by the researchers of the DUNE group. The data used in this project are stored in 2 _"response.drdf"_ files, each generated after a simulation of 1000 events (i.e. charged particle interaction in the detector with photons production) with 60 and 58 camera configuration. They are organized as a list, where each element is composed of 2 objects: the number of the event and a dictionary. The dictionary gives us information on the photons arrived on the camera: the keys are the names of the cameras and the values are matrices 32x32 where each element represents the number of photons arrived in a pixel. This number isn't integer since it considers the electronic signal. 

These data are loaded and rearranged in the first section _Simulated Data - Preprocessing_.

# 2.2 True Data
Together with each simulated file _"response.drdf"_, there is the file _"sensors.root"_ with the MC truth of data, that will represent the labels for CNN training. It is a _ROOT_ file: each camera is a ROOT Tree, and each Tree has some variables, organized in TLeaves. The variable of our interest is only _innerPhotons_, which tells us how many photons produced within the camera are detected by the sensor. As before, due to the large dimensions of the file, only the useful data were taken and then saved into a numpy file. 

The data have been rearrenged to have a format consistent with the simulated ones and then labelled. The label criterion is based on a consideration on the ratio #inner_photons/#total_photons: if it's larger than 0.1, the camera is considered dazzled, and a 1 is assigned to it, otherwise, is not dazzled. This parameter was considered since sometimes the number of inner photons seems to be large but the photons produced in the remaining part of the detector are way larger. This can occur in situations as the third case presented in Section 1.2.1, when the particle starts to emit before the camera and continues inside. So, if we have looked only at the absolute number of photons we would have discarded a camera that can be useful for the reconstruction.

These modifications were done since these data have to represent only the state of the camera, i.e. dazzled/not dazzled. So at the end, an array of 0 and 1 was obtained.

# 2.3 Dataset features and rearrangements
This dataset is very imbalanced towards the not-dazzled cameras, with a percentage of 99.7% - 0.3%. So, with this kind of data, a neural network would be very good in finding the not-dazzled cameras, but only because they are in larger amount. For this reason, I applied an augmentation on the dazzled cameras, increasing their abundance up 35% with respect to the total number of events.
Moreover, I applied a cut on the cameras with less than 40 photons, since they don't give useful information for the track reconstruction, reducing the dataset of 12%.
Then, I split the dataset into 3: train dataset of $\approx 10^5$ events, validation dataset of $\approx 10^3$ events and test dataset of $\approx 10^4$ events. The augmentation dataset was attached to the train dataset. 

# 3. CNN MODEL and EXAMPLE
In the section _CNN model_, there is the construction of CNN, through a _Sequential_ model. 
The optimizer is 'adam', the loss function is a BinaryCrossentropy, since is a binary classification problem and the metric chosen is F1Score since I want to reduce both the number of FN and FP.
An important feature added to the model is the class_weight in the model.fit() function. This was another attempt to solve the imbalancing problem. In this way, the model give more weight and importance to the minority class. The model is trained for 10 epochs. Below the performances of the model during these epochs are shown, in the two plots. 

<p align="center">
![668239b0-ece9-4c00-bafb-cd9b5dcc6fa0](https://github.com/giacomo-santoni/SC-project/assets/133137485/2c183e87-0ce4-4ea5-82bc-8ee4a2f236fb)
</p>

Then, the model is tested on a different dataset, the test dataset, and the results are: 
- test F1 score: ~ 98.93%, test loss ~ 4.7%;

with a confusion matrix of: 


METTERE GRAFICI SU ANDAMENTO 

# 4. CONCLUSION


The CNN model seems good, let's see if increasing the dimensions of the dataset the performances increase.

