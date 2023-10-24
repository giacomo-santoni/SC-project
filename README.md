# SOFTWARE & COMPUTING for NUCLEAR and SUBNUCLEAR PHYSICS project
## Abstract
This project is part of the workflow of the simulation chain for the GRAIN detector of the SAND calorimeter in the DUNE experiment. GRAIN, a LAr detector, detects scintillating photons produced inside the Ar volume using cameras, devices formed by a sensor (matrix of SiPM) and an Hadamard mask. These photons can be produced outside or inside the camera: depending on the number of photons produced inside, the camera will be defined as dazzled or not-dazzled. Since the dazzled cameras can't be used in the current reconstruction algorithm, a classification that separates these two classes of images is needed.
This project tries to accomplish this task through a convolutional neural network. 

## 1. INTRODUCTION
### 1.1 Experiment overview
The Deep Underground Neutrino Experiment (DUNE) is a long-baseline neutrino oscillation experiment that is being built in the United States. It will consists of two detectors: a Near Detector, close to Fermilab in Illinois and a Far Detector, in South Dakota, 1300 km away. Below a schematic design of the experiment is presented.
<p align="center">
<img width="758" alt="Screenshot 2023-10-19 alle 09 32 51" src="https://github.com/giacomo-santoni/SC-project/assets/133137485/0a0f477e-2ce3-4b4d-aa48-10de63a9adee">
</p>
Basically, the motivation for the design of this experiment is a more precise study of some neutrinos' properties: the mass hierarchy, the determination of CP-violating phase $\delta_{CP}$, the measurement of the octant $\theta_{23}$ and precise calculations of all the mixing angles.
Moreover, it will contribute in the study of proton lifetime and in Beyond Standard Model physics.

### 1.2 The detector 
As said before, DUNE is provided with two main detectors since it's an neutrino oscillation experiment. <br> The Far Detector is a LArTPC, made of 4 modules of 10 ktons each. These modules combine tracking and calorimetry, allowing us to identify $\nu_{mu}$ interactions and reconstruct particle's energies. <br> The Near Detector is made of multiple sub-detectors, as it shown in the scheme below. A LArTPC is present, to be consistent with the Far one. In the Phase I, The Muon Spectrometer (TMS), a detector that measures the momentum and charge sign of the muons will be built; in Phase II it will be substituted with ND-GAr, a magnetized high-pressure gaseous argon TPC with a surrounding calorimeter. It allows particle-by-particle charge and momentum reconstruction. These two detectors can move off-axis, so they can study interaction of neutrinos at different energy spectra. 
<p align="center">
<img width="594" alt="Screenshot 2023-10-19 alle 10 28 15" src="https://github.com/giacomo-santoni/SC-project/assets/133137485/2b4a1099-72e3-4a93-b9d8-9773b1b9a63d">
</p>

SAND (System for on-Axis Neutrino Detection) is the third detector and it's fixed on-axis. It has the aim of measuring neutrino beam spectrum and performing tracking and calorimetry measurements. Below, the design it is shown. The idea of this system comes from an existing magnet and ECAL of KLOE detector, used in DAFNE experiment at INFN LNF Laboratory. <br> SAND consists in two subdetectors: a Straw Tube Target tracker (STT) for momentum measurements and the GRAIN (GRanular Argon for Interactions of Neutrinos) detector. 
<p align="center">
<img width="537" alt="Screenshot 2023-10-19 alle 10 42 23" src="https://github.com/giacomo-santoni/SC-project/assets/133137485/8eef6263-4dba-47e3-958c-4a41c12c48a6">
</p>

#### 1.2.1 GRAIN detector
The GRAIN detector, filled with ~ 1 ton of Liquid Argon, is placed upstream in the SAND volume. It will provide inclusive Ar interactions to find systematic uncertainties from nuclear effects located on-axis, cross-calibrating with other detectors.<br> Since the construction of a LAr TPC in the ND is not easy due to numerous events and the pile-up that occur, a new detection technique is developed: the tracking and calorimetry system of GRAIN is based on the exploitation of the LAr scintillation light through imaging. Indeed, charged particles in LAr ionize and excite Ar atoms. Then, with the subsequent de-excitation, a photon emission is induced. 

Matrices of SiPMs are placed as photosensors, each provided with a Hadamard's mask. The combination of mask and sensor is the so-called camera. The photodetectors aim to detect scintillation photons produced by the de-excitation of Ar atoms after interacting with charged particles. Capturing these photons, it should be possible to reconstruct the track of the charged particles in LAr, as is done in classic bubble chambers. Below is reported the GRAIN geometry.

<p align="center">
<img width="185" alt="image" src="https://github.com/giacomo-santoni/SC-project/assets/133137485/6b086b6d-ca07-4a65-8c1d-39f18c07093f">
</p>

The approach chosen in this detector is the Coded Aperture Imaging technique, which requires, as said before, a mask in front of each sensor. The images formed on the sensors are convolutions of images from each hole. Below is reported an image that sketch some typical scenarios that can occur. If the photons are produced before the mask, then on the sensor we will see an image similar to the mask pattern (top image); otherwise, if the photons are produced between the mask and the camera, we will see a large count of photons accumulated in the camera (centered image). Sometimes, it can happen that a particle starts to emit photons outside the camera and continues inside, causing an unclear pattern on the sensor (bottom image). 

<p align="center">
<img width="523" alt="Screenshot 2023-10-18 alle 11 11 18" src="https://github.com/giacomo-santoni/SC-project/assets/133137485/45bb503d-fc04-4d3a-9f1c-c1094b7fbc52">
</p>

For the reconstruction task, the dazzled cameras can't be used in the current algorithm. For this reason, this project aims to classify the cameras, distinguishing the good ones from the dazzled ones and allowing us to discard the latter ones. Since, as shown above, the classification it's not always clear, a CNN written in Python was used to accomplish this task. The code has been uploaded in this repo, moduled in 4 different files to be clearer: Preprocessing.py, RootPreprocessing.py, Useful_functions.py, training_model.py

# 2. DATASET
# 2.1 Simulated Data

Is the function for loading _drdf_ files. It takes as input the name of the _drdf_ file and it returns a list, where each element is composed of 2 objects: the number of the event and a dictionary. The dictionary gives us information on the photons arrived on the camera: the keys are the names of the cameras and the values are matrices 32x32 where each element represents the number of photons that hit a pixel. This number isn't an integer since it is extracted from the electronic signal.


In this project, simulated data are used since the experiment is still being built. They are in _.drdf_ format, created by the researchers of the DUNE group. Data are stored in 2 _"response.drdf"_ files, each generated after a simulation of 1000 events (i.e. charged particle interaction in the detector with photons production) with 60 and 58 camera configuration. They are organized as a list, where each element is composed of 2 objects: the number of the event and a dictionary. The dictionary gives us information on the photons arrived on the camera: the keys are the names of the cameras and the values are matrices 32x32 where each element represents the number of photons arrived in a pixel. This number isn't integer since it considers the electronic signal. Since the files are too The code for the import of these data is in **Preprocessing.py**. 

Thus, these data are stored in a matrix of (cameras x events). This matrix has then been arranged in an array, where each element is a matrix representing a single camera 32x32. These rearrangements are presented in **Preprocessing.py**.

# 2.2 True Data

> These data are stored in a _ROOT_ file: each camera is a _ROOTTree_ and inside each Tree we are interested in the **innerPhotons** branch, that tells us how many photons produced within the camera are detected by the sensor. 

> **OpenRootFile function**<br>It takes in input the name of the _drdf_ and _ROOT_ file and read the branch. Then it arranges the data of inner photons in a matrix of dimensions EVENTS(rows)xCAMERAS(cols), where each element is the number of inner photons.

Together with each simulated file _"response.drdf"_, there is the file _"sensors.root"_ with the "truth" of data, that will represent the labels for CNN training. It is a _ROOT_ file: each camera is a ROOT Tree, and each Tree has some variables, organized in TLeaves. The variable of our interest is only _innerPhotons_, which tells us how many photons produced within the camera are detected by the sensor. These data are imported into the code in **RootPreprocessing.py**. In their original format, they look like this: 

![innerPhotons](https://github.com/giacomo-santoni/SC-project/assets/133137485/1e487172-6256-47aa-b413-8db6b020923e)

So, to handle these data, they have been reorganized in a matrix (events x cameras). To be consistent with the simulated dataset, the matrix has been transposed and flattened, obtaining an array, where each element is the number of inner photons. Then, the data have been labelled considering the ratio #inner_photons/#total_photons: if it's larger than 0.1, the camera is considered dazzled, and a 0 is assigned to it, otherwise, is not dazzled. This parameter was considered instead of the absolute value of innerPhotons, since sometimes the number of inner photons seems to be large but the photons produced also in the detector are way larger. So, looking only at the absolute number of photons we would discard a camera that can be useful for the reconstruction.

These modifications were done since these data have to represent only the state of the camera, i.e. dazzled/not dazzled. So at the end, an array of 0 and 1 was obtained, named _"ev_cam_state"_, that can tell the truth about the state of the camera. This handling is done in **RootPreprocessing.py**.

# 2.3 Dataset features and rearrangement
This dataset is very imbalanced towards the not-dazzled cameras, with a percentage of 99.7% - 0.3%. So, with this kind of data, a neural network would be very good in finding the not-dazzled cameras, but only because they are in larger amount. For this reason, I applied an augmentation on the dazzled cameras. 
Moreover, I applied a cut on the cameras with less than 40 photons, since they don't give useful information for the track reconstruction, reducing the dataset to ...
Then, I split the dataset into 3: train dataset of $\approx 10^5$ events, validation dataset of $\approx 10^3$ events and test dataset of $\approx 10^4$ events. The augmentation dataset was attached to the train dataset, yielding to an abundance of 65% not dazzled/35% dazzled. 

# 4. CNN MODEL and RESULTS
In the module **training_model.py**, there is the construction of CNN, through a _Sequential_ model, that presents:

output_bias = keras.initializers.Constant(initial_bias)
model = models.Sequential([
layers.Conv2D(32, (3,3), activation='relu',input_shape = input_shape[1:]),
layers.MaxPooling2D((2,2)),
layers.Flatten(),#(input_shape = input_shape[1:]),
layers.Dense(128, activation='relu', bias_initializer=output_bias),
layers.Dense(64, activation='relu'),
layers.Dense(32,activation='relu'),
layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss=loss_func, metrics=metric)

The optimizer is 'adam', the loss function is a BinaryCrossentropy, since is a binary classification problem and the metric chosen is F1Score since I want to reduce both the number of FN and FP.
An important feature added to the model is the class_weight in the model.fit() function. This was another attempts in order to solve the imbalancing problem. In this way, the model give more weight and importance to the minority class. Training the model for 10 epochs, the results are:
DA SISTEMARE!!!!!
- training accuracy: ~ 98.98%, training loss ~ 4.5%;
- validation accuracy: ~ 98.77%, validation loss ~ 5.5%;
- test accuracy: ~ 98.93%, test loss ~ 4.7%;

METTERE GRAFICI SU ANDAMENTO 

# 5. CONCLUSION
The CNN model seems good, let's see if increasing the dimensions of the dataset the performances increase.

