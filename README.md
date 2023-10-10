# SOFTWARE & COMPUTING for NUCLEAR and SUBNUCLEAR PHYSICS project
# 1. INTRODUCTION
This project is part of the workflow of the track reconstruction in the GRAIN detector of the SAND calorimeter in the DUNE experiment. Its aim consists in classifying two types of images, observing the state of a single camera: if it is “blinded”, i.e. reached by too many photons, or not. 

The GRAIN detector, filled with ~ 1 ton of Liquid Argon, is covered in total by almost 60 pixel-segmented photodetectors, each provided with a Hadamard's mask. The combination of mask and sensor is the so-called camera. The photodetectors aim to detect scintillation photons produced by the de-excitation of Ar atoms after interacting with charged particles. Capturing these photons, it should be possible to reconstruct the track of the charged particles in LAr, as is done in classic bubble chambers.

The approach chosen in this detector is the Coded Aperture Imaging technique, which requires, as said before, a mask in front of each sensor. The images formed on the sensors are convolutions of images from each hole. So, if the photons are produced before the mask, then we will see a pattern similar to the mask pattern on the camera; otherwise, if the photons are produced between the mask and the camera, we will see a large count of photons in the camera. Sometimes, it can happen that a particle starts to emit photons outside the camera and continues inside.  

For the reconstruction task, the blinded cameras are unuseful, since don't give us information about the passage of the particle (????). For this reason, this project aims to classify the cameras, distinguishing the good ones from the blinded ones and allowing us to discard the latter ones. To accomplish this task, a CNN written in Python was used. The code has been uploaded in this repo, moduled in 5 different files to be clearer: import_drdf.py, functions.py, scaling.py, root_file.py, cnn_model.py.

# 2. DATASET
# 2.1 Simulated Data
In this code, simulated data are used since the experiment is still being built. They are in _.drdf_ format, created by the researchers of the DUNE group. Data are stored in the _"response.drdf"_ and _"response_pde25.drdf"_ files. They are organized as a list, where each element is composed of 2 objects: the number of the event and a dictionary. The dictionary gives us information on the photons arrived on the camera: the keys are the names of the cameras and the values are matrices where each element represents the number of photons arrived in a pixel. In each file 1000 events (i.e. charged particle interaction in the detector with photon production) are generated, the cameras are almost 60, depending on the simulation, and they have 32x32 pixels. The code for the import of these data is in **Preprocessing.py**. 
Below two typical images of the file are reported: the right one is a normal camera and the left one is a blinded camera. The functions for the plotting are defined in **Plots.py**.

![ev_0_cam_1](https://github.com/giacomo-santoni/SC-project/assets/133137485/25a9b943-60e5-4cca-9ec6-d2557ce180a6)                               ![blindcam](https://github.com/giacomo-santoni/SC-project/assets/133137485/eab6400d-084f-4fa2-915d-9771940680f2)

Thus, these data are stored in a matrix of (cameras x events). This matrix has been arranged in an array, where each element is a matrix representing a single camera 32x32. These rearrangements are presented in **Preprocessing.py**.

# 2.2 True Data
Together with each simulated file _"response.drdf"_, there is the file _"sensors.root"_ with the "truth" of data, that will represent the labels for CNN training. It is a _ROOT_ file: each camera is a ROOT Tree, and each Tree has some variables, organized in TLeaves. The variable of our interest is only _innerPhotons_, which tells us how many photons are produced between the mask and the camera. These data are imported into the code in **RootPreprocessing.py**. In their original format, they look like this: 

![innerPhotons](https://github.com/giacomo-santoni/SC-project/assets/133137485/1e487172-6256-47aa-b413-8db6b020923e)


So, to handle these data, they have been reorganized in a matrix (events x cameras). To be consistent with the simulated dataset, the matrix has been transposed and flattened, obtaining an array, where each element is the number of inner photons. Then, the data have been labelled considering the ratio #inner_photons/#total_photons: if it's larger than 0.1, the camera is considered blinded, and a 0 is assigned to it, otherwise, is not blinded. This parameter was considered instead of the absolute value of innerPhotons, since sometimes the number of inner photons seems to be large but the photons produced also in the detector are way larger. So, looking only at the absolute number of photons we would discard a camera that can be useful for the reconstruction.

These modifications were done since these data have to represent only the state of the camera, i.e. blinded/not blinded. So at the end, an array of 0 and 1 was obtained, named _"ev_cam_state"_, that can tell the truth about the state of the camera. This handling is done in **RootPreprocessing.py**.

# 3. DATA RESCALING
Before passing the data to the model of a CNN, they were rescaled. The simulated dataset presents mostly a number of photons of the order of 10, except for some outliers, which are 2 or 3 orders of magnitude larger. So, the scaling had to take this into account, to maintain the correct distribution of the data. For this reason, in **Preprocessing.py** the _RobustScaler_ algorithm from _sk.learn_ ML library was used: it can scale the data preserving the distribution. The scaling was applied to the all the cameras relative to one event, since the maximum number of photons is common for the whole event. The algorithm takes in input a 2-dim array and works per column, for this reason the matrix was rearranged as a matrix of 1 column x number of rows relative to pixel hit by photon. _RobustScaler_ works according to the following formula <img width="296" alt="Screenshot 2023-08-29 alle 15 26 16" src="https://github.com/giacomo-santoni/SC-project/assets/133137485/4bd36bad-3fe2-426c-8afe-1952f0239f7d">.
After the scaling the number of photons inside these matrices changed being actually rescaled, such that the outliers can approach the median. Cameras look like: 

![not blind](https://github.com/giacomo-santoni/SC-project/assets/133137485/08688fad-36c3-419f-a707-5583e669cd50) ![blind](https://github.com/giacomo-santoni/SC-project/assets/133137485/01bb4192-2e36-485a-81e5-1ca03e8cad1a)

Then, the data were reorganized to be consistent with ROOT data, and so a 1-dim array was obtained, named _"all_images_scaled_1d"_.

# 4. CNN MODEL and RESULTS
In the module **training_model.py**, there is the construction of CNN. First the dataset was divided into three sets: train_ds, val_ds, test_ds from the simulated data with the respective labels from the ROOT ones, with the SplitDataset() and PrepareDataset() functions.
Once the data were ready, a _Sequential_ model was built, that presents:

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

The loss function is a BinaryCrossentropy, since is a binary classification problem. 
An important feature added to the model is the class_weight in the model.fit() function. This has been done since the dataset is very imbalanced towards the not blinded cameras: without this precaution we would have a very good accuracy since the network can classify very well not blinded cameras, but it has large difficulties in classifying the blinded ones. Training the model for 10 epochs, the results are:
DA SISTEMARE!!!!!
- training accuracy: ~ 98.98%, training loss ~ 4.5%;
- validation accuracy: ~ 98.77%, validation loss ~ 5.5%;
- test accuracy: ~ 98.93%, test loss ~ 4.7%;

Considering the weights of the two classes, the count of False Negatives drops a lot, but on the other hand the False Positive counts increase.
So, on one side, this is good because we can exclude a great part of blinded cameras, but on the other side, another problem arises since a lot of good cameras will be excluded, being predicted as blinded, as is shown in the Confusion Matrix below: 
![cm_test_dataset](https://github.com/giacomo-santoni/SC-project/assets/133137485/7c276d14-1009-4dc6-bb71-7bd3c9ba35f0)

To try to face the second problem, the dataset was enlarged adding other events: an effective reduction in the number of FP was observed, as it is shown below:

<img width="535" alt="cm_larger_dataset" src="https://github.com/giacomo-santoni/SC-project/assets/133137485/71717842-d750-4aed-a74d-439b6769cc35">




# 5. CONCLUSION
The CNN model seems good, let's see if increasing the dimensions of the dataset the performances increase.

