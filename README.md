# MACHINE LEARNING METHOD for DAZZLED CAMERAS RECOGNITION

> N.B. Before running **cnn_model.ipynb**, please refer to the _Before the execution_ paragraph for instructions on downloading data and for the required libraries. 

## Abstract
This project is part of the workflow of the simulation chain for the GRAIN detector within the SAND calorimeter of the DUNE experiment. GRAIN is a LAr (Liquid Argon) detector that detects scintillating photons produced inside the Ar volume using specialized cameras. These cameras include a sensor (matrix of SiPMs) and a Hadamard mask. The photons are produced from outside or inside the camera: depending on the number of photons internally generated, the camera is categorized as either dazzled or non-dazzled. Due to limitations in the current reconstruction algorithm, the dazzled cameras cannot be used. Hence, a classification that separates these two sets of images is needed. This project tries to achieve this objective by employing a convolutional neural network. The model presented here reaches a _F1Score_ value of $\sim$ 0.82.

The _Introduction_ provides an overview of the experiment within which this project is situated. The _Dataset_ section presents the data organization and the necessary adjustments made. In the _Code execution_ section, the model is outlined. The final section includes the _Conclusions_.

## Before the execution
For a successful project execution, pay attention to the following lines.

Since the files in their original format are too heavy to be uploaded in the GitHub repo or Google Drive, the necessary information was taken from the original files and saved into _numpy_ files. Then, these files were uploaded to Google Drive.
  
To download these files, just run the first box of the notebook. But before, if you don't have `gdown` already installed, you can install it by running:
```
pip install gdown
```
This package allows to download files.

Moreover, you need `tensorflow` and `scikit-learn` libraries installed. If they aren't already installed, you can do so by running:
```
pip install tensorflow
pip install -U scikit-learn
```
At last, you can clone this repository by running the command: 
```
git clone https://github.com/giacomo-santoni/SC-project.git
```

The rest of the code is explained in the _Code execution_ section and directly in the notebook.

## 1. INTRODUCTION
### 1.1 The DUNE Experiment
The Deep Underground Neutrino Experiment (DUNE) is a long-baseline neutrino oscillation experiment currently under construction in the USA. It will comprise two detectors: a Near Detector, located near Fermilab in Illinois, and a Far Detector, situated in South Dakota, 1300 km away. Below, a schematic design of the experiment is provided.
<p align="center">
<img width="758" alt="Screenshot 2023-10-19 alle 09 32 51" src="https://github.com/giacomo-santoni/SC-project/assets/133137485/0a0f477e-2ce3-4b4d-aa48-10de63a9adee">
</p>
<p align="center">
  <em>Schematic design of DUNE experiment.</em>
</p>

The experiment aims to provide a more precise understanding of some neutrino properties, including the mass hierarchy, determination of CP-violating phase &delta;<sub>CP</sub>, measurement of the octant &theta;<sub>23</sub>, and accurate calculations of all the mixing angles.
Additionally, it will contribute to proton lifetime studies and Beyond Standard Model physics. 
According to preliminary predictions, the detectors will work at full capacity by the end of this decade ($\sim$ 2030).

### 1.2 The detector 
As mentioned earlier, DUNE is provided with two main detectors. <br> The Far Detector is a Liquid Argon Time Projection Chamber (LArTPC) that combines tracking and calorimetry, allowing us to identify interactions involving $\nu_{\mu}$ and reconstruct particle's energies. <br> The Near Detector comprises multiple sub-detectors, as shown in the scheme below. A LArTPC is present to be consistent with the Far one. In Phase I, The Muon Spectrometer (TMS) will be constructed to measure muon momentum and charge sign; in Phase II, it will be replaced by ND-GAr, a magnetized high-pressure gaseous argon TPC with a surrounding calorimeter. It allows particle-by-particle charge and momentum reconstruction. These two detectors can move off-axis to study neutrino interactions at different energy spectra. 
<p align="center">
<img width="594" alt="Screenshot 2023-10-19 alle 10 28 15" src="https://github.com/giacomo-santoni/SC-project/assets/133137485/2b4a1099-72e3-4a93-b9d8-9773b1b9a63d">
<em></em>
</p>
<p align="center">
  <em>Near detector design in two different configurations.</em>
</p>

The System for on-Axis Neutrino Detection (SAND) is the third detector, and it is fixed on-axis. It is designed to measure the neutrino beam spectrum and to perform tracking and calorimetry measurements. Below, the design is shown. The idea of this system comes from an existing magnet and ECAL of the KLOE detector used in the DAFNE experiment at INFN LNF Laboratory. <br> SAND consists of two subdetectors: a Straw Tube Target tracker (STT) for momentum measurements and the GRAIN (GRanular Argon for Interactions of Neutrinos) detector. 
<p align="center">
<img width="537" alt="Screenshot 2023-10-19 alle 10 42 23" src="https://github.com/giacomo-santoni/SC-project/assets/133137485/8eef6263-4dba-47e3-958c-4a41c12c48a6">
</p>
<p align="center">
  <em>Design of the SAND calorimeter, with GRAIN and STT subdetectors.</em>
</p>

#### 1.2.1 GRAIN detector
The GRAIN detector, filled with ~ 1 ton of Liquid Argon, is positioned upstream within the SAND volume. It will provide inclusive Ar interactions to find systematic uncertainties from nuclear effects located on-axis, cross-calibrating with other detectors.

<p align="center">
<img width="185" alt="image" src="https://github.com/giacomo-santoni/SC-project/assets/133137485/6b086b6d-ca07-4a65-8c1d-39f18c07093f">
</p>
<p align="center">
  <em>Design of GRAIN detector configuration.</em>
</p>

Given the challenges related to the construction of an LAr TPC in the Near Detector due to numerous events and the pile-up, a novel detection technique has been developed. The tracking and calorimetry system of GRAIN is based on the exploitation of the LAr scintillation light through imaging. Indeed, charged particles in LAr ionize and excite Ar atoms, resulting in photon emission during subsequent de-excitation.
Hence, matrices of SiPMs are placed as photosensors, each equipped with a Hadamard mask. The combination of mask and sensor constitutes a ‘camera’. The chosen approach to study this detector is the Coded Aperture Imaging technique. The light patterns observed by the sensors are convolutions of images from each hole. We can have different situations depending on the photon emission. Basically, there are three situations, as shown in the figure below:
* the photons are emitted before the camera, and there will be a clear mask pattern on the sensor (top image)
* the photons are emitted inside the camera from a particle that hits the camera, then there will be a peak of photons in one single point on the sensor (central image)
* the photons are emitted both inside and outside the camera from an oblique particle: this leads to an unclear pattern on the sensor (bottom image)

<p align="center">
<img width="567" alt="Screenshot 2024-01-30 alle 12 00 21" src="https://github.com/giacomo-santoni/SC-project/assets/133137485/d31fd23e-1ccd-4a63-8247-44b563bae293">
</p>
<p align="center">
  <em>Three possible patterns on the camera, depending on where photons are produced.</em>
</p>

The current reconstruction algorithm is not implemented to work with dazzled cameras. Hence, this project aims to classify them, distinguishing the good ones from the dazzled ones and allowing the exclusion of the latter. Since, as shown above, the differences are not always evident, a Python-based CNN is employed to accomplish this task. Currently, this classification is achieved using the MonteCarlo truth, but a data-driven classification will be necessary once the experiment will be operative.
The code has been uploaded to this repository as a VSCode Jupyter Notebook. It is divided into five sections: 
* _Simulated Data - Preprocessing_, where the simulated data are loaded and rearranged;
* _ROOT "True" Data - RootPreprocessing_, where the data from MonteCarlo simulations are loaded and prepared;
* _Data Rearrangement_, where data are prepared for the training;
* _CNN Model_ where the model is built and data are trained;
* _Results_ where some results are reported to evaluate the model performance.

# 2. DATASET
## 2.1 Simulated Data
Simulated data are used in this project, as the experiment is still under construction. They are in _.drdf_ format, created by the researchers of the DUNE group. The data used in this project are stored in 2 _"response.drdf"_ files, each generated after a simulation of 1000 events (i.e., charged particle interaction in the detector with photons production) with 60 and 58 camera configurations. They are organized as a list, where each element consists of 2 objects: the event number and a dictionary. The dictionary gives us information on the photons detected by the camera: the keys are the names of the cameras, and the values are matrices 32x32, where each element represents the number of photons arrived in a pixel. This is not an integer number since it comes from the electronic signal. It has been extrapolated and stored in the simulated numpy file.

These data are loaded and rearranged in the first section _Simulated Data - Preprocessing_.

## 2.2 True Data
Together with each simulated file _"response.drdf"_, there is the file _"sensors.root"_ containing the MC truth of data that will represent the labels for CNN training. It is a _ROOT_ file: each camera is represented as a ROOT Tree, and each Tree has some variables organized in TLeaves. The relevant variable is _innerPhotons_, which indicates how many photons produced within the camera are detected by the sensor. As before, due to the large dimensions of the file, only the useful data were taken and then saved into a numpy file. 

The data have been rearranged to have a format consistent with the simulated ones and then labelled. The labelling criterion is based on the ratio $`\frac{N_{inner photons}}{N_{total photons}}`$: if it's larger than 0.1, the camera is considered dazzled (_D_), and a 1 is assigned to it, otherwise, it is not-dazzled (_ND_), and labelled with 0. This criterion was chosen to account for situations like the third case presented in Section 1.2.1 when the particle starts to emit before the camera and continues inside. So, if we had looked only at the absolute number of photons, we would have discarded a camera that could be useful for the reconstruction.

These adjustments were made since these data have to represent only the state of the camera, i.e., dazzled/not dazzled. So, in the end, an array of 0 and 1 was obtained.

## 2.3 Dataset features and rearrangements
This dataset is highly imbalanced towards _ND_ cameras, with a distribution of 99.7% _ND_ - 0.3% _D_. So, with this kind of data, a neural network would be optimal in finding the _ND_ cameras, but only because they are in higher amounts. For this reason, an augmentation to the _D_ cameras was applied, increasing their abundance to 35% with respect to the total number of events.
Moreover, a threshold was set for cameras with less than 40 photons since they don't provide information for track reconstruction, reducing the dataset by 12%.
Then, the dataset was split into three subsets: a training dataset of $\sim 10^5$ events, a validation dataset of $\sim 10^3$ events, and a test dataset of $\sim 10^4$ events. The augmentation dataset was attached to the training and validation datasets. 

# 3. CODE EXECUTION
In the section _CNN model_, there is the construction of CNN through a _Sequential_ model.
The model's parameters are gathered inside the _CNNparameters_ dictionary. The batch size is set 32, as often is done in neural networks. The input shape is (32,32,1) because the cameras have 32x32 pixels and one colour channel. The number of epochs is set to 10, however, it is managed by the _EarlyStopping_ callback, which stops the training when there is no more improvement in the learning. Since it is a binary classification problem, the _BinaryCrossentropy_ is a well-suited loss function. The optimizer is _'adam'_, as suggested in 1. The chosen metric is _F1Score_ to minimize both the number of false negatives (FN) and false positives (FP). 
It considers both the precision and the recall:
$$F1 = 2 \cdot \frac{precision \cdot recall}{precision + recall} = \frac{TP}{TP + \frac{1}{2}(FP + FN)},$$
where $`precision = \frac{TP}{TP + FP}`$ and $`recall = \frac{TP}{TP + FN}`$. In classification problems, $TP$ indicates true positives (well-identified D cameras), $FP$ false positives (_ND_ cameras classified as _D_), $FN$ false negatives (_D_ cameras classified as _ND_) and $TN$ true negatives (well-identified _ND_ cameras).
An important feature added to the model is the *class_weight* in the model.fit() function. This was another attempt to solve the imbalancing problem. In this way, the model gives more weight and importance to the minority class.

The notebook provides the output of each code segment along with corresponding comments.

# 4. RESULTS and NEXT STEPS
After the data preprocessing and the building of the CNN architecture, the model was trained for several epochs managed by the _EarlyStopping_ callback. During the training, the _F1Score_ metric and the loss function have been monitored for both the training and the validation datasets. The score increases approaching $\sim$ 0.998, while the loss decreases to less than $\sim 0.02$, as shown below and in the notebook. The better value of the validation score and loss is probably due to the regularization option in the model, that acts only on training set increasing the loss function value. This is a technique to mitigate overfitting: it constrains the network’s weights to take only small values, making the weight distribution more regular by adding a cost to the loss function when it is associated with large weights. In particular, in the _L2 regularization_, here used, the cost added is proportional to the square of the value of the weight coefficients.
The confusion matrix, reported below and in the notebook, shows that the predicted _ND_ cameras include some true _D_ ones. Nevertheless, the purity of the dataset increases after the algorithm application. At the beginning the purity is $`p_i = \frac{TN+FP}{N_{totcameras}} = 0.9971`$; after the algorithm processing it becomes $`p_f = \frac{TN}{TN+FN} = 0.9995`$. This shows an improvement that leads to a better track reconstruction since the relative abundance of the _D_ cameras is reduced. The model obtains a _F1Score_ value of $\sim 0.82$. 

<p align="center">
<img width="450" alt="c251ca60-22e9-4682-a054-bceea19fd918" src="https://github.com/giacomo-santoni/SC-project/assets/133137485/040679b1-fdef-4ecf-b4d0-8c035f6adc87">
</p>
<p align="center">
  <em>Trends of F1Score metric and loss function over 10 epochs for training and validation datasets.</em>
</p>

<p align="center">
<img width="300" alt="Screenshot 2024-02-02 alle 10 50 53" src="https://github.com/giacomo-santoni/SC-project/assets/133137485/85d5a3a1-9713-4f30-b545-b6f704685f3d">
</p>
<p align="center">
  <em>Confusion matrix obtained from the model.</em>
</p>

Despite the good results, further steps can be taken to improve the model's performance. It must be noted that the data are not so clearly distinguishable between dazzled and non-dazzled since sometimes even a _D_-labelled camera looks like an _ND_-one, or vice versa, making the task very complex. A possible improvement can come from a redefinition in the labelling criterion, defining three categories. Beyond the non-dazzled cameras, a further distinction among the dazzled cameras can be made by considering the ones where the particle directly hit the sensor and the ones where the particle emitted photons within the camera without hitting the sensor.

# BIBLIOGRAPHY
1. _An overview of gradient descent optimization algorithms_, [https://doi.org/10.48550/arXiv.1609.04747]
