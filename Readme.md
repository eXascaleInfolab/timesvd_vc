# Timesvd_vc: Predict fashion trend using both visual and time components

EU project 732328: "Fashion Brain".

D5.3: "Early demo for trend prediction".

## Getting started

### Installation:

``` bash 
git clone https://github.com/../timesvd_vc
cd timesvd_vc/
```

### Description of the "src" directory

The source code contains the following python files:
 - run.py (file for running the experiments)
 - timeSVDpp.py (the implementation of timeSVD++ model)
 - TVBPR.py (the implementation of TVBPR model)
 - timeSVD_VC.py (the implementation of timeSVD_VC model)

The following folders with regard to the data needed are provided:
 - datasets: the folder conatining the datasets used as a training set
 - image_features: Download the file image_features_Men.b from this link https://drive.google.com/open?id=1fexvEuk1MGKQL4KNYiTzEcLxxv0vWqL1 and add it to this folder.
 

### Running the code 


In order to run an experiment, the run.py file is used. Other files reprsent the models that have been implemented.
The arguments needed for the run.py file are the following: 
 - 1st argument: the name of the model, which can take the following values: timeSVDpp, TVBPR, or timeSVD_VC
 - 2nd argument: # of iterations, which takes any integer number 
 - 3rd argument : # of epochs, which takes any integer number
 - 4th argument: # of non-visual factors, which takes any integer number
 - 5th argumnet: # of visual factors, which takes any integer number
 - 6th argument: the dataset being used, which takes one of the following values: 390_actions, 780_actions, 1560_actions, 2340_actions, 4099_actions) 

Example:
 - Running an experiment of timeSVD++ with the following parameters: 
	- model name = timeSVDpp
	- #iterations = 100, 
	- #epochs = 10, 
	- #non-visual factors = 20, 
	- #visual factors = 20,
	- dataset = 390_actions

 - In terminal and already being in the "timesvd_vc/" direcotry, run the following command:
    ``` bash 
      python run.py timeSVDpp 100 10 20 20 390_actions
    ```














