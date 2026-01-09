# NEWER
Code for NEWER: Neural Estimation of Wavelet-Embedded Representations

## Code setup and install

### Clone github repo:
git clone https://github.com/kumar-vishesh/NEWER.git

### Create a conda env and setup with all required packages via the requirements.txt file

````
conda create NEWER_env
conda install requirements.txt
````

## Run the code

Note that all default architectures used for comparison in the paper are available in the ``archs/`` directory. To train and INR with that arch on a given datapoint (a 2D .npy file) simply run the train.py file with the name of the architecture and path to the datapoint passed as CLI arguements. 

For example to train NEWER on the 1st synthetic datapoint at full resolution run the following:

````
python train.py --arch NEWER --data synthetic_data/datapoint01.npy
````

the output from this run will then be save in the ``results/`` directory inside a timestamped subdirectory. There are 3 additional CLI flags you can use:

````
--downsample (bool, whether to train at 2x downsample, false by default)
--n_epochs (int, number of epochs to train for, 5000 by default)
--debug    (bool, save a list of the loss during training and intermediate verision of the model, false by default)
````