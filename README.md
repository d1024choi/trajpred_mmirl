# Official implementation codes for the paper "Future trajectory prediction via RNN and maximum margin inverse reinforcement learning, ICMLA, 2018".


## Setup
All the codes were developed on Ubuntu 16.04 with Python 3.5 and Tensorflow 1.10.0. 

## Train New Models

To train the network from scratch, run the followings. The parameters of the trained networks will be stored at the folder ***save***.
```sh
$ python kitti_train.py
```
**kitti_train.py** havs a number of command-line flags that you can use to configure the model architecture, hyperparameters, and input / output settings. You can find the descriptions in the files.


To test the trained model, run the followings. The program will automatically read the parameters of the train networks in the folder ***save***.
```sh
$ python kitti_validation_simple.py
```

## Dataset and Pretrained Models
Download the dataset and pre-trained models from https://drive.google.com/drive/folders/1IWRvvtgPWlsA9DvvpssiODAkOTS8cN12?usp=sharing.
Place the folder and files in the link at the main root. The folder "save" contains the trained network parameters.

## Citation
```
@article{Choi,
author = {D Choi, TH An, K Ahn, and J Choi},
title = {Future trajectory prediction via RNN and maximum margin inverse reinforcement learning},
book = {2018 17th IEEE International Conference on Machine Learning and Applications (ICMLA)},
year = 2018
}
```
