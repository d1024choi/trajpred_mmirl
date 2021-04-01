# Official implementation codes for the paper "Future trajectory prediction via RNN and maximum margin inverse reinforcement learning, ICMLA, 2018".


## Setup
All the codes were developed on Ubuntu 16.04 with Python 3.5 and Tensorflow 1.10.0. 

## Train New Models

To train the network from scratch, run the followings. The parameters of the trained networks will be stored at the folder ***saved_1_0***.
```sh
$ python crowd_train.py --dataset_num 1 --exp_id 0
$ python sdd_train.py --dataset_num 1 --exp_id 0
```
**crowd_train.py** and **sdd_train.py** have a number of command-line flags that you can use to configure the model architecture, hyperparameters, and input / output settings. You can find the descriptions in the files.


To test the trained model, run the followings. The program will automatically read the parameters of the train networks in the folder ***saved_1_0***.
```sh
$ python crowd_test.py --dataset_num 1 --exp_id 0
$ python sdd_test.py --dataset_num 1 --exp_id 0
```

## Pretrained Models
Download the pre-trained models from https://www.dropbox.com/sh/nbxr12n6i3jgoi5/AABH3URiRQ_wwYU--lGDIHZTa?dl=0
Each model in the downloaded folder is the result with the best gamma parameter reported in the paper.

## Citation
```
@article{Choi,
author = {D Choi, TH An, K Ahn, and J Choi},
title = {Future trajectory prediction via RNN and maximum margin inverse reinforcement learning},
book = {2018 17th IEEE International Conference on Machine Learning and Applications (ICMLA)},
year = 2018
}
```
