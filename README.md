## TCCF: Tracking Based on Convolutional Neural Network and Correlation Filters

### Introduction
TCCF is an online visual tracking algorithm which is based on correaltion filters and a pre-trained convolutional neural network. This package contains the source code to reproduce the experimental results of TCCF reported in our ICIG 2017 paper, which can be download in this repository. The source code is mainly written in MATLAB.

### Usage

* Supported OS: the source code was tested on 64-bit Ubuntu 14.04.3 LTS Linux OS, and it should also be executable in other linux distributions.

* Dependencies: 
 * Caffe (http://caffe.berkeleyvision.org/) framework and all its dependencies. 
 * Cuda enabled GPUs

* Installation: 
 1. Install caffe: Compile the source code in the ./caffe directory and the matlab interface following the [installation instruction of caffe](http://caffe.berkeleyvision.org/installation.html).
 2. Download the 16-layer VGG network from https://gist.github.com/ksimonyan/211839e770f7b538e2d8, and put the caffemodel file under the ./model directory.
 3. Run the demo code demo_TCCF.m. You can customize your own test sequences following this example.

### Citing Our Work

If you find TCCF useful in your research, please consider to cite our paper:

        @inproceedings{liu2017tccf,
          title={TCCF: Tracking Based on Convolutional Neural Network and Correlation Filters},
          author={Liu, Qiankun and Liu, Bin and Yu, Nenghai},
          booktitle={International Conference on Image and Graphics},
          pages={316--327},
          year={2017},
          organization={Springer}
}


### All rights reserved. 

