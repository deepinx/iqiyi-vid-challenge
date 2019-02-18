# The Code for IQIYI-VID Challenge

This is the code for IQIYI-VID(IQIYI video person identification) Challenge. The enhanced SSH (ESSH) from [enhanced-ssh-mxnet](https://github.com/deepinx/enhanced-ssh-mxnet) is applied for face detection and alignment. [Insightface](https://github.com/deepinsight/insightface) scheme is used for face recognition.

Pre-trained models can be downloaded on [baiducloud](https://pan.baidu.com/s/1vS_Uv9l5XZLeMwXPs_XzOA) or [googledrive](https://drive.google.com/open?id=1_8-GcZbYNGvm_msyuwqDw4u9mGnHqSQP).

## Environment

-   Python 2.7 
-   Ubuntu 18.04
-   Mxnet-cu90 (=1.3.0)

## Installation

1.  Prepare the environment.

2.  Clone the repository.
    
3.  Type  `make`  to build necessary cxx libs.

4.  Download the pre-trained model and place it in *`./model/`*

5.  Download the IQIYI-VID Datasets from [IQIYI_VID](http://challenge.ai.iqiyi.com/detail?raceId=5afc36639689443e8f815f9e) and unzip them to your disk. 

## Usage

1. Use ``python detection.py`` for face detection of train+val datasets and serialize storage. 

2. Use ``python feature.py`` to extract face features from the above serialize detection results.

3. Use ``python genfeat.py`` to re-save the extracted model features for training the MLP network.

4. Run ``train_mlp.py`` to train the MLP network for face ID recognition.

5. Run ``python predict.py`` to predict face ID using the trained MLP network and save results.

6. Run ``python output.py`` to output the final submissions for IQIYI-VID Challenge.

## License

MIT LICENSE


## Reference

```
@article{deng2018arcface,
title={ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
author={Deng, Jiankang and Guo, Jia and Niannan, Xue and Zafeiriou, Stefanos},
journal={arXiv:1801.07698},
year={2018}
}

@inproceedings{Najibi2017SSH,
  title={SSH: Single Stage Headless Face Detector},
  author={Najibi, Mahyar and Samangouei, Pouya and Chellappa, Rama and Davis, Larry S.},
  booktitle={IEEE International Conference on Computer Vision},
  year={2017},
}
```

## Acknowledgment

The code is adapted based on an intial fork from the [insightface](https://github.com/deepinsight/insightface) repository.

