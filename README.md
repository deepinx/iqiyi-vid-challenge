## Code for IQIYI-VID Challenge based on ESSH and Insightface

Recently [www.iqiyi.com](http://www.iqiyi.com) released a great video person dataset called [IQIYI_VID](http://challenge.ai.iqiyi.com/detail?raceId=5afc36639689443e8f815f9e) and also launched a person search competition on it. It is a very large and real dataset worth trying to verify your face model accuracy precisely.

This repository contains the code for IQIYI-VID(IQIYI video person identification) Challenge. The methods are implemented in Python and MXNet. The Enhanced SSH (ESSH) from [enhanced-ssh-mxnet](https://github.com/deepinx/enhanced-ssh-mxnet) is applied for face detection and alignment. [Insightface](https://github.com/deepinsight/insightface) scheme is used for face recognition.

Pre-trained models can be downloaded on [BaiduCloud](https://pan.baidu.com/s/1vS_Uv9l5XZLeMwXPs_XzOA) or [GoogleDrive](https://drive.google.com/open?id=1_8-GcZbYNGvm_msyuwqDw4u9mGnHqSQP).

## Environment

This repository has been tested under the following environment:

-   Python 2.7 
-   Ubuntu 18.04
-   Mxnet-cu90 (==1.3.0)

## Installation

1.  Prepare the environment.

2.  Clone the repository.
    
3.  Type  `make`  to build necessary cxx libs.

4.  Download the pre-trained model and place it in *`./model/`*

5.  Download the IQIYI-VID Datasets from [IQIYI_VID](http://challenge.ai.iqiyi.com/detail?raceId=5afc36639689443e8f815f9e) and unzip them to `data/iqiyi_vid` directory. 

## Usage

1.  Detect faces on train+val dataset and test dataset respectively using ESSH model.  Model `model-r50-gg` is used to judge the quality of the detected faces.
```
python detect.py --model ./model/model-r50-gg/model,0 --output ./output/det_trainval --dataset ./data/iqiyi_vid --gpu 0 --stage trainval
python detect.py --model ./model/model-r50-gg/model,0 --output ./output/det_test --dataset ./data/iqiyi_vid --gpu 0 --stage test
```

2. Extract features to the detected faces of train+val and test dataset respectively using `model-r100-gg` model.
```
python feature.py --model ./model/model-r100-gg/model,0 --input ./output/det_trainval --output ./output/feat_trainval  --gpu 0
python feature.py --model ./model/model-r100-gg/model,0 --input ./output/det_test --output ./output/feat_test --gpu 0
```
3.  Re-save the extracted face features for training the MLP network.
```
python genfeat.py --inputs ./output/feat_trainval --output ./output/trainval
```
4. Train the MLP network for face ID recognition using train+val datasets.
```
python train_mlp.py --data ./output/trainval --prefix ./model/iqiyi --ckpt 1 --network r50 --lr 0.2 --per-batch-size 1024
```
5.  Predict face ID from features of the test dataset using the pre-trained MLP network.
```
python predict.py --model ./model/iqiyi,40 --gpu 0 --inputs ./output/feat_test --output ./output/pred_test
```
6. Run ``python submit.py`` to generate the final submissions for IQIYI-VID Challenge.

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

