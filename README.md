# LiRCDepth

Pytorch implementation of LiRCDepth: Lightweight Radar-Camera Depth Estimation via Knowledge Distillation and Uncertainty Guidance (Accepted by ICASSP 2025)

Models have been tested using Python 3.7/3.8, Pytorch 1.10.1+cu111

## Setting up dataset
To set up the dataset, please refer to the [CaFNet repo](https://github.com/harborsarah/CaFNet).

## Training LiRCDepth
To train LiRCDepth on the nuScenes dataset, you may run:
```
python main_student.py arguments_train_nuscenes_student.txt arguments_test_nuscenes.txt
```

## Evaluating LiRCDepth
To evaluate LiRCDepth the nuScenes dataset, you may run:
```
python test_student.py arguments_test_nuscenes_student.txt 
```
You may replace the path dirs in the arguments files.

## Acknowledgement
Our work builds on and uses code from [radar-camera-fusion-depth](https://github.com/nesl/radar-camera-fusion-depth), [bts](https://github.com/cleinc/bts). We'd like to thank the authors for making these libraries and frameworks available.

## Citation
If you use this work, please cite our paper:

