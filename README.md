# Domain-Generalization
Eliminating Human Identity-sensitive Information for Domain Generalization on Gaze Estimation

# Dataset preprocessing
The datasets in our paper are open access. You can download at the following link. Remember to cite the corresponding literatures. 
1. [ETH-XGaze](https://ait.ethz.ch/xgaze?query=eth)
2. [Gaze360](http://gaze360.csail.mit.edu/)
3. [MPIIFaceGaze](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/its-written-all-over-your-face-full-face-appearance-based-gaze-estimation)
4. [Eyediap](https://www.idiap.ch/en/scientific-research/data/eyediap).

You can refer to [Review and Benchmark](https://phi-ai.buaa.edu.cn/Gazehub/#benchmarks) for the dataset preprocessing.

# Start training 
We select ETH-XGaze and Gaze360 as source domain for model training, you need to change the hyper-parameters in file `config/train/eth.yaml` or `config/train/gaze360.yaml`. You can write code in terminal
```
python train/train.py -s config/train/eth.yaml
```
or 
```
python train/train.py -s config/train/gaze360.yaml
```

# Testing
MPII and Eyediap are selected as target domain for model testing, you need to change the hyper-parameters in file `config/test/mpii.yaml` or `config/test/eyediap.yaml`. We have four cross-domain evaluations. Taking "ETH-XGaze -> MPII" as an example, you can write code in terminal
```
python test/total.py -s config/train/eth.yaml -t config/test/mpii.yaml
```
We open source the training weights and log files of four experiments:  
| Tasks | E-M  | E-D | G-M | G-D |
| :---------:| :---------: | :---------: | :---------: | :---------: |
| Accuracy(°) | [6.28](https://drive.google.com/drive/folders/13-pi2KcZmG_G11PwINfMplE70KDN73Q0) | [6.91](https://drive.google.com/drive/folders/1CpXbNPqkJCRL0dBaOO6hzP_0q2dKBVUk) | [6.48](https://drive.google.com/drive/folders/12mebMWkOj1JsbIHGIi0RuEHjahKKHtMy) | [8.54](https://drive.google.com/drive/folders/1f2ZnvW02lho2zkD5Neobyyq69bVnUJE_) |

# References
The main references are as follows
1. [Review and Benchmark](https://phi-ai.buaa.edu.cn/Gazehub/#benchmarks)
```python
@article{cheng2024review,
  title={Appearance-based gaze estimation with deep learning: A review and benchmark},
  author={Cheng, Yihua and Wang, Haofei and Bao, Yiwei and Lu, Feng},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024},
  publisher={IEEE}
}
```  
2. [GazeTR](https://github.com/yihuacheng/GazeTR)
```python
@InProceedings{cheng2022gazetr,
  title={Gaze Estimation using Transformer},
  author={Yihua Cheng and Feng Lu},
  journal={International Conference on Pattern Recognition (ICPR)},
  year={2022}
}
```
3. [Domain Genralization for Gaze Estimation](https://ojs.aaai.org/index.php/AAAI/article/view/25406)
```python
@inproceedings{xu2023learning,
  title={Learning a generalized gaze estimator from gaze-consistent feature},
  author={Xu, Mingjie and Wang, Haofei and Lu, Feng},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={37},
  number={3},
  pages={3027--3035},
  year={2023}
}
```
4. [DomainDrop](https://arxiv.org/abs/2308.10285)
   [Code](https://github.com/lingeringlight/DomainDrop/blob/main/train_domain.py)
```python
@inproceedings{guo2023domaindrop,
  title={DomainDrop: Suppressing Domain-Sensitive Channels for Domain Generalization},
  author={Guo, Jintao and Qi, Lei and Shi, Yinghuan},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2023}
}
```
