# Domain-Generalization
Eliminating Human Identity-sensitive Information for Domain Generalization on Gaze Estimation

# Dataset preprocessing
The dataset preprocessing are shown in ```https://phi-ai.buaa.edu.cn/Gazehub/3D-dataset/```

# Start training 
We select ETH-XGaze and Gaze360 as source domain for model training, you need to change the dataset path in `config/train/eth.yaml` or `config/train/gaze360.yaml`. You can write code in terminal
```
python train/train.py -s config/train/eth.yaml
```
or 
```
python train/train.py -s config/train/gaze360.yaml
```
