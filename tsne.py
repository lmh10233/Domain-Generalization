import os, sys
import random
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2, yaml, copy
from easydict import EasyDict as edict
import ctools, gtools
import argparse
import torchvision
from resnet import resnet18
from regressor import Gaze_regressor
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm


def seed_torch(seed=3407):  # 114514, 3407
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    
def main(train, test):
    seed_torch()
    
    fine_tune = True
    total_training_time = 0  # 初始化总训练时间
    
    test_reader = importlib.import_module("reader." + test.reader)
    torch.cuda.set_device(test.device)

    # params = test.params
    data = test.data
    load = test.load
    
    if train.data.name == "eth":
        n_classes = 75
    elif train.data.name == "gaze360":
        n_classes = 54
        
    model = resnet18(domains=n_classes)
    mlp = Gaze_regressor()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    mlp = mlp.to(device)
    
    if data.isFolder: 
        data, _ = ctools.readfolder(data) 
    
    model_path = os.path.join(train.save.metapath, train.save.folder, 'checkpoint/res18')
    model_name = train.save.model_name
    
    # test_data, test_folder = ctools.readfolder(data, [test.person])
    # test_dataset = test_reader.loader(test_data, 500, num_workers=4, shuffle=True)
    
    test_dataset = test_reader.loader(data, 256, num_workers=4, shuffle=False)

    model_statedict = torch.load(
        os.path.join(model_path, 
            f"Iter_80_dg.pt"), 
        map_location={f"cuda:{train.device}": f"cuda:{test.device}"}
    )

    model.to(device); model.load_state_dict(model_statedict["resnet18"], strict = False); model.eval()
    mlp.to(device); mlp.load_state_dict(model_statedict["mlp"]); mlp.eval()
    
    feature_list = []
    gt_list = []
    print("开始特征整合！")
    with torch.no_grad():
        for j, (data, label) in enumerate(test_dataset):

            for key in data:
                if key != 'name': data[key] = data[key].cuda()

            names =  data["name"]
            gts = label.cuda()

            features, _ = model(data["face"])
            gazes = mlp(features)

            gt_list.append(gts)
            feature_list.append(features)
            
    feature_mpii = torch.cat(feature_list, dim=0)
    label_mpii = torch.cat(gt_list, dim=0)
    print(f"特征整合完毕！特征维度：{feature_mpii.shape}, 标签维度：{label_mpii.shape}")
    
    image_features_np = feature_mpii.detach().cpu().numpy()
    labels_np = label_mpii.detach().cpu().numpy()
    
    np.save("T-SNE/image_features.npy", image_features_np)
    np.save("T-SNE/labels.npy", labels_np)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    image_features_2d = tsne.fit_transform(image_features_np)
    
    np.save("T-SNE/result.npy", image_features_2d)
    
    print("降维完毕")


def t_SNE():
    image_features_2d = np.load('T-SNE/result.npy')
    labels_np = np.load('T-SNE/labels.npy')
    
    # 绘制 t-SNE 可视化
    viridis = cm.get_cmap('rainbow', 20)
    plt.figure(figsize=(10, 8))
    plt.scatter(image_features_2d[:, 0], image_features_2d[:, 1], c=-labels_np[:, 0]+labels_np[:, 1], s=8, cmap=viridis)
    plt.axis('off')
    # plt.savefig("tsne_visualization.pdf", bbox_inches='tight')
    # plt.savefig("tsne_visualization.pdf")
    plt.savefig("tsne_visualization.png")
    
    plt.show()
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pytorch Basic Model Training')

    parser.add_argument('-s', '--source', type=str, default='config/train/gaze360.yaml',
                        help = 'config path about training')

    parser.add_argument('-t', '--target', type=str, default='config/test/mpii.yaml',
                        help = 'config path about test')
    
    parser.add_argument('-c', '--cluster', type=int, default=1)
    args = parser.parse_args()

    # Read model from train config and Test data in test config.
    train_conf = edict(yaml.load(open(args.source), Loader=yaml.FullLoader))

    test_conf = edict(yaml.load(open(args.target), Loader=yaml.FullLoader))

    if args.cluster:
        main(train_conf.train, test_conf.test)
    else:     
        t_SNE()