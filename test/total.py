import os, sys
base_dir = os.getcwd()
sys.path.insert(0, base_dir)
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
from resnet import resnet18, resnet50, MMD_loss
from regressor import Gaze_regressor
import torchvision
from warmup_scheduler import GradualWarmupScheduler
import time
    
def main(train, test, model_type):
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
    
    if model_type == 'res18':
        model = resnet18(domains=n_classes)
    if model_type == 'res50':
        model = resnet50(domains=n_classes)
        
    mlp = Gaze_regressor()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    mlp = mlp.to(device)
    
    if data.isFolder: 
        data, _ = ctools.readfolder(data) 
    
    test_dataset = test_reader.loader(data, 80, num_workers=4, shuffle=False)

    begin = load.begin_step
    end = train.params.epoch
    step = load.steps
    
    model_path = os.path.join(train.save.metapath, train.save.folder, 'checkpoint/ours')
    model_name = train.save.model_name
    
    logpath = os.path.join(train.save.metapath, data.name, test.savename, f'total')
    
    if not os.path.exists(logpath):
        os.makedirs(logpath)
           
    for saveiter in range(begin, end+step, step):
        # 测试性能
        print(f"Test {saveiter}")
        
        model_statedict = torch.load(
            os.path.join(model_path, 
                f"Iter_{saveiter}_{model_name}.pt"), 
            map_location={f"cuda:{train.device}": f"cuda:{test.device}"}
        )

        model.load_state_dict(model_statedict['resnet18']); model.eval()
        mlp.load_state_dict(model_statedict['mlp']); mlp.eval
            
        logname = f"{saveiter}.log"
        outfile = open(os.path.join(logpath, logname), 'w')
        outfile.write("name results gts\n")

        length = len(test_dataset); accs = 0; count = 0
        with torch.no_grad():
            for j, (data, label) in enumerate(test_dataset):

                for key in data:
                    if key != 'name': data[key] = data[key].cuda()

                names =  data["name"]
                gts = label.cuda()

                features, _ = model(data["face"])
                gazes = mlp(features)

                for k, gaze in enumerate(gazes):

                    gaze = gaze.cpu().detach().numpy()
                    gt = gts.cpu().numpy()[k]

                    count += 1                
                    if train.save.folder == "eth":
                        accs += gtools.angular(
                                    gtools.gazeto3d_cvt(gaze),
                                    gtools.gazeto3d(gt)
                                )
                    else:
                        accs += gtools.angular(
                                    gtools.gazeto3d(gaze),
                                    gtools.gazeto3d(gt)
                                )   

                    name = [names[k]]
                    gaze = [str(u) for u in gaze] 
                    gt = [str(u) for u in gt] 
                    log = name + [",".join(gaze)] + [",".join(gt)]
                    outfile.write(" ".join(log) + "\n")
            print(count)
            loger = f"[{saveiter}] Total Num: {count}, avg: {accs/count}"
            outfile.write(loger)
            print(loger)
        outfile.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pytorch Basic Model Training')

    parser.add_argument('-s', '--source', type=str,
                        help = 'config path about training')

    parser.add_argument('-t', '--target', type=str,
                        help = 'config path about test')
    
    parser.add_argument('-m', '--model', type=str, default='res18',
                        help = 'model type')

    args = parser.parse_args()

    # Read model from train config and Test data in test config.
    train_conf = edict(yaml.load(open(args.source), Loader=yaml.FullLoader))

    test_conf = edict(yaml.load(open(args.target), Loader=yaml.FullLoader))

    main(train_conf.train, test_conf.test, args.model)
    
    
