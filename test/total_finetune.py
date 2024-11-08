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
from resnet import resnet18, MMD_loss
from regressor import Gaze_regressor
import torchvision
from warmup_scheduler import GradualWarmupScheduler
import time


def seed_torch(seed=3407):  # 114514, 3407
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    
def main(train, test):
    seed_torch()
    
    fine_tune = True
    total_training_time = 0  # 初始化总训练时间
    
    test_reader = importlib.import_module("reader." + test.reader)
    torch.cuda.set_device(test.device)

    data = test.data
    load = test.load
    
    if train.data.name == "eth":
        n_classes = 75
    elif train.data.name == "gaze360":
        n_classes = 69
        
    model = resnet18(domains=n_classes)
    mlp = Gaze_regressor()
    
    model_path = os.path.join(train.save.metapath, train.save.folder, 'checkpoint/res18')
    model_name = train.save.model_name
    
    model_statedict = torch.load(
        os.path.join(model_path, f"Iter_10_{model_name}.pt"), 
        map_location={f"cuda:{train.device}": f"cuda:{test.device}"})
    
    print(os.path.join(model_path, f"Iter_10_{model_name}.pt"))

    model.load_state_dict(model_statedict['resnet18'])
    mlp.load_state_dict(model_statedict['mlp']); mlp.eval()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    mlp = mlp.to(device)
    
    if data.isFolder: 
        data, _ = ctools.readfolder(data) 
    
    finetune_dataset = test_reader.loader(data, 80, num_workers=4, shuffle=False, fine_tune=True)
    
    parameters= list(model.parameters()) + list(mlp.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.0001, betas=(0.9, 0.95))
  
    scheduler = torch.optim.lr_scheduler.StepLR( 
                optimizer,
                step_size=20, 
                gamma=0.5
            )
    
    savepath = os.path.join(test.save.metapath, test.save.folder, f"checkpoint")
    
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        
    optimizer.zero_grad()
    optimizer.step()
    scheduler.step()

    for epoch in range(1, 10 + 1):
        for i, finetune_data in enumerate(finetune_dataset):
            f_data, anno = finetune_data
            image = f_data['face'].to(device)
            label = anno.to(device)

            feature, _ = model(image)
            loss = mlp.loss(feature, label)
            
            optimizer.zero_grad()                
            loss.backward()
            optimizer.step()
        
            if i % 20 == 0:
                print(f"[{epoch}/10] loss: {loss}")
            
        state_dict = {
                "resnet18": model.state_dict(),
                "mlp": mlp.state_dict()
            }
        if (epoch) % test.save.step == 0:
            torch.save(
                    state_dict, 
                    os.path.join(
                        savepath, 
                        f"Iter_{epoch}_{test.save.model_name}.pt"
                        )
                    )
            scheduler.step()
    
    begin = load.begin_step
    end = train.params.epoch
    step = load.steps
    
    test_dataset = test_reader.loader(data, 80, num_workers=4, shuffle=False)

    model_path = os.path.join(test.save.metapath, test.save.folder, 'checkpoint')
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

    args = parser.parse_args()

    # Read model from train config and Test data in test config.
    train_conf = edict(yaml.load(open(args.source), Loader=yaml.FullLoader))

    test_conf = edict(yaml.load(open(args.target), Loader=yaml.FullLoader))

    main(train_conf.train, test_conf.test)
    
    
