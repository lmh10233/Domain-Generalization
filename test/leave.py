import os, sys
base_dir = os.getcwd()
sys.path.insert(0, base_dir)
from torch.utils.data import SubsetRandomSampler
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
from warmup_scheduler import GradualWarmupScheduler


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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
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
    
    test_data, test_folder = ctools.readfolder(data, [test.person])

    testname = test_folder[test.person] 
    
    test_dataset = test_reader.loader(test_data, 500, num_workers=4, shuffle=True)
    
    logpath = os.path.join(train.save.metapath, 
                           data.name, f'{test.savename}/{testname}')

    if not os.path.exists(logpath):
        os.makedirs(logpath)
        
    begin = load.begin_step
    end = train.params.epoch
    step = load.steps
    
    model_path = os.path.join(train.save.metapath, train.save.folder, f'checkpoint/baseline')
    model_name = train.save.model_name

    all_acc = []
    for saveiter in range(begin, end+step, step):
        # 测试性能
        print(f"Test {saveiter}")
        
        model_statedict = torch.load(
            os.path.join(model_path, 
                f"Iter_{saveiter}_{model_name}.pt"), 
            map_location={f"cuda:{train.device}": f"cuda:{test.device}"}
        )
    
        model.to(device); model.load_state_dict(model_statedict["resnet18"]); model.eval()
        mlp.to(device); mlp.load_state_dict(model_statedict["mlp"]); mlp.eval()
    
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

            loger = f"[{saveiter}] Total Num: {count}, avg: {accs/count}"
            outfile.write(loger)
            print(loger)
        all_acc.append(accs/count)
        outfile.close()
    print(f"mean accuracy: {np.mean(all_acc):.2f}, best accuracy: {np.min(all_acc):.2f}, std: {np.sqrt(np.var(all_acc)):.2f}")   
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pytorch Basic Model Training')

    parser.add_argument('-s', '--source', type=str,
                        help = 'config path about training')

    parser.add_argument('-t', '--target', type=str,
                        help = 'config path about test')
    
    parser.add_argument('-p', '--person', type=int,
                    help = 'the num of subject for test')

    args = parser.parse_args()
    train_conf = edict(yaml.load(open(args.source), Loader=yaml.FullLoader))

    test_conf = edict(yaml.load(open(args.target), Loader=yaml.FullLoader))
    test_conf = test_conf.test

    test_conf.person = args.person
    main(train_conf.train, test_conf)
    
    
