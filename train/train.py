import sys, os
base_dir = os.getcwd()
sys.path.insert(0, base_dir)
import random
import copy
import yaml
import cv2
import copy
import torch
import torchvision
from torch import nn
import importlib
import ctools
import argparse
import time
import numpy as np
from resnet import resnet18, resnet50, MMD_loss, FocalLoss
from regressor import Gaze_regressor
from warmup_scheduler import GradualWarmupScheduler
from scheduler import cosine_schedule
from easydict import EasyDict as edict
from torch.nn.functional import one_hot


def seed_torch(seed=3407):  # 114514, 3407
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    
def select_layers(layer_wise_prob):
    discriminator_layers = [1, 2, 3, 4]
    # layer_wise_prob: prob for layer-wise dropout
    layer_index = np.random.randint(len(discriminator_layers), size=1)[0]
    layer_select = discriminator_layers[layer_index]
    layer_drop_flag = [0, 0, 0, 0]
    if random.random() <= layer_wise_prob:
        layer_drop_flag[layer_select - 1] = 1
    return layer_drop_flag    
    

def main(config, model_type):
    seed_torch()
    total_training_time = 0  # 初始化总训练时间
    dataloader = importlib.import_module("reader." + config.reader)
    
    data = config.data
    save = config.save
    params = config.params
    
    if data.name == 'gaze360':
        n_classes = 54
    elif data.name == 'eth':
        n_classes = 75
        
    print(f"{data.name}训练集包含{n_classes}个人")
    
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
    
    dataset = dataloader.loader(
                    data,
                    params.batch_size, 
                    shuffle=True, 
                    num_workers=8,
                    trans=False
                )
    aug_dataset = dataloader.loader(
                    data,
                    params.batch_size, 
                    shuffle=True, 
                    num_workers=8,
                    trans=True
                )
            
    parameters= list(model.parameters()) + list(mlp.parameters())
    optimizer = torch.optim.Adam(parameters, lr=params.lr, betas=(0.9, 0.95))
  
    scheduler = torch.optim.lr_scheduler.StepLR( 
                optimizer, 
                step_size=params.decay_step, 
                gamma=params.decay
            )
    
    if params.warmup:
        scheduler = GradualWarmupScheduler( 
                        optimizer, 
                        multiplier=1, 
                        total_epoch=params.warmup, 
                        after_scheduler=scheduler
                    )
    
    savepath = os.path.join(save.metapath, save.folder, f"checkpoint")

    if not os.path.exists(savepath):
        os.makedirs(savepath)
    
    length = len(dataset); total = length * params.epoch
    timer = ctools.TimeCounter(total)
    
    domain_discriminator_flag = 1
    
    # id_criterion = nn.CrossEntropyLoss().cuda()
    id_criterion = FocalLoss().cuda()
    mmd_loss_op = MMD_loss().cuda()
    
    optimizer.zero_grad()
    optimizer.step()
    scheduler.step()
    
    print("Starting Training")
    with open(os.path.join(savepath, "train_log"), 'w') as outfile:
        outfile.write(ctools.DictDumps(config) + '\n')
        
        class_total = 0.0
        FL_domain_loss = [0.0 for i in range(5)]
        domain_right = [0.0 for i in range(5)]
        
        for epoch in range(1, params.epoch + 1):
            for i, (origindata, augdata) in enumerate(zip(dataset, aug_dataset)):
                layer_drop_flag = select_layers(layer_wise_prob=0.8)
                
                data, anno = origindata
                image = data['face'].to(device)
                label = anno.to(device)
                
#######################################ours##################################################
                person_class = data['id']
                person_class = [int(cl) for cl in person_class]
                person_id = torch.tensor(person_class).to(device)
                aug_data, aug_anno = augdata
                aug_image = aug_data['face'].to(device)
                aug_label = aug_anno.to(device)
                
                face = torch.cat((image, aug_image)).to(device)
                person_id_two = torch.cat((person_id, person_id)).to(device)
                
                id_feature, id_logic = model(face, person_id_two, layer_drop_flag)
                batch_size = int(id_feature.shape[0] / 2)
                ori_feature1 = id_feature[:batch_size]
                ori_feature2 = id_feature[batch_size:]
                ori_loss1 = mlp.loss(ori_feature1, label)
                ori_loss2 = mlp.loss(ori_feature2, aug_label)
                mmd_loss = mmd_loss_op(ori_feature1, ori_feature2)
                loss = ori_loss1 + ori_loss2 + mmd_loss
#######################################resnet##################################################
                # ori_feature1, _ = model(image)
                # ori_loss1 = mlp.loss(ori_feature1, label)
                # loss = ori_loss1
#######################################resnet##################################################

                domain_losses_avg = torch.tensor(0.0).to(device)
                if domain_discriminator_flag == 1:
                    domain_losses = []
                    for k, logit in enumerate(id_logic):
                        domain_loss = id_criterion(logit, person_id_two)
                        domain_losses.append(domain_loss)
                        FL_domain_loss[k] += domain_loss
                    domain_losses = torch.stack(domain_losses, dim=0)
                    domain_losses_avg = domain_losses.mean(dim=0)
                    loss += domain_losses_avg

                optimizer.zero_grad()                
                loss.backward()
                optimizer.step()
                rest = timer.step()/3600
                
                domain_right_batch = [torch.tensor(0.0).cuda() for i in range(5)]
                if domain_discriminator_flag == 1:
                    accuracy = []
                    for k, logit in enumerate(id_logic):
                        _, predict_1 = torch.max(logit, 1)
                        # 计算准确率
                        domain_right_batch[k] = torch.sum(predict_1 == person_id_two)
                        domain_right[k] += domain_right_batch[k]
                        # correct_1 = (predict_1 == person_id_two).sum()
                        # acc_1 = torch.tensor(correct_1.item() / person_id_two.size(0) * 100)
                        # accuracy.append(acc_1)
                
                data_shape = data["face"].shape[0]
                class_total += data_shape
                acc = [float(right / class_total) for right in domain_right]
                
                if i % 20 == 0:
                    if domain_discriminator_flag == 1:
                        print(acc)
                    log = f"[{epoch}/{params.epoch}]: " + \
                          f"[{i}/{length}] " +\
                          f"lr:{ctools.GetLR(optimizer)} " +\
                          f"ori_loss1:{ori_loss1:.6f} " +\
                          f"ori_loss2:{ori_loss2:.6f} " +\
                          f"mmd_loss:{mmd_loss:.6f} " +\
                          f"domain_loss:{domain_losses_avg:.6f} " +\
                          f"rest time:{rest:.2f}h"
                    
                    print(log); outfile.write(log + "\n")
                    sys.stdout.flush(); outfile.flush()
                    
            domain_acc = [float(right / class_total) for right in domain_right]
            print(domain_acc)
            
            state_dict = {
                "resnet18": model.state_dict(),
                "mlp": mlp.state_dict()
            }
            if (epoch) % save.step == 0:
                torch.save(
                        state_dict, 
                        os.path.join(
                            savepath, 
                            f"Iter_{epoch}_{save.model_name}.pt"
                            )
                        )
            scheduler.step()
        
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Pytorch Basic Model Training')

    parser.add_argument('-s', '--train', type=str,
                        help='The source config for training.')
    parser.add_argument('-m', '--model', type=str, default='res18',
                        help='Model_type.')

    args = parser.parse_args()

    config = edict(yaml.load(open(args.train), Loader=yaml.FullLoader))

    main(config.train, args.model)
