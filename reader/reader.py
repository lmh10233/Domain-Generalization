import os
import cv2 
import torch
import random
import numpy as np
from easydict import EasyDict as edict
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision.transforms import ToPILImage
from transforms.transform import Transform
from torchvision import datasets
from PIL import Image


def Decode_MPII(line):
    anno = edict()
    line[0] = line[0].replace('\\', r'/')
    line[1] = line[1].replace('\\', r'/')
    line[2] = line[2].replace('\\', r'/')
    anno.face, anno.lefteye, anno.righteye = line[0], line[1], line[2]
    anno.name = line[3]

    anno.gaze3d, anno.head3d = line[5], line[6]
    anno.gaze2d, anno.head2d = line[7], line[8]
    anno.id = None
    return anno

def Decode_Diap(line):
    anno = edict()
    line[0] = line[0].replace('\\', r'/')
    line[1] = line[1].replace('\\', r'/')
    line[2] = line[2].replace('\\', r'/')
    anno.face, anno.lefteye, anno.righteye = line[0], line[1], line[2]
    anno.name = line[3]

    anno.gaze3d, anno.head3d = line[4], line[5]
    anno.gaze2d, anno.head2d = line[6], line[7]
    anno.id = None

    return anno

def Decode_Gaze360(line):
    anno = edict()
    line[0] = line[0].replace('\\', r'/')
    line[1] = line[1].replace('\\', r'/')
    line[2] = line[2].replace('\\', r'/')
    # anno.face, anno.lefteye, anno.righteye = line[0], line[1], line[2]
    anno.face = line[0]
    anno.name = line[3]

    anno.gaze3d = line[4]
    anno.gaze2d = line[5]
    anno.id = line[-1]
    return anno

def Decode_ETH(line):
    anno = edict()
    line[0] = line[0].replace('\\', r'/')
    line[1] = line[1].replace('\\', r'/')
    line[2] = line[2].replace('\\', r'/')
    anno.face = line[0]
    anno.gaze2d = line[1]
    anno.head2d = line[2]
    anno.name = line[3]
    anno.id = line[7]
    return anno

def Decode_RTGene(line):
    anno = edict()
    line[0] = line[0].replace('\\', r'/')
    anno.face = line[0]
    anno.gaze2d = line[6]
    anno.head2d = line[7]
    anno.name = line[0]
    anno.id = None
    return anno

def Decode_GazeCapture(line):
    anno = edict()
    anno.face = line[0]
    anno.gaze2d = line[1]
    anno.head2d = line[2]
    anno.name = line[3]
    return anno

def Decode_Dict():
    mapping = edict()
    mapping.mpiigaze = Decode_MPII
    mapping.eyediap = Decode_Diap
    mapping.gaze360 = Decode_Gaze360
    mapping.eth = Decode_ETH
    mapping.gazecapture = Decode_GazeCapture
    return mapping


def long_substr(str1, str2):
    substr = ''
    for i in range(len(str1)):
        for j in range(len(str1)-i+1):
            if j > len(substr) and (str1[i:i+j] in str2):
                substr = str1[i:i+j]
    return len(substr)


def Get_Decode(name):
    mapping = Decode_Dict()
    keys = list(mapping.keys())
    name = name.lower()
    score = [long_substr(name, i) for i in keys]
    key  = keys[score.index(max(score))]
    return mapping[key]
    

class trainloader(Dataset): 
  def __init__(self, dataset, trans):

    # Read source data
    self.dataset = dataset
    self.data = edict() 
    self.data.line = []
    self.data.root = dataset.image
    self.data.decode = Get_Decode(dataset.name)
    self.trans = trans
    
    if isinstance(dataset.label, list):

      for i in dataset.label:

        with open(i) as f: line = f.readlines()

        if dataset.header: line.pop(0)

        self.data.line.extend(line)

    else:

      with open(dataset.label) as f: self.data.line = f.readlines()

      if dataset.header: self.data.line.pop(0)

    ## build transforms
    if self.trans:
        self.transform = Transform(input_size=224)
        
    else:
        self.transform = None
        
  def __len__(self):

    return len(self.data.line)


  def __getitem__(self, idx):

    # Read souce information
    line = self.data.line[idx]
    line = line.strip().split(" ")
    anno = self.data.decode(line)

    # img = Image.open(os.path.join(self.data.root, anno.face)).convert('RGB')
    img = cv2.imread(os.path.join(self.data.root, anno.face))
    if self.transform:
        img = self.transform(img)
        img = np.array(img)
    img = img.transpose(2, 0, 1)
    # img = (img - img.min()) / (img.max() - img.min())
    img = img / 255 
    img = torch.from_numpy(img).type(torch.FloatTensor)

    label = np.array(anno.gaze2d.split(",")).astype("float")
    label = torch.from_numpy(label).type(torch.FloatTensor)
    
    data = edict()
    data.face = img
    data.name = anno.name
    if anno.id:
        data.id = anno.id
    
    return data, label


def loader(source, batch_size, shuffle=True, num_workers=0, trans=False, fine_tune=False):
    dataset = trainloader(source, trans)
    g=torch.Generator()
    g.manual_seed(0)
    
    if fine_tune:
        # percent_to_select = 100
        num_total = len(dataset)
        num_to_select = 100
        # num_to_select = int(num_total * (percent_to_select / 100.0))
        indices = torch.randperm(num_total)[:num_to_select]
        sampler = SubsetRandomSampler(indices)
        dataset = torch.utils.data.Subset(dataset, indices)  # sampler=sampler
    
    print(f"-- [Read Data]: Source: {source.label}")
    print(f"-- [Read Data]: Total num: {len(dataset)}")
    load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, generator=g)
    return load

