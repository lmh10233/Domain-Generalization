import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import math


class Gaze_regressor(nn.Module):
    def __init__(self, in_dim=1000, hidden_dim=256, out_dim=2, drop=0.5):
        super(Gaze_regressor, self).__init__()

        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(in_dim, hidden_dim)
        self.last_layer = nn.Linear(hidden_dim, out_dim)
        self.act = nn.ReLU(inplace=True)
        self.loss_op = nn.L1Loss()

    def forward(self, x):
        x = self.fc(x)
        x = self.drop(x)
        x = self.last_layer(x)
        return x
    
    def loss(self, x_in, label):
        gaze = self.forward(x_in)
        loss = self.loss_op(gaze, label)
        return loss
        


if __name__ == '__main__':
    model = Gaze_regressor()
    print(model)