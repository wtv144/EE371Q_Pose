import torch
import torch.nn as nn

class vmse(nn.Module):
    def __init__(self):
        super(vmse,self).__init__()

    def forward(self, c_gt, c_pred, vis):
        #assume vis is going to be a 1d tensor of length 16
        #c_gt is the ground truth, pred is our model prediction
        d = torch.square(c_gt-c_pred)
        #apply visibility
        v2 = torch.stack([vis,vis], dim=2)
        d2 = d*v2
        #reduction sum
        return d2.sum()/vis.sum()