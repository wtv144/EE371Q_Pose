import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pickle
import torch
from jointset import jointset
from vmse import vmse
from resnet import resnet, resnet34, resnet18, resnet101
import glob
import matplotlib.pyplot as plt
import cv2 
import numpy as np 
from utils import eval_PCK, plotoverlay

train_dir = "./data/train2017"
#train_dir = "/datasets/coco/coco_train2014"
train_a_path = "./data/annotations/person_keypoints_train2017.json"
#train_a_path = "/datasets/coco/annotations/person_keypoints_train2014.json"
val_dir = "./data/val2017"
#val_dir = "/datasets/coco/coco_val2014"
val_a_path ="./data/annotations/person_keypoints_val2017.json"
#val_a_path ="/datasets/coco/annotations/person_keypoints_val2014.json" 
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.ToTensor(), normalize])
#train_data = jointset(train_dir,train_a_path,True, transform)
val_data = jointset(val_dir,val_a_path,False, transform)
#train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle= False)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def denormalize_coords(pts,bw=192,bh=256,bc = (96,128)):
        pts[:,0]*= bw
        pts[:,1]*=bh
        pts[:,0]+=bc[0]
        pts[:,1]+=bc[1]
        return pts
def pck_wrapper(pred, gt):
    total = 0
    for i in range(pred.shape[0]):
        idx_loss = eval_PCK(pred[i],gt[i])
        total+= idx_loss
    return total/pred.shape[0]
files = glob.glob("checkpoints/*.pt")
for f in files:
    print(f)
    model = resnet()
    if "18" in f:
        print("model 18")
        model = resnet18()
    elif "34" in f:
        print("model 34")

        model = resnet34()
    elif "101" in f:
        print("model 101")
        model = resnet101()
    model.to(device)
    criterion = vmse()
    optimizer = torch.optim.Adam(model.parameters())
    model.eval()
    valid_loss = []
    checkpoint = torch.load(f)
    model.load_state_dict(checkpoint['model_state_dict'],strict=False )
    with torch.no_grad():
        for idx, data in enumerate(val_loader,0):
            inputs, labels, visibility, orig= data
            inputs = inputs.to(device)
            labels = labels.to(device)
            visibility = visibility.to(device)
            labels = labels.type(torch.float32) #did this bc of error of found dtype double but expected float
            outputs = model(inputs)
            img = orig.squeeze().detach().cpu().numpy()
            outputs = outputs.squeeze().detach().cpu().numpy().astype(np.float32)
            labels = labels.squeeze().detach().cpu().numpy().astype(np.float32)
            labels = denormalize_coords(labels)
            outputs = denormalize_coords(outputs, 192, 256, (96,128))
            #plotoverlay(outputs, img)
            loss = pck_wrapper(outputs,labels)
            valid_loss.append(loss)
    print(f, " : ", sum(valid_loss)/len(valid_loss))
