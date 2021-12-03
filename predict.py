import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pickle
import torch
from jointset import jointset
from vmse import vmse
from resnet import resnet, resnet34, resnet18
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = resnet()
model.to(device)
model.eval()
criterion = vmse()
optimizer = torch.optim.Adam(model.parameters())
checkpoint_f = "./checkpoints/model.pt"
checkpoint = torch.load(checkpoint_f)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
resize = transforms.Resize((256,192))
transform = transforms.Compose([resize, transforms.ToTensor(), normalize])
fdir = "golf_frames"
files = glob.glob(fdir + "/" +"*.jpg")
#files = glob.glob("tennis*.jpg")
files = sorted(files)
def denormalize_coords(pts,bw=192,bh=256,bc = (96,128)):
        pts[:,0]*= bw
        pts[:,1]*=bh
        pts[:,0]+=bc[0]
        pts[:,1]+=bc[1]
        return pts

def recenter_coords(pts,orig_shape):
    pts[:,0]*= orig_shape[1]/192
    pts[:,1]*= orig_shape[0]/256
    return pts
from utils import plotoverlay
pred_list = []
start = time.time()
for f in files:
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    im = cv2.imread(f)
    im = cv2.filter2D(src=im, ddepth = -1, kernel = kernel) #sharpen the image 
    pil_im = Image.fromarray(np.uint8(im))
    t = transform(pil_im)
    ex = t.unsqueeze(dim=0).to(device)
    preds = model(ex)
    preds = preds.squeeze().detach().cpu().numpy().astype(np.float32)
    preds = denormalize_coords(preds, 192, 256, (96,128))
    im2 = np.array(pil_im.resize((192,256)))

    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    #plotoverlay(preds,im2)

    preds2 = recenter_coords(preds.copy(),im.shape)
    fpath = "res_dir/" +f.split("/")[-1] 
    print(fpath)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    #plotoverlay(preds2,im,fpath )
    pred_list.append(preds)
end = time.time()    
print ("Time elapsed:", end - start)
'''
with open("predictions.pkl", 'wb') as f:
        pickle.dump(pred_list,f)
'''