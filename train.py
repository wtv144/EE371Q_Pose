from jointset import jointset
from resnet import resnet 
import numpy as np
import cv2 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pickle
import argparse
from vmse import vmse
parser = argparse.ArgumentParser()
parser.add_argument('--load_model',  action = 'store_true')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#train_dir = "./data/train2017"
train_dir = "/datasets/coco/coco_train2014"
#train_a_path = "./data/annotations/person_keypoints_train2017.json"
train_a_path = "/datasets/coco/annotations/person_keypoints_train2014.json"
#val_dir = "./data/val2017"
val_dir = "/datasets/coco/coco_val2014"
#val_a_path ="./data/annotations/person_keypoints_val2017.json"
val_a_path ="/datasets/coco/annotations/person_keypoints_val2014.json" 
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.ToTensor(), normalize])
train_data = jointset(train_dir,train_a_path,True, transform)
val_data = jointset(val_dir,val_a_path,False, transform)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
val_loader = DataLoader(val_data, batch_size=128, shuffle= False)
model = resnet()
model.to(device)
criterion = vmse()
optimizer = torch.optim.Adam(model.parameters())
epochs = 60
checkpoint_f = "./checkpoints/model.pt"
epoch = 0 
def store_model(epoch):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_f)
def load_model():
    global epoch, model, optimizer
    checkpoint = torch.load(checkpoint_f)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']+1
    print("model loaded")
training_losses = []
valid_losses = []
def load_losses():
    global training_losses, valid_losses
    with open("./checkpoints/losses.pkl", 'rb') as f:
        loss_dict = pickle.load(f)
        training_losses = loss_dict["training"]
        valid_losses = loss_dict['valid']
        
def store_losses():
    d = {"training":training_losses, "valid":valid_losses}
    with open("./checkpoints/losses.pkl", 'wb') as f:
        pickle.dump(d,f)
    
if args.load_model:
    print("Loading model")
    load_model()
    load_losses()

def train():
    model.train()
    train_loss = 0
    for idx, data in enumerate(train_loader,0):
            inputs, labels, visibility= data
            inputs = inputs.to(device)
            labels = labels.to(device)
            visibility = visibility.to(device)
            labels = labels.type(torch.float32) #did this bc of error of found dtype double but expected float
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(labels,outputs,visibility) 
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
    training_losses.append(train_loss/len(train_loader))
    print(training_losses[-1])
def valid():
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for idx, data in enumerate(val_loader,0):
            inputs, labels, visibility= data
            inputs = inputs.to(device)
            labels = labels.to(device)
            visibility = visibility.to(device)
            labels = labels.type(torch.float32) #did this bc of error of found dtype double but expected float
            outputs = model(inputs)
            loss = criterion(labels, outputs, visibility) 
            valid_loss+=loss.item()
    valid_losses.append(valid_loss/len(val_loader))
        
    
for e in range(epoch, epochs):
        print("Epoch no.", e)
        train()
        #valid()
        store_model(e)
        store_losses()

            