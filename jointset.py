from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from torchvision import transforms, utils
import os
import torch
from pycocotools.coco import COCO
from PIL import Image
import numpy as np 
import pandas as pd
import math 
import cv2
import argparse
class jointset(Dataset):
    def __init__(self,img_dir, a_path, is_train, transform, isTest = False):
        self.img_dir = img_dir
        self.a_path = a_path
        self.is_train = is_train 
        #build the df of images 
        coco = COCO(a_path)
        images_df, persons_df = self.convert_to_df(coco)
        df = pd.merge(images_df, persons_df, right_index=True, left_index=True)
        df = df[df['num_keypoints'] > 14]
        self.df = df[df['is_crowd'] == 0 ]
        self.transform = transform
        if isTest:
            self.df = self.df.iloc[:1000]

    def __getitem__(self, idx):
        a = self.df[idx:idx+1]
        path = a['path'].values[0]
        bbox = a['bbox'].values[0]
        x1 = math.floor(bbox[0])
        x2 = math.ceil(x1 + bbox[2])
        y1 = math.floor(bbox[1])
        y2 = math.ceil(y1 + bbox[3])
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = Image.open(path)
        #w,h = img.size
        #img = np.array(img)
        img = img[y1:y2,x1:x2]
        pts = np.array(a['keypoints'].values[0])
        pts = pts.reshape(-1,3)
        pts = pts[:,:2]
        visibility = torch.zeros(17)
        ptss= pts.copy()
        x_scale = 192/img.shape[1]
        y_scale = 256/img.shape[0]
        img = cv2.resize(img,(192,256))
        ptss[:,0] = np.floor((ptss[:,0] - x1)*x_scale)
        ptss[:,1] = np.floor((ptss[:,1] - y1 ) * y_scale)       
        for i in range(ptss.shape[0]):
            if ptss[i,0] > 0 and ptss[i,0] <= 192 and ptss[i,1]> 0 and ptss[i,1]<= 256:
                visibility[i] = 1
        #shift coords for scaling
        ptss= ptss.astype(np.float64)
        ptss = self.normalize_coords(ptss.copy(), 192,256, ( 96,128))
        #get indicators for visibility
        if self.transform:
            img2 = self.transform(img.copy())
        return img2, torch.tensor(ptss), visibility, img 

    def normalize_coords(self,pts, bw,bh,bc):
        pts[:,0] = (pts[:,0] - bc[0])/bw
        pts[:,1] = (pts[:,1] - bc[1])/bh
        return pts
    def denormalize_coords(self,pts,bw,bh,bc):
        pts[:,0]*= bw
        pts[:,1]*=bh
        pts[:,0]+=bc[0]
        pts[:,1]+=bc[1]
        return pts
    def __len__(self):
        return self.df.shape[0]

    def convert_to_df(self,coco):
        images_data = []
        persons_data = []
        # iterate over all images
        for img_id, img_fname, w, h, meta in self.get_meta(coco):
            images_data.append({
                'image_id': int(img_id),
                'path': os.path.join(self.img_dir,img_fname),
                'width': int(w),
                'height': int(h)
            })
            # iterate over all metadata
            for m in meta:
                persons_data.append({
                    'image_id': m['image_id'],
                    'is_crowd': m['iscrowd'],
                    'bbox': m['bbox'],
                    'area': m['area'],
                    'num_keypoints': m['num_keypoints'],
                    'keypoints': m['keypoints'],
                })
        # create dataframe with image paths
        images_df = pd.DataFrame(images_data)
        images_df.set_index('image_id', inplace=True)
        # create dataframe with persons
        persons_df = pd.DataFrame(persons_data)
        persons_df.set_index('image_id', inplace=True)
        return images_df, persons_df
    def get_meta(self,coco):
        ids = list(coco.imgs.keys())
        for i, img_id in enumerate(ids):
            img_meta = coco.imgs[img_id]
            ann_ids = coco.getAnnIds(imgIds=img_id)
            # basic parameters of an image
            img_file_name = img_meta['file_name']
            w = img_meta['width']
            h = img_meta['height']
            # retrieve metadata for all persons in the current image
            anns = coco.loadAnns(ann_ids)

            yield [img_id, img_file_name, w, h, anns]