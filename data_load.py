import os
from torch.utils import data
from torchvision import transforms
from PIL import Image
import random
import torch.nn as nn
from torch.autograd import Variable
import torch


class KITTI_loader(data.Dataset):
    def __init__(self,root,transform = None,transform_t = None,train = True,KITTI = True):
            "makes directory list which lies in the root directory"
            if KITTI:
                dir_path = list(map(lambda x:os.path.join(root,x),os.listdir(root)))
                dir_path_deep=[]
                left_path=[]
                right_path=[]
                self.left_image_paths = []
                self.right_image_paths = []


                for dir_sub in dir_path:
                    dir_sub_dir=[]

                    dir_sub_list = [file for file in os.listdir(dir_sub) if not file.endswith(".txt")]
                    for name in dir_sub_list:
                        dir_sub_dir.append(name) 

                    dir_sub_path = list(map(lambda x:os.path.join(dir_sub,x),dir_sub_dir))
                    dir_sub_path.sort()  
                    for dir_sub_sub in dir_sub_path:

                        dir_path_deep.append(dir_sub_sub)   

                for dir_sub_sub in dir_path_deep:
                    left_name = os.listdir(os.path.join(dir_sub_sub,'image_02/data'))
                    right_name = os.listdir(os.path.join(dir_sub_sub,'image_03/data'))
                    left_name.sort()
                    right_name.sort()
                    for left_image_name in left_name:
                        self.left_image_paths.append(os.path.join(dir_sub_sub,'image_02/data',left_image_name))         
                    for right_image_name in right_name:
                        self.right_image_paths.append(os.path.join(dir_sub_sub,'image_03/data',right_image_name))         
                self.transform = transform
                self.transform_t = transform_t
                self.train = train
                self.KITTI = KITTI
 
    def __getitem__(self,index):
           
        if self.KITTI == True:
            
            left_path = self.left_image_paths[index]
            right_path = self.right_image_paths[index]
                
            left = Image.open(left_path).convert('RGB')
            right = Image.open(right_path).convert('RGB')
            data=[]
        if self.transform is not None:
            data.append(self.transform(left))
            data.append(self.transform(right))
        return data

    def __len__(self):
        
        return len(self.left_image_paths)
