import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
import pandas as pd
import pydicom, numpy as np, os, cv2
# from matplotlib import pyplot as plt
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image
# torch.cuda.set_device(1)

datasetRoot = '/home/edwardxji6/covid-19/siim-covid19-detection/'

def read_xray(path, voi_lut = True, fix_monochrome = True):
    # Original from: https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
    dicom = pydicom.read_file(path)
    
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to 
    # "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
               
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
        
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
        
    return data

class covid19Dataset(data.Dataset):
    def __init__(self, libraryfile='', transform=None, rootFolder = ''):
        lib = pd.read_csv(libraryfile)
        if(rootFolder == ''):
            raise NotImplementedError('Where is the dataset root folder???')
        self.imagePathes = lib['path'].to_list()
        self.targets = lib['type'].to_list()
        print('Number of images: {}'.format(len(self.imagePathes)))
        self.transform = transform
        self.rootFolder = rootFolder
    def __getitem__(self,index):
        img = read_xray(self.rootFolder + self.imagePathes[index])
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
        img = clahe.apply(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = Image.fromarray(img)
        # print(type(img))
        if self.transform is not None:
            img = self.transform(img)
        target = self.targets[index]
        return img, target
    def __len__(self):
        return len(self.imagePathes)