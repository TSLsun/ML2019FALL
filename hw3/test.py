import os
import sys
# import cv2
import json
import math
import copy
import numpy as np
import pandas as pd
import torch 
# print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
from PIL import Image

# train_img_dir = sys.argv[1]
# train_csv = sys.argv[2]
test_img_dir = sys.argv[1]
sample_submission = './sample_submission.csv'
output_csv = sys.argv[2]

use_cuda = torch.cuda.is_available()
DEVICE = torch.device('cuda' if use_cuda else 'cpu')
print("Device:", DEVICE)

def train_valid_split(X, y=None, valid_size=0.2, random_state=42):
    n_train = int((1 - valid_size) * len(X))
    n_test = len(X) - n_train
    n_samples = len(X)

    rng = np.random.RandomState(random_state)
    permutation = rng.permutation(n_samples)
    ind_test = permutation[:n_test]
    ind_train = permutation[n_test:(n_test + n_train)]

    X_train = X[ind_train]
    X_valid = X[ind_test]
    if y is not None:
        y_train = y[ind_train]
        y_valid = y[ind_test]
        return X_train, X_valid, y_train, y_valid
    else:
        return X_train, X_valid


class test_hw3(Dataset):

    def __init__(self, data_dir, sample_submission, transform):
        self.data_dir = data_dir
        self.name = pd.read_csv(sample_submission).to_numpy()
        self.transform = transform
    
    def __getitem__(self, index):
        pic_file = '{:0>4d}.jpg'.format(self.name[index][0])
        # img = cv2.imread(os.path.join(self.data_dir, pic_file), cv2.IMREAD_GRAYSCALE)
        # img = np.expand_dims(img, 0)
        img = Image.open(os.path.join(self.data_dir, pic_file)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return self.name.shape[0]


class resnext50_32x4d(nn.Module):
    def __init__(self):
        super(resnext50_32x4d, self).__init__()
        self.resnet = nn.Sequential(*list(models.resnext50_32x4d(pretrained=True).children())[:-1])
        self.fc = nn.Linear(2048,7)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == '__main__':

    transform = transforms.Compose([
        transforms.RandomAffine(15, translate=(0.1,0.1), scale=(0.9,1.1), shear=10, fillcolor=0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize([mean], [std], inplace=False)
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize([mean], [std], inplace=False)
        ])

    inception_transform = transforms.Compose([
        transforms.Resize(299),
        transforms.ToTensor(),
        #transforms.Normalize([mean], [std], inplace=False)
        ])


    model_1 = resnext50_32x4d()
    model_2 = resnext50_32x4d()
    model_3 = resnext50_32x4d()

    model = models.inception_v3(pretrained=True)
    model.aux_logits=False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 7)
    
    model_path_1 = 'models/resnext50_epoch150-epoch_150_64.15.pth'
    model_path_2 = 'models/resnext50_epoch300-epoch_267_trainAcc_0.997.pth'
    model_path_3 = 'models/resnext50_epoch300-max_val_acc.pkl'
    model_path_4 = 'models/inceptionV3_earlystop30-epoch_46_trainAcc_0.991.pth'

    model_1.load_state_dict(torch.load(model_path_1))
    model_2.load_state_dict(torch.load(model_path_2))
    model_3.load_state_dict(torch.load(model_path_3))
    model.load_state_dict(torch.load(model_path_4))

    ### predict
    test_dataset = test_hw3(test_img_dir, sample_submission, test_transform)
    test_dataset_inception = test_hw3(test_img_dir, sample_submission, inception_transform)
    test_loader = DataLoader(test_dataset, batch_size=256)
    test_loader_inception = DataLoader(test_dataset_inception, batch_size=256)


    ## load model
    model_1 = model_1.to(DEVICE)
    model_2 = model_2.to(DEVICE)
    model_3 = model_3.to(DEVICE)
    model = model.to(DEVICE)
    model_1.eval()
    model_2.eval()
    model_3.eval()
    model.eval()

    num_elements = len(test_loader.dataset)
    num_batches = len(test_loader)
    resnext_out = torch.zeros((num_elements,7))
    inception_out = torch.zeros((num_elements,7))
    with torch.no_grad():
        for batch_idx, img in enumerate(test_loader):
            start = batch_idx * test_loader.batch_size
            end = start + test_loader.batch_size
            if batch_idx == num_batches - 1:
                end = num_elements

            img = img.to(DEVICE)
            out_1 = model_1(img)
            out_2 = model_2(img)
            out_3 = model_3(img)
            avg_out = (out_1 + out_2 + out_3) / 3.0 
            
            resnext_out[start:end] = avg_out
            # _, pred_label = torch.max(avg_out, 1)
            # predictions[start:end] = pred_label

        for batch_idx, img in enumerate(test_loader_inception):
            start = batch_idx * test_loader_inception.batch_size
            end = start + test_loader_inception.batch_size
            if batch_idx == num_batches - 1:
                end = num_elements

            img = img.to(DEVICE)
            out = model(img)
            
            inception_out[start:end] = out
    
    _, pred_label = torch.max((resnext_out+inception_out)/2.0, 1)
    predictions = pred_label


    with open(output_csv, 'w') as f:
            f.write('id,label\n')
            for i, v in enumerate(predictions):
                f.write('%d,%d\n' %(i, v.item()))

    for i in range(7):
        cnt = (predictions==i).sum().item()
        print(f"{i}: {cnt:4d} {cnt*100.0/predictions.size(0):4.2f}%")
