import os
import sys
import glob
import argparse
import re
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable, Function
from torch.optim import RMSprop, Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
import torchvision
from torchvision import models
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# from torchsummary import summary

class SourceDataset(Dataset):
    def __init__(self, data, label, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.data = data
        self.labels = label

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # print(img.shape)
        if img.shape[0] != 1:
            #print(img)
            img = Image.fromarray(np.uint8(np.asarray(img.transpose((1, 2, 0)))))

        elif img.shape[0] == 1:
            im = np.uint8(np.asarray(img))
            # print(np.vstack([im,im,im]).shape)
            im = np.vstack([im, im, im]).transpose((1, 2, 0))
            img = Image.fromarray(im)

        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

class TargetDataset(Dataset):
    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = data

    def __getitem__(self, index):
        img = self.data[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # print(img.shape)
        if img.shape[0] != 1:
            #print(img)
            img = Image.fromarray(np.uint8(np.asarray(img.transpose((1, 2, 0)))))
        #
        elif img.shape[0] == 1:
            im = np.uint8(np.asarray(img))
            # print(np.vstack([im,im,im]).shape)
            im = np.vstack([im, im, im]).transpose((1, 2, 0))
            img = Image.fromarray(im)

        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.data)

class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)

def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(8192, 3072)
        self.bn1_fc = nn.BatchNorm1d(3072)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), stride=2, kernel_size=3, padding=1)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), stride=2, kernel_size=3, padding=1)
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), 8192)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        return x

class Classifier(nn.Module):
    def __init__(self, prob=0.5):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(8192, 3072)
        self.bn1_fc = nn.BatchNorm1d(3072)
        self.fc2 = nn.Linear(3072, 2048)
        self.bn2_fc = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(2048, 10)
        self.bn_fc3 = nn.BatchNorm1d(10)
        self.prob = prob

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        x = self.fc3(x)
        return x

resnet_dict = {"ResNet18": models.resnet18, "ResNet34": models.resnet34,
               "ResNet50": models.resnet50, "ResNet101": models.resnet101,
               "ResNet152": models.resnet152, "ResNeXt50": models.resnext50_32x4d,
               "ResNeXt101": models.resnext101_32x8d}
# in_features: 2048

class ResNetGenerator(nn.Module):
    def __init__(self, model_name):
        super(ResNetGenerator, self).__init__()
        model = resnet_dict[model_name](pretrained=True)
        self.feature_layers = nn.Sequential(*list(model.children())[:-2])
        self.avgpool = model.avgpool
        self.in_features = model.fc.in_features

    def forward(self, x):
        x = self.feature_layers(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), self.in_features)
        x = F.dropout(x, training=self.training)
        return x

densenet_dict = {"Densenet121": models.densenet121, "Desnsenet169": models.densenet169,
                 "Desnsenet161": models.densenet161, "Densenet201": models.densenet201}
# in_features: 1024, 1664, 2208, 1920

class DensenetGenerator(nn.Module):
    def __init__(self, model_name):
        super(DensenetGenerator, self).__init__()
        model = densenet_dict[model_name](pretrained=True)
        self.feature_layers = model.features
        self.in_features = model.classifier.in_features

    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), self.in_features)
        x = F.dropout(x, training=self.training)
        return x

class FlexClassifier(nn.Module):
    def __init__(self, in_features=2048, hidden_num=2048, num_class=10, prob=0.5):
        super(FlexClassifier, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_num)
        self.bn1_fc = nn.BatchNorm1d(hidden_num)
        self.fc2 = nn.Linear(hidden_num, num_class)
        self.bn_fc2 = nn.BatchNorm1d(num_class)
        self.prob = prob

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = self.fc2(x)
        return x
def print_dist(pred, n=10):
    for i in range(n):
        print(f"{(pred==i).sum()}", end=" ")
    print("", flush=True)
    for i in range(n):
        print(f"{(pred==i).sum()*100.0/(1.0 * len(pred)):.1f}%", end=" ")
    print("", flush=True)

if __name__ == "__main__":
   # detect is gpu available.
    use_cuda = torch.cuda.is_available()
    DEVICE = torch.device('cuda' if use_cuda else 'cpu')
    # CHECKPOINT_DIR = 'checkpoint'
    # if not os.path.exists(CHECKPOINT_DIR):
        # os.mkdir(CHECKPOINT_DIR)
    # if not os.path.exists('record'):
        # os.mkdir('record')
    if not os.path.exists('pred'):
        os.mkdir('pred')

    # source_image = np.load('data/trainX.npy')
    # source_label = np.load('data/trainY.npy')
    # target_image = np.load("data/testX.npy")
    source_image = np.load(sys.argv[1])
    source_label = np.load(sys.argv[2])
    target_image = np.load(sys.argv[3])

    checkpoint_dir = sys.argv[4]

    source_image = np.transpose(source_image, (0, 3, 1, 2))
    target_image = np.transpose(target_image, (0, 3, 1, 2))
    # x_train, x_valid, y_train, y_valid = train_test_split(source_image, source_label, test_size=0.25, random_state=42)

    transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    sourceDataset = SourceDataset(source_image, source_label, transform=transform)
    targetDataset = TargetDataset(target_image, transform)

    # source_dataloader = DataLoader(sourceDataset, batch_size=128, shuffle=True, num_workers=4)
    target_dataloader = DataLoader(targetDataset, batch_size=200, shuffle=False, num_workers=4)

    # G = Generator()
    # C1 = Classifier()
    # C2 = Classifier()

    # pred_dir = "k4...bs32/chechpoint"
    # torch.save(self.G, f'{self.chechpoint_dir}/model_epoch{epoch:03d}_G.pt')
    saved_epochs = []
    for pred_file in glob.glob(os.path.join(checkpoint_dir, "*_G.pt")):
        saved_epochs.append(int(re.search("(\d+)_G.pt$", pred_file).group(1)))

    for resume_epoch in sorted(saved_epochs):
        print(f"Epoch {resume_epoch}:")

        G = torch.load(f'{checkpoint_dir}/model_epoch{resume_epoch:03d}_G.pt', map_location=DEVICE)
        C1 = torch.load(f'{checkpoint_dir}/model_epoch{resume_epoch:03d}_C1.pt', map_location=DEVICE)
        C2 = torch.load(f'{checkpoint_dir}/model_epoch{resume_epoch:03d}_C2.pt', map_location=DEVICE)

        G = G.to(DEVICE)
        C1 = C1.to(DEVICE)
        C2 = C2.to(DEVICE)

        G.eval()
        C1.eval()
        C2.eval()

        num_elements = len(target_dataloader.dataset)
        num_batches = len(target_dataloader)
        predictions_1 = torch.zeros(num_elements).int()
        predictions_2 = torch.zeros(num_elements).int()
        predictions_3 = torch.zeros(num_elements).int()
        # print(num_elements, flush=True)
        with torch.no_grad():
            for batch_idx, img_t in enumerate(target_dataloader):
                start = batch_idx * target_dataloader.batch_size
                end = start + target_dataloader.batch_size
                if batch_idx == num_batches - 1:
                    end = num_elements

                img_t = img_t.to(DEVICE)
                feat_t = G(img_t)
                output_t1 = C1(feat_t)
                output_t2 = C2(feat_t)
                output_ensemble = output_t1 + output_t2
                pred_label_1 = torch.max(output_t1, 1)[1]
                pred_label_2 = torch.max(output_t2, 1)[1]
                pred_label_3 = torch.max(output_ensemble, 1)[1]
                predictions_1[start:end] = pred_label_1.int()
                predictions_2[start:end] = pred_label_2.int()
                predictions_3[start:end] = pred_label_3.int()

        # Check distruibution
        print("Pred1: ")
        print_dist(predictions_1)
        print("Pred2: ")
        print_dist(predictions_2)
        print("Pred3: ")
        print_dist(predictions_3)
        print()

        # Generate your submission
        df = pd.DataFrame({'id': np.arange(0, len(predictions_1)), 'label': predictions_1})
        df.to_csv(f'pred/epoch{resume_epoch:03d}_pred_F1.csv',index=False)

        df = pd.DataFrame({'id': np.arange(0, len(predictions_2)), 'label': predictions_2})
        df.to_csv(f'pred/epoch{resume_epoch:03d}_pred_F2.csv',index=False)

        df = pd.DataFrame({'id': np.arange(0, len(predictions_3)), 'label': predictions_3})
        df.to_csv(f'pred/epoch{resume_epoch:03d}_pred_F1_F2_ensemble.csv',index=False)
