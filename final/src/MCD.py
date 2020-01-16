import os
import sys
import argparse
import collections
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

class PairedData(object):
    def __init__(self, data_loader_A, data_loader_B, max_dataset_size):
        self.data_loader_A = data_loader_A
        self.data_loader_B = data_loader_B
        self.stop_A = False
        self.stop_B = False
        self.max_dataset_size = max_dataset_size

    def __iter__(self):
        self.stop_A = False
        self.stop_B = False
        self.data_loader_A_iter = iter(self.data_loader_A)
        self.data_loader_B_iter = iter(self.data_loader_B)
        self.iter = 0
        return self

    def __next__(self):
        A, A_paths = None, None
        # B, B_paths = None, None
        B = None
        try:
            A, A_paths = next(self.data_loader_A_iter)
        except StopIteration:
            if A is None or A_paths is None:
                self.stop_A = True
                self.data_loader_A_iter = iter(self.data_loader_A)
                A, A_paths = next(self.data_loader_A_iter)

        try:
            # B, B_paths = next(self.data_loader_B_iter)
            B = next(self.data_loader_B_iter)
        except StopIteration:
            # if B is None or B_paths is None:
            if B is None:
                self.stop_B = True
                self.data_loader_B_iter = iter(self.data_loader_B)
                B = next(self.data_loader_B_iter)

        if (self.stop_A and self.stop_B) or self.iter > self.max_dataset_size:
            self.stop_A = False
            self.stop_B = False
            raise StopIteration()
        else:
            self.iter += 1
            # return {'S': A, 'S_label': A_paths, 'T': B, 'T_label': B_paths}
            return {'S': A, 'S_label': A_paths, 'T': B}

class UnalignedDataLoader():
    def initialize(self, data_loader_s, data_loader_t):
        self.dataset_s = data_loader_s.dataset
        self.dataset_t = data_loader_t.dataset
        self.paired_data = PairedData(data_loader_s, data_loader_t, float("inf"))

    def name(self):
        return 'UnalignedDataLoader'

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return min(max(len(self.dataset_s), len(self.dataset_t)), float("inf"))

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
        # self.fc1 = nn.Linear(8192, 3072)
        # self.bn1_fc = nn.BatchNorm1d(3072)
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

class Solver(object):
    def __init__(self, datasets, batch_size=64, source='real', target='fake',
                 interval=100, optimizer='adam', learning_rate=2e-4,
                 num_k=4, checkpoint_dir=None, save_epoch=10, use_abs_diff=True):
        self.batch_size = batch_size
        self.source = source
        self.target = target
        self.num_k = num_k
        self.checkpoint_dir = checkpoint_dir
        self.save_epoch = save_epoch
        # self.use_abs_diff = use_abs_diff
        print('dataset loading', flush=True)
        self.datasets = datasets
        print('load finished!', flush=True)

        self.G = Generator().to(DEVICE)
        self.C1 = Classifier().to(DEVICE)
        self.C2 = Classifier().to(DEVICE)

        self.interval = interval

        self.set_optimizer(which_opt=optimizer, lr=learning_rate)
        self.lr = learning_rate

    def set_optimizer(self, which_opt='momentum', lr=0.001, momentum=0.9):
        if which_opt == 'momentum':
            self.opt_g = optim.SGD(self.G.parameters(),
                                   lr=lr, weight_decay=0.0005,
                                   momentum=momentum)
            self.opt_c1 = optim.SGD(self.C1.parameters(),
                                    lr=lr, weight_decay=0.0005,
                                    momentum=momentum)
            self.opt_c2 = optim.SGD(self.C2.parameters(),
                                    lr=lr, weight_decay=0.0005,
                                    momentum=momentum)

        if which_opt == 'adam':
            self.opt_g = optim.Adam(self.G.parameters(),
                                    lr=lr, weight_decay=0.0005)
            self.opt_c1 = optim.Adam(self.C1.parameters(),
                                     lr=lr, weight_decay=0.0005)
            self.opt_c2 = optim.Adam(self.C2.parameters(),
                                     lr=lr, weight_decay=0.0005)

    def reset_grad(self):
        self.opt_g.zero_grad()
        self.opt_c1.zero_grad()
        self.opt_c2.zero_grad()

    def ent(self, output):
        return - torch.mean(output * torch.log(output + 1e-6))

    def discrepancy(self, out1, out2):
        return torch.mean(torch.abs(F.softmax(out1, dim=1) - F.softmax(out2, dim=1)))

    def train(self, epoch, record_file=None):
        criterion = nn.CrossEntropyLoss()
        self.G.train()
        self.C1.train()
        self.C2.train()
        torch.cuda.manual_seed(42)

        for batch_idx, data in enumerate(self.datasets):
            img_t = data['T']
            img_s = data['S']
            label_s = data['S_label']
            if img_s.size()[0] < self.batch_size or img_t.size()[0] < self.batch_size:
                break
            img_s = img_s.to(DEVICE)
            img_t = img_t.to(DEVICE)
            imgs = Variable(torch.cat((img_s, \
                                       img_t), 0))
            label_s = Variable(label_s.long().to(DEVICE))

            img_s = Variable(img_s)
            img_t = Variable(img_t)
            self.reset_grad()
            feat_s = self.G(img_s)
            output_s1 = self.C1(feat_s)
            output_s2 = self.C2(feat_s)

            loss_s1 = criterion(output_s1, label_s)
            loss_s2 = criterion(output_s2, label_s)
            loss_s = loss_s1 + loss_s2
            loss_s.backward()
            self.opt_g.step()
            self.opt_c1.step()
            self.opt_c2.step()
            self.reset_grad()

            feat_s = self.G(img_s)
            output_s1 = self.C1(feat_s)
            output_s2 = self.C2(feat_s)
            feat_t = self.G(img_t)
            output_t1 = self.C1(feat_t)
            output_t2 = self.C2(feat_t)

            loss_s1 = criterion(output_s1, label_s)
            loss_s2 = criterion(output_s2, label_s)
            loss_s = loss_s1 + loss_s2
            loss_dis = self.discrepancy(output_t1, output_t2)
            loss = loss_s - loss_dis
            loss.backward()
            self.opt_c1.step()
            self.opt_c2.step()
            self.reset_grad()

            for i in range(self.num_k):
                feat_t = self.G(img_t)
                output_t1 = self.C1(feat_t)
                output_t2 = self.C2(feat_t)
                loss_dis = self.discrepancy(output_t1, output_t2)
                loss_dis.backward()
                self.opt_g.step()
                self.reset_grad()
            if batch_idx > 500:
                return batch_idx

            if batch_idx % self.interval == 0:
                print('Train Epoch: {:3d} \tLoss1: {:.6f}\t Loss2: {:.6f}\t Discrepancy: {:.6f}'.format(epoch, loss_s1.item(), loss_s2.item(), loss_dis.item()), flush=True)
                if record_file:
                    record = open(record_file, 'a')
                    record.write('%s,%s,%s\n' % (loss_dis.item(), loss_s1.item(), loss_s2.item()))
                    record.close()
        return batch_idx

    def save(self, epoch):
        if epoch % self.save_epoch == 0:
            torch.save(self.G, f'{self.checkpoint_dir}/model_epoch{epoch:03d}_G.pt')
            torch.save(self.C1, f'{self.checkpoint_dir}/model_epoch{epoch:03d}_C1.pt')
            torch.save(self.C2, f'{self.checkpoint_dir}/model_epoch{epoch:03d}_C2.pt')
    
if __name__ == "__main__":
   # detect is gpu available.
    use_cuda = torch.cuda.is_available()
    DEVICE = torch.device('cuda' if use_cuda else 'cpu')
    CHECKPOINT_DIR = 'checkpoint'
    if not os.path.exists(CHECKPOINT_DIR):
        os.mkdir(CHECKPOINT_DIR)
    if not os.path.exists('record'):
        os.mkdir('record')

    # source_image = np.load('data/trainX.npy')
    # source_label = np.load('data/trainY.npy')
    # target_image = np.load("data/testX.npy")

    arg_names = [ 'py',
                  'source_x', 'source_y', 'target_x', 
                  'batch_size', 'epoch', 'save_epoch', 'num_k',
                  'optimizer', 'lr',
                ]
    args = dict(zip(arg_names, sys.argv))
    print(args)

    source_image = np.load(sys.argv[1])
    source_label = np.load(sys.argv[2])
    target_image = np.load(sys.argv[3])

    BATCH_SIZE = int(args['batch_size'])
    EPOCH = int(args['epoch'])
    SAVE_EPOCH = int(args['save_epoch'])
    num_K = int(args['num_k'])
    OPTIM = args['optimizer']
    LR = float(args['lr'])

    source_image = np.transpose(source_image, (0, 3, 1, 2))
    target_image = np.transpose(target_image, (0, 3, 1, 2))

    transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    sourceDataset = SourceDataset(source_image, source_label, transform=transform)
    targetDataset = TargetDataset(target_image, transform)

    source_dataloader = DataLoader(sourceDataset, batch_size=BATCH_SIZE,
                                   shuffle=True, num_workers=4)
    target_dataloader = DataLoader(targetDataset, batch_size=BATCH_SIZE,
                                   shuffle=True, num_workers=4)

    dataloader = UnalignedDataLoader()
    dataloader.initialize(source_dataloader, target_dataloader)

    datasets = dataloader.load_data()

    solver = Solver(datasets, batch_size=BATCH_SIZE, num_k=num_K,
                    optimizer=OPTIM, learning_rate=LR,
                    checkpoint_dir=CHECKPOINT_DIR, save_epoch=SAVE_EPOCH)

    record_num = 0
    record_train = f'record/k_{num_K}_{record_num}.csv' # loss_dis, loss_s1, loss_s2
    while os.path.exists(record_train):
        record_num += 1
        record_train = f'record/k_{num_K}_{record_num}.csv'

    count = 0
    for t in range(1, EPOCH+1):
        num = solver.train(t, record_file=record_train)
        count += num
        if t % 1 == 0:
            solver.save(t)
        # if count >= 20000:
            # break