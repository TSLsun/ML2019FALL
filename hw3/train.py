import os
import sys
import cv2
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

# import matplotlib
# matplotlib.use("AGG")
# import matplotlib.pyplot as plt
# from torchsummary import summary

train_img_dir = sys.argv[1]
train_csv = sys.argv[2]
# test_img_dir = sys.argv[3]
# sample_submission = sys.argv[4]
# output_csv = "./pred.csv"

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

def load_label(train_csv):
    label = pd.read_csv(train_csv).to_numpy()
    return train_valid_split(label, valid_size=0.2, random_state=42)
    # return label


class train_hw3(Dataset):

    def __init__(self, data_dir, label, transform):
        self.data_dir = data_dir
        self.label = label
        self.transform = transform
    
    def __getitem__(self, index):
        pic_file = '{:0>5d}.jpg'.format(self.label[index][0])
        # img = cv2.imread(os.path.join(self.data_dir, pic_file), cv2.IMREAD_GRAYSCALE)
        # img = np.expand_dims(img, 0)
        img = Image.open(os.path.join(self.data_dir, pic_file)).convert('RGB')
        img = self.transform(img)
        return img, self.label[index, 1]

    def __len__(self):
        return self.label.shape[0]

# class test_hw3(Dataset):

    # def __init__(self, data_dir, sample_submission, transform):
        # self.data_dir = data_dir
        # self.name = pd.read_csv(sample_submission).to_numpy()
        # self.transform = transform
    
    # def __getitem__(self, index):
        # pic_file = '{:0>4d}.jpg'.format(self.name[index][0])
        # # img = cv2.imread(os.path.join(self.data_dir, pic_file), cv2.IMREAD_GRAYSCALE)
        # # img = np.expand_dims(img, 0)
        # img = Image.open(os.path.join(self.data_dir, pic_file)).convert('RGB')
        # img = self.transform(img)
        # return img

    # def __len__(self):
        # return self.name.shape[0]


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


def train(model, train_loader, valid_loader, EPOCH=10):
    model = model.to(DEVICE)
    # print(model, flush=True)
    # print(summary(model, (3,48,48)))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    model_min_val_loss = './model/min_val_loss.pkl'
    model_max_val_acc = './model/max_val_acc.pkl'
    model_best_acc_file = None
    history = {
            'loss': [],
            'val_loss': [],
            'acc': [],
            'val_acc': [],
        }
    min_val_cost = 1e10
    max_val_acc = 0
    max_train_acc = 0
    cnt = 0
    early_stop_steps = 50
    
    print()
    ### train
    for epoch in range(EPOCH):
        print(f"\nEpoch: {epoch+1}/{EPOCH}")
        model.train()
        train_loss = 0
        correct = 0
        for batch_idx, (data, label) in enumerate(train_loader):
            data, label = data.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, label)
            train_loss += loss.item() * data.size(0)
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(output, 1)
            correct += torch.sum(preds == label) 

        train_loss /= len(train_loader.dataset)
        epoch_acc = (correct.double() / len(train_loader.dataset)).item()    

        print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            train_loss, correct, len(train_loader.dataset), 100. * epoch_acc), flush=True)

        model.eval()
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                val_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss

                _, preds = torch.max(output, 1)
                val_correct += torch.sum(preds == target) 

        val_loss /= len(valid_loader.dataset)
        epoch_val_acc = (val_correct.double() / len(valid_loader.dataset)).item()
        print('Valid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            val_loss, val_correct, len(valid_loader.dataset), 100. * epoch_val_acc), flush=True )

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['acc'].append(epoch_acc)
        history['val_acc'].append(epoch_val_acc)

        if epoch_val_acc > max_val_acc:
            max_val_acc = epoch_val_acc
            cnt = 0
            # save best model val_acc
            torch.save(model.state_dict(), model_max_val_acc)
            print('\tmax val_acc model saved to %s' % model_max_val_acc, flush=True)
            if val_loss < min_val_cost:
                min_val_cost = val_loss
                torch.save(model.state_dict(), model_min_val_loss)
                print('\tmin val_loss model saved to %s' % model_min_val_loss, flush=True)

        elif val_loss < min_val_cost:
            min_val_cost = val_loss
            cnt = 0
            # save best model val_loss
            torch.save(model.state_dict(), model_min_val_loss)
            print('\tmin val_loss model saved to %s' % model_min_val_loss, flush=True)
        else:
            if epoch_acc > max_train_acc:
                max_train_acc = epoch_acc
                cnt = 0
                # save best model train acc 
                if epoch_acc > 0.99:
                    # checkpoint_path = f'./model/epoch_{epoch+1}_{100*epoch_val_acc:4.2f}.pth'
                    checkpoint_path = f'./model/epoch_{epoch+1}_trainAcc_{epoch_acc:.3f}.pth'
                    torch.save(model.state_dict(), checkpoint_path)
                    print(f"\ttrain acc > 0.99 model saved to {checkpoint_path}", flush=True)
                    model_best_acc_file = checkpoint_path
            else:
                cnt += 1

        if cnt > early_stop_steps:
            print(f"early stop at Epoch {epoch}!\nval_loss {min_val_cost:.3f} or val_acc {max_val_acc:.3f} don't decrease in {early_stop_steps} iters")
            break 
    
    if os.path.isfile(model_max_val_acc):
        model.load_state_dict(torch.load(model_max_val_acc))
    elif os.path.isfile(model_min_val_loss):
        model.load_state_dict(torch.load(model_min_val_loss))
    elif model_best_acc_file is not None:
        model.load_state_dict(torch.load(model_best_acc_file))
    else:
        print("No model saved...")
    # return model, history, min_val_cost
    if model_best_acc_file is not None:
        best_train_acc_model = copy.deepcopy(model)
        best_train_acc_model.load_state_dict(torch.load(model_best_acc_file))
    else:
        best_train_acc_model = None
    return model, history, best_train_acc_model

    

### comment this !
# def plot_history(h, preName="", save=False):
    # plt.figure()
    # epochs = range(len(h['loss']))
    # plt.plot(epochs, h['loss'], label='train')
    # plt.plot(epochs, h['val_loss'], label='valid')
    # plt.title("loss")
    # plt.legend()
    # if save:
        # plt.savefig(preName+"_loss.png")
    # else:
        # plt.show()

    # plt.figure()
    # epochs = range(len(h['acc']))
    # plt.title("accuarcy")
    # plt.plot(epochs, h['acc'], label='train')
    # plt.plot(epochs, h['val_acc'], label='valid')
    # plt.legend()
    # if save:
        # plt.savefig(preName+"_acc.png")
    # else:
        # plt.show()


if __name__ == '__main__':

    labels = pd.read_csv(train_csv).to_numpy()
    train_label, valid_label = train_valid_split(labels, valid_size=2000/len(labels), random_state=77)

    for c in range(7):
        cnt = (train_label==c).sum()
        val_cnt = (valid_label==c).sum()
        print(f"{c}: {cnt:4d} ({cnt*100.0/len(train_label):4.2f}%) \
                {c}: {val_cnt:4d} ({val_cnt*100.0/len(valid_label):4.2f}%)")

    transform = transforms.Compose([
        transforms.RandomAffine(15, translate=(0.1,0.1), scale=(0.9,1.1), shear=10, fillcolor=0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize([mean], [std], inplace=False)
        ])

    train_dataset = train_hw3(train_img_dir, train_label, transform)
    valid_dataset = train_hw3(train_img_dir, valid_label, transform)
    train_loader = DataLoader(train_dataset, batch_size=256) 
    valid_loader = DataLoader(valid_dataset, batch_size=256) 

    print("train_dataset:", len(train_loader.dataset))
    print("valid_dataset:", len(valid_loader.dataset), flush=True)

    if not os.path.exists('./model/'):
        os.makedirs('./model/')

    model = resnext50_32x4d()
    model_name = "resnext50_32x4d"
    model, history, acc99model = train(model, train_loader, valid_loader, EPOCH=300)
    
    # plot_history(history, model_name, save=True)

    with open('history.json', 'w') as fp:
        json.dump(history, fp)

    ### predict
    # test_dataset = test_hw3(test_img_dir, sample_submission, transform)
    # test_loader = DataLoader(test_dataset, batch_size=256)

    # ## load best model
    # # model = best_model
    # # model.load_state_dict(torch.load(best_model_file))
    # # print("Load", model_save_file, "to predict...")
    # model = model.to(DEVICE)
    # model.eval()

    # num_elements = len(test_loader.dataset)
    # num_batches = len(test_loader)
    # predictions = torch.zeros(num_elements)
    # with torch.no_grad():
        # for batch_idx, img in enumerate(test_loader):
            # start = batch_idx * test_loader.batch_size
            # end = start + test_loader.batch_size
            # if batch_idx == num_batches - 1:
                # end = num_elements

            # img = img.to(DEVICE)
            # out = model(img)
            # _, pred_label = torch.max(out, 1)
            # predictions[start:end] = pred_label

    # with open(output_csv, 'w') as f:
            # f.write('id,label\n')
            # for i, v in enumerate(predictions):
                # f.write('%d,%d\n' %(i, v.item()))

    # for i in range(7):
        # cnt = (predictions==i).sum().item()
        # print(f"{i}: {cnt:4d} {cnt*100.0/predictions.size(0):4.2f}%")

    # if acc99model is not None:
        # acc99model = acc99model.to(DEVICE)
        # acc99model.eval()

        # num_elements = len(test_loader.dataset)
        # num_batches = len(test_loader)
        # predictions = torch.zeros(num_elements)
        # with torch.no_grad():
            # for batch_idx, img in enumerate(test_loader):
                # start = batch_idx * test_loader.batch_size
                # end = start + test_loader.batch_size
                # if batch_idx == num_batches - 1:
                    # end = num_elements

                # img = img.to(DEVICE)
                # out = acc99model(img)
                # _, pred_label = torch.max(out, 1)
                # predictions[start:end] = pred_label

        # with open('trainAcc99_pred.csv', 'w') as f:
                # f.write('id,label\n')
                # for i, v in enumerate(predictions):
                    # f.write('%d,%d\n' %(i, v.item()))

        # print("\ntrain_acc 0.99 model result")
        # for i in range(7):
            # cnt = (predictions==i).sum().item()
            # print(f"{i}: {cnt:4d} {cnt*100.0/predictions.size(0):4.2f}%")


