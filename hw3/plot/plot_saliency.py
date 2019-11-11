#%%
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
from torchvision import transforms, utils
import torchvision.models as models
import PIL
from PIL import Image
# import matplotlib
# matplotlib.use("AGG")
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_color_map
from torchsummary import summary
import flashtorch
from flashtorch.saliency import Backprop
import seaborn as sns
from torch.autograd import Variable

#%%
class CNN(nn.Module):
    def __init__(self, input_shape=(1,48,48), conv1_filters=96, conv2_filters=256, conv3_filters=256, num_units=256, dropout=0.5, num_classes=7):
        super(CNN, self).__init__()
        self.conv_base = nn.Sequential(
            nn.Conv2d(1, conv1_filters, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(conv1_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),

            nn.Conv2d(conv1_filters, conv2_filters, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(conv2_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),

            nn.Conv2d(conv2_filters, conv3_filters, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(conv3_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),
        )
        
        n_size = self._get_conv_output(input_shape)

        self.fc_base = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(n_size, num_units),
            nn.ReLU(inplace=True),

            nn.Dropout(p=dropout),
            nn.Linear(num_units, num_units),
            nn.ReLU(inplace=True),
        )

    def _get_conv_output(self, shape):
        bs = 1
        inp = torch.rand(bs, *shape)
        out = self._forward_features(inp)
        return int(np.prod(out.size()[1:]))
    
    def _forward_features(self, x):
        x = self.conv_base(x)
        return x
    
    def forward(self, x):
        x = self.conv_base(x)
        x = x.view(x.size(0), -1)
        x = self.fc_base(x)
        return x

class train_hw3(Dataset):

    def __init__(self, data_dir, label, transform):
        self.data_dir = data_dir
        self.label = label
        self.transform = transform
    
    def __getitem__(self, index):
        pic_file = '{:0>5d}.jpg'.format(self.label[index][0])
        img = cv2.imread(os.path.join(self.data_dir, pic_file), cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, 0)
        return torch.FloatTensor(img), self.label[index, 1]
        # img = Image.open(os.path.join(self.data_dir, pic_file)).convert('RGB')
        # img = self.transform(img)
        # return img, self.label[index, 1]

    def __len__(self):
        return self.label.shape[0]

class test_hw3(Dataset):

    def __init__(self, data_dir, sample_submission, transform):
        self.data_dir = data_dir
        self.name = pd.read_csv(sample_submission).to_numpy()
        self.transform = transform
    
    def __getitem__(self, index):
        pic_file = '{:0>4d}.jpg'.format(self.name[index][0])
        img = cv2.imread(os.path.join(self.data_dir, pic_file), cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, 0)
        return torch.FloatTensor(img)
        # img = Image.open(os.path.join(self.data_dir, pic_file)).convert('RGB')
        # img = self.transform(img)
        # return img

    def __len__(self):
        return self.name.shape[0]

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
#%%

use_cuda = torch.cuda.is_available()
DEVICE = torch.device('cuda' if use_cuda else 'cpu')
print("Device:", DEVICE)

#%%
train_img_dir = "data/train_img"
train_csv = "data/train.csv"
test_img_dir = "data/test_img"
sample_submission = "data/sample_submission"
output_csv = "./pred.csv"
#%%
model_pth = "jobs/CNN_3_model/model/max_val_acc.pkl"

model = CNN(input_shape=(1,48,48), conv1_filters=64, conv2_filters=128, conv3_filters=512, num_units=64, dropout=0.5, num_classes=7)
model.load_state_dict(torch.load(model_pth, map_location=DEVICE))
#%%
print(model.conv_base[8].weight.cpu().detach().clone().shape)

# %%
labels = pd.read_csv(train_csv).to_numpy()
train_label, valid_label = train_valid_split(labels, valid_size=2000/len(labels), random_state=77)

# %%
transform = transforms.Compose([
    # transforms.RandomAffine(15, translate=(0.1,0.1), scale=(0.9,1.1), shear=10, fillcolor=0),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize([mean], [std], inplace=False)
    ])

train_dataset = train_hw3(train_img_dir, train_label, transform)
valid_dataset = train_hw3(train_img_dir, valid_label, transform)
train_loader = DataLoader(train_dataset, batch_size=64) 
valid_loader = DataLoader(valid_dataset, batch_size=512) 

# %%
plt.figure()
plt.axis('off')
plt.title("Input tensor")
plt.imshow(train_dataset.__getitem__(1)[0].numpy().reshape(48,48), cmap='gray')

# %%
def compute_saliency_maps(X, y, model):
    """
    X表示图片, y表示分类结果, model表示使用的分类模型
    
    Input : 
    - X : Input images : Tensor of shape (N, C, H, W)
    - y : Label for X : LongTensor of shape (N,)
    - model : A pretrained CNN that will be used to computer the saliency map
    
    Return :
    - saliency : A Tensor of shape (N, H, W) giving the saliency maps for the input images
    """
    # 确保model是test模式
    model.eval()
    # 确保X是需要gradient
    X.requires_grad_()
    saliency = None
    logits = model.forward(X)
    logits = logits.gather(1, y.view(-1, 1)).squeeze() # 得到正确分类
    logits.backward(torch.FloatTensor([1.]*y.size(0))) # 只计算正确分类部分的loss
    saliency = abs(X.grad.data) # 返回X的梯度绝对值大小
    saliency, _ = torch.max(saliency, dim=1)
    return saliency.squeeze()



#%%
label_name = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise","Neutral"]
# %%
tl = [(data, lab) for data, lab in  train_loader]
# %%
tl[0]
# %%
saliency = compute_saliency_maps(tl[0][0], tl[0][1], model)
saliency.shape
#%%
plt.imshow(tl[0][0][0].detach().numpy().reshape(48,48))
#%%
plt.imshow(saliency[0])

# %%
#%%
idx = 9
plt.title(label_name[tl[0][1][idx].item()])
plt.imshow(tl[0][0][idx].detach().numpy().reshape(48,48), cmap='gray')
plt.axis("off")
plt.savefig("jobs/CNN_3_model/plot/input.jpg")

plt.imshow(saliency[idx], cmap='gray')
plt.axis("off")
plt.savefig("jobs/CNN_3_model/plot/saliency.jpg")
# %%
idx = 62
subplots = [
            # (title, [(image1, cmap, alpha), (image2, cmap, alpha)])
            (label_name[tl[0][1][idx].item()],
             [(tl[0][0][idx].detach().numpy().reshape(48,48),
              'gray',
              None)]),
            ('Gradients',
             [(saliency[idx],
              'gray',
              None)])
            # ('Max gradients',
            #  [(format_for_plotting(standardize_and_clip(max_gradients)),
            #   cmap,
            #   None)]),
            # ('Overlay',
            #  [(format_for_plotting(denormalize(input_)), None, None),
            #   (format_for_plotting(standardize_and_clip(max_gradients)),
            #    cmap,
            #    alpha)])
        ]

fig = plt.figure(figsize=(8, 4))
for i, (title, images) in enumerate(subplots):
    ax = fig.add_subplot(1, len(subplots), i + 1)
    ax.set_axis_off()
    ax.set_title(title)

    for image, cmap, alpha in images:
        ax.imshow(image, cmap=cmap, alpha=alpha)
plt.savefig("jobs/CNN_3_model/plot/"+label_name[tl[0][1][idx].item()]+".png")