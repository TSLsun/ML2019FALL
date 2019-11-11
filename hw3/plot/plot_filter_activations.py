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
model.eval()
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


#%%
# visualize all filters (filter weight)
kernels = model.conv_base[8].weight.cpu().detach().clone()
kernels = kernels - kernels.min()
kernels = kernels / kernels.max()

# %%
def vistensor(tensor, ch=0, allkernels=False, nrow=8, padding=1): 
    '''
    vistensor: visuzlization tensor
        @ch: visualization channel 
        @allkernels: visualization all tensores
    ''' 
    
    n,c,w,h = tensor.shape
    if allkernels: tensor = tensor.view(n*c,-1,w,h )
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)
        
    rows = np.min( (tensor.shape[0]//nrow + 1, 64 )  )    
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure( figsize=(nrow,rows) )
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    # plt.savefig("./plot/kernel.png")
    plt.show()

#%%
ik = 8
kernel = model.conv_base[ik].weight.data.clone()
print(kernel.shape)
vistensor(kernel, ch=0, nrow=16, allkernels=False)

# %%
def convert_to_grayscale(im_as_arr):
    """
        Converts 3d image to grayscale
    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)
    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im


def save_gradient_images(gradient, file_name):
    """
        Exports the original gradient image
    Args:
        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        file_name (str): File name to be exported
    """
    if not os.path.exists('./results'):
        os.makedirs('./results')
    # Normalize
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    # Save image
    path_to_file = os.path.join('./results', file_name + '.jpg')
    save_image(gradient, path_to_file)


def save_class_activation_images(org_img, activation_map, file_name):
    """
        Saves cam activation map and activation map on the original image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    if not os.path.exists('./results'):
        os.makedirs('./results')
    # Grayscale activation map
    heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map, 'hsv')
    # Save colored heatmap
    path_to_file = os.path.join('./results', file_name+'_Cam_Heatmap.png')
    save_image(heatmap, path_to_file)
    # Save heatmap on iamge
    path_to_file = os.path.join('./results', file_name+'_Cam_On_Image.png')
    save_image(heatmap_on_image, path_to_file)
    # SAve grayscale heatmap
    path_to_file = os.path.join('./results', file_name+'_Cam_Grayscale.png')
    save_image(activation_map, path_to_file)


def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image


def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr


def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)


def preprocess_image(pil_im, resize_im=True):
    """
        Processes image for CNNs
    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    # Resize image
    if resize_im:
        pil_im.thumbnail((224, 224))
    im_as_arr = np.float32(pil_im)
    # print(im_as_arr.shape)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        # im_as_arr[channel] -= mean[channel]
        # im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


def recreate_image(im_as_var):
    """
        Recreates images from a torch variable, sort of reverse preprocessing
    Args:
        im_as_var (torch variable): Image to recreate
    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    # reverse_mean = [-0.485, -0.456, -0.406]
    # reverse_std = [1/0.229, 1/0.224, 1/0.225]
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    # print(im_as_var.shape)
    # for c in range(im_as_var.shape[1]):
        # recreated_im[c] /= reverse_std[c]
        # recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0) # W,H,C
    if recreated_im.shape[2] == 3:
        return recreated_im
    return recreated_im.reshape(recreated_im.shape[1], recreated_im.shape[1])


#%%
class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = torch.tensor(output,requires_grad=True).to(DEVICE)
    def close(self):
        self.hook.remove()

class FilterVisualizer():
    def __init__(self, model, size=56, upscaling_steps=12, upscaling_factor=1.2, ch=1):
        self.size, self.upscaling_steps, self.upscaling_factor = size, upscaling_steps, upscaling_factor
        self.model = model
        self.ch = ch
    def visualize(self, layer, filter, lr=0.1, opt_steps=20, blur=None):
        sz = self.size
        img = np.uint8(np.random.uniform(150, 180, (sz, sz, self.ch)))  # generate random image
        activations = SaveFeatures(list(self.model.children())[layer])  # register hook
        # processed_image = preprocess_image(img, False)

        for _ in range(self.upscaling_steps):  # scale the image up upscaling_steps times
            img = img.reshape(sz,sz,self.ch)
            img_var = preprocess_image(img, False)
            optimizer = torch.optim.Adam([img_var], lr=lr, weight_decay=1e-6)
            for n in range(opt_steps):  # optimize pixel values for opt_steps times
                optimizer.zero_grad()
                self.model(img_var)
                loss = -activations.features[0, filter].mean()
                loss.backward()
                optimizer.step()
            img = recreate_image(img_var)
            self.output = img
            sz = int(self.upscaling_factor * sz)  # calculate new image size
            img = cv2.resize(img, (sz, sz), interpolation = cv2.INTER_CUBIC)  # scale image up
            if blur is not None: img = cv2.blur(img,(blur,blur))  # blur image to reduce high frequency patterns
            # print(self.output.shape)
        self.save(layer, filter)
        activations.close()
        
    def save(self, layer, filter):
        # plt.imsave("layer_"+str(layer)+"_filter_"+str(filter)+".jpg", np.clip(self.output, 0, 1))
        plt.imsave("layer_"+str(layer)+"_filter_"+str(filter)+".png", self.output)
#%%
layer = 8
filter = 414

FV = FilterVisualizer(model=model.conv_base, size=48, ch=1, upscaling_steps=3, upscaling_factor=1.2)
FV.visualize(layer, filter, blur=1)

img = PIL.Image.open("layer_"+str(layer)+"_filter_"+str(filter)+".png")
plt.imshow(np.array(img))
plt.show()





#%%
img = PIL.Image.open("data/train_img/00037.jpg")
img = transform(img).reshape(1,1,48,48).to(DEVICE)
# img.numpy().reshape(48,48,1)
plt.imshow(img.numpy().reshape(48,48), cmap='gray')
plt.show()


# %%
cnn_layer = 8
activations = SaveFeatures(model.conv_base[cnn_layer])
model.conv_base(img)
# %%
total_filters_in_layer = 512
mean_act = [activations.features[0,i].mean().item() for i in range(total_filters_in_layer)]

print("max act:", np.argmax(mean_act))

thresh = 0.46
filter_pos_over_thresh = [i for i in range(total_filters_in_layer) if mean_act[i]>thresh]
print(f"act > thresh({thresh})", filter_pos_over_thresh)

#%%
filter_pos = np.argmax(mean_act)
plt.figure(figsize=(7,5))
act = plt.plot(mean_act,linewidth=2.)
extraticks=[filter_pos]
ax = act[0].axes
ax.set_xlim(0,500)
plt.axvline(x=filter_pos, color='grey', linestyle='--')
ax.set_xlabel("feature map")
ax.set_ylabel("mean activation")
ax.set_xticks([0,200,400] + extraticks)
plt.show()

# %%
