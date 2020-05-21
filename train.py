import imageio
from torch.utils.data import DataLoader
from torchvision import transforms
import PIL
import matplotlib.pyplot as plt
import numpy as np
import sys
import scipy.ndimage
import scipy.misc
import time
import math
import tables
import random
from sklearn.metrics import confusion_matrix
import torch
from torch import nn

from livelossplot import PlotLosses
from sklearn.metrics import accuracy_score

import cv2

from unet import UNet #code borrowed from https://github.com/jvanvugt/pytorch-unet

import torch.nn.functional as F
#import pydensecrf.densecrf as dcrf
from PIL import Image
#from v2.UNETnew import R2U_Net, UNetnew

dataname = "nuclei_segmentation"

ignore_index = -100  #Unet has the possibility of masking out pixels in the output image, we can specify the index value here (though not used)
gpuid = 0

# -------------------------------------- unet params -------------------------------------------------------------------
#these parameters get fed directly into the UNET class, and more description of them can be discovered there
n_classes = 2        #number of classes in the data mask that we'll aim to predict
in_channels = 3      #input channel of the data, RGB = 3
padding = True       #should levels be padded
depth = 5            #depth of the network
wf = 3               #wf (int): number of filters in the first layer is 2**wf
up_mode = 'upconv'   #should we simply upsample the mask, or should we try and learn an interpolation
batch_norm = True    #should we use batch normalization between the layers

# -------------------------------------- training params ---------------------------------------------------------------
batch_size = 5
patch_size = 256
num_epochs = 100
edge_weight = False            #edges tend to be the most poorly segmented given how little area they occupy in the training set, this paramter boosts their values along the lines of the original UNET paper
phases = ["train", "val"]      #how many phases did we create databases for?
validation_phases = ["val"]    #when should we do valiation? note that validation is time consuming, so as opposed to doing for both training and validation, we do it only for vlaidation at the end of the epoch




def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent+.00001)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


class Dataset(object):
    def __init__(self, fname, img_transform=None, mask_transform=None, edge_weight=True):
        self.fname = fname
        self.edge_weight = edge_weight
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.tables = tables.open_file(self.fname)
        self.numpixels = self.tables.root.numpixels[:]
        self.nitems = self.tables.root.img.shape[0]
        self.tables.close()
        self.img = None
        self.mask = None

    def __getitem__(self, index):

        with tables.open_file(self.fname, 'r') as db:
            self.img = db.root.img
            self.mask = db.root.mask
            img = self.img[index, :, :, :]
            mask = self.mask[index, :, :]

        if self.edge_weight:
            weight = scipy.ndimage.morphology.binary_dilation(mask == 1, iterations=2) & ~mask
        else:
            weight = np.ones(mask.shape, dtype=mask.dtype)
        mask = mask[:, :, None].repeat(3, axis=2)
        weight = weight[:, :, None].repeat(3,
                                           axis=2)
        img_new = img
        mask_new = mask
        weight_new = weight
        seed = random.randrange(sys.maxsize)
        if self.img_transform is not None:
            random.seed(seed)
            img_new = self.img_transform(img)
        if self.mask_transform is not None:
            random.seed(seed)
            mask_new = self.mask_transform(mask)
            mask_new = np.asarray(mask_new)[:, :, 0].squeeze()
            random.seed(seed)
            weight_new = self.mask_transform(weight)
            weight_new = np.asarray(weight_new)[:, :, 0].squeeze()
        return img_new, mask_new, weight_new

    def __len__(self):
        return self.nitems


print(torch.cuda.get_device_properties(gpuid))
torch.cuda.set_device(gpuid)
device = torch.device(f'cuda:{gpuid}' if torch.cuda.is_available() else 'cpu')
model = UNet(n_class=n_classes, in_channels=in_channels, filterCountInFirstLayer=8, param_1=1, param_2=2, param_3=3, param_4=4, param_6=6, param_8=8, param_12=12, param_16=16, param_24=24).to(device)
#model = R2U_Net(img_ch=3, output_ch=2, t=3).to(device)

print(f"total params: \t{sum([np.prod(p.size()) for p in model.parameters()])}")

img_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(size=(patch_size, patch_size), pad_if_needed=True),
    # transforms.RandomResizedCrop(size=patch_size),
    # transforms.RandomRotation(180),
    # transforms.RandomGrayscale(),
    transforms.ToTensor()
])

mask_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(size=(patch_size, patch_size), pad_if_needed=True),
    # transforms.RandomResizedCrop(size=patch_size, interpolation=PIL.Image.NEAREST),
    # transforms.RandomRotation(180),
])

dataset = {}
dataLoader = {}
for phase in phases:
    dataset[phase] = Dataset(f"./{dataname}_{phase}.pytable", img_transform=img_transform,
                             mask_transform=mask_transform, edge_weight=edge_weight)
    dataLoader[phase] = DataLoader(dataset[phase], batch_size=batch_size,
                                   shuffle=False, num_workers=0, pin_memory=True)


#val_size = 0
#for datasetx in dataset["val"]:
    #[img, mask, mask_weight] = datasetx

    #imageio.imsave("test_" + str(val_size) + ".png", np.moveaxis(img.numpy(), 0, -1))

    #fig = plt.imshow(mask, interpolation='nearest')
    #plt.axis('off')
    #fig.axes.get_xaxis().set_visible(False)
    #fig.axes.get_yaxis().set_visible(False)
    #plt.savefig("test_mask_" + str(val_size) + ".png", bbox_inches='tight', pad_inches=0)

    #val_size = val_size + 1


(img, patch_mask, patch_mask_weight) = dataset["train"][1]
fig, ax = plt.subplots(1, 4, figsize=(10, 4))
ax[0].imshow(np.moveaxis(img.numpy(), 0, -1))
ax[1].imshow(patch_mask == 1)
ax[2].imshow(patch_mask_weight)
ax[3].imshow(patch_mask)
fig.savefig("simpleinput.png")

optim = torch.optim.Adam(model.parameters())
nclasses = dataset["train"].numpixels.shape[1]
class_weight = dataset["train"].numpixels[1, 0:2]
class_weight = torch.from_numpy(1 - class_weight/class_weight.sum()).type('torch.FloatTensor').to(device)
# criterion = nn.CrossEntropyLoss(weight=class_weight, ignore_index=ignore_index, reduce=False)
criterion = nn.CrossEntropyLoss()
#writer = SummaryWriter()
best_loss_on_test = np.Infinity
edge_weight = torch.tensor(edge_weight).to(device)
# edge_weight = torch.tensor(edge_weight)


liveloss = PlotLosses()


start_time = time.time()
for epoch in range(num_epochs):
    all_acc = {key: 0 for key in phases}
    all_loss = {key: torch.zeros(0).to(device) for key in phases}
    cmatrix = {key: np.zeros((2, 2)) for key in phases}

    total = 0
    val_accuracy = 0
    train_accuracy = 0
    logs = {}

    for phase in phases:
        if phase == 'train':
            model.train()
        else:
            model.eval()
        for ii, (X, y, y_weight) in enumerate(dataLoader[phase]):
            X = X.to(device)
            y_weight = y_weight.type('torch.FloatTensor').to(device)
            y = y.type('torch.LongTensor').to(device)
            with torch.set_grad_enabled(phase == 'train'):
                prediction = model(X)
                loss_matrix = criterion(prediction, y)
                # loss = (loss_matrix * (edge_weight**y_weight)).mean()
                loss = loss_matrix.mean()
                if phase == "train":
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    train_loss = loss
                all_loss[phase] = torch.cat((all_loss[phase], loss.detach().view(1, -1)))


                #Calculate accuracy
                p = prediction[:, :, :, :].detach().cpu().numpy()
                cpredflat = np.argmax(p, axis=1).flatten()
                yflat=y.cpu().numpy().flatten()
                cmatrix[phase] = cmatrix[phase]+confusion_matrix(yflat, cpredflat, labels=range(n_classes))


        all_loss[phase] = all_loss[phase].cpu().numpy().mean()

        if phase in validation_phases:
            print(f'{phase}/acc', all_acc[phase], epoch)
            print(f'{phase}/TN',  cmatrix[phase][0, 0], epoch)
            print(f'{phase}/TP',  cmatrix[phase][1, 1], epoch)
            print(f'{phase}/FP',  cmatrix[phase][0, 1], epoch)
            print(f'{phase}/FN',  cmatrix[phase][1, 0], epoch)
            print(f'{phase}/TNR', cmatrix[phase][0, 0]/(cmatrix[phase][0, 0]+cmatrix[phase][0, 1]), epoch)
            print(f'{phase}/TPR', cmatrix[phase][1, 1]/(cmatrix[phase][1, 1]+cmatrix[phase][1, 0]), epoch)
            val_accuracy =  100 * (cmatrix[phase][0, 0] + cmatrix[phase][1, 1]) / (cmatrix[phase][0, 0] + cmatrix[phase][1, 1] + cmatrix[phase][0, 1] + cmatrix[phase][1, 0])
            #print(f'{phase}/accuracy:', accuracy, epoch)
        else:
            train_accuracy = 100 * (cmatrix[phase][0, 0] + cmatrix[phase][1, 1]) / (cmatrix[phase][0, 0] + cmatrix[phase][1, 1] + cmatrix[phase][0, 1] + cmatrix[phase][1, 0])


    print('%s ([%d/%d] %d%%), train loss: %.4f val loss: %.4f train_accuracy %4.f val_accuracy %4.f' % (timeSince(start_time, (epoch+1) / num_epochs),
                                                 epoch+1, num_epochs, (epoch+1) / num_epochs * 100, all_loss["train"], all_loss["val"], train_accuracy, val_accuracy), end="")

    prefix = ''
    logs[prefix + 'log loss'] = all_loss["train"]
    logs[prefix + 'accuracy'] = train_accuracy

    prefix = 'val_'

    logs[prefix + 'log loss'] = all_loss["val"]
    logs[prefix + 'accuracy'] = val_accuracy

    liveloss.update(logs)


    if all_loss["val"] < best_loss_on_test:
        best_loss_on_test = all_loss["val"]
        print("  **")
        state = {'epoch': epoch + 1,
         'model_dict': model.state_dict(),
         'optim_dict': optim.state_dict(),
         'best_loss_on_test': all_loss,
         'n_classes': n_classes,
         'in_channels': in_channels,
         'padding': padding,
         'depth': depth,
         'wf': wf,
         'up_mode': up_mode, 'batch_norm': batch_norm}
        torch.save(state, f"{dataname}_model.pth")
    else:
        print("")


liveloss.draw()


# -------------------------------------- generate output ---------------------------------------------------------------

checkpoint = torch.load(f"{dataname}_model.pth")
model.load_state_dict(checkpoint["model_dict"])

[img, mask, mask_weight] = dataset["val"][4]
output = model(img[None, ::].to(device))
output = output.detach().squeeze().cpu().numpy()
output = np.moveaxis(output, 0, -1)
fig, ax = plt.subplots(1, 4, figsize=(10, 4))
ax[0].imshow(output[:, :, 1])
ax[1].imshow(np.argmax(output, axis=2))
ax[2].imshow(mask)
ax[3].imshow(np.moveaxis(img.numpy(), 0, -1))
fig.savefig("result.png")


def plot_kernels(tensor, num_cols=8, cmap="gray"):
    if not len(tensor.shape) == 4:
        raise Exception("assumes a 4D tensor")
    num_kernels = tensor.shape[0]
    num_rows = 1 + num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols, num_rows))
    for i in range(tensor.shape[0]):
        ax1 = fig.add_subplot(num_rows, num_cols, i+1)
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig("kernel.png")


class LayerActivations():
    features = None

    def __init__(self, layer):
        self.hook = layer.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.cpu()

    def remove(self):
        self.hook.remove()


#w = model.up_path[0].conv_block.block[3]
#plot_kernels(w.weight.detach().cpu(), 8)



val_size = 0
i = 0
for datasetx in dataset["val"]:
    [img, mask, mask_weight] = datasetx

    imageio.imsave("test_" + str(val_size) + ".png", np.moveaxis(img.numpy(), 0, -1))

    plt.clf()

    fig = plt.imshow(mask, interpolation='nearest')
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig("test_mask_" + str(val_size) + ".png", bbox_inches='tight', pad_inches=0)
    plt.rcParams.update({'figure.max_open_warning': 0})
    val_size = val_size + 1

    plt.clf()
    output = model(img[None, ::].to(device))
    output = output.detach().squeeze().cpu().numpy()
    output = np.moveaxis(output, 0, -1)
    fig, ax = plt.subplots(1, 4, figsize=(10, 4))

    ax[0].imshow(output[:, :, 1])
    ax[1].imshow(np.argmax(output, axis=2))
    ax[2].imshow(mask)
    ax[3].imshow(np.moveaxis(img.numpy(), 0, -1))
    fig.savefig("result" + str(i) + ".png")
    i = i+1

train_size = 0
for datasetx in dataset["train"]:
    [img, mask, mask_weight] = datasetx
    imageio.imsave("train_" + str(train_size) + ".png", np.moveaxis(img.numpy(), 0, -1))

    plt.clf()

    fig = plt.imshow(mask, interpolation='nearest')
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig("train_mask_" + str(train_size) + ".png", bbox_inches='tight', pad_inches=0)
    plt.rcParams.update({'figure.max_open_warning': 0})
    train_size = train_size + 1

print("val_size: " + str(val_size) + " train_size: " + str(train_size))