#!/usr/bin/env python
# coding: utf-8

# In[2]:


from load_data import *
import load_data
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import os
from image import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import torch.nn.functional as F
from utils import *
from torch.utils.data import DataLoader
from tqdm import tqdm

plt.rcParams["axes.grid"] = False


# In[7]:


from load_data import *
import load_data
import gc
import matplotlib.pyplot as plt
import patch_config
from torch import autograd
plt.rcParams["axes.grid"] = False
plt.axis('off')

cfgfile = "cfg/yolo.cfg"
weightfile = "weights/yolo.weights"
printfile = "non_printability/30values.txt"


mode = 'base'

print('LOADING MODELS')
darknet_model = Darknet(cfgfile)
darknet_model.load_weights(weightfile)
patch_size = darknet_model.height
img_size = darknet_model.height
darknet_model = darknet_model.eval().cuda()
patch_applier = PatchApplier().cuda()
patch_transformer = PatchStopTransformer().cuda()
prob_extractor = MaxProbExtractor(11, 80, patch_config.patch_configs[mode]()).cuda() # 11 for stopsign
nps_calculator = NPSCalculator(printfile, patch_size)
nps_calculator = nps_calculator.cuda()
total_variation = TotalVariation().cuda()
print('MODELS LOADED')
print(patch_size)
print(img_size)


# In[11]:


# dataset predifine 
# read the image(jpg) and its label(txt) with same name 

# print('LOADING MODELS')
# darknet_model = Darknet(cfgfile)
# darknet_model.load_weights(weightfile)
# patch_size = darknet_model.height  # 416
# img_size = darknet_model.height  # 416
import cv2

def pad_and_scale(img, lab, imgsize):
    """

    Args:
        img:

    Returns:

    """
    w,h = img.size
    if w==h:
        padded_img = img
    else:
        dim_to_pad = 1 if w<h else 2
        if dim_to_pad == 1:
            padding = (h - w) / 2
            padded_img = Image.new('RGB', (h,h), color=(127,127,127))
            padded_img.paste(img, (int(padding), 0))
            lab[:, [1]] = (lab[:, [1]] * w + padding) / h
            lab[:, [3]] = (lab[:, [3]] * w / h)
        else:
            padding = (w - h) / 2
            padded_img = Image.new('RGB', (w, w), color=(127,127,127))
            padded_img.paste(img, (0, int(padding)))
            lab[:, [2]] = (lab[:, [2]] * h + padding) / w
            lab[:, [4]] = (lab[:, [4]] * h  / w)
    padded_img = padded_img.resize((imgsize,imgsize))     #choose here
    # padded_img = np.resize(padded_img, (imgsize,imgsize, c))
    return padded_img, lab


class StopSignDataset(Dataset):
    def __init__(self, root, train=True, image_size=416):
       self.root = root
       self.path_img = root + '/img/' # path-to-the-images
       self.path_lab = root + '/label/' # path-to-the-labels

       self.images = [os.path.join(self.path_img, img) for img in sorted(os.listdir(self.path_img))]
       #print(self.images)
       self.labels = [os.path.join(self.path_lab, lab) for lab in sorted(os.listdir(self.path_lab))]
       #print(self.labels)
    #    print(len(self.images))
       self.nSamples  = len(self.images)
       self.nLabels = len(self.labels)
       if self.nSamples != self.nLabels:
           raise 'NumberNotMatchError!'

       self.train = train # True: train; False: test
       self.image_size = image_size
       self.class_names = load_class_names('data/coco.names')
    #    self.shape = shape
    #    self.seen = seen
    #    self.batch_size = batch_size
    #    self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        # do not forget to add the data argumentation when training but not validating
        # jitter = 0.2
        # hue = 0.1
        # saturation = 1.5 
        # exposure = 1.5
        # img, label = data_argumentation(imgpath, self.shape, jitter, hue, saturation, exposure)

        # get the sample
        img_idx = self.images[index]
        lab_idx = self.labels[index]
        # print(img_idx)
        image = Image.open(img_idx).convert('RGB')
        # image = cv2.imread(img_idx)
        if os.path.getsize(lab_idx) > 0:
            label_df = pd.read_csv(lab_idx, header=None, sep=',') # .to_numpy() # .squeeze(0)
            label_df.iloc[0, 4] = self.class_names.index(label_df.iloc[0, 4])
            # print(label_df)
            label = label_df.astype(np.float16).to_numpy()
            # print(label)
        else:
            label = np.array([[11.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float16)
            # return 
        # label ToTensor
        label = torch.from_numpy(label).float()
        # print(label)

        # Transform/Argumentation
        image, label = self.transform(image, label)

        # image ToTensor
        transform_ = transforms.ToTensor()
        image = transform_(image)

        return image, label
    
    def transform(self, image, label):
        ymin, xmin, ymax, xmax, _ = label.squeeze(0)
        # print(label.squeeze(0))
        # print(xmin, ymin, xmax, ymax)
        # label = torch.tensor([[xmin, ymin, xmax-xmin, ymax-ymin]])  # note the order of label
        label = torch.tensor([[(xmax+xmin)/2.0, (ymax+ymin)/2.0, xmax-xmin, ymax-ymin]])
        label = label / 1024.0
        # print(label)
        image, label = pad_and_scale(image, label, self.image_size)
        # print(label)
        # label = F.normalize(label, dim=1)
        label = torch.cat((torch.tensor([[_]]), label), dim=1)
        # print(label)
        return image, label
        
    
# test the dataset class
root = './test/SceneImages/'
dataset = StopSignDataset(root, train=True, image_size=416)  # 416 is the darknet.height
image, label = dataset[111]
print(label)
print(image.shape)


# In[12]:


# params setting
# dataloader
root_ = './test/SceneImages'
training_set = StopSignDataset(root_, train=True, image_size=img_size)
validation_set = StopSignDataset(root_, train=True, image_size=img_size)
train_dataloader = DataLoader(training_set, batch_size=32, shuffle=True, num_workers=0)
test_dataloader = DataLoader(validation_set, batch_size=32, shuffle=True, num_workers=0)

# Initial the adversarial patch 
patch_img = Image.open("saved_patches/before.jpg").convert('RGB') # patch is random choosing with some pointed pattern
# patch_img = cv2.imread("saved_patches/cake.jpg")
# patch_img = cv2.resize(patch_img, (patch_size,patch_size))
# print(patch_img.shape)

# patch to tensor with same size of stop sign
tf = transforms.Resize((patch_size,patch_size)) # should be sett as the size of the stopsign
patch_img = tf(patch_img)
tf = transforms.ToTensor()
adv_patch_cpu = tf(patch_img)
adv_patch_cpu.requires_grad_(True)
print(adv_patch_cpu.shape)

n_epochs = 2
epoch_length = len(train_dataloader)
optimizer = optim.Adam([adv_patch_cpu], lr = 0.01)

# training 
for epoch in range(n_epochs):
    ffw1 = time.time()
    adv_patch = adv_patch_cpu.cuda()
    losses = np.array([])
    for i_batch, img_lab_list in tqdm(enumerate(train_dataloader), desc=f'Running epoch {epoch}',
                                      total=epoch_length):
        # print(i_batch, len(img_lab_list))
        img_batch, lab_batch = img_lab_list
        img_batch = img_batch.cuda()
        lab_batch = lab_batch.cuda()
        # print(adv_patch.shape)
        # print(img_batch.shape)
        # print(lab_batch.shape)

        # forward 
        # print(adv_patch.shape)
        adv_batch_t =  patch_transformer(adv_patch, lab_batch, img_size)
        p_img_batch = patch_applier(img_batch, adv_batch_t)
        p_img_batch = F.interpolate(p_img_batch,(darknet_model.height, darknet_model.width))
        #print('running patched images through model')

        darknet_model = darknet_model.cpu()
        p_img_batch = p_img_batch.cpu()
        output = darknet_model(p_img_batch)
        output = output.cuda()
        
            
        # boxes = get_region_boxes(output, 0.5, darknet_model.num_classes, 
        #                         darknet_model.anchors, darknet_model.num_anchors)[0]
        # # print('boxex:', boxes)
        # boxes = nms(boxes,0.4)
        # # print('boxex:', boxes)
        # class_names = load_class_names('data/coco.names')
        # squeezed = p_img_batch.squeeze(0)
        #img = transforms.ToPILImage('RGB')(squeezed.detach().cpu())
        # img.save('./saved_patches/checkpoints.png')
        img = p_img_batch[1, :, :,]
        img = transforms.ToPILImage()(img.detach().cpu())

        max_prob = prob_extractor(output)
        # print('extracting max probs')
        nps = nps_calculator(adv_patch)
        # print('calculating tv')
        tv = total_variation(adv_patch)

        det_loss = torch.mean(max_prob)
        # loss = det_loss

        # printable loss
        nps_loss = nps*0.1
        tv_loss = tv*0.00005
        loss = det_loss + nps_loss + tv_loss
        # img_batch.retain_grad()
        # adv_batch_t.retain_grad()
        # adv_patch.retain_grad()

        # print('loss to patch', torch.autograd.grad(loss,img_batch))
        loss.backward()
        # tl1 = time.time()
        # print('adv_patch.grad',adv_patch_cpu.grad)
        # print('adv_batch_t.grad',adv_batch_t.grad)
        # print('img_batch.grad',img_batch.grad)
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.empty_cache()

        losses = np.append(losses, loss.detach().cpu().numpy())
        ffw2 = time.time()

    if (epoch) % 1 == 0:
        print('  EPOCH NR: ', epoch)
        print('EPOCH LOSS: ', losses.mean())
        print('  DET LOSS: ', det_loss.detach().cpu().numpy())
        print('  NPS LOSS: ', nps_loss.detach().cpu().numpy())
        print('   TV LOSS: ', tv_loss.detach().cpu().numpy())
        print('EPOCH TIME: ', ffw2-ffw1)
        # del adv_batch_t, output, max_prob
        im = transforms.ToPILImage('RGB')(adv_patch_cpu)
        plt.imshow(im)
        plt.show()
        plt.imshow(img)
        plt.show()
    # else:
    #     print('EPOCH LOSS: ', loss.detach().cpu().numpy())

    im = transforms.ToPILImage('RGB')(adv_patch_cpu)
    #im.save('saved_patches/exp/patch_cake_{:d}.jpg'.format(epoch))
    
boxes = get_region_boxes(output,0.5,darknet_model.num_classes, 
                        darknet_model.anchors, darknet_model.num_anchors)[0]
boxes = nms(boxes,0.4)
print(boxes)
#im.save('saved_patches/patch_cake.jpg')


# In[13]:


im.save('saved_patches/patch_lion.jpg') # save the generated patch


# In[14]:


# test input clean image

from utils import *
patch_size = darknet_model.height
img_size = darknet_model.height

img_path = "test/SceneImages/img/000110.png"
lab_path = "test/SceneImages/label/000110.txt"

image = Image.open(img_path).convert('RGB')
class_names = load_class_names('data/coco.names')

label_df = pd.read_csv(lab_path, header=None, sep=',') 
print(label_df)
label_df.iloc[0, 4] = class_names.index(label_df.iloc[0, 4])
label = label_df.astype(np.float16).to_numpy()
print(label)
ymin, xmin, ymax, xmax, _ = label.squeeze(0)

label = torch.tensor([[(xmax+xmin)/2.0, (ymax+ymin)/2.0, xmax-xmin, ymax-ymin]])
label = torch.cat((torch.tensor([[_]]), label), dim=1).squeeze(0)
print(label)
label = label / 1024.0
print(label)
# print(label.shape)
# print(label)
if label.dim() == 1:
    label = label.unsqueeze(0)
# print(label.shape)
image, label = pad_and_scale(image, label, img_size)
transform = transforms.ToTensor()
image = transform(image)

batch_size = 1    
image = image.unsqueeze(0)
label = label.unsqueeze(0)
print('image shape', image.shape)
print('label shape', label.shape)
img_batch = image.expand(batch_size,-1,-1,-1)
lab_batch = label.expand(batch_size,-1,-1)
print('after expand image shape', image.shape)
print('after expand label shape', label.shape)

img_batch = img_batch.cuda()
lab_batch = lab_batch.cuda()
darknet_model = darknet_model.cuda()
output = darknet_model(img_batch)
#output = darknet_model(p_img_batch)
boxes = get_region_boxes(output, 0.5, darknet_model.num_classes, 
                        darknet_model.anchors, darknet_model.num_anchors)[0]
boxes = nms(boxes,0.4)
class_names = load_class_names('data/coco.names')
squeezed = img_batch.squeeze(0)
#squeezed = p_img_batch[0,:,:,:]
print(squeezed.shape)
img = transforms.ToPILImage('RGB')(squeezed.detach().cpu())
plotted_image = plot_boxes(img, boxes, class_names=class_names)
plt.imshow(plotted_image)
plt.show()

print(boxes)


# In[15]:


# test-- After attack
from PIL import Image

adv_patch = Image.open("./saved_patches/patch_lion.jpg")#  .convert('RGB')
adv_patch = adv_patch.resize((img_size, img_size))
# print(adv_patch)


# adv_patch = cv2.imread("./saved_patches/exp/patch_cake_0.jpg")
# adv_patch = np.resize(adv_patch, (img_size, img_size, 3))
# print(adv_patch)

# adv_patch = np.array(Image.fromarray(adv_patch))

transform = transforms.ToTensor()
adv_patch = transform(adv_patch).cuda()
print(adv_patch.shape)

img_size = img_batch.size(-1)
print(img_size)
print(label.cuda().shape)

# transform to patch that can be adapted to img directly
adv_batch_t = patch_transformer(adv_patch, lab_batch.cuda(), img_size, do_rotate=False, rand_loc=False)
# print('adv patch t shaep:', adv_batch_t.shape)

p_img_batch = patch_applier(img_batch, adv_batch_t)
# print('p_img_batch shaep:', p_img_batch.shape)
p_img_batch = F.interpolate(p_img_batch,(darknet_model.height, darknet_model.width))
# print('after interpolate p_img_batch shaep:', p_img_batch.shape)
    
output = darknet_model(p_img_batch)
boxes = get_region_boxes(output,0.5,darknet_model.num_classes, 
                        darknet_model.anchors, darknet_model.num_anchors)[0]
boxes = nms(boxes,0.4)
class_names = load_class_names('data/coco.names')
# print('before squzzezd shape:', squeezed.shape)
squeezed = p_img_batch.squeeze(0)
# print('squzzezd shape:', squeezed.shape)
img = transforms.ToPILImage('RGB')(squeezed.detach().cpu())
plotted_image = plot_boxes(img, boxes, class_names=class_names)
plt.imshow(plotted_image)
# plt.imshow(adv_patch)
plt.show()

print(boxes)


# In[16]:


# test input clean image

from utils import *
patch_size = darknet_model.height
img_size = darknet_model.height

img_path = "test/SceneImages/img/000023.png"
lab_path = "test/SceneImages/label/000023.txt"

image = Image.open(img_path).convert('RGB')
class_names = load_class_names('data/coco.names')

label_df = pd.read_csv(lab_path, header=None, sep=',') 
label_df.iloc[0, 4] = class_names.index(label_df.iloc[0, 4])
label = label_df.astype(np.float16).to_numpy()
ymin, xmin, ymax, xmax, _ = label.squeeze(0)

label = torch.tensor([[(xmax+xmin)/2.0, (ymax+ymin)/2.0, xmax-xmin, ymax-ymin]])
label = torch.cat((torch.tensor([[_]]), label), dim=1).squeeze(0)
label = label / 1024.0
# print(label.shape)
# print(label)
if label.dim() == 1:
    label = label.unsqueeze(0)
# print(label.shape)
image, label = pad_and_scale(image, label, img_size)
transform = transforms.ToTensor()
image = transform(image)

batch_size = 1    
image = image.unsqueeze(0)
label = label.unsqueeze(0)
img_batch = image.expand(batch_size,-1,-1,-1)
lab_batch = label.expand(batch_size,-1,-1)

img_batch = img_batch.cuda()
lab_batch = lab_batch.cuda()

output = darknet_model(img_batch)
#output = darknet_model(p_img_batch)
boxes = get_region_boxes(output, 0.5, darknet_model.num_classes, 
                        darknet_model.anchors, darknet_model.num_anchors)[0]
boxes = nms(boxes,0.4)
class_names = load_class_names('data/coco.names')
squeezed = img_batch.squeeze(0)
#squeezed = p_img_batch[0,:,:,:]
print(squeezed.shape)
img = transforms.ToPILImage('RGB')(squeezed.detach().cpu())
plotted_image = plot_boxes(img, boxes, class_names=class_names)
plt.imshow(plotted_image)
plt.show()

print(boxes)


# In[17]:


# test-- After attack
from PIL import Image

adv_patch = Image.open("./saved_patches/patch_lion.jpg")#  .convert('RGB')
adv_patch = adv_patch.resize((img_size, img_size))
# print(adv_patch)


# adv_patch = cv2.imread("./saved_patches/exp/patch_cake_0.jpg")
# adv_patch = np.resize(adv_patch, (img_size, img_size, 3))
# print(adv_patch)

# adv_patch = np.array(Image.fromarray(adv_patch))

transform = transforms.ToTensor()
adv_patch = transform(adv_patch).cuda()
print(adv_patch.shape)

img_size = img_batch.size(-1)
print(img_size)
print(label.cuda().shape)

# transform to patch that can be adapted to img directly
adv_batch_t = patch_transformer(adv_patch, lab_batch.cuda(), img_size, do_rotate=False, rand_loc=False)
# print('adv patch t shaep:', adv_batch_t.shape)

p_img_batch = patch_applier(img_batch, adv_batch_t)
# print('p_img_batch shaep:', p_img_batch.shape)
p_img_batch = F.interpolate(p_img_batch,(darknet_model.height, darknet_model.width))
# print('after interpolate p_img_batch shaep:', p_img_batch.shape)
    
output = darknet_model(p_img_batch)
boxes = get_region_boxes(output,0.5,darknet_model.num_classes, 
                        darknet_model.anchors, darknet_model.num_anchors)[0]
boxes = nms(boxes,0.4)
class_names = load_class_names('data/coco.names')
# print('before squzzezd shape:', squeezed.shape)
squeezed = p_img_batch.squeeze(0)
# print('squzzezd shape:', squeezed.shape)
img = transforms.ToPILImage('RGB')(squeezed.detach().cpu())
plotted_image = plot_boxes(img, boxes, class_names=class_names)
plt.imshow(plotted_image)
# plt.imshow(adv_patch)
plt.show()

print(boxes)


# In[18]:


# Evalate on all the data
from utils import *
image_root = './test/SceneImages/img/'
label_root = './test/SceneImages/label/'
imgs = sorted(os.listdir(image_root))
labs = sorted(os.listdir(label_root))
#print(imgs)
#print(labs)
adv_patch = Image.open("./saved_patches/patch_lion.jpg")
adv_patch = adv_patch.resize((img_size, img_size))
class_names = load_class_names('data/coco.names')
count = 0
countlabel = []

transform = transforms.ToTensor()
adv_patch = transform(adv_patch).cuda()
print(adv_patch.shape)

# process iamges under director
for idx in range(len(os.listdir(image_root))):
    img_idx = os.path.join(image_root, imgs[idx])
    lab_idx = os.path.join(label_root, labs[idx])
    #break

    image = Image.open(img_idx).convert('RGB')
    label_df = pd.read_csv(lab_idx, header=None, sep=',') 
    label_df.iloc[0, 4] = class_names.index(label_df.iloc[0, 4])
    label = label_df.astype(np.float16).to_numpy()
    ymin, xmin, ymax, xmax, _ = label.squeeze(0)

    label = torch.tensor([[(xmax+xmin)/2.0, (ymax+ymin)/2.0, xmax-xmin, ymax-ymin]])
    label = torch.cat((torch.tensor([[_]]), label), dim=1).squeeze(0)
    label = label / 1024.0
    
    # print(label.shape)
    # print(label)
    if label.dim() == 1:
        label = label.unsqueeze(0)
    # print(label.shape)
    image, label = pad_and_scale(image, label, img_size)
    transform = transforms.ToTensor()
    image = transform(image)

    batch_size = 1    
    image = image.unsqueeze(0)
    label = label.unsqueeze(0)
    img_batch = image.expand(batch_size,-1,-1,-1)
    lab_batch = label.expand(batch_size,-1,-1)

    img_batch = img_batch.cuda()
    lab_batch = lab_batch.cuda()

    adv_batch_t = patch_transformer(adv_patch, lab_batch.cuda(), img_size, do_rotate=False, rand_loc=False)
    # print('adv patch t shaep:', adv_batch_t.shape)

    p_img_batch = patch_applier(img_batch, adv_batch_t)
    # print('p_img_batch shaep:', p_img_batch.shape)
    p_img_batch = F.interpolate(p_img_batch,(darknet_model.height, darknet_model.width))
    # print('after interpolate p_img_batch shaep:', p_img_batch.shape)
        
    output = darknet_model(p_img_batch)
    boxes = get_region_boxes(output,0.5,darknet_model.num_classes, 
                            darknet_model.anchors, darknet_model.num_anchors)[0]
    boxes = nms(boxes,0.4)
    if boxes != []:
        count += 1
        countlabel.append(boxes)
    class_names = load_class_names('data/coco.names')
    # print('before squzzezd shape:', squeezed.shape)
    squeezed = p_img_batch.squeeze(0)
    # print('squzzezd shape:', squeezed.shape)
    img = transforms.ToPILImage('RGB')(squeezed.detach().cpu())
    plotted_image = plot_boxes(img, boxes, class_names=class_names)
    plt.imshow(plotted_image)
    # plt.imshow(adv_patch)
    plt.show()
    #img.save(f'./outputs/attack_result_{idx}.png')
print('attack failed: {:d}, total {:d}'.format(count, len(imgs)))


# In[20]:


print('attack failed: {:d}, total {:d}'.format(count, len(imgs)))


# In[ ]:




