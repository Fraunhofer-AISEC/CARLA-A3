#!/usr/bin/env python
# coding: utf-8

# In[1]:

from pathlib import Path
from importlib.resources import path
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
from sklearn.model_selection import train_test_split

plt.rcParams["axes.grid"] = False


# In[2]:


from load_data import *
import gc
import matplotlib.pyplot as plt
import patch_config
from utils import * 
import numpy as np
plt.rcParams["axes.grid"] = False
plt.axis('off')

cfgfile = "cfg/yolo.cfg"
weightfile = "weights/yolov2.weights"
printfile = "non_printability/30values.txt"

def seed_everything(seed: int):
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed = 3407
mode = 'base'

# seed everything
seed_everything(seed)

print('LOADING MODELS')
darknet_model = Darknet(cfgfile)
darknet_model.load_weights(weightfile)
patch_size = darknet_model.height
img_size = darknet_model.height
darknet_model = darknet_model.eval().cuda()
patch_applier = PatchApplier().cuda()
patch_transformer_train = PatchStopTransformer().cuda()
patch_transformer_train.train(True)
patch_transformer_test = PatchStopTransformer().cuda()
patch_transformer_test.train(False)
prob_extractor = MaxProbExtractor(11, 80, patch_config.patch_configs[mode]()).cuda() # 11 for stopsign
nps_calculator = NPSCalculator(printfile, patch_size)
nps_calculator = nps_calculator.cuda()
total_variation = TotalVariation().cuda()
print('MODELS LOADED')
print(patch_size)
print(img_size)


# In[3]:


# seed everything
seed_everything(seed)

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
       self.path_img = root + '/image/'
       self.path_lab = root + '/label/'
       self.idx_list = []

       self.images = [os.path.join(self.path_img, img) for img in os.listdir(self.path_img)]
       self.labels = [os.path.join(self.path_lab, lab) for lab in os.listdir(self.path_lab)]

    #    print('before sorted:', self.images)
    #    print('before sorted:', self.labels)

       self.images = sorted(self.images)
       self.labels = sorted(self.labels)

    #    print('after sorted:',self.images)
    #    print('after sorted:',self.labels)
       
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

        # pix_shift = torch.randint(-3, 3, (2, ))
        label = torch.tensor([[(xmax+xmin)/2.0, (ymax+ymin)/2.0, xmax-xmin, ymax-ymin]])
        # pix_shift = torch.cat((pix_shift, torch.zeros(2,))).unsqueeze(0)
        # label = label + pix_shift

        # print(label)
        label = label / 1024.0
        # print(label)
        image, label = pad_and_scale(image, label, self.image_size)
        # print(label)
        label = torch.cat((torch.tensor([[_]]), label), dim=1)
        # print(label)
        return image, label
        
    
# test the dataset class
root = './test/SceneImages2/'
dataset = StopSignDataset(root, train=True, image_size=416)  # 416 is the darknet.height
image, label = dataset[108]
print(label)
print(image.shape)


# In[4]:


# seed everything
seed_everything(seed)

# dataset and dataloader setting 
# train test split 
root_ = './test/SceneImages2/'
root_img = root_ + 'image/'
idx_list = []
images = [os.path.join(root_img, img) for img in os.listdir(root_img)]
# print(images)
nSamples = len(images)
for i in range(nSamples):
    idx_list.append(i)
        
# fix the random state and train_idx and test_idx
train_idx, test_idx = train_test_split(idx_list, shuffle=True, test_size=0.2, random_state=42)
print('idx_list:', idx_list)
print('train_idx:', train_idx)
print('test_idx:', test_idx)
print('length:', len(train_idx)+len(test_idx))

# dataloader
the_whole_set = StopSignDataset(root_, train=True, image_size=img_size)
whole_dataloader = DataLoader(the_whole_set, batch_size=8, shuffle=True, num_workers=0)
# validation_set = StopSignDataset(root_, train=True, image_size=img_size)
training_set = torch.utils.data.Subset(the_whole_set, train_idx)
validation_set = torch.utils.data.Subset(the_whole_set, test_idx)
train_dataloader = DataLoader(training_set, batch_size=4, shuffle=True, num_workers=0)
val_dataloader = DataLoader(validation_set, batch_size=1, shuffle=True, num_workers=0) # bs should be 1 to save chpt img


# In[5]:


# seed everything
seed_everything(seed)

# params setting
# Initial the adversarial patch 
patch_img = Image.open("saved_patches/cake_2040.jpg").convert('RGB') # patch is random choosing with some pointed pattern

# different resolution experiment
# resize, save, read, resize to model 
# tf_original = transforms.Resize((patch_size,patch_size)) # should be sett as the size of the stopsign


# random noise patch
# def get_white_noise_image(width, height):
#     pil_map = Image.new("RGB", (width, height), 255)
#     random_grid = map(lambda x: (
#             int(random.random() * 256),
#             int(random.random() * 256),
#             int(random.random() * 256)
#         ), [0] * width * height)
#     pil_map.putdata(list(random_grid))
#     return pil_map

# patch_img = get_white_noise_image(patch_size,patch_size)
# patch_img.show()

# patch to tensor with same size of stop sign
tf = transforms.Resize((patch_size,patch_size)) # should be sett as the size of the stopsign
patch_img = tf(patch_img)
tf = transforms.ToTensor()
adv_patch_cpu = tf(patch_img)
adv_patch_cpu.requires_grad_(True)
print(adv_patch_cpu.shape)

n_epochs = 0
epoch_length = len(the_whole_set)
print(epoch_length)
optimizer = optim.Adam([adv_patch_cpu], lr = 0.01)
loss_epoch = []

# training 
for epoch in range(n_epochs):
    ffw1 = time.time()
    adv_patch = adv_patch_cpu.cuda()
    losses = np.array([])
    for i_batch, img_lab_list in tqdm(enumerate(whole_dataloader), desc=f'Running epoch {epoch}'):
        # print(i_batch, len(img_lab_list))
        img_batch, lab_batch = img_lab_list
        print(f"img_batch.shape {img_batch.shape}, lab_batch.shape {lab_batch.shape}")
        img_batch = img_batch.cuda()
        lab_batch = lab_batch.cuda()
        
        # forward adv img batch
        adv_batch_t =  patch_transformer_train(adv_patch, lab_batch, img_size)
        p_img_batch = patch_applier(img_batch, adv_batch_t)
        # print('before interpolate:',p_img_batch.shape)
        # why interpolate here
        p_img_batch = F.interpolate(p_img_batch, (darknet_model.height, darknet_model.width))
        # print('after interpolate:', p_img_batch.shape)
        #print('running patched images through model')

        
        p_img_batch = p_img_batch.cpu()
        img_batch = img_batch.cpu()


        output_adv = darknet_model.cpu()(p_img_batch)
        output_clean = darknet_model.cpu()(img_batch)

        output_adv = output_adv.cuda()
        output_clean = output_clean.cuda()


            
        # boxes_adv = get_region_boxes(output_adv, 0.5, darknet_model.num_classes, 
        #                         darknet_model.anchors, darknet_model.num_anchors)[0]
        # boxes_clean = get_region_boxes(output_clean, 0.5, darknet_model.num_classes, 
        #                         darknet_model.anchors, darknet_model.num_anchors)[0]
        # # print('boxex:', boxes)
        # boxes_adv = nms(boxes_adv,0.4)
        # boxes_clean= nms(boxes_clean,0.4)
        
        # batch size should be 1 when apply this 
        # print('boxex:', boxes)
        # class_names = load_class_names('data/coco.names')
        # squeezed = p_img_batch.squeeze(0)
        # img = transforms.ToPILImage('RGB')(squeezed.detach().cpu())
        # img.save(f'./saved_patches/checkpoints/checkpoints_{i_batch}.png')

        max_prob_adv = prob_extractor(output_adv)
        max_prob_clean = prob_extractor(output_clean)
        # print('extracting max probs adv:', max_prob_adv)
        # print('extracting max probs clean:', max_prob_clean)
        max_prob_adv = torch.where(max_prob_clean > 0.4, max_prob_adv, 0.0)
        # print('weighted max probs adv:', max_prob_adv)

        ################################# loss computing #########################################
        nps = nps_calculator(adv_patch)
        tv = total_variation(adv_patch)

        # obj loss
        det_loss = torch.mean(max_prob_adv)

        # printable loss
        nps_loss = nps*0.1

        # total variance
        tv_loss = tv*0.00005
        loss = det_loss + nps_loss + tv_loss
        ################################# loss computing #########################################

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.empty_cache()

        losses = np.append(losses, loss.detach().cpu().numpy())
        ffw2 = time.time()

    if (epoch) % 10 == 0:
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

    loss_epoch.append(losses.mean())
    # else:
    #     print('EPOCH LOSS: ', loss.detach().cpu().numpy())

    im = transforms.ToPILImage('RGB')(adv_patch_cpu)
    im.save('saved_patches/exp/patch_cake_{:d}.jpg'.format(epoch))
    
# boxes = get_region_boxes(output,0.5,darknet_model.num_classes, 
#                         darknet_model.anchors, darknet_model.num_anchors)[0]
# boxes = nms(boxes,0.4)
# print(boxes)
#im.save('saved_patches/patch_cake/cake_2040_masked_0824.jpg')


# In[ ]:


# plot the losses curve
# loss_epoch = np.array(loss_epoch)
# epoch = np.arange(0, n_epochs, dtype=int)
# plt.plot(epoch, loss_epoch, 'b')
# plt.show()


# In[ ]:


# test single input
img_path = "test/SceneImages2/image/000087.png"
lab_path = "test/SceneImages2/label/000087.txt"

# patch_transformer_test = PatchStopTransformer(train=False)

image = Image.open(img_path).convert('RGB')
class_names = load_class_names('data/coco.names')

label_df = pd.read_csv(lab_path, header=None, sep=',') 
label_df.iloc[0, 4] = class_names.index(label_df.iloc[0, 4])
label = label_df.astype(np.float16).to_numpy()
ymin, xmin, ymax, xmax, _ = label.squeeze(0)

label = torch.tensor([[(xmax+xmin)/2.0, (ymax+ymin)/2.0, xmax-xmin, ymax-ymin]])
label = label / 1024.0
label = torch.cat((torch.tensor([[_]]), label), dim=1).squeeze(0)
# print(label.shape)
print(label)

if label.dim() == 1:
    label = label.unsqueeze(0)
# print(label.shape)
image, label = pad_and_scale(image, label, img_size)
transform = transforms.ToTensor()
image = transform(image)

adv_patch = Image.open("./saved_patches/patch_cake/cake_2040_masked_0824.jpg") # .convert('RGB')
adv_patch = adv_patch.resize((img_size, img_size))
# print(adv_patch)

transform = transforms.ToTensor()
adv_patch = transform(adv_patch).cuda()
print(adv_patch.shape)

batch_size = 1    
image = image.unsqueeze(0)
label = label.unsqueeze(0)
img_batch = image.expand(batch_size,-1,-1,-1)
lab_batch = label.expand(batch_size,-1,-1)

img_batch = img_batch.cuda()
lab_batch = lab_batch.cuda()

adv_batch_t = patch_transformer_test(adv_patch, lab_batch.cuda(), img_size, do_rotate=False, rand_loc=False)
print('adv patch t shape:', adv_batch_t.shape)

p_img_batch = patch_applier(img_batch, adv_batch_t)
print('p_img_batch shape:', p_img_batch.shape)
p_img_batch = F.interpolate(p_img_batch, (darknet_model.height, darknet_model.width))
print('after interpolate p_img_batch shaep:', p_img_batch.shape)

image = p_img_batch.squeeze(0)
image = transforms.ToPILImage('RGB')(image.detach().cpu())
image.save("./saved_patches/demo.png")


output = darknet_model.cuda()(p_img_batch)
print(p_img_batch.shape)
print(output.shape)
boxes = get_region_boxes(output,0.5,darknet_model.num_classes, 
                    darknet_model.anchors, darknet_model.num_anchors)[0]
print('boxes:', boxes)
boxes = nms(boxes,0.4)
print('boxes after:',boxes)

# boxes = label
# print('boxes:', boxes)
# boxes = nms(boxes,0.4)

class_names = load_class_names('data/coco.names')
# print('before squzzezd shape:', squeezed.shape)
squeezed = p_img_batch.squeeze(0)
print(squeezed.shape)
# print('squzzezd shape:', squeezed.shape)
img = transforms.ToPILImage('RGB')(squeezed.detach().cpu())
plotted_image = plot_boxes(img, boxes, class_names=class_names)
img.show()
plt.imshow(img)


# In[ ]:


# test images in batches
# initial
from utils import *
image_root = './test/effective/image/'
label_root = './test/effective/label/'
imgs = os.listdir(image_root)
labs = os.listdir(label_root)
adv_patch = Image.open("./saved_patches/patch_cake/cake_2040_masked_0824.jpg")
adv_patch = adv_patch.resize((img_size, img_size))
class_names = load_class_names('data/coco.names')
count = 0  # fail attack
sum = 0 # need to attack 

transform = transforms.ToTensor()
adv_patch = transform(adv_patch).cuda()
print(adv_patch.shape)

# process iamges under director
for idx in range(len(sorted(os.listdir(image_root)))):
    img_idx = os.path.join(image_root, imgs[idx])
    lab_idx = os.path.join(label_root, labs[idx])
    # print(lab_idx)

    image = Image.open(img_idx).convert('RGB')
    try:
        label_df = pd.read_csv(lab_idx, header=None, sep=',')   
    except:
        continue
    
    sum += 1
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


    image_original_dix = str.split(str.split(img_idx, '/')[-1], '.')[0]


    # adv samples
    adv_batch_t = patch_transformer_test(adv_patch, lab_batch.cuda(), img_size, do_rotate=False, rand_loc=False)
    # print('adv patch t shaep:', adv_batch_t.shape)
    p_img_batch = patch_applier(img_batch, adv_batch_t)
    # print('p_img_batch shaep:', p_img_batch.shape)
    p_img_batch = F.interpolate(p_img_batch,(darknet_model.height, darknet_model.width))
    # print('after interpolate p_img_batch shaep:', p_img_batch.shape)
        
    output_adv = darknet_model(p_img_batch)
    boxes_adv = get_region_boxes(output_adv,0.5,darknet_model.num_classes, 
                            darknet_model.anchors, darknet_model.num_anchors)[0]
    boxes_adv = nms(boxes_adv,0.4)
    # print('before squzzezd shape:', squeezed.shape)
    squeezed_adv = p_img_batch.squeeze(0)
    # print('squzzezd shape:', squeezed.shape)
    img_adv = transforms.ToPILImage('RGB')(squeezed_adv.detach().cpu())
    plotted_image_adv = plot_boxes(img_adv, boxes_adv, class_names=class_names)
    if boxes_adv != []:
        # clean images
        output_clean = darknet_model(img_batch)
        #output = darknet_model(p_img_batch)
        boxes_clean = get_region_boxes(output_clean, 0.5, darknet_model.num_classes, 
                                darknet_model.anchors, darknet_model.num_anchors)[0]
        boxes_clean = nms(boxes_clean,0.4)
        class_names = load_class_names('data/coco.names')
        squeezed_clean = img_batch.squeeze(0)
        img_clean = transforms.ToPILImage('RGB')(squeezed_clean.detach().cpu())
        plotted_image_clean = plot_boxes(img_clean, boxes_clean, class_names=class_names)
        # plotted_image_clean.show()
        Path("outputs/cake_2040_masked_0824/clean").mkdir(parents=True, exist_ok=True)
        plotted_image_clean.save(f'./outputs/cake_2040_masked_0824/clean/attack_result_{image_original_dix}_clean.png')

        fw = open("./outputs/cake_2040_masked_0824/test_log.txt", 'a+') 
        count += 1
        fw.write("attack fails sample {:d}:".format(count))  
        fw.write("\n")    
        fw.write('idx:')
        fw.write(image_original_dix)
        fw.write("\n")    
        fw.write('clean bounding box:')
        fw.write(str(boxes_clean))
        fw.write('\n')
        fw.write('adv bounding box:')
        fw.write(str(boxes_adv))
        fw.write('\n')
        fw.write('\n')
        
    # img.save(f'./outputs/attack_result_{idx}.png')

    
    Path("outputs/cake_2040_masked_0824/attack/").mkdir(parents=True, exist_ok=True)
    img_adv.save(f'./outputs/cake_2040_masked_0824/attack/attack_result_{image_original_dix}.png')
   
fw.close() 
print('attack failed: {:d}, total {:d}, Success rate {:.4f}'.format(count, sum, 1 - count/sum))



# In[ ]:




