#!/usr/bin/env python
# coding: utf-8

# In[1]:


from load_data import *
import load_data
import matplotlib.pyplot as plt

plt.rcParams["axes.grid"] = False


# In[2]:


from load_data import *
import load_data
import gc
import matplotlib.pyplot as plt
import patch_config
from torch import autograd
plt.rcParams["axes.grid"] = False
plt.axis('on')

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


# In[3]:


# Single image training

from utils import *



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
    return padded_img, lab

# Initial the adversarial patch 
patch_img = Image.open("saved_patches/before.jpg").convert('RGB')
tf = transforms.Resize((patch_size,patch_size)) # should be set as the size of the stopsign

patch_img = tf(patch_img)
tf = transforms.ToTensor()
adv_patch_cpu = tf(patch_img)

adv_patch_cpu.requires_grad_(True)





img_path = "test/img/stopsign.png"
lab_path = "test/lab/stopsign.txt"

imgsize = darknet_model.height
image = Image.open(img_path).convert('RGB')
label = np.loadtxt(lab_path)
label = torch.from_numpy(label).float()
if label.dim() == 1:
    label = label.unsqueeze(0)
image, label = pad_and_scale(image, label, imgsize)
transform = transforms.ToTensor()
image = transform(image)

batch_size = 24    
image = image.unsqueeze(0)
label = label.unsqueeze(0)
img_batch = image.expand(batch_size,-1,-1,-1)
lab_batch = label.expand(batch_size,-1,-1)

img_batch = img_batch.cuda()
lab_batch = lab_batch.cuda()

n_epochs = 100

#random mode
#adv_patch_cpu = torch.full((3,patch_size,patch_size),0.5)
#adv_patch_cpu = torch.rand((3,patch_size,patch_size))
#adv_patch_cpu.requires_grad_(True)

optimizer = optim.Adam([adv_patch_cpu], lr = 0.01)

tl1 = time.time()
for epoch in range(n_epochs):
    ffw1 = time.time()
    adv_patch = adv_patch_cpu.cuda()
    tl0 = time.time()
    #print('batch load time: ', tl0-tl1)
    img_size = img_batch.size(-1)
    #print('transforming patches')
    t0 = time.time()
    adv_batch_t =  patch_transformer(adv_patch, lab_batch, img_size)
    #print('applying patches')
    t1 = time.time()
    p_img_batch = patch_applier(img_batch, adv_batch_t)
    p_img_batch = F.interpolate(p_img_batch,(darknet_model.height, darknet_model.width))
    #print('running patched images through model')
    t2 = time.time()

    output = darknet_model(p_img_batch)
    #print('does output require grad? ',output.requires_grad)
    #print('extracting max probs')
    t3 = time.time()
    max_prob = prob_extractor(output)
    #print('does max_prob require grad? ',max_prob.requires_grad)
    #print('calculating nps')
    t4 = time.time()
    nps = nps_calculator(adv_patch)
    t5 = time.time()
    #print('calculating tv')
    tv = total_variation(adv_patch)
    t6 = time.time()
    '''
    print('---------------------------------')
    print('   patch transformation : %f' % (t1-t0))
    print('      patch application : %f' % (t2-t1))
    print('        darknet forward : %f' % (t3-t2))
    print(' probability extraction : %f' % (t4-t3))
    print('        nps calculation : %f' % (t5-t4))
    print('        total variation : %f' % (t6-t5))
    print('---------------------------------')
    print('     total forward pass : %f' % (t6-t0))

    print(torch.mean(max_prob))
    print(nps)
    print(tv)
    '''

    det_loss = torch.mean(max_prob)
    nps_loss = nps*0.1
    tv_loss = tv*0.00005
    loss = det_loss + nps_loss + tv_loss

    #img_batch.retain_grad()
    #adv_batch_t.retain_grad()
    #adv_patch.retain_grad()

    #print('loss to patch', torch.autograd.grad(loss,img_batch))
    loss.backward()
    tl1 = time.time()
    #print('adv_patch.grad',adv_patch_cpu.grad)
    #print('adv_batch_t.grad',adv_batch_t.grad)
    #print('img_batch.grad',img_batch.grad)
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.empty_cache()
    ffw2 = time.time()
    if (epoch)%1 == 0:
        print('  EPOCH NR: ', epoch)
        print('EPOCH LOSS: ', loss.detach().cpu().numpy())
        print('  DET LOSS: ', det_loss.detach().cpu().numpy())
        print('  NPS LOSS: ', nps_loss.detach().cpu().numpy())
        print('   TV LOSS: ', tv_loss.detach().cpu().numpy())
        print('EPOCH TIME: ', ffw2-ffw1)
        boxes = get_region_boxes(output,0.5,darknet_model.num_classes, 
        darknet_model.anchors, darknet_model.num_anchors)[0]
        boxes = nms(boxes,0.4)

        print(boxes)
        
        im = transforms.ToPILImage('RGB')(adv_patch_cpu)
        plt.imshow(im)
        plt.show()
    #del adv_batch_t, output, max_prob

    
boxes = get_region_boxes(output,0.5,darknet_model.num_classes, 
darknet_model.anchors, darknet_model.num_anchors)[0]
boxes = nms(boxes,0.4)

print(boxes)
        
im = transforms.ToPILImage('RGB')(adv_patch_cpu)
plt.imshow(im)
plt.show()


# In[5]:


im.save('saved_patches/patchlion.jpg')
print(boxes)


# In[6]:


# test input clean image

from utils import *
patch_size = darknet_model.height
img_size = darknet_model.height

img_path = "test/img/stopsign.png"
lab_path = "test/lab/stopsign.txt"

image = Image.open(img_path).convert('RGB')
label = np.loadtxt(lab_path)
label = torch.from_numpy(label).float()
if label.dim() == 1:
    label = label.unsqueeze(0)
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
boxes = get_region_boxes(output,0.5,darknet_model.num_classes, 
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


# In[ ]:


# test-- After attack


adv_patch = Image.open("saved_patches/patchlion.jpg").convert('RGB')
adv_patch = adv_patch.resize((img_size,img_size))
#before = np.array(PIL.Image.fromarray(adv_patch).resize((msize, msize)))




#adv_patch = after_adv
#adv_patch = PIL.Image.fromarray(adv_patch).resize((300,300))

transform = transforms.ToTensor()
adv_patch = transform(adv_patch).cuda()
#print(np.size(adv_patch))

img_size = img_batch.size(-1)
#print(img_size)
adv_batch_t = patch_transformer(adv_patch, lab_batch.cuda(), img_size, do_rotate=False, rand_loc=False)



p_img_batch = patch_applier(img_batch, adv_batch_t)
p_img_batch = F.interpolate(p_img_batch,(darknet_model.height, darknet_model.width))
    
output = darknet_model(p_img_batch)
boxes = get_region_boxes(output,0.5,darknet_model.num_classes, 
                        darknet_model.anchors, darknet_model.num_anchors)[0]
boxes = nms(boxes,0.4)
class_names = load_class_names('data/coco.names')
squeezed = p_img_batch.squeeze(0)
print(squeezed.shape)
img = transforms.ToPILImage('RGB')(squeezed.detach().cpu())
plotted_image = plot_boxes(img, boxes, class_names=class_names)
plt.imshow(plotted_image)
#plt.imshow(adv_patch)
plt.show()


# In[15]:


print(boxes)


# In[ ]:




