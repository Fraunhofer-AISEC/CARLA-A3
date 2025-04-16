import numpy as np
import PIL
import torch
from matplotlib import pyplot

from details.CV2Manager import CV2Manager
from details.yolo.utils import get_region_boxes
from details.yolo.wrapper import Yolo

import matplotlib; matplotlib.use("TkAgg")

img = "camoutput/000050.png"
lbl = "camoutput/000050.txt"


od = Yolo()

x = pyplot.imread(img)
x = x[:,:,:3]
x = x*255
x=x.astype(np.uint8)


def helper(x):
    bbox, cls, scores, clsname, _ = od.detect(x)

    annX = CV2Manager.drawBoundingBox(
        x.copy(), bbox, cls, scores, clsname
    )
    pyplot.figure()
    pyplot.imshow(annX)
    pyplot.show(block=False)
    pyplot.pause(0.001)



helper(x)

xx = np.array(PIL.Image.open(img).resize((608, 608)))[:,:,:3]
xx_input = torch.tensor(xx.transpose(2, 0, 1)).div(255).float().unsqueeze(0)
ypred = od.model.forward(xx_input)
y = torch.zeros_like(ypred) # Ground truth



# fill the ground truth matrix
gt = np.loadtxt(lbl)
nGrids = 19
gt[1:5] *= nGrids

label, x_c, y_c, w, h = gt
grid_x_idx = int(np.floor(x_c))
grid_y_idx = int(np.floor(y_c))
cls = np.zeros(80)
cls[int(label)] = 5 # arbitarily large number


def inv_sigmoid(x):
    return np.log(x/(1-x))

bx = x_c
by = y_c
bw = w
bh = h
cx = grid_x_idx
cy = grid_y_idx
tx = inv_sigmoid(bx-cx)
ty = inv_sigmoid(by-cy)
# inv_sigmoid or not is the same. Why?
pw = od.model.anchors[2]
ph = od.model.anchors[3]
tw = np.log(bw/pw)
th = np.log(bh/ph)

y = y.squeeze()
# y[0:4 , grid_x_idx, grid_y_idx] = torch.tensor([x_c, y_c, w, h])
y[0+85:4+85 , grid_y_idx, grid_x_idx] = torch.tensor([tx, ty, tw, th])
y[4+85, grid_y_idx, grid_x_idx] = 1 #objectness score, arbitarily large number
y[5+85:5+80+85,grid_y_idx, grid_x_idx] = torch.tensor(cls)
y = y.unsqueeze(0)




# convert ground truth matrix back to bbox format
bbox_pred = get_region_boxes(ypred, 0.5, od.model.num_classes, od.model.anchors, od.model.num_anchors)[0]
bbox_gt   = get_region_boxes(y, 0.5, od.model.num_classes, od.model.anchors, od.model.num_anchors)[0]

def bbox_rel_to_abs(bbox, width, height):
    boundingbox = []
    class_idx = []
    score = []
    scores_array = []
    classname = []


    for box in bbox:
        x1 = int(torch.nan_to_num(torch.round((box[0] - box[2]/2.0) * width)))
        y1 = int(torch.nan_to_num(torch.round((box[1] - box[3]/2.0) * height)))
        x2 = int(torch.nan_to_num(torch.round((box[0] + box[2]/2.0) * width)))
        y2 = int(torch.nan_to_num(torch.round((box[1] + box[3]/2.0) * height)))

        boundingbox.append([y1, x1, y2, x2])
        score.append(box[5])
        class_idx.append(box[6])
        classname.append("stop sign")
        scores_array.append(box[7].cpu().detach().numpy())


    return np.array(boundingbox), np.array(score), np.array(class_idx), np.array(classname)

bb, sc, clsidx, clsname = bbox_rel_to_abs(bbox_gt, xx.shape[0], xx.shape[1])
annX = CV2Manager.drawBoundingBox(xx.copy(), bb, clsidx, sc, clsname)
pyplot.figure()
pyplot.imshow(annX)
pyplot.show(block=False)
pyplot.pause(0.001)

opt = torch.optim.Adam(od.model.parameters())

# naive loss function. not the actual loss function used by Yolo
MSE = torch.nn.MSELoss()

for i in range(100):
    ypred = od.model.forward(xx_input)
    loss = MSE(ypred, y)
    print(f"epoch {i:03d}, loss = {loss.item():.2f}")
    loss.backward()
    opt.step()
    opt.zero_grad()

helper(x)
pass
