from pathlib import Path

from matplotlib import pyplot

file = Path(__file__)
lib = "adversarial_yolo_master"
dependency = file.parent.joinpath(lib).as_posix()
import sys

if dependency not in sys.path:
    sys.path.append(dependency)


import time

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms
from tqdm import tqdm

from ..CV2Manager import CV2Manager
from .adversarial_yolo_master.darknet import Darknet
from .adversarial_yolo_master.load_data import (MaxProbExtractor,
                                                NPSCalculator, PatchApplier,
                                                PatchStopTransformer,
                                                TotalVariation)
from .adversarial_yolo_master.patch_config import patch_configs


def pad_and_scale(img, lab, imgsize):
   
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
    # Choose here
    padded_img = padded_img.resize((imgsize,imgsize))     
    return padded_img, lab



class Attacker():
    def __init__(self) -> None:
        self.cfgfile = "cfg/yolo.cfg"
        self.cfgfile = file.parent.joinpath(lib, self.cfgfile).as_posix()
        self.weightfile = "weights/yolo.weights"
        self.weightfile = file.parent.joinpath(lib, self.weightfile).as_posix()
        self.printfile = "non_printability/30values.txt"
        self.printfile = file.parent.joinpath(lib, self.printfile).as_posix()
        # Stop sign
        self.targetClassIndex = 11 
        self.mode = "base"
        # Coco labels
        self.numClass = 80 

        self.darknetModel = Darknet(self.cfgfile)
        self.darknetModel.load_weights(self.weightfile)
        self.darknetModel = self.darknetModel.eval()
        self.patchSize = self.darknetModel.height
        

        self.probExtractor = MaxProbExtractor(
            self.targetClassIndex, 
            self.numClass, 
            patch_configs[self.mode]()
        )
        self.npsCalculator = NPSCalculator(self.printfile, self.patchSize)
        self.totalVariation = TotalVariation()
        self.patchTransformer = PatchStopTransformer()
        self.patchApplier = PatchApplier()
        

    def generateAttackPatch(
        self,
        initialPatch = None,
        numEpoch = 10,
        outputDir = "advpatch",
        outputName = "advpatch.png"
        ) -> np.array:
        
        Path(outputDir).mkdir(parents=True, exist_ok=True)
        
        
        # Load initial patch
        if initialPatch is None:
            self.adv_patch = torch.rand(3, self.patchSize, self.patchSize)

        # Load the initial patch given the image file
        else: 
            self.adv_patch = Image.open(initialPatch).convert("RGB")
            transform = transforms.Resize((self.patchSize, self.patchSize))
            self.adv_patch = transform(self.adv_patch)
            transform = transforms.ToTensor()
            self.adv_patch = transform(self.adv_patch)

        self.adv_patch.requires_grad_(True)
        

        # Load scene image
        img_path = F"{dependency}/test/img/stopsign.png"
        lab_path = F"{dependency}/test/lab/stopsign.txt"
        image = Image.open(img_path).convert('RGB')
        label = np.loadtxt(lab_path)
        label = torch.from_numpy(label).float()
        if label.dim() == 1:
            label = label.unsqueeze(0)
        image, label = pad_and_scale(image, label, self.patchSize)
        transform = transforms.ToTensor()
        image = transform(image)

        self.batch_size = 24
        image = image.unsqueeze(0)
        label = label.unsqueeze(0)
        img_batch = image.expand(self.batch_size,-1,-1,-1)
        lab_batch = label.expand(self.batch_size,-1,-1)

        img_batch = img_batch.cuda()
        lab_batch = lab_batch.cuda()

        optimizer = torch.optim.Adam([self.adv_patch], lr = 0.01)

        for epoch in range(numEpoch):
            
            ffw1 = time.time()
            adv_patch_local = self.adv_patch.cuda()
            tl0 = time.time()
            
            img_size = img_batch.size(-1)
            
            t0 = time.time()
            
            lab_batch = lab_batch.cuda()
            adv_batch_t =  self.patchTransformer(adv_patch_local, lab_batch, img_size)
            
            t1 = time.time()
            img_batch=img_batch.cuda()
            p_img_batch = self.patchApplier(img_batch, adv_batch_t)
            p_img_batch = F.interpolate(p_img_batch,(self.darknetModel.height, self.darknetModel.width))
            
            t2 = time.time()
            p_img_batch = p_img_batch.cpu()
            output = self.darknetModel(p_img_batch)
           
            t3 = time.time()
            output = output.cuda()
            max_prob = self.probExtractor(output)
            max_prob = max_prob.cpu()
            output = output.cpu()
            
            
            t4 = time.time()
            adv_patch_local = adv_patch_local.cpu()
            nps = self.npsCalculator(adv_patch_local)
            t5 = time.time()
            
            tv = self.totalVariation(adv_patch_local)
            t6 = time.time()
            
            det_loss = torch.mean(max_prob)
            nps_loss = nps*0.1
            tv_loss = tv*0.00005
            loss = det_loss + nps_loss + tv_loss

            
            loss.backward()
            tl1 = time.time()
            
            
            
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
                            
            CV2Manager.showImage("Adv Patch", np.array(transforms.ToPILImage()(self.adv_patch)))
            patch_img = np.array(transforms.ToPILImage()(self.adv_patch))

        a = self.maskStopSign(patch_img)
        pyplot.imsave(Path(outputDir).joinpath(outputName).as_posix(), a)
        
        return a
        
    def maskStopSign(self, image):
        self.baseMaskFile = file.parent.joinpath(lib, "masks/base_mask.txt").as_posix()
        self.textMaskFile = file.parent.joinpath(lib, "masks/text_mask.txt").as_posix()
        baseMask = np.loadtxt(self.baseMaskFile)
        textMask = np.loadtxt(self.textMaskFile)

        baseMask = np.array(Image.fromarray(baseMask).resize((self.patchSize, self.patchSize))).clip(0, 1)
        textMask = np.array(Image.fromarray(textMask).resize((self.patchSize, self.patchSize))).clip(0, 1)
        after = np.multiply(image, np.stack([baseMask]*3, 2))
        after = np.maximum(after/255, np.stack([textMask]*3, 2))

        alpha = baseMask + textMask
        alpha = alpha.clip(0, 1)

        return np.dstack((after, alpha))


if __name__ == "__main__":

    attacker = Attacker()
