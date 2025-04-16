from .darknet import Darknet
from .utils import do_detect, load_class_names
from ..ObjectDetectorBase import ObjectDetectorBase

import cv2
import torch
import numpy as np

class Yolo(ObjectDetectorBase):    
    def __init__(
        self,
        cudaFlag = False,
        yoloConfigFile = 'details/attack/adversarial_yolo_master/cfg/yolov2.cfg',
        yoloWeightFile = 'details/attack/adversarial_yolo_master/weights/yolo.weights',
        classNameFile = 'details/attack/adversarial_yolo_master/data/coco.names'
    ) -> None:
        super().__init__()
        self.model = Darknet(yoloConfigFile)
        self.model.print_network()
        self.model.load_weights(yoloWeightFile)
        self.classNames = load_class_names(classNameFile)
        self.cudaFlag = cudaFlag
        if cudaFlag:
            self.model.cuda()

    def detect(
        self,
        img,
        scoreThreshold = 0.5,
        bboxIOUthreshold = 0.4
    ):
        resized_img = cv2.resize(img, (self.model.width, self.model.height))
        bounding_boxes = do_detect(
            model=self.model,
            img=resized_img, 
            conf_thresh=scoreThreshold,
            nms_thresh=bboxIOUthreshold,
            use_cuda=self.cudaFlag
        )
        boundingbox = []
        class_idx = []
        score = []
        scores_array = []
        classname = []

        height, width, _ = img.shape

        for box in bounding_boxes:
            x1 = int(torch.nan_to_num(torch.round((box[0] - box[2]/2.0) * width)))
            y1 = int(torch.nan_to_num(torch.round((box[1] - box[3]/2.0) * height)))
            x2 = int(torch.nan_to_num(torch.round((box[0] + box[2]/2.0) * width)))
            y2 = int(torch.nan_to_num(torch.round((box[1] + box[3]/2.0) * height)))

            boundingbox.append([y1, x1, y2, x2])
            score.append(box[5])
            class_idx.append(box[6])
            classname.append(self.classNames[box[6]])
            scores_array.append(box[7].cpu().detach().numpy())
        
        return (
            np.array(boundingbox),
            np.array(class_idx),
            np.array(score), 
            np.array(classname),
            np.array(scores_array)
        )