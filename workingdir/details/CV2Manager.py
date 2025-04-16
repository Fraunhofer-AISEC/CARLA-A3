import cv2
import numpy as np
from .yolo.utils import plot_boxes_cv2
from object_detection.utils.visualization_utils import visualize_boxes_and_labels_on_image_array
from PIL import Image
import os
class CV2Manager:

    def __init__(self) -> None:
        pass

    @classmethod
    def drawBoundingBox(cls, img, bboxes, classindex, scores, classname):
        if bboxes is None:
            return img

        classindex = np.array(classindex)
        if (classindex.ndim == 2):
            # assume sorted in ascending order
            classindex = classindex[:, -1] 

        scores = np.array(scores)
        if (scores.ndim == 2): 
            # assume sorted in ascending order
            scores = scores[:, -1] 

        classname = np.array(classname)
        if (classname.ndim == 2):
            # assume sorted in ascending order
            classname = classname[:,-1] 

        categoryIndex = {i: {"id": i, "name":j} for (i, j) in zip(classindex, classname)}
        overlay_img = visualize_boxes_and_labels_on_image_array(
            img,
            bboxes,
            classindex,
            scores,
            categoryIndex,
            use_normalized_coordinates=False,
            max_boxes_to_draw=None,
            min_score_thresh=0.5,
            line_thickness=1
        )
        return overlay_img

    @classmethod
    def showDetection(cls, winTitle, img, bboxes, classindex, scores, classname,imgid,dir,saveCameraImageFlag,patch_indx):
        overlay_img = cls.drawBoundingBox(img, bboxes, classindex, scores, classname)
        cls.showImage(winTitle, overlay_img)
        if saveCameraImageFlag:
            img = Image.fromarray(img)
            img.save(os.path.join(dir, F"{patch_indx}_"+F"{imgid:06d}.png"))
            

    @staticmethod
    def showImage(winTitle, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow(winTitle, img)
        cv2.waitKey(1)