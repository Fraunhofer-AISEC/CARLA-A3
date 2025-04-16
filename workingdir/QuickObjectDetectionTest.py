import cv2
import numpy as np
from matplotlib import pyplot
from PIL import Image

from details.CV2Manager import CV2Manager
from details.faster_rcnn.wrapper import FasterRCNN
from details.yolo.wrapper import Yolo

import matplotlib; matplotlib.use("TkAgg")
"""
Object Detection code test

"""

def main(filename : str, objectDetector, h, w):
    
    img = np.array(Image.open(filename))[:,:,:3]
    img = cv2.resize(img, (h, w))

    CV2 = CV2Manager()
    boundingboxes, classes, scores, classname, _= objectDetector.detect(img)
    annotatedImg = CV2.drawBoundingBox(
        img, boundingboxes, classes, scores, classname
    )
    pyplot.figure(1); pyplot.imshow(img)
    pyplot.figure(2); pyplot.imshow(annotatedImg)
    pyplot.show(block=False)
    pyplot.pause(0.001)
    pyplot.imsave(
        f"{filename}_annotated.png",
        annotatedImg
    )

if __name__ == "__main__":
    
    w, h = 1200, 1200

    frcnn = FasterRCNN(inputWidth=w, inputHeight=h)
    yolo = Yolo()
    
    basename = "camoutput/%06d.png"
    for i in range(3, 70):
        file = basename % i
        main(file, frcnn, h, w)
