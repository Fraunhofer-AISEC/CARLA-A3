from distutils.log import warn
import warnings

import numpy as np

from .annotation import AnnotationList
from .odmetriclib.BoundingBox import BoundingBox
from .odmetriclib.BoundingBoxes import BoundingBoxes
from .odmetriclib.Evaluator import Evaluator
from .odmetriclib.utils import (BBFormat, BBType, CoordinatesType,
                                MethodAveragePrecision)

np.seterr(divide='ignore', invalid='ignore')

def rejectOverlappingBbox(
    al : AnnotationList,
    iouThreshold : float = 0.5
):
    """
    Reject overlapping bounding boxes in an 
    annotation list
    """
    classScores = [i.classScore for i in al.annotationList]
    if len(classScores) == 0:
        return al

    classScores = np.array(classScores)
    if classScores.ndim == 2:
        classScores = classScores[:,-1]
    
    sortedID = np.argsort(classScores) 
    # ascending scores
    sortedID = sortedID[::-1] 
    # descending scores
    sortedClassScore = classScores[sortedID] 
    sortedAL = np.array(al.annotationList)[sortedID].tolist()

    N = len(sortedAL)
    output = []
    for i in range(N):
        if sortedClassScore[i] > 0:
            output.append(sortedAL[i])
            for j in range(i+1, N):
                bbox1 = sortedAL[i].boundingBox
                bbox2 = sortedAL[j].boundingBox
                if bbox_iou(bbox1, bbox2) > iouThreshold:
                    # remove overlap bbox
                    sortedClassScore[j] = 0 
    al.annotationList = output
    return al

def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = min(box1[0]-box1[2]/2.0, box2[0]-box2[2]/2.0)
        Mx = max(box1[0]+box1[2]/2.0, box2[0]+box2[2]/2.0)
        my = min(box1[1]-box1[3]/2.0, box2[1]-box2[3]/2.0)
        My = max(box1[1]+box1[3]/2.0, box2[1]+box2[3]/2.0)
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea/uarea


def computeAP(
    detection: AnnotationList,
    groundtruth: AnnotationList
):
    bboxes = BoundingBoxes()
    classname = {}

    addBB(bboxes, detection, BBType.Detected)
    addBB(bboxes, groundtruth, BBType.GroundTruth)

    evaluator = Evaluator()
    
    output = evaluator.GetPascalVOCMetrics(
        bboxes,  # Object containing all bounding boxes (ground truths and detections)
        IOUThreshold=0.3,  # IOU threshold
        method=MethodAveragePrecision.EveryPointInterpolation,  
    )  
# Plot the interpolated precision curve
    for i in output:
        i["classname"] = classname.get(i["class"], None)

    return output


def addBB(
    bboxes : BoundingBoxes, 
    annList : AnnotationList,
    bbType : BBType
):
    for ann in annList.annotationList:
        
        classIntLabel, classTextLabel, classScore = ann.classIntLabel, ann.classTextLabel, ann.classScore
        if isinstance(classIntLabel, np.ndarray):
            classIntLabel = classIntLabel[-1]
        if isinstance(classScore, np.ndarray):
            classScore = classScore[-1]
        if isinstance(classTextLabel, np.ndarray):
            classTextLabel = classTextLabel[-1]
        
        
        
        bboxes.addBoundingBox(
            BoundingBox(
                imageName=str(ann.imageID),
                classId=classIntLabel,
                classConfidence=classScore,
                x=ann.boundingBox[0],
                y=ann.boundingBox[1],
                w=ann.boundingBox[2],
                h=ann.boundingBox[3],
                typeCoordinates=CoordinatesType.Absolute,
                bbType=bbType,
                format=BBFormat.XYX2Y2,
                imgSize=None
            )
    )


class EvalStat():
    
    def __init__(self) -> None:
        self.detBbox = None
        self.detClass = None
        self.gtBbox = None
        self.gtClass = None
        self.bboxes = BoundingBoxes()

        # Key metrics stat
        # Adversarial accuracy
        self.AA = 0 
        # Average confidence of adversarial class
        self.ACAC = 0 
        # Average confidence of true class
        self.ACTC = 0
        # Noise tolerance estimation 
        self.NTE = 0 
        # True Positive Rate
        self.TPR = 0 

        # auxiliary
        self.frameCount = 0
        # Frame num that contain ground truth, total positive
        self.gtFrameCount = 0 
        # Count the number of frames where stop sign BB is missing (either not proposed, or mis-located), false negative
        self.missBbCount = 0 
        # Count the number of frames where stop sign BB is correctly located (satisfy IoU), but may or may not be correctly classified
        self.successfulBbCount = 0
        # Count the number of frames where the class is correctly classified 
        self.successfulDetCount = 0 # 
        # Count the number of frames where stop sign BB is correctly located but wrongly classified 
        self.successfulAttackCount = 0 
        # Assert self.MissedBbCount + Self.SuccessfulBbCount = Total # of gt frames
        # Keep track of the adversarial accuracy of those successful attacks
        self.successfulAttackAdvAcc = [] 
        # Keep track of the ground truth accuracy of those successful attacks
        self.successfulAttackGtAcc = [] 
        # Keep track of the tolerance between adversarial class and the max of other class
        # s.t., len(Self.SuccessfulAttackAdvAcc) == len(self.SuccessfulAttackGtAcc) == self.SccessfulAttackCount == len(self.Tolerance)
        self.tolerance = [] 
                            
    
    def updateStat(
        self, 
        iFrame,
        detAnnList,
        gtAnnList,
        iouThreshold = 0.5
    ):
        if iFrame > self.frameCount:
            self.frameCount += 1



        addBB(self.bboxes, detAnnList, BBType.Detected)
        addBB(self.bboxes, gtAnnList, BBType.GroundTruth)
        
        # No ground truth in the frame
        if not gtAnnList.annotationList: 
            return

        self.gtFrameCount += 1
        # Check whether detAnnList contains at least one BB that has IOU with GT >= threshold
        # detAnnList is sorted by detection confidence in descending order
        # In case of multiple detections, only count the detection of highest confidence
        detAnnList = rejectOverlappingBbox(detAnnList)
        iou_matched = False
        for det in detAnnList.annotationList:
            y1_det, x1_det, y2_det, x2_det = det.boundingBox
            
            for gt in gtAnnList.annotationList:
                y1_gt, x1_gt, y2_gt, x2_gt = gt.boundingBox   
                iou =  bbox_iou(
                    [x1_det, y1_det, x2_det, y2_det],
                    [x1_gt, y1_gt, x2_gt, y2_gt],
                    x1y1x2y2 = True
                )
                # IOU(gt, det) > threshold == correct location of det bbox
                if iou > iouThreshold:
                    iou_matched = True
                    self.successfulBbCount += 1

                    # Further check if classification is incorrect
                    # incorrect == successful adversarial attack
                    if (det.classIntLabel == gt.classIntLabel):
                        self.successfulDetCount += 1
                    else:
                        self.successfulAttackCount += 1
                        if det.classScoresArray is not None:
                            scores = det.classScoresArray
                            advScore, gtScore = scores[det.classIntLabel], scores[gt.classIntLabel]
                            self.successfulAttackAdvAcc.append(advScore)
                            self.successfulAttackGtAcc.append(gtScore)
                            scores_sorted = np.sort(scores)
                            self.tolerance.append(np.abs(scores_sorted[-1] - scores_sorted[-2]))

        if iou_matched == False:
            self.missBbCount += 1

        assert (self.missBbCount + self.successfulBbCount) == self.gtFrameCount
        assert (self.successfulDetCount + self.successfulAttackCount) == self.successfulBbCount
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.AA = np.nan_to_num(
                np.divide(self.successfulAttackCount+self.missBbCount, self.gtFrameCount)
            )
            self.ACAC = np.nan_to_num(np.mean(self.successfulAttackAdvAcc))
            self.ACTC = np.nan_to_num(np.mean(self.successfulAttackGtAcc))
            self.NTE = np.nan_to_num(np.mean(self.tolerance))
            self.TPR = np.nan_to_num(np.divide(self.successfulDetCount, self.gtFrameCount))
        
            



    def computeMetrics(self, iouThreshold = 0.5):
        evaluator = Evaluator()
      
        output = evaluator.GetPascalVOCMetrics(
            self.bboxes,
            IOUThreshold=iouThreshold,
            method=MethodAveragePrecision.EveryPointInterpolation
        )
        return output
