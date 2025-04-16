from typing import List
from matplotlib.pyplot import cla
import numpy as np

class Annotation:

    def __init__(
        self,
        imageID,
        boundingBox : List[int],
        classIntLabel : int,
        classTextLabel : str,
        classScore : float,
        isGroundTruth : bool,
        classScoresArray : np.ndarray or None,

    ) -> None:
        self.imageID = imageID
        self.boundingBox = boundingBox
        self.classIntLabel = classIntLabel
        self.classTextLabel = classTextLabel
        self.classScore = classScore
        self.isGroundTruth = isGroundTruth
        self.classScoresArray = classScoresArray
        



class AnnotationList:

    def __init__(self) -> None:
        self.annotationList : List[Annotation]= []

    def fromArray(
        self,
        imageID,
        isGroundTruth,
        boundingBox : List[List[float]],
        classIntLabel : List[int],
        classScore : List[float],
        classTextLabel : List[str] = None,
        classScoresArray : np.ndarray = None,
    ) -> "AnnotationList":
        
        N1, N2, N3 = len(boundingBox), len(classIntLabel), len(classScore)
        assert( (N1==N2) and (N2==N3) )
        
        if classTextLabel is None:
            classTextLabel = np.full(N1, None)
        if classScoresArray is None:
            classScoresArray = np.full(N1, None)

        for bbox, clabel, ctlabel, cscore, scorearr  in zip(boundingBox, classIntLabel, classTextLabel, classScore, classScoresArray):
            self.addImageAnnotation(
                imageID, bbox, clabel, ctlabel, cscore, isGroundTruth, scorearr
            )


    def getAnnotationFromImage(self, imageID) -> "AnnotationList":
        pass

    def extendAnnotationList(self, other : "AnnotationList"):
        self.annotationList.extend(other)
    
    def addImageAnnotation(
        self,
        imageID,
        boundingBox : List[float or int],
        classIntLabel : int,
        classTextLabel : str or None, 
        classScore : float,
        isGroundTruth : bool,
        classScoreArray : np.ndarray
    ) -> None:
        self.annotationList.append(
            Annotation(
                imageID, boundingBox, classIntLabel, classTextLabel, 
                classScore, isGroundTruth, classScoreArray
            )
        )


if __name__ == "__main__":
    al = AnnotationList()
    al.extendAnnotationList(["abc", "def"])

    print(al)
