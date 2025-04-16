from .yolo.wrapper import Yolo
from .faster_rcnn.wrapper import FasterRCNN
from .SimWeather import Weather

class SimConfig():
    def __init__(self,resX=800,resY=800,patch_indx=0) -> None:
        self.numSimFrame = 50
        self.map = "Town02"
        self.dashCamResolutionX = 600
        self.dashCamResolutionY = 600
        self.dashCamFOV = 90
        self.playerCarSpeed = (10, 0, 0)        # (x, y, z)
        self._od = FasterRCNN(
                cudaFlag=False,
                inputHeight=self.dashCamResolutionX,
                inputWidth=self.dashCamResolutionY                
            )
        self.distanceToStopSign = 25
        self.weather = Weather()
        self.cameraImageOutputDir = "camoutput"
        self.camera_od_ImageOutputDir = "cam_od_output"
        self.cameraImageOutputDir_w_bb = "cam_gt_bb"
        self.metLogDir = "metlog"
        self.gtLabelIndex = 12 if isinstance(self.objectDetector, FasterRCNN) else 11
        self.gtClassName = "stop sign"
        self.patch_indx = patch_indx

    @property
    def objectDetector(self):
        return self._od

    @objectDetector.setter
    def objectDetector(self, value):
        self._od = value
        self.gtLabelIndex = 11 if isinstance(self.objectDetector, Yolo) else 12




