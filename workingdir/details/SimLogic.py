import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import List

import carla
import numpy as np
from matplotlib import pyplot
from prettytable import PrettyTable

from details.annotation import AnnotationList
from details.camgeo import CameraGeometry
from details.CV2Manager import CV2Manager
from details.injector import Injector
from details.metric import EvalStat
from details.SersorImage import Image
from details.simconfig import SimConfig

import matplotlib; matplotlib.use("TkAgg")


class SimLogic:

    def __init__(self, config:SimConfig) -> None:
        self.isInitialized = False
        self.simConfig = config

    def init(self):
        # set fixed config
        if not self.isInitialized:
            self.lastTimeStamp = 0
            self.startTime = datetime.today().strftime('%Y%m%d%H%M%S')
            
            self.dashCamMotionBlur = 0
            self.dashCamMotionMaxDistortion = 0
            self.dashCamPostProcess = True
            self.dashCamLocation = (1.8, 0, 1.2)    # (x, y, z)
            self.spectatorLocation = (-8, 0, 5)     # (x, y, z)
            
            self.odAnnotationGoundTruth = AnnotationList()
            self.odAnnotation = AnnotationList()            
            self.evalStat = EvalStat()
            self.statTable = None
            self.metrics = {}

            self.CV2Manager = CV2Manager()
            self.cameraImage = Image()
            self.cameraImageID = 0
            self.saveCameraImageFlag = True


            self.client = carla.Client("localhost", 2000)
            self.client.set_timeout(300.0)
            self.client.load_world("Town02")
            self.world = self.client.get_world()
            self.spectator = self.world.get_spectator()
            self.setSynchronous(True)
            self.setUserConfig()
            self.isInitialized = True
    
    def setUserConfig(self):
        # Set user customizable config
        self.dashCamResolutionX = self.simConfig.dashCamResolutionX
        self.dashCamResolutionY = self.simConfig.dashCamResolutionY
        self.dashCamFOV = self.simConfig.dashCamFOV
        self.playerCarSpeed = self.simConfig.playerCarSpeed
        self.objectDetector = self.simConfig.objectDetector
        self.numSimFrames = self.simConfig.numSimFrame
        self.metLogDir = self.simConfig.metLogDir
        self.cameraImageOutputDir = self.simConfig.cameraImageOutputDir
        self.gtLabelIndex = self.simConfig.gtLabelIndex
        self.gtClassName = self.simConfig.gtClassName
        self.camera_od_ImageOutputDir = self.simConfig.camera_od_ImageOutputDir
        self.cameraImageOutputDir_w_bb = self.simConfig.cameraImageOutputDir_w_bb
        self.weather = self.simConfig.weather.toCarlaWeather()
        self.patch_indx = self.simConfig.patch_indx
        # Replace with your stop sign coordinates
        # Use this when converting coordinates between the simulator and Python script: Python_API_coord = Carla_coord * 0.01
        self.spawnPointTransform = carla.Transform(
            carla.Location(x=-7.53, y=283.65 - self.simConfig.distanceToStopSign, z=0.5),  
            carla.Rotation(pitch=0, roll=0, yaw=90)                                        
        )
        self.cameraGeometry = CameraGeometry(
            self.dashCamResolutionX,
            self.dashCamResolutionY,
            self.dashCamFOV
        )
        Path(self.metLogDir).mkdir(parents=True, exist_ok=True)
        Path(self.cameraImageOutputDir).mkdir(parents=True, exist_ok=True)


    def setSynchronous(
        self,
        boolFlag : bool = True,
        fixedDeltaSeconds : float = 0.05
    ):
        # Set synchronization
        self.worldSettings = self.world.get_settings()
        self.worldSettings.synchronous_mode = boolFlag
        self.worldSettings.fixed_delta_seconds = fixedDeltaSeconds
        self.world.apply_settings(self.worldSettings)
        self.trafficManager = self.client.get_trafficmanager()
        self.trafficManager.set_synchronous_mode(boolFlag)

    def spawnPlayerCar(self):
        carBP = self.world.get_blueprint_library().filter('vehicle').find('vehicle.mercedes.coupe_2020')
        self.playerCar = self.world.spawn_actor(carBP, self.spawnPointTransform)
        self.playerCar.set_autopilot(True)
        self.playerCar.set_light_state(carla.VehicleLightState.All)
        self.setCarConstantSpeed()
        


    def spawnSensors(self):
        camBP = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camBP.set_attribute('image_size_x', str(self.dashCamResolutionX))
        camBP.set_attribute('image_size_y', str(self.dashCamResolutionY))
        camBP.set_attribute('fov', str(self.dashCamFOV))
        camBP.set_attribute('motion_blur_intensity', str(self.dashCamMotionBlur))
        camBP.set_attribute('motion_blur_max_distortion', str(self.dashCamMotionMaxDistortion))
        camBP.set_attribute('enable_postprocess_effects', str(self.dashCamPostProcess))


        self.dashCamTransform = carla.Transform(carla.Location(*self.dashCamLocation))
        self.dashCam = self.world.spawn_actor(camBP, self.dashCamTransform, attach_to=self.playerCar)
        self.dashCam.listen(self.cameraImage.updateImage)

        self.dummySpectatorTransform = carla.Transform(carla.Location(*self.spectatorLocation))
        self.dummySpectator = self.world.spawn_actor(camBP, self.dummySpectatorTransform, attach_to=self.playerCar)


    def spawnNpcCar(self):
        transform = carla.Transform(self.spawnPointTransform.location, self.spawnPointTransform.rotation)
        transform.location.y += 8.0
        bp = self.world.get_blueprint_library().filter('vehicle').find('vehicle.ford.ambulance')
        npcCar = self.world.try_spawn_actor(bp, transform)
        if npcCar is not None:
            npcCar.set_autopilot(True)

    def spectatorFollowPlayerCar(self):
        spectator_transform =  self.dummySpectator.get_transform()
        self.spectator.set_transform(spectator_transform)


    
    def dashCamObjectDetection(self):
        if self.cameraImage.image is None:
            return

        img = self.cameraImage.toNumpy()
        ## Frame drop check
        delta = self.cameraImage.image.timestamp - self.lastTimeStamp
        self.lastTimeStamp = self.cameraImage.image.timestamp
       
        boundingboxes, classes, scores, classname, scores_array = self.objectDetector.detect(img)
        self.CV2Manager.showDetection("Live Detection", img, boundingboxes, classes, scores, classname,self.cameraImageID,self.camera_od_ImageOutputDir,self.saveCameraImageFlag,self.patch_indx)
        
        self.updateDetection(boundingboxes, classes, scores, classname, scores_array)

    def saveCameraImage(self, img):
        if self.saveCameraImageFlag is False:
            return

        pyplot.imsave(
            os.path.join(self.cameraImageOutputDir, F"{self.cameraImageID:06d}.png"),
            img
        )

    def showGroundTruthBoundingBox(self):
        if self.cameraImage.image is None:
            return

        boundingbox, classes, scores, classname = [], [], [], []

        img = self.cameraImage.toNumpy()
        self.saveCameraImage(img)
        # Note: do not use get_actors() api. It will return the collision bounding box which is not what we want      
        stopSigns = self.world.get_environment_objects() 
        # Note: Change to sm_1 (or sm_0 if currently sm_1) if the bounding box is drawn against the pole
        stopSignsBB = [s.bounding_box for s in stopSigns if "advstopsign" in s.name.lower() and "sm_0" in s.name.lower()] 
        # Use Carla BBS API
        stopSignsBoundingBox = self.world.get_level_bbs(carla.CityObjectLabel.TrafficSigns) 

        for i, bbox in enumerate(stopSignsBB):
            (x_min, y_min), (x_max, y_max) = self.cameraGeometry.bounding_box_to_pixel_coordinate(bbox, self.dashCam, 50)
            if x_min is None:
                continue

            
            boundingbox.append([y_min, x_min, y_max, x_max])
            classes.append(self.gtLabelIndex)
            scores.append(1.0)
            classname.append(self.gtClassName)


        CV2Manager.showDetection(
            "Ground Truth", img,
            np.array(boundingbox),
            np.array(classes),
            np.array(scores),
            np.array(classname),
            self.cameraImageID,
            self.cameraImageOutputDir_w_bb,
            True,
            self.patch_indx
        )
        self.updateGroundTruth(boundingbox, classes, scores, classname)
        self.saveGtBbox(boundingbox,classes, classname)


    def saveGtBbox(self,
        boundingBox : List[List[int]], 
        classIdx : List[int], 
        className : List[str]):

        filename = os.path.join(self.cameraImageOutputDir, F"{self.cameraImageID:06d}.txt")
        with open(filename, mode="w+") as fout:
            for bbox, cidx, cname in zip(boundingBox, classIdx, className):
                
                # New bbox coordinate <class index> <x_center> <y_center> <width> <height>
                fout.write("%d %f %f %f %f\n" % (cidx, 
                                                 (bbox[1]+bbox[3]) / 2.0 /self.dashCamResolutionX, # X
                                                 (bbox[0]+bbox[2]) / 2.0 /self.dashCamResolutionY, # Y
                                                 abs(bbox[3] - bbox[1]) / self.dashCamResolutionX, # rel. X = width
                                                 abs(bbox[2] - bbox[0]) / self.dashCamResolutionY, # rel. Y = height                                         
                                                )
                          )
    
    def updateDetection(self, boundingbox, classes, scores, classname, scores_array):
        self.odAnnotation = AnnotationList()
        self.odAnnotation.fromArray(
            self.cameraImageID, False, boundingbox, classes, scores, classname, scores_array
        )

    def updateGroundTruth(self, boundingbox, classes, scores, classname):
        self.odAnnotationGoundTruth = AnnotationList()
        self.odAnnotationGoundTruth.fromArray(
            self.cameraImageID, True, boundingbox, classes, scores, classname, None
        )

    def updateEvalStat(self):
        self.evalStat.updateStat(
            self.cameraImageID,
            self.odAnnotation,
            self.odAnnotationGoundTruth,
        )
        
        self.metrics["frame"] = self.metrics.get("frame", []) + [self.evalStat.frameCount]
        self.metrics["gtFrame"] = self.metrics.get("gtFrame", []) + [self.evalStat.gtFrameCount]
        self.metrics["sucBB"] = self.metrics.get("sucBB", []) + [self.evalStat.successfulBbCount]
        self.metrics["sucDet"] = self.metrics.get("sucDet", []) + [self.evalStat.successfulDetCount]
        self.metrics["sucAtt"] = self.metrics.get("sucAtt", []) + [self.evalStat.successfulAttackCount]
        self.metrics["missBB"] = self.metrics.get("missBB", []) + [self.evalStat.missBbCount]
        self.metrics["AA"] = self.metrics.get("AA", []) + [self.evalStat.AA]
        self.metrics["ACAC"] = self.metrics.get("ACAC", []) + [self.evalStat.ACAC]
        self.metrics["ACTC"] = self.metrics.get("ACTC", []) + [self.evalStat.ACTC]
        self.metrics["NTE"] = self.metrics.get("NTE", []) + [self.evalStat.NTE]
        self.metrics["TPR"] = self.metrics.get("TPR", []) + [self.evalStat.TPR]
        

    def printEvalStat(self):
        if self.statTable  is None:
            
            self.statTable = PrettyTable()
        


        data = [[
            self.evalStat.frameCount,
            self.evalStat.gtFrameCount,
            self.evalStat.successfulBbCount,
            self.evalStat.successfulDetCount,
            self.evalStat.successfulAttackCount,
            self.evalStat.missBbCount,
            
            self.evalStat.TPR,
            self.evalStat.AA,
            self.evalStat.ACAC,
            self.evalStat.ACTC,
            self.evalStat.NTE,            
        ]]

        data = [[f'{x:1.2f}' for row in data for x in row]]
        columns = ('Frame#', 'GT Frame#', "Suc BB#", "Suc Det#",  "Suc Att#", "Miss BB#",  "TPR", 
                   'AA', 'ACAC', 'ACTC', 'NTE')

        self.statTable.field_names = columns
        self.statTable.add_rows(data)
        print(self.statTable)


    def computeMetric(self):
        out = self.evalStat.computeMetrics()
        for i in out:
            if i["class"] == self.gtLabelIndex:
                self.metrics["AP"] = i["AP"]
        pass

    def showMetric(self):
        pyplot.figure("Metrics")
        pyplot.clf()
        ncol, nrow = 1, 2

        def helper(keys, xlim, ylim, percentage = False):
            for key in keys:
                x = np.array(self.metrics.get("frame", []))
                y = np.array(self.metrics.get(key, []))
                y = y * 100 if percentage else y
                pyplot.plot(x, y)
                pyplot.xlim(xlim)
                pyplot.ylim(ylim)


        keys = ["gtFrame", "sucBB", "sucDet", "sucAtt", "missBB"]
        ax = pyplot.subplot(nrow, ncol, 1)
        helper(keys, [-1, self.numSimFrames + 5], [-0.05, self.numSimFrames + 5])
        pyplot.ylabel("Cumulative occurrence")
        ax.legend(keys)

        keys = ["TPR", "AA", "ACAC", "ACTC", "NTE"]
        ax = pyplot.subplot(nrow, ncol, 2)
        helper(keys, [-0.05, self.numSimFrames + 5], [-5, 105], True)
        pyplot.xlabel("Frame")
        pyplot.ylabel("Percentage")
        ax.legend(keys)
        
        
        pyplot.suptitle(F"Image Frame {self.cameraImageID}")
        pyplot.show(block=False)
        pyplot.pause(0.001)

    def logMetrics(self, metLogDir = None):
        if metLogDir is None:
            metLogDir = self.logDir
        path = Path().joinpath(metLogDir) \
                     .joinpath(datetime.today().strftime("%Y%m%d"))        
        path.mkdir(parents=True, exist_ok=True)
        abpath = path.joinpath(f"{self.startTime}.html").as_posix()
        with open(abpath, "w+") as file:
            file.write(
                self.statTable.get_html_string(attributes={"class":"table"}, format=True)
            )

    def saveSim(
        self, 
        destDir : str = None, 
        name : str = None
    ):
        if destDir is None:
            destDir = self.metLogDir
        if name is None:
            name = self.startTime
        path = Path(destDir)  
        path.mkdir(parents=True, exist_ok=True)
        abpath = path.joinpath(f"{name}").as_posix()
        with open(abpath, "wb+") as file:
            pickle.dump(self.metrics, file)

    def printCarSpeed(self):
        print(
            self.playerCar.get_velocity()
        )

    def setCarConstantSpeed(self):
        
        self.playerCar.enable_constant_velocity(carla.Vector3D(*self.playerCarSpeed))

    def setWeather(self):
        self.world.set_weather(self.weather)

    def tick(self):
        self.cameraImageID += 1
        self.world.tick()

    def eventLoop(self):
        
        while self.cameraImageID < self.numSimFrames:

            self.spectatorFollowPlayerCar()
            self.tick()
            self.dashCamObjectDetection()
            self.showGroundTruthBoundingBox()
            self.updateEvalStat()
            self.printEvalStat()
            self.showMetric()
        

    def cleanUp(self):
        self.dashCam.stop()
        try:
            self.dashCam.destroy()
            self.playerCar.destroy()
        except:
            pass

    def injectPatch(self, patchImg : np.array):
        Injector.inject(self.client, patchImg)

    def run(
        self,
        patchImg : np.array = None,
        metLogDir : str = None
    ):
        self.init()
        self.setWeather()
        self.spawnPlayerCar()
        self.spawnSensors()
        if patchImg is not None:
            self.injectPatch(patchImg)
        self.eventLoop()
        self.computeMetric()
        self.logMetrics(metLogDir)
        pass

if __name__ == "__main__":

    sim = SimLogic()    
    sim.run()
    
