from details.analytic import compMet
from details.simconfig import SimConfig
from details.SimLogic import SimLogic
from details.faster_rcnn.wrapper import FasterRCNN
import traceback

"""
Run simulation on Carla comparing multiple dashcam resolution settings 
and observing effectiveness of adversarial patches applied over a stop sign. 

"""

if __name__ == "__main__":
    # Edit the values of resolution here
    res = [(400, 400), (600, 600), (800, 800), (1000, 1000), (1200, 1200)]
    
    for i, (resX, resY) in enumerate(res, start=1):
            simConfig = SimConfig()
            simConfig.numSimFrame = 60
            simConfig.dashCamResolutionX = resX
            simConfig.dashCamResolutionY = resY
            simConfig.objectDetector = FasterRCNN(inputHeight=resY, inputWidth=resX)
            sim = SimLogic(simConfig)
            try:
                sim.run(patchImg="data/Chen_Patch_1.png", metLogDir="metlog")
                sim.saveSim(destDir="simlog", name=f"sim{i}_{resX}x{resY}")
                sim.cleanUp()
            except:
                print(traceback.format_exc())
                sim.cleanUp()    
                sim.setSynchronous(False)
                exit(0) 
    # To exit Carla without causing a crash
    try:
        simlist = [f"simlog/sim{i}_{resX}x{resY}" for i, (resX, resY) in enumerate(res, start=1)]
        compMet(simlist, outputDir="complog")
        sim.setSynchronous(False)
        exit(0)
    except:
        print(traceback.format_exc())
        sim.setSynchronous(False)
        exit(0)
