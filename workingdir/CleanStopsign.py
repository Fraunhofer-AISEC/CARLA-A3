from details.simconfig import SimConfig
from details.SimLogic import SimLogic
from details.faster_rcnn.wrapper import FasterRCNN
import traceback

"""
Run simulation with a clean stop sign
"""
# Replace the Adversarial stop sign with the default "BP_StopSign" before running this script
if __name__ == "__main__":

    
    simConfig = SimConfig()
    simConfig.dashCamResolutionX, simConfig.dashCamResolutionY = 600, 600
    simConfig.objectDetector = FasterRCNN(
        inputWidth=simConfig.dashCamResolutionX,
        inputHeight=simConfig.dashCamResolutionY
    )
    simConfig.numSimFrame = 60
    sim = SimLogic(simConfig)
    try:
        sim.run(metLogDir="metlog")
        sim.saveSim(destDir="simlog", name=f"clean_stop_sign1")
        sim.cleanUp()    
        sim.setSynchronous(False)
    except:
        print(traceback.format_exc())
        sim.cleanUp()   
        sim.setSynchronous(False)