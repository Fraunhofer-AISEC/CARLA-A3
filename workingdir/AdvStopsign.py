from details.simconfig import SimConfig
from details.SimLogic import SimLogic
from details.faster_rcnn.wrapper import FasterRCNN
import argparse, os

""" 
Run simulation on Carla with an adversarial patch. 
You can pass the path using -p 

Run using an adversarial patch from the repo
- "python3 AdvStopsign.py"

Run using your own adversarial patch
- "python3 AdvStopsign.py -p /path_to_adversarial_patch_image/"

"""

def check_args(path):
    if path is None:
        # Default
        path="data/Chen_Patch_1.png"
        return path
    
    elif path is not None:
        # Path passed
        if os.path.isfile(path) and path.lower().endswith(('.png', '.jpg')):
            return path
        else:
            print("Invalid path")
    


if __name__ == "__main__":
    try:

        parser = argparse.ArgumentParser(description="Process some input arguments.")
        parser.add_argument('-p', '--path', type=str, help='The path to the adversarial patches', required=False)
        
        args = parser.parse_args()
        
        patch_img = check_args(args.path)

        simConfig = SimConfig()
        simConfig.dashCamResolutionX, simConfig.dashCamResolutionY = 800, 800
        simConfig.objectDetector = FasterRCNN(
            inputWidth=simConfig.dashCamResolutionX,
            inputHeight=simConfig.dashCamResolutionY
        )
        simConfig.numSimFrame = 60
        sim = SimLogic(simConfig)
        sim.run(patchImg= patch_img, metLogDir="metlog")
        sim.saveSim(destDir="simlog", name=f"adv_stop_sign1")
        
        # To exit Carla without causing a crash
        sim.cleanUp()    
        sim.setSynchronous(False) 
        exit(0)
    except:
        sim.cleanUp()    
        sim.setSynchronous(False)
        exit(0)

