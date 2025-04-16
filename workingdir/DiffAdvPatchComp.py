from pathlib import Path
from details.SimWeather import Weather
import traceback
from details.analytic import compMet
from details.faster_rcnn.wrapper import FasterRCNN
from details.simconfig import SimConfig
from details.SimLogic import SimLogic
import argparse, os



""" 
Run simulation on Carla comparing multiple adversarial patches applied over a stop sign. 

You can pass the path using -p or a directory with all the images

Run using an adversarial patch from the repo
- "python3 AdvStopsign.py"

Run using your own adversarial patch
- "python3 AdvStopsign.py -p /path_to_adversarial_patch_image/"

"""

def check_args(path, dir):
    
    if path is None and dir is None:
        # Default
        dir="./data"
        files = [os.path.relpath(os.path.join(dir, f), start=".") for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
        return files
    elif path is None and dir is not None:
        # Directory passed 
        if os.path.isdir(dir):
            files = [os.path.relpath(os.path.join(dir, f), start=".") for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f)) and f.lower().endswith(('.png', '.jpg')) ]
            
            return files
        else:
            print("Invalid Directory")
    elif path is not None and dir is None:
        # Path passed
        if os.path.isfile(path) and path.lower().endswith(('.png', '.jpg')):
            return [path]
        else:
            print("Invalid path")
    elif path is not None and dir is not None:
        print("Please pass either the path or the directory and not both")



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Process some input arguments.")
    parser.add_argument('-p', '--path', type=str, help='The path to an adversarial patch image file', required=False)
    parser.add_argument('-d', '--directory', type=str, help='The path to the directory with all the patches', required=False)
    
    
    args = parser.parse_args()
    
    advpatches = check_args(args.path, args.directory)

    for i, patch in enumerate(advpatches, start=1):
        simConfig = SimConfig(patch_indx=i)
        simConfig.numSimFrame=50
        simConfig.dashCamResolutionX = 800
        simConfig.dashCamResolutionY = 800
        simConfig.objectDetector = FasterRCNN(
            inputWidth=simConfig.dashCamResolutionX,
            inputHeight=simConfig.dashCamResolutionY
        )       
        sim = SimLogic(simConfig)
        # To exit Carla without causing a crash
        try: 
            sim.run(patchImg=patch,metLogDir="metlog")
            sim.saveSim(destDir="simlog", name=f"sim_patch{i}")
            sim.cleanUp()
        except:
            print(traceback.format_exc())
            sim.cleanUp()    
            sim.setSynchronous(False)
            exit(0)       
    try: 
        simlist = [f"simlog/sim_patch{i}" for i, _ in enumerate(advpatches, start=1)]
        compMet(simlist, outputDir="complog")
        sim.setSynchronous(False)
        exit(0)
    except:
        print(traceback.format_exc())
        sim.setSynchronous(False)
