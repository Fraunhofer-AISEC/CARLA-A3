from details.analytic import compMet
from details.simconfig import SimConfig
from details.SimLogic import SimLogic
from details.SimWeather import PresetWeather, Weather
from details.faster_rcnn.wrapper import FasterRCNN
import numpy as np
import traceback

"""
Run simulation on Carla comparing multiple environment weather settings 
and observing effectiveness of adversarial patches applied over a stop sign. 

"""

if __name__ == "__main__":
    
    # Uncomment to select weather conditions to run the experiment

    weather = [ 
        Weather(cloudiness=30),
        PresetWeather.ClearNoon,
        # PresetWeather.ClearSunset,
        
        PresetWeather.CloudyNoon,        
        # PresetWeather.CloudySunset,
        # PresetWeather.CloudyNight,
        
        PresetWeather.HardRainyNoon,
        # # PresetWeather.HardRainySunset,        
        # PresetWeather.HardRainyNight,
                
        # PresetWeather.MidRainyNoon,
        # PresetWeather.MidRainySunset,
        # PresetWeather.MidRainyNight,
        
        PresetWeather.SoftRainyNoon,
        # PresetWeather.SoftRainySunset,
        # PresetWeather.SoftRainyNight,

        PresetWeather.WetCloudyNoon,
        # PresetWeather.WetCloudySunset,
        # PresetWeather.WetCloudyNight,

        PresetWeather.WetNoon,
        PresetWeather.WetSunset,
        PresetWeather.WetNight,        
    ]
    
    for i, w in enumerate(weather, start=1):
        simConfig = SimConfig()
        simConfig.numSimFrame = 60
        simConfig.weather = w 
        simConfig.dashCamResolutionX = 800
        simConfig.dashCamResolutionY = 800
        simConfig.objectDetector = FasterRCNN(
            inputWidth=simConfig.dashCamResolutionX,
            inputHeight=simConfig.dashCamResolutionY
        )       
        sim = SimLogic(simConfig)
        try:
            sim.run(patchImg="data/Chen_Patch_1.png", metLogDir="metlog")
            sim.saveSim(destDir="simlog", name=f"sim{i}_adv_weather{i}")
            sim.cleanUp()
        except:
            print(traceback.format_exc())
            sim.cleanUp()    
            sim.setSynchronous(False)
            exit(0) 
    # To exit Carla without causing a crash
    try:
        simlist = [f"simlog/sim{i}_adv_weather{i}" for i, _ in enumerate(weather, start=1)]
        compMet(simlist, outputDir="complog")
        sim.setSynchronous(False)
        exit(0)
    except:
        print(traceback.format_exc())
        sim.setSynchronous(False)
        exit(0)
