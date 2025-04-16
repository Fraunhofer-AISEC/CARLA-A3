from details.attack.wrapper import Attacker
import os, argparse

"""
Generate Adversarial image patches
"""
def check_args(path):
    if path is None:
        # Default
        path="advpatch/advbase0.jpg"
        return path
    
    elif path is not None:
        # Path passed
        if os.path.isfile(path) and path.lower().endswith(('.png', '.jpg')):
            return path
        else:
            print("Invalid path")
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process some input arguments.")
    parser.add_argument('-p', '--path', type=str, help='The path to an adversarial patch base image file', required=False)
    
    args = parser.parse_args()
    
    advbase = check_args(args.path)

    att = Attacker()
    att.generateAttackPatch(
        initialPatch=advbase,
        numEpoch=200,
        outputDir="advpatch",
        outputName="advpatch0.png"
    )
