# Adversarial YOLO
This repository is based on the marvis YOLOv2 inplementation: https://github.com/marvis/pytorch-yolo2



# What you need
We use Python 3.6.
Make sure that you have a working implementation of PyTorch installed, to do this see: https://pytorch.org/
To visualise progress we use tensorboardX which can be installed using pip:
```
pip install tensorboardX tensorboard
```
No installation is necessary, you can simply run the python code straight from this directory.

Make sure you have the YOLOv2 MS COCO weights:
```
mkdir weights; curl https://pjreddie.com/media/files/yolov2.weights -o weights/yolo.weights
```


# Generating a patch with single input

You can generate this patch by running:
```
Attack_Stopsign.ipynb
```

# Generate a patch with a batch of input

You can generate this patch by running:
```
Attack_Stopsign_Batch_1.ipynb
```

Before that, you should put the images into one folder and the corresponding labels into another one. Then, replace the path in the corresponding place on the notebook.