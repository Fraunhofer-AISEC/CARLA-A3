Adversarial attack results
==================================

Here we document the results from running different adversarial attack scenarios with Carla. We compare the results from the original experiment published [here](https://www.youtube.com/watch?v=7C4aAekBbiE) with the results we have obtained. We use the following metric to measure the effectiveness of the attack:

| Metric | Definition                                                                                   |
| ------ | -------------------------------------------------------------------------------------------- |
| Adversarial Accuracy (AA)     |  Counts successful misclassification and successful bounding box miss. |
| Average Confidence of Adversarial Class (ACAC)   |  Average confidence score from the object detection (OD) model when an object is misclassified as an adversarial class.                                                     |
| Average Confidence of True Class (ACTC)   |   Average confidence score from the object detection (OD) model for the true class .                    |
| Noise Tolerance Estimation (NTE)  |  Mean of difference in confidence between the adversarial class and the next most confident class. How confident is the model about the current adversarial class wrt all other classes                                                                  |
| True positive classification rate (TPR)    | True positive classification rate of OD model                                                |

### Experiment Conditions

We run our experiments on Carla 0.9.15, the default weather conditions can be seen [here](https://gitlab.cc-asp.fraunhofer.de/harsha.ramesh/adv_ss_attack/-/blame/main/workingdir/details/SimWeather.py?ref_type=heads#L4) along with the config details [here](https://gitlab.cc-asp.fraunhofer.de/harsha.ramesh/adv_ss_attack/-/blame/main/workingdir/details/simconfig.py?ref_type=heads#L6).

### Comparison of different attack patches

We compare the results when running 3 adversarial stop sign patches from [this](https://github.com/shangtse/robust-physical-attack) github repo with the dashcam resolution set to 800x800: 

![](media/image14.png)

### Carla 0.9.15

|     Patch#   |   TPR        |    AA  |   ACAC |   ACTC |   NTE  |
| ------------ | ------------ | ------ | ------ | ------ | ------ |
|  chen_patch1 | 0.6111       | 0.3889 | 0.7879 | 0.0105 | 0.7699 |
| chen_patch2  | 0.6852       | 0.3148 | 0.8008 | 0.0027 | 0.7959 |
| chen_patch3  | 0.7593       | 0.2407 | 0      | 0      | 0      |

### Original Experiment (Carla 0.9.13)

|     Patch#  | TPR    | AA     | ACAC   | ACTC   | NTE    |
| ----------- | ------------------- | ------ | ------ | ------ | ------ |
| chen_patch1 | 0.6667              | 0.3333 | 0.8313 | 0.0016 | 0.7797 |
| chen_patch2 | 0.8519              | 0.1481 | 0.8562 | 0.0012 | 0.8538 |
| chen_patch3 | 0.7963              | 0.2037 | 0      | 0      | 0      |


### Comparison of attack effectiveness under different weather conditions

Both the experiments were run while applying [Chen_patch_1](workingdir/data/Chen_Patch_1.png) as the adversarial patch with the dashcam resolution set to 800x800. We compare the results between our experiment and the original experiment () below:

### Carla 0.9.15 

| Weather    |   TPR                                  |    AA  |   ACAC |   ACTC |   NTE  |
| ---------- | -------------------------------------- | ------ | ------ | ------ | ------ |
| Clear      | 0.7593                                 | 0.2407 | 0.6697 | 0.0363 | 0.6247 |
| Cloudy     | 0.7407                                 | 0.2593 | 0.5914 | 0.0222 | 0.5509 |
| Hard rain  | 0.7593                                 | 0.2407 | 0.7596 | 0.0065 | 0.7358 |
| Soft rain  | 0.7593                                 | 0.2407 | 0.6477 | 0.0131 | 0.6174 |
| Wet cloudy | 0.7037                                 | 0.2963 | 0.6604 | 0.0148 | 0.629  |
| Wet clear  | 0.6981                                 | 0.3019 | 0.6524 | 0.0263 | 0.6202 |


### Original Experiment (Carla 0.9.13)
| Weather    | TPR                           | AA     | ACAC   | ACTC   | NTE    |
| ---------- | ----------------------------- | ------ | ------ | ------ | ------ |
| Clear      | 0.6842                        | 0.3158 | 0.8905 | 0.0011 | 0.8573 |
| Cloudy     | 0.7407                        | 0.2593 | 0.7494 | 0.0092 | 0.7291 |
| Hard rain  | 0.8889                        | 0.1111 | 0      | 0      | 0      |
| Soft rain  | 0.7222                        | 0.2778 | 0.8946 | 0.0006 | 0.8763 |
| Wet cloudy | 0.7407                        | 0.2593 | 0.7459 | 0.0099 | 0.7259 |
| Wet clear  | 0.7037                        | 0.2963 | 0.7698 | 0.0039 | 0.743  |



### Observations

Based on the above results we can see that there are some variations in the results between the 2 experiments for the same scenario.
These variations can be because of various factors such as:
1. Angle of Sunlight.
2. Weather.
3. Starting point of the vehicle.
4. Carla Material properties.
5. Variations in the Intersection over Union (IoU) between the bounding boxes applied by the object detector model and the ground truth.
6. Variations in the classification confidence of the object detector.