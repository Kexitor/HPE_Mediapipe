# Human pose estimation and classification using Mediapipe

This programm classifies poses (walk, fall, fallen, sitting) using Mediapipe for human pose estimation. This programm prototype can only classify 1 person in frame due to Mediapipe limitations. With 37 training videos and 11 test videos it showed about 88% of accuracy on classyfing fall, fallen and walking poses and about 32 FPS on RTX 3060 12GB and i5 12400F.

Example of work: 

https://github.com/Kexitor/HPE_Mediapipe/assets/55799671/876018a0-cce9-4e44-9aba-5ce3fed332a5

https://www.youtube.com/watch?v=yeEJ6y-gU10

https://www.youtube.com/watch?v=uedp3CnXWmM

## Used videos for training and testing:

https://www.youtube.com/watch?v=8Rhimam6FgQ

http://fenix.ur.edu.pl/mkepski/ds/uf.html

## Used lib versions:

Python updated to version 3.10.11

Required libs can be installed:

```
pip install -r requirements.txt
```

<!--Python==3.7.8

matplotlib==3.5.3

mediapipe-0.9.0.1

numpy==1.21.6

opencv-python==4.7.0.72

pandas==1.3.5

scikit-learn==1.0.2

scipy==1.7.3-->


## How to use:

`pose_estimator_training_data_generator.py` used to make data for training. Also by this file is used `data_lists.py` and numerous of videos to generate CSV file data. All params for generating CSV are in code.


Example of usage:
```
python pose_estimator_training_data_generator.py
```

`pose_estimator_usage_example.py` is used to classify pose of person on video. This file uses CSV generated by previos command. In current version file `pm_37vtrain_mp.pkl` already contains training data.

Example of usage:
```
python pose_estimator_usage_example.py
```

If you want to train your own model look through `pose_estimator_train_example.py`.

Example of usage:
```
python pose_estimator_train_example.py
```

In order to make you experience with my code better (provided some docstrings), you can modify my classes in following files:

* `MediapipePoseEstimator.py`
* `MediapipeUtilities.py`
* `PoseClassifierTrain.py`
* `TrainingDataGenerator.py`

You can check my other prototypes: 

https://github.com/Kexitor/HPE_YOLOv7

https://github.com/Kexitor/HPE_Torchvision


