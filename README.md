# Human pose estimation and classification using Mediapipe

This programm classifies poses (walk, fall, fallen, sitting) using Mediapipe for human pose estimation. This programm prototype can only classify 1 person in frame due to Mediapipe limitations. With 37 training videos and 11 test videos it showed about 88% of accuracy on classyfing fall, fallen and walking poses. and about 32 FPS on RTX 3060 12GB and i5 12400F.

Example of work: 


https://github.com/Kexitor/HPE_Mediapipe/assets/55799671/876018a0-cce9-4e44-9aba-5ce3fed332a5


## Used videos for training and testing:

https://www.youtube.com/watch?v=8Rhimam6FgQ

http://fenix.ur.edu.pl/mkepski/ds/uf.html

## Used lib versions:

keras==2.10.0

Keras-Preprocessing==1.1.2

matplotlib==3.5.3

matplotlib-inline==0.1.6

numba==0.56.4

numpy==1.21.6

opencv-python==4.6.0.66

pandas==1.3.5

scikit-learn==1.0.2

scipy==1.7.3

tensorflow==2.10.0

tensorflow-estimator==2.10.0

tensorflow-gpu==2.10.0

tensorpack==0.11

torch==1.13.1+cu117

torchaudio==0.13.1+cu116

## How to use:

`data_generator.py` used to make data for training. Also by this file is used `data_lists.py` and numerous of videos to generate CSV file data. All params for generating CSV are in code.


Example of usage:
```
python data_generator.py
```
`pose_estimator_mp.py` is used to classify pose of person on video. This file uses CSV generated by previos command. In current version file `pm_37vtrain_mp.pkl` already contains training data. If you want train model by yourself uncomment lines 95-96.

Example of usage:
```
pose_estimator_mp.py
```




