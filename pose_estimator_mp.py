import cv2
import mediapipe as mp
import numpy as np
import time
import pandas as pd
import re
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import pickle


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# 37 tr + 11 te = 71%
# 30 tr + 11 te = 52%


def csv_converter(path, fname):
    df = pd.read_csv(path + fname, sep=";")
    df_ = pd.DataFrame(data=df)
    # string = df_.bp1[10]
    # numbers = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", string)

    all_coordinates = []
    all_poses = []
    headers_names = df_.columns.values
    # print(len(df_))
    for col_num in range(len(df_)):
        # print(df_[headers_names[col_num]].values)
        # print(df_.loc[col_num].values)
        all_poses.append([df_.loc[col_num].values[2]])
        all_coordinates.append([])
        for i in range(3, len(df_.loc[col_num].values)):
            nums = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", df_.loc[col_num].values[i])
            for num in nums:
                all_coordinates[col_num].append(float(num))

    return all_poses, all_coordinates


def pose_to_num(poses_):
    poses_list = ["walk", "fall", "fallen", "sitting"]
    all_poses_num = []
    for pose in poses_:
        if pose[0] == "walk":
            all_poses_num.append(["0"])
        if pose[0] == "fall":
            all_poses_num.append(["1"])
        if pose[0] == "fallen":
            all_poses_num.append(["2"])
        if pose[0] == "sitting":
            all_poses_num.append(["3"])

    return all_poses_num


def get_pose_from_num(pose_number):
    if pose_number[0] == "0":
        return "walk"
    if pose_number[0] == "1":
        return "fall"
    if pose_number[0] == "2":
        return "fallen"
    if pose_number[0] == "3":
        return "sitting"
    else:
        return "code_error"

def get_coords_line(kps):
    coords_line = []
    for kp in kps:
        coords_line.append(kp[0])
        coords_line.append(kp[1])
    return coords_line


def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        curr_frequency = List.count(i)
        if (curr_frequency >= counter):
            counter = curr_frequency
            num = i
    if counter == 1:
        return List[-1]
    return num


def get_keypoints(landmarks, w, h):
    kps = []
    kps2 = []
    # print(landmarks[mp_pose.PoseLandmark.NOSE])
    # print("ok")
    for i in range(len(landmarks.landmark)):
        kps.append((landmarks.landmark[i].x * w, landmarks.landmark[i].y * h))
    return kps


path = ""  # "videos/csv_files/"
filename = "37vtrain_mp.csv"# "37vid_data_train.csv" "37vid_data_train.csv"
train_poses, train_coords = csv_converter(path, filename)
train_poses_num = pose_to_num(train_poses)
# NN = MLPClassifier(solver='lbfgs', activation='logistic', alpha=1e-5, hidden_layer_sizes=(150, 10), random_state=1,
#                    max_iter=10000).fit(train_coords, train_poses_num)
#
# pkl_filename = "pm_37vtrain_mp.pkl"
# with open(pkl_filename, 'wb') as file:
#     pickle.dump(NN, file)
NN = ""
with open("pm_37vtrain_mp.pkl", 'rb') as file:
    NN = pickle.load(file)

# For webcam input:
vid_path = "videos/50wtf.mp4"  # "videos/50wtf.mp4" "videos/pw3_.mp4"
cap = cv2.VideoCapture(vid_path)
# fourcc = cv2.VideoWriter_fourcc(*'MP4V')
# out = cv2.VideoWriter("50wtf_mp.mp4", fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

width = cap.get(3)
height = cap.get(4)

frame_count = 0  # to count total frames
total_fps = 0  # to get the final frames per second
fps_time = 0
frame_n = 0
fps = cap.get(cv2.CAP_PROP_FPS)
vid_fps = cap.get(cv2.CAP_PROP_FPS)
pose_label = "none"
keypoints = []

with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break
        frame_number = frame_n / vid_fps
        frame_n += 1

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        coords_line = []
        try:
            # print("###################")
            keypoints = get_keypoints(results.pose_landmarks, width, height)
            coords_line = get_coords_line(keypoints)
            # print(coords_line)
            # print(results.pose_landmarks)
            if 67 >= len(coords_line) >= 1:
                pose_code = NN.predict([coords_line])
                pose_label = get_pose_from_num(pose_code)
            cv2.putText(image,
                        "pose: %s" % (pose_label),
                        (int(keypoints[0][0]), int(keypoints[0][1]) - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
        except:
            pass
        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        # Flip the image horizontally for a selfie-view display.
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.putText(image,
                    "NN: %s" % (pose_label),
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        # mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        # out.write(image)
        cv2.imshow('MediaPipe Pose', image)
        fps_time = time.time()
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
# out.release()

