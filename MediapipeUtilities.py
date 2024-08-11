import cv2
import mediapipe as mp
import time
import pandas as pd
import re
import pickle
import sklearn


class MediapipeUtilities:
    def csv_converter(self, path: str, file_name: str):
        """
        Converts data from csv for training format

        :param path: file directory/folder without file name
        :param file_name: file name
        :return: lists of poses and list of coordinates
        """
        df = pd.read_csv(path + file_name, sep=";")
        df_ = pd.DataFrame(data=df)
        all_coordinates = []
        all_poses = []
        for col_num in range(len(df_)):
            all_poses.append([df_.loc[col_num].values[2]])
            all_coordinates.append([])
            for i in range(3, len(df_.loc[col_num].values)):
                nums = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", df_.loc[col_num].values[i])
                for num in nums:
                    all_coordinates[col_num].append(float(num))

        return all_poses, all_coordinates

    def pose_to_num(self, poses_: list[str]):
        """
        Converts poses to numerous identifier

        :param poses_: list of poses (markup)
        :return: list of poses converted to numerous identifier
        """
        # poses_list = ["walk", "fall", "fallen", "sitting"]
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

    def get_pose_from_num(self, pose_number):
        """
        Converts numerous identifier to pose

        :param pose_number: pose identifier
        :return: pose text label
        """
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

    def get_coords_line(self, kps):
        coords_line = []
        for kp in kps:
            coords_line.append(kp[0])
            coords_line.append(kp[1])
        return coords_line

    def get_keypoints(self, landmarks, w, h):
        kps = []
        kps2 = []
        for i in range(len(landmarks.landmark)):
            kps.append((landmarks.landmark[i].x * w, landmarks.landmark[i].y * h))
        return kps

    def keypoints_parser(self, kps, dt_line):
        human = kps
        for points in human:
            dt_line.append((round(points[0], 2), round(points[1], 2)))
        return dt_line
