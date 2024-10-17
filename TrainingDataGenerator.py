import cv2
import mediapipe as mp
import time
import csv
from data_lists import train_data, test_data
import sklearn
from MediapipeUtilities import MediapipeUtilities


class TrainingDataGenerator:
    def __init__(self):
        self.pretrained_model = None
        self.__mp_pose = None
        self.__mp_drawing_styles = None
        self.__mp_drawing = None
        self.csv_save_path = None
        self.video_data_path = None
        self.markup = None

    def __init_mediapipe(self):
        """
        Initialize model from sklearn for pose-estimation
        """
        self.__mp_drawing = mp.solutions.drawing_utils
        self.__mp_drawing_styles = mp.solutions.drawing_styles
        self.__mp_pose = mp.solutions.pose

    def init_main_params(self, markup, csv_save_path: str, video_data_path: str):
        """
        Initializes data for training

        :param markup: look for more information in file data_lists.py
        :param csv_save_path: path to csv generated csv
        :param video_data_path: directory/folder with train/test videos
        :return: generated csv save path with extension ".csv"
        """
        self.csv_save_path = csv_save_path + ".csv"
        self.video_data_path = video_data_path
        self.markup = markup
        print(f"CSV full save path: {self.csv_save_path}")

        return self.csv_save_path

    def __init_csv_file(self):
        """
        Initializes csv file
        """
        with open(self.csv_save_path, 'w', newline='') as csvfile:
            # For each point of interest (POI) I take relative X and Y coordinates and convert them to absolute
            spam_writer = csv.writer(csvfile, delimiter=';')
            spam_writer.writerow(
                ['time', "vname", 'pose', 'bp0', 'bp1', 'bp2', 'bp3', 'bp4', 'bp5', 'bp6', 'bp7', 'bp8',
                 'bp9', 'bp10', 'bp11', 'bp12', 'bp13', 'bp14', 'bp15', 'bp16', 'bp17', 'bp18', 'bp19',
                 'bp20', 'bp21', 'bp22', 'bp23', 'bp24', 'bp25', 'bp26', 'bp27', 'bp28', 'bp29', 'bp30',
                 'bp31', 'bp32'])

    def generate_train_csv(self, vid_counter=18):
        """
        Generates coordinates for each video in folder

        :param vid_counter: number of videos, which will be taken from folder (depends on you data)
        """
        # Vid_counter can be changed differing on type of data (test up to 18 or train up to 30)
        # you are using from data_lists
        self.__init_mediapipe()
        self.__init_csv_file()
        total_frames_count = 0
        frame_sum = 0

        # Cycle for each chosen video
        for video_n in range(vid_counter):
            vid_path = self.video_data_path + self.markup[video_n][3]
            print(vid_path)

            # Strange_falls - falls, which are hard to estimate in markup
            # Following 4 string can be un/commented
            strange_falls = ["50wtf9.mp4", "50wtf12.mp4", "50wtf16.mp4",
                             "50wtf28.mp4", "50wtf31.mp4", "50wtf47.mp4", "50wtf49.mp4"]
            if self.markup[video_n][3] in strange_falls:
                continue

            cap = cv2.VideoCapture(vid_path)
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_sum += length
            fps_time = 0
            frame_n = 0
            vid_fps = cap.get(cv2.CAP_PROP_FPS)
            video_width = cap.get(3)
            video_height = cap.get(4)

            # Cycle for each frame of video
            with self.__mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                while cap.isOpened():
                    success, image = cap.read()
                    if not success:
                        print("Ignoring empty camera frame.")
                        # If loading a video, use 'break' instead of 'continue'.
                        break
                    frame_number = frame_n / vid_fps
                    frame_n += 1
                    data_line = []
                    total_frames_count += 1
                    data_line.append(round(frame_number, 2))
                    data_line.append(vid_path)

                    time_1 = self.markup[video_n][1]
                    time_2 = self.markup[video_n][2]
                    init_pose = self.markup[video_n][0]
                    pose_label = "none"
                    if frame_number < time_1:
                        pose_label = init_pose
                    if time_1 <= frame_number < time_2:
                        pose_label = "fall"
                    if frame_number >= time_2:
                        pose_label = "fallen"
                    data_line.append(pose_label)

                    print(data_line)
                    print(frame_n)
                    print(vid_fps)
                    print(pose_label)
                    print("#####")

                    # To improve performance, optionally mark the image as not writeable to
                    # pass by reference.
                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = pose.process(image)

                    # Getting human coordinates
                    try:
                        print("###################")
                        key_points = MediapipeUtilities().get_keypoints(results.pose_landmarks, video_width, video_height)
                        data_line = MediapipeUtilities().keypoints_parser(key_points, data_line)
                    except:
                        pass
                    # Draw the pose annotation on the image.
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    self.__mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        self.__mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.__mp_drawing_styles.get_default_pose_landmarks_style())

                    cv2.putText(image,
                                "FPS: %f" % (1.0 / (time.time() - fps_time)),
                                (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 2)

                    cv2.imshow('MediaPipe Pose', image)
                    fps_time = time.time()

                    # Writing human coordinates, 67 is count of 33 POI which MP can get multiplied by 2 coordinates
                    if 67 > len(data_line) > 3:
                        # data_line[2] = "none"
                        with open(self.csv_save_path, 'a', newline='') as csvfile:
                            spam_writer = csv.writer(csvfile, delimiter=';')
                            spam_writer.writerow(data_line)
                    else:
                        print("#########################################################")
                        print("skip")
                        print("skip")
                        print("skip")
                        print("skip")
                        print("#########################################################")

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            cap.release()
