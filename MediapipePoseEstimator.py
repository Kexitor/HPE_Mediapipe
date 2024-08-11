import cv2
import mediapipe as mp
import time
from MediapipeUtilities import MediapipeUtilities


class MediapipePoseEstimator:
    def __init__(self):
        self.pretrained_model = None
        self.video_capture = None
        self.video_width = None
        self.video_height = None
        self.__mp_drawing = None
        self.__mp_drawing_styles = None
        self.__mp_pose = None
        self.__output_writer = None

    def init_video(self, file_path: str):
        """
        Initialize video for estimation

        :param file_path: path to video which is to be pose-estimated
        """
        # vid_path = "videos/50wtf.mp4"  # "videos/50wtf.mp4" "videos/pw3_.mp4"
        self.video_capture = cv2.VideoCapture(file_path)
        self.video_width = self.video_capture.get(3)
        self.video_height = self.video_capture.get(4)

    def init_mediapipe(self, model):
        """
        Initialize model from sklearn for pose-estimation

        :param model: example sklearn.neural_network._multilayer_perceptron.MLPClassifier(required_model_config).fit(required_train_data)
        """
        self.__mp_drawing = mp.solutions.drawing_utils
        self.__mp_drawing_styles = mp.solutions.drawing_styles
        self.__mp_pose = mp.solutions.pose
        self.pretrained_model = model

    def __init_output_video_writer(self, output_save_path: str):
        """
        Initializes video writer

        :param output_save_path: path to generated mp4 file
        :return: cv2.VideoWriter
        """
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        output_writer = cv2.VideoWriter(output_save_path + ".mp4", fourcc, 30.0, (int(self.video_capture.get(3)), int(self.video_capture.get(4))))

        return output_writer

    def estimate_pose(self, output_save_path=None):
        """
        Shows video with realtime pose estimation working models (mediapipe for pose estimation + sklearn for pose classification)

        :param output_save_path: path to generated mp4 file
        """
        fps_time = 0
        frame_n = 0
        pose_label = "none"

        if output_save_path:
            self.__output_writer = self.__init_output_video_writer(output_save_path)

        with self.__mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as pose:
            while self.video_capture.isOpened():
                success, image = self.video_capture.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    break
                frame_n += 1

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)

                # Classifying pose for identified human
                try:
                    key_points = MediapipeUtilities().get_keypoints(results.pose_landmarks, self.video_width, self.video_height)
                    coordinates_line = MediapipeUtilities().get_coords_line(key_points)
                    if 67 >= len(coordinates_line) >= 1:
                        pose_code = self.pretrained_model.predict([coordinates_line])
                        pose_label = MediapipeUtilities().get_pose_from_num(pose_code)
                    cv2.putText(image,
                                "pose: %s" % pose_label,
                                (int(key_points[0][0]), int(key_points[0][1]) - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), 2)
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
                cv2.putText(image,
                            "NN: %s" % pose_label,
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)
                # mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
                if output_save_path:
                    self.__output_writer.write(image)

                cv2.imshow('MediaPipe Pose', image)
                fps_time = time.time()
                if cv2.waitKey(5) & 0xFF == 27:
                    break

        self.video_capture.release()
        if output_save_path:
            self.__output_writer.release()
