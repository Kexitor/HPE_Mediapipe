from MediapipePoseEstimator import MediapipePoseEstimator
import pickle


path_to_pretrained_model = "pretrained_models/pm_37vtrain_mp.pkl"
recorded_video_save_path = "generated_videos/test1"
estimated_video_file_path = "videos/50wtf.mp4"

with open(path_to_pretrained_model, 'rb') as file:
    pretrained_model = pickle.load(file)

video_pose_classifier = MediapipePoseEstimator()
video_pose_classifier.init_video(estimated_video_file_path)
video_pose_classifier.init_mediapipe(pretrained_model)
video_pose_classifier.estimate_pose()  # recorded_video_save_path could be passed as argument
