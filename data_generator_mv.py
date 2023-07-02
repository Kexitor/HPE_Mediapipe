import cv2
import mediapipe as mp
import time
import csv
from data_lists import train_data, test_data
import sklearn

global total_frames_count
total_frames_count = 0



def keypoints_parser(kps, dt_line):
    human = kps
    for points in human:
        dt_line.append((round(points[0], 2), round(points[1], 2)))
    return dt_line


def get_keypoints(landmarks, w, h):
    kps = []
    for i in range(len(landmarks.landmark)):
        kps.append((landmarks.landmark[i].x * w, landmarks.landmark[i].y * h))
    return kps


def pose_estimation_video(data_path, markup, frame_sum):
    # cap = cv2.VideoCapture(filename)
    # VideoWriter for saving the video
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # out = cv2.VideoWriter("fname.mp4", fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))
    global total_frames_count
    vid_counter = 18
    csvname = "30videos_data.csv" # '7vid_data_test.csv' # str(vid_counter) + 'vid_data_test.csv' #  '10vid_data_train.csv'
    with open(csvname, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';')
        spamwriter.writerow(['time', "vname", 'pose', 'bp0', 'bp1', 'bp2', 'bp3', 'bp4', 'bp5', 'bp6', 'bp7', 'bp8',
                             'bp9', 'bp10', 'bp11', 'bp12', 'bp13', 'bp14', 'bp15', 'bp16', 'bp17', 'bp18', 'bp19',
                             'bp20', 'bp21', 'bp22', 'bp23', 'bp24', 'bp25', 'bp26', 'bp27', 'bp28', 'bp29', 'bp30',
                             'bp31', 'bp32'])
    for video_n in range(vid_counter):
        vid_path = data_path + markup[video_n][3]
        print(vid_path)
        strange_falls = ["50wtf9.mp4", "50wtf12.mp4", "50wtf16.mp4",
                         "50wtf28.mp4", "50wtf31.mp4", "50wtf47.mp4", "50wtf49.mp4"]
        if markup[video_n][3] in strange_falls:
            continue
        cap = cv2.VideoCapture(vid_path)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_sum += length
        frame_count = 0  # to count total frames
        total_fps = 0  # to get the final frames per second
        fps_time = 0
        frame_n = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        vid_fps = cap.get(cv2.CAP_PROP_FPS)
        pose_label = "none"
        width = cap.get(3)
        height = cap.get(4)

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
                data_line = []
                total_frames_count += 1
                data_line.append(round(frame_number, 2))
                data_line.append(vid_path)

                time_1 = markup[video_n][1]
                time_2 = markup[video_n][2]
                init_pose = markup[video_n][0]  # "sitting" # "walk"
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
                try:
                    print("###################")
                    keypoints = get_keypoints(results.pose_landmarks, width, height)
                    data_line = keypoints_parser(keypoints, data_line)
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

                # mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
                cv2.imshow('MediaPipe Pose', image)
                fps_time = time.time()
                if 67 > len(data_line) > 3:
                    # data_line[2] = "none"
                    with open(csvname, 'a', newline='') as csvfile:
                        spamwriter = csv.writer(csvfile, delimiter=';')
                        spamwriter.writerow(data_line)
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
    return frame_sum


frames_count = 0
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
data_path_ = "videos/cuts_test/"
frames_count = pose_estimation_video(data_path_, test_data, frames_count)

print(total_frames_count, " ", frames_count)
