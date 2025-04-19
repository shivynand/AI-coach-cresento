import cv2
import mediapipe as mp
import numpy as np

mp.pose = mp.solutions.pose
mp.drawing = mp.solutions.drawing_utils
pose = mp.pose.Pose()

video_path = "ideal_shot.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file '{video_path}'")
    exit()

frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

ball_pos = np.array([0.5,0.9])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Knee angle (right knee)
        knee = np.array([landmarks[mp.pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp.pose.PoseLandmark.RIGHT_KNEE].y])
        hip = np.array([landmarks[mp.pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp.pose.PoseLandmark.RIGHT_HIP].y])
        ankle = np.array([landmarks[mp.pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp.pose.PoseLandmark.RIGHT_ANKLE].y])

        v1 = hip - knee
        v2 = ankle - knee
        knee_angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        # Plant foot distance
        plant_foot = np.array([landmarks[mp.pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp.pose.PoseLandmark.LEFT_ANKLE].y])
        distance_pixels = np.linalg.norm(plant_foot - ball_pos)
        distance_cm = distance_pixels * 180 / (frame_height * 0.8) * 100

        print(f"Knee angle: {knee_angle:.2f}º, Plant foot distance: {distance_cm:.2f}cm")
        mp.drawing.draw_landmarks(frame, results.pose_landmarks, mp.pose.POSE_CONNECTIONS)

        cv2.imshow('Shot analysis', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
cap.release()
cv2.destroyAllWindows()
pose.close()
