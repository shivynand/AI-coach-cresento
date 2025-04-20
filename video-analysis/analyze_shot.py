import cv2
import mediapipe as mp
import numpy as np

mp.pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp.pose.Pose()

video_path = "new_ideal_shot.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file '{video_path}'")
    exit()

frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def detect_ball_position(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=50, param2=30,minRadius=15,maxRadius=50)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # Filter circles in bottom 50% of frame, since this is where football resides
        valid_circles = [c for c in circles if c[1] > frame_height * 0.5]
        if valid_circles:
            x,y,r = valid_circles[0]
            return np.array([x,y], dtype=np.float32), (x,y,r)
    return None, None

# Find contact frame (i.e. when ball is about to be kicked)
min_distance = float('inf')
contact_frame = 0
frame_data = []

for n in range(total_frames):
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    ball_pos, _ = detect_ball_position(frame_bgr)
    if results.pose_landmarks and ball_pos is not None:
        landmarks = results.pose_landmarks.landmark
        foot = np.array([landmarks[mp.pose.PoseLandmark.RIGHT_FOOT_INDEX].x * frame_width, landmarks[mp.pose.PoseLandmark.RIGHT_FOOT_INDEX].y * frame_height])
        distance = np.linalg.norm(foot - ball_pos)
        frame_data.append((n, distance))
        if distance < min_distance:
            min_distance = distance
            contact_frame = n

cap.release()

if not frame_data:
    print("Error: No valid frames with ball and keypoints detected")
    pose.close()
    exit()

# Find metrics for contact frame
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, contact_frame)
ret,frame = cap.read()
if not ret:
    print("Error: Could not read contact frame")
    cap.release()
    pose.close()
    exit()

frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = pose.process(frame_rgb)
frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

ball_pos, ball_draw = detect_ball_position(frame_bgr)
if ball_pos is not None and results.pose_landmarks:
    landmarks = results.pose_landmarks.landmark

    # Knee angle for right leg
    hip = np.array([landmarks[mp.pose.PoseLandmark.RIGHT_HIP].x * frame_width, landmarks[mp.pose.PoseLandmark.RIGHT_HIP].y * frame_height])
    knee = np.array([landmarks[mp.pose.PoseLandmark.RIGHT_KNEE].x * frame_width, landmarks[mp.pose.PoseLandmark.RIGHT_KNEE].y * frame_height])
    ankle = np.array([landmarks[mp.pose.PoseLandmark.RIGHT_ANKLE].x * frame_width, landmarks[mp.pose.PoseLandmark.RIGHT_ANKLE].y * frame_height])

    v1 = hip - knee
    v2 = ankle - knee
    cos_angle = np.dot(v1,v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    knee_angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    # Plant foot distance (left foot)
    plant_foot = np.array([landmarks[mp.pose.PoseLandmark.LEFT_ANKLE].x * frame_width, landmarks[mp.pose.PoseLandmark.LEFT_ANKLE].y * frame_height])
    distance_pixels = np.linalg.norm(plant_foot - ball_pos)
    distance_cm = distance_pixels * 180 / (frame_height * 0.8)

    # Print and save metrics
    print(f"Contact frame: {contact_frame}, Knee angle: {knee_angle:.2f}º, Plant foot distance: {distance_cm:.2f}cm")
    with open('ideal_metrics.txt', 'w') as f:
        f.write(f"{contact_frame}, {distance_cm}, {knee_angle}\n")

    # Visualise contact frame
    if ball_draw:
        x,y,r = ball_draw
        cv2.circle(frame_bgr, (x,y), r, (0,255,0), 2)
        cv2.circle(frame_bgr, (x,y), r, (0,0,255), -1)

    mp_drawing.draw_landmarks(frame_bgr, results.pose_landmarks, mp.pose.POSE_CONNECTIONS)
    cv2.putText(frame_bgr, f"Contact frame: {contact_frame}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Contact frame', frame_bgr)
    cv2.waitKey(0)
    cv2.imwrite('contact_frame.jpeg', frame_bgr)
else:
    print("Error: No ball or keypoints in frame")

cap.release()
cap.destroyAllWindows()
pose.close()
