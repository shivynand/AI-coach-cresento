import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8)

def detect_ball_position(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=50, param2=30, minRadius=15, maxRadius=100)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        valid_circles = [c for c in circles if c[1] > frame_height * 0.5]
        if valid_circles:
            x, y, r = valid_circles[0]
            return np.array([x, y], dtype=np.float32), (x, y, r)
    return None, None

# Load video
video_paths = [("better_ideal_shot.mp4", "ideal"), ("testing_shot.mp4", "test")]

# Open file to save metrics for all frames
with open('metrics_all_frames.txt', 'w') as f:
    # Write header
    f.write("shot_type,frame,hip_x,hip_y,knee_x,knee_y,ankle_x,ankle_y,left_hip_x,left_hip_y,left_knee_x,left_knee_y,plant_foot_x,plant_foot_y,ball_pos_x,ball_pos_y,knee_angle,plant_foot_distance\n")

    for video_path, shot_type in video_paths:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            continue

        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        with open('metrics_all_frames.txt', 'a') as f:
            collect_metrics = True
            contact_frame = None
            last_knee_angle = 0.0
            last_distance_cm = 0.0
            previous_ball_pos = None
            min_distance = float('inf')

            # Process each frame
            for n in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                ball_pos, ball_draw = detect_ball_position(frame_bgr)
                if ball_pos is not None:
                    previous_ball_pos = ball_pos
        
                if ball_pos is not None and results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark

                    # Check keypoint visibility
                    required_landmarks = [
                        mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE,
                        mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.LEFT_HIP,
                        mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE,
                        mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
                    ]
                    if all(landmarks[i].visibility > 0.7 for i in required_landmarks):
                        # Right leg keypoints (kicking leg)
                        hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * frame_width,
                                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * frame_height])
                        knee = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x * frame_width,
                                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y * frame_height])
                        ankle = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x * frame_width,
                                        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y * frame_height])
                        print(f"Frame {n}, Hip: {hip}, Knee: {knee}, Ankle: {ankle}")

                        # Validate right leg keypoints (y: ankle > knee > hip)
                        if not (ankle[1] > knee[1] > hip[1]):
                            print(f"Frame {n}, Warning: Right leg keypoints misaligned, attempting correction")
                            knee = (hip + ankle) / 2
                            knee[1] = (hip[1] + ankle[1]) / 2

                        # Calculate knee angle
                        v1 = hip - knee
                        v2 = ankle - knee
                        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                        knee_angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
                        knee_angle = min(knee_angle, 360 - knee_angle)

                        # Left leg keypoints (plant foot)
                        left_hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * frame_width,
                                            landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * frame_height])
                        left_knee = np.array([landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x * frame_width,
                                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y * frame_height])
                        plant_foot = np.array([landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x * frame_width,
                                            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * frame_height])
                        print(f"Frame {n}, Left Hip: {left_hip}, Left Knee: {left_knee}, Plant foot: {plant_foot}, Ball pos: {ball_pos}")

                        # Validate left leg keypoints (y: ankle > knee > hip)
                        if not (plant_foot[1] > left_knee[1] > left_hip[1]):
                            print(f"Frame {n}, Warning: Left leg keypoints misaligned, attempting correction")
                            left_knee = (left_hip + plant_foot) / 2
                            left_knee[1] = (left_hip[1] + plant_foot[1]) / 2
                            if abs(plant_foot[1] - ball_pos[1]) > 100:
                                plant_foot[1] = ball_pos[1] - 50

                        # Calculate plant foot distance
                        distance_pixels = np.linalg.norm(plant_foot - ball_pos)
                        ball_radius_pixels = ball_draw[2] if ball_draw is not None else 25
                        print(f"Frame {n}, Ball radius: {ball_radius_pixels}px")
                        pixels_per_cm = ball_radius_pixels / 11  # Ball diameter 22 cm
                        distance_cm = distance_pixels / pixels_per_cm

                        # Cross-check with player height
                        player_height_pixels = abs(left_hip[1] - plant_foot[1])
                        distance_cm_alt = distance_pixels * 180 / player_height_pixels  # 1.8m player
                        print(f"Frame {n}, Alternative distance (player height): {distance_cm_alt:.2f}cm")

                        # Detect contact frame
                        foot = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x * frame_width, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * frame_height])
                        foot_to_ball_distance = np.linalg.norm(foot - ball_pos)
                        if foot_to_ball_distance < 50 and collect_metrics:
                            print(f"Contact frame detected at frame {n}, stopping metric collection")
                            contact_frame = n
                            collect_metrics = False

                        # Collect and save metrics if before or at contact
                        if collect_metrics:
                            print(f"Frame {n}, Knee angle: {knee_angle:.2f}°, Plant foot distance: {distance_cm:.2f}cm")
                            f.write(f"{n},{hip[0]},{hip[1]},{knee[0]},{knee[1]},{ankle[0]},{ankle[1]},"
                                    f"{left_hip[0]},{left_hip[1]},{left_knee[0]},{left_knee[1]},{plant_foot[0]},{plant_foot[1]},"
                                    f"{ball_pos[0]},{ball_pos[1]},{knee_angle},{distance_cm}\n")
                            last_knee_angle = knee_angle
                            last_distance_cm = distance_cm

                        # Visualize frame with annotations
                        if ball_draw:
                            x, y, r = ball_draw
                            cv2.circle(frame_bgr, (x, y), r, (0, 255, 0), 2)
                            cv2.circle(frame_bgr, (x, y), 5, (0, 0, 255), -1)

                    mp_drawing.draw_landmarks(frame_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    cv2.putText(frame_bgr, f"[{shot_type}] Frame: {n}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if collect_metrics:
                        cv2.putText(frame_bgr, f"Knee angle: {knee_angle:.2f}°", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame_bgr, f"Plant foot dist: {distance_cm:.2f}cm", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame_bgr, f"Knee angle: {last_knee_angle:.2f}° (Stopped)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame_bgr, f"Plant foot dist: {last_distance_cm:.2f}cm (Stopped)", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('Frame Analysis', frame_bgr)
                    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                            break
                else:
                    print(f"[{shot_type}] Frame {n}, Error: Keypoints not visible")
            else:
                print(f"[{shot_type}] Frame {n}, Error: No ball or keypoints detected")

    cap.release()
    cv2.destroyAllWindows()

pose.close()