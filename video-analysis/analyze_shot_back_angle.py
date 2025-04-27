import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from collections import deque

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

def detect_elements(frame, previous_ball_pos=None, previous_goal_box=None, frame_idx=0):
    # Detect player, ball, and goal using YOLOv8
    results = model(frame, verbose=False)
    player_box = None
    ball_box = None
    goal_box = previous_goal_box
    goal_confidence = 0.0

    # Process YOLOv8 detections
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf)
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])

            # Player detection
            if cls == 0 and conf >= 0.3:  # Class 0 is person
                player_box = (x_min, y_min, x_max, y_max)

            # Ball detection
            elif cls == 32 and conf >= 0.3:  # Class 32 is sports ball
                ball_box = (x_min, y_min, x_max, y_max)

            # Goal detection: Look for a large rectangular structure in the upper half
            # YOLOv8 doesn't have a "goal" class, so we'll look for a generic object that fits the goal's characteristics
            if frame_idx < 50 or goal_box is None:  # Only detect goal in first 50 frames or if not found
                if conf >= 0.4:  # Higher confidence threshold for goal
                    # Check if the object is in the upper half of the frame
                    if y_max < frame.shape[0] * 0.5:
                        # Check if the object has a goal-like shape (wider than tall)
                        width = x_max - x_min
                        height = y_max - y_min
                        aspect_ratio = width / height if height > 0 else 0
                        if 1.5 < aspect_ratio < 4.0 and width > frame.shape[1] * 0.3:  # Goal-like dimensions
                            if goal_box is None or conf > goal_confidence:
                                goal_box = (x_min, y_min, x_max, y_max)
                                goal_confidence = conf

    # Predict ball position if YOLO fails
    if ball_box is None and previous_ball_pos is not None:
        search_radius = 100
        x, y = previous_ball_pos
        roi_x_min = max(0, int(x - search_radius))
        roi_y_min = max(0, int(y - search_radius))
        roi_x_max = min(frame.shape[1], int(x + search_radius))
        roi_y_max = min(frame.shape[0], int(y + search_radius))
        roi = frame[roi_y_min:roi_y_max, roi_x_min:roi_x_max]
        if roi.size == 0:
            return player_box, ball_box, goal_box
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                   param1=50, param2=10, minRadius=5, maxRadius=30)
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (cx, cy, r) in circles:
                dist = np.linalg.norm(np.array([cx + roi_x_min, cy + roi_y_min]) - previous_ball_pos)
                if dist < search_radius:
                    ball_box = (cx + roi_x_min - r, cy + roi_y_min - r,
                                cx + roi_x_min + r, cy + roi_y_min + r)
                    break

    # Fallback: If goal not detected, assume it's in the upper center of the frame
    if goal_box is None and frame_idx >= 50:
        goal_width = int(frame.shape[1] * 0.5)
        goal_height = int(frame.shape[0] * 0.3)
        x_min = (frame.shape[1] - goal_width) // 2
        x_max = x_min + goal_width
        y_min = 0
        y_max = goal_height
        goal_box = (x_min, y_min, x_max, y_max)

    return player_box, ball_box, goal_box

def calculate_metrics_at_contact(frame, frame_width, frame_height, player_box, ball_box, goal_box):
    if player_box is None or ball_box is None or goal_box is None:
        return None, None, None, None

    # Get pose landmarks
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    if not results.pose_landmarks:
        return None, None, None, None
    landmarks = results.pose_landmarks.landmark

    # Identify kicking leg and plant foot
    right_knee = np.array([
        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x * frame_width,
        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y * frame_height
    ])
    left_knee = np.array([
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x * frame_width,
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y * frame_height
    ])
    right_ankle = np.array([
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x * frame_width,
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y * frame_height
    ])
    left_ankle = np.array([
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x * frame_width,
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * frame_height
    ])
    right_hip = np.array([
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * frame_width,
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * frame_height
    ])
    left_hip = np.array([
        landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * frame_width,
        landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * frame_height
    ])
    x_min, y_min, x_max, y_max = ball_box
    ball_center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])

    # Determine kicking leg (closer ankle to ball)
    right_dist = np.linalg.norm(right_ankle - ball_center)
    left_dist = np.linalg.norm(left_ankle - ball_center)
    if right_dist < left_dist:
        kicking_knee = right_knee
        kicking_hip = right_hip
        kicking_ankle = right_ankle
        plant_ankle = left_ankle
    else:
        kicking_knee = left_knee
        kicking_hip = left_hip
        kicking_ankle = left_ankle
        plant_ankle = right_ankle

    # Calculate knee angle (hip-knee-ankle)
    v1 = kicking_hip - kicking_knee
    v2 = kicking_ankle - kicking_knee
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    knee_angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    # Calculate plant foot distance in pixels
    plant_foot_dist_pixels = np.linalg.norm(plant_ankle - ball_center)

    # Convert to centimeters using goal width as reference (standard goal width = 732 cm)
    goal_width_pixels = goal_box[2] - goal_box[0]
    pixels_per_cm = goal_width_pixels / 732
    plant_foot_dist_cm = plant_foot_dist_pixels / pixels_per_cm

    # Determine kicking leg label for feedback
    kicking_leg = "right" if right_dist < left_dist else "left"

    return knee_angle, plant_foot_dist_pixels, plant_foot_dist_cm, kicking_leg

def generate_feedback(knee_angle, plant_foot_dist_cm, final_position, intended_shot, kicking_leg):
    feedback = []
    if final_position == "miss":
        feedback.append("You missed the goal entirely. Focus on aligning your body towards the goal.")
        feedback.append(f"Your plant foot was {plant_foot_dist_cm:.1f} cm from the ball. Ensure it's positioned closer (around 20-30 cm) for better control.")
    elif final_position != intended_shot:
        vertical_regions = {
            "top": ["top-left", "top-center", "top-right"],
            "center": ["center-left", "center", "center-right"],
            "bottom": ["bottom-left", "bottom-center", "bottom-right"]
        }
        intended_vertical = next(v for v, regions in vertical_regions.items() if intended_shot in regions)
        actual_vertical = next(v for v, regions in vertical_regions.items() if final_position in regions)
        if intended_vertical != actual_vertical:
            if intended_vertical == "top" and actual_vertical in ["center", "bottom"]:
                feedback.append(f"You aimed for a {intended_shot} shot but hit {final_position}. Your {kicking_leg} knee angle ({knee_angle:.1f}°) might be too low. Extend your knee more (aim for 140-160°) to lift the ball higher.")
            elif intended_vertical == "bottom" and actual_vertical in ["center", "top"]:
                feedback.append(f"You aimed for a {intended_shot} shot but hit {final_position}. Your {kicking_leg} knee angle ({knee_angle:.1f}°) might be too high. Bend your knee more (aim for 90-120°) to keep the ball low.")
        horizontal_regions = {
            "left": ["top-left", "center-left", "bottom-left"],
            "center": ["top-center", "center", "bottom-center"],
            "right": ["top-right", "center-right", "bottom-right"]
        }
        intended_horizontal = next(h for h, regions in horizontal_regions.items() if intended_shot in regions)
        actual_horizontal = next(h for h, regions in horizontal_regions.items() if final_position in regions)
        if intended_horizontal != actual_horizontal:
            if intended_horizontal == "left" and actual_horizontal in ["center", "right"]:
                feedback.append(f"You aimed for a {intended_shot} shot but hit {final_position}. Your plant foot was {plant_foot_dist_cm:.1f} cm from the ball. Position it slightly to the right of the ball to aim left more accurately.")
            elif intended_horizontal == "right" and actual_horizontal in ["center", "left"]:
                feedback.append(f"You aimed for a {intended_shot} shot but hit {final_position}. Your plant foot was {plant_foot_dist_cm:.1f} cm from the ball. Position it slightly to the left of the ball to aim right more accurately.")
    else:
        feedback.append("Great shot! You hit the intended target!")
    
    return feedback

# Main processing with real-time visualization
video_path = "back_angle_shot_new.mp4"
intended_shot = "top-left"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file '{video_path}'")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Initialize variables for analysis
contact_frame = None
contact_detected = False
trajectory = []
final_position = "miss"
knee_angle = None
plant_foot_dist_pixels = None
plant_foot_dist_cm = None
kicking_leg = None
previous_ball_pos = None
previous_goal_box = None

cv2.namedWindow("Shot Analysis", cv2.WINDOW_AUTOSIZE)

# Process video in a single pass with real-time visualization
frame_idx = 0
while frame_idx < total_frames:
    skip = 5 if not contact_detected else 3
    frame_idx = frame_idx + skip if frame_idx + skip < total_frames else total_frames - 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        break

    # Detect elements
    player_box, ball_box, goal_box = detect_elements(frame, previous_ball_pos, previous_goal_box, frame_idx)
    previous_goal_box = goal_box
    
    # Initialize frame for visualization
    display_frame = frame.copy()

    # Draw detected elements
    if player_box:
        x_min, y_min, x_max, y_max = player_box
        cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.putText(display_frame, "Player", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    if ball_box:
        x_min, y_min, x_max, y_max = ball_box
        ball_center = (int((x_min + x_max) / 2), int((y_min + y_max) / 2))
        cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(display_frame, "Ball", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if contact_detected:
            trajectory.append(ball_center)
        previous_ball_pos = np.array(ball_center)
    if goal_box:
        x_min, y_min, x_max, y_max = goal_box
        cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        cv2.putText(display_frame, "Goal", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    else:
        cv2.putText(display_frame, "Goal not detected", (10, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Draw trajectory
    for i in range(1, len(trajectory)):
        cv2.line(display_frame, trajectory[i-1], trajectory[i], (255, 255, 0), 2)

    # Detect contact frame
    if not contact_detected and player_box and ball_box:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            right_ankle = np.array([
                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x * frame_width,
                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y * frame_height
            ])
            left_ankle = np.array([
                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x * frame_width,
                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * frame_height
            ])
            x_min, y_min, x_max, y_max = ball_box
            ball_center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
            right_dist = np.linalg.norm(right_ankle - ball_center)
            left_dist = np.linalg.norm(left_ankle - ball_center)
            if min(right_dist, left_dist) < 50:
                for i in range(max(0, frame_idx-5), min(total_frames, frame_idx+6)):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    if not ret:
                        break
                    player_box, ball_box, goal_box = detect_elements(frame, previous_ball_pos, previous_goal_box, i)
                    previous_goal_box = goal_box
                    if not player_box or not ball_box:
                        continue
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(frame_rgb)
                    if not results.pose_landmarks:
                        continue
                    landmarks = results.pose_landmarks.landmark
                    right_ankle = np.array([
                        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x * frame_width,
                        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y * frame_height
                    ])
                    left_ankle = np.array([
                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x * frame_width,
                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * frame_height
                    ])
                    x_min, y_min, x_max, y_max = ball_box
                    ball_center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
                    right_dist = np.linalg.norm(right_ankle - ball_center)
                    left_dist = np.linalg.norm(left_ankle - ball_center)
                    if min(right_dist, left_dist) < 50:
                        contact_frame = i
                        contact_detected = True
                        knee_angle, plant_foot_dist_pixels, plant_foot_dist_cm, kicking_leg = calculate_metrics_at_contact(frame, frame_width, frame_height, player_box, ball_box, goal_box)
                        # Slow down at contact frame
                        for _ in range(30):
                            display_frame = frame.copy()
                            if player_box:
                                x_min, y_min, x_max, y_max = player_box
                                cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                            if ball_box:
                                x_min, y_min, x_max, y_max = ball_box
                                cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                            if goal_box:
                                x_min, y_min, x_max, y_max = goal_box
                                cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                            cv2.putText(display_frame, "Contact Detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            if knee_angle is not None:
                                cv2.putText(display_frame, f"Knee Angle: {knee_angle:.1f}°", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            if plant_foot_dist_cm is not None:
                                cv2.putText(display_frame, f"Plant Foot Dist: {plant_foot_dist_cm:.1f} cm", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            cv2.imshow("Shot Analysis", display_frame)
                            cv2.waitKey(33)
                        break
                if contact_detected:
                    frame_idx = contact_frame
                    continue

    # Determine final position near the end
    if frame_idx >= total_frames - 9 and ball_box and goal_box:
        x_min, y_min, x_max, y_max = ball_box
        ball_center = (int((x_min + x_max) / 2), int((y_min + y_max) / 2))
        gx_min, gy_min, gx_max, gy_max = goal_box
        if gx_min <= ball_center[0] <= gx_max and gy_min <= ball_center[1] <= gy_max:
            g_width = gx_max - gx_min
            g_height = gy_max - gy_min
            x_rel = (ball_center[0] - gx_min) / g_width
            y_rel = (ball_center[1] - gy_min) / g_height
            if y_rel < 0.33:
                if x_rel < 0.33:
                    final_position = "top-left"
                elif x_rel < 0.67:
                    final_position = "top-center"
                else:
                    final_position = "top-right"
            elif y_rel < 0.67:
                if x_rel < 0.33:
                    final_position = "center-left"
                elif x_rel < 0.67:
                    final_position = "center"
                else:
                    final_position = "center-right"
            else:
                if x_rel < 0.33:
                    final_position = "bottom-left"
                elif x_rel < 0.67:
                    final_position = "bottom-center"
                else:
                    final_position = "bottom-right"

    # Add text overlays
    cv2.putText(display_frame, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if contact_detected:
        cv2.putText(display_frame, f"Contact Frame: {contact_frame}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if knee_angle is not None:
            cv2.putText(display_frame, f"Knee Angle: {knee_angle:.1f}°", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if plant_foot_dist_cm is not None:
            cv2.putText(display_frame, f"Plant Foot Dist: {plant_foot_dist_cm:.1f} cm", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Ball Position: {final_position}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow("Shot Analysis", display_frame)
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

# Fallback if contact frame wasn't detected
if not contact_detected:
    contact_frame = total_frames // 2
    contact_detected = True
    cap.set(cv2.CAP_PROP_POS_FRAMES, contact_frame)
    ret, frame = cap.read()
    if ret:
        player_box, ball_box, goal_box = detect_elements(frame, previous_ball_pos, previous_goal_box, frame_idx)
        knee_angle, plant_foot_dist_pixels, plant_foot_dist_cm, kicking_leg = calculate_metrics_at_contact(frame, frame_width, frame_height, player_box, ball_box, goal_box)

# Generate and display feedback immediately
if knee_angle is not None and plant_foot_dist_cm is not None and kicking_leg is not None:
    feedback = generate_feedback(knee_angle, plant_foot_dist_cm, final_position, intended_shot, kicking_leg)
    print("\nAnalysis Summary:")
    print(f"Contact Frame: {contact_frame}")
    print(f"Knee Angle at Contact: {knee_angle:.1f}°")
    print(f"Plant Foot Distance: {plant_foot_dist_cm:.1f} cm")
    print(f"Ball Final Position: {final_position}")
    print("\nFeedback to Improve Your Shot:")
    for i, tip in enumerate(feedback, 1):
        print(f"{i}. {tip}")
else:
    print("\nCould not calculate metrics at contact frame.")

# Visualize trajectory on last frame
cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
ret, frame = cap.read()
if ret:
    _, _, goal_box = detect_elements(frame, previous_ball_pos, previous_goal_box, frame_idx)
    if goal_box:
        cv2.rectangle(frame, (goal_box[0], goal_box[1]), (goal_box[2], goal_box[3]), (0, 255, 0), 2)
        cv2.putText(frame, f"Final Position: {final_position}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    for i in range(1, len(trajectory)):
        cv2.line(frame, trajectory[i-1], trajectory[i], (255, 255, 0), 2)
    if trajectory:
        cv2.circle(frame, trajectory[-1], 5, (0, 0, 255), -1)
    cv2.imwrite("trajectory_plot.png", frame)

cap.release()
cv2.destroyAllWindows()
pose.close()
print("\nPlot saved: 'trajectory_plot.png'")