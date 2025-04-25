import pandas as pd
import matplotlib.pyplot as plt

ideal_metrics = pd.read_csv('metrics_ideal_shot.txt')
test_metrics = pd.read_csv('metrics_test_shot.txt')

ideal_contact_frame = ideal_metrics['frame'].max()
test_contact_frame = test_metrics['frame'].max()

print(f"Ideal shot contact frame: {ideal_contact_frame}")
print(f"Test shot contact frame: {test_contact_frame}")

num_frames_to_compare = 10
ideal_start_frame = max(0, ideal_contact_frame - num_frames_to_compare + 1)
test_start_frame = max(0, test_contact_frame - num_frames_to_compare + 1)

ideal_data = ideal_metrics[ideal_metrics['frame'] >= ideal_start_frame].copy()
test_data = test_metrics[test_metrics['frame'] >= test_start_frame].copy()

ideal_data['relative_frame'] = ideal_data['frame'] - ideal_contact_frame
test_data['relative_frame'] = test_data['frame'] = test_contact_frame

ideal_contact = ideal_metrics[ideal_metrics['frame'] == ideal_contact_frame].iloc[0]
test_contact = test_metrics[test_metrics['frame'] == test_contact_frame].iloc[0]

# Knee angle comparison
ideal_knee_angle = ideal_contact['knee_angle']
test_knee_angle = test_contact['knee_angle']
knee_angle_diff = test_knee_angle - ideal_knee_angle

# Plant foot distance comparison
ideal_plant_foot = ideal_contact['plant_foot_distance']
test_plant_foot = test_contact['plant_foot_distance']
plant_foot_diff = test_plant_foot - ideal_plant_foot

# Generate feedback
feedback = []
if knee_angle_diff > 10:
    feedback.append("Bend your knee more for a more powerful shot")
elif knee_angle_diff < -10:
    feedback.append("Straighten your knee slightly for better balanced shot")

if plant_foot_diff > 5:
    feedback.append("Your plant foot is too far from ball, bring it closer to the ball.")
elif plant_foot_diff < -5:
    feedback.append("Your plant foot is too close to ball, position it further for better accuracy and balance.")

print('\nFeedback to improve your shot:')
if feedback:
    for i, tip in enumerate(feedback, 1):
        print(f"{i}. {tip}")
else:
    print("Great job! Your shot closely matches ideal form")

plt.figure(figsize=(10,6))

# Knee angle plot
plt.subplot(2,1,1)
plt.plot(ideal_data['relative_frame'], ideal_data['knee_angle'], label="Ideal Shot", color='blue')
plt.plot(test_data['relative_frame'], test_data['knee_angle'], label="Test Shot", color='red')
plt.axvline(x=0, color='black', linestyle='--', label='Contact Frame')
plt.xlabel('Frames relative to contact')
plt.ylabel('Knee angle (º)')
plt.title('Knee angle comparison')
plt.legend()
plt.grid(True)

# Plant foot distance plot
plt.subplot(2,1,2)
plt.plot(ideal_data['relative_frame'], ideal_data['plant_foot_distance'], label="Ideal Shot", color='blue')
plt.plot(test_data['relative_frame'], test_data['plant_foot_distance'], label="Test Shot", color='red')
plt.axvline(x=0, color='black', linestyle='--', label="Contact Frame")
plt.xlabel('Frames relative to contact')
plt.ylabel('Plant foot distance (cm)')
plt.title('Plant foot distance comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('shot_comparison.png')
print("\nComparison plot saved as 'shot_comparsion.png'")
