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

ideal_data = ideal_metrics[ideal_metrics['frame'] >= ideal_start_frame]
test_data = test_metrics[test_metrics['frame'] >= test_start_frame]

ideal_contact = ideal_metrics[ideal_metrics['frame'] == ideal_contact_frame].iloc[0]
test_contact = test_metrics[test_metrics['frame'] == test_contact_frame].iloc[0]

