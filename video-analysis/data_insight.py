import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('metrics_all_frames.txt')
print(data[['frame', 'knee_angle', 'plant_foot_distance']])

plt.plot(data['frame'], data['knee_angle'], label='Knee Angle (º)') 
plt.plot(data['frame'], data['plant_foot_distance'], label='Plant Foot Distance (cm)')
plt.xlabel ('Frame')
plt.legend()
plt.savefig('motion_analysis.png')
