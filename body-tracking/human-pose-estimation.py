import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as pimg

mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
img = pimg.imread("sample-image.jpg")

plt.imshow(img)
results = mp_pose.process(img)
print(results.pose_landmarks)

mp_drawing = mp.solutions.drawing_utils

mp_drawing.draw_landmarks(img, results.pose_landmarks,
                          mp.solutions.pose.POSE_CONNECTIONS,
                          mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=3, circle_radius=2),
                          mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=3, circle_radius=2)
                          )
plt.imshow(img)