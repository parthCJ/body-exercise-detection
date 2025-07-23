import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils # drawing the utils
mp_pose = mp.solutions.pose # making the pose.

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3) as pose:
    while cap.isOpened():
        ret, Image = cap.read()
        Image = cv2.cvtColor(Image, cv2.COLOR_BGR2RGB) # converting the color
        Image.flags.writeable = False
        results = pose.process(Image)
        Image.flags.writeable = True
        Image = cv2.cvtColor(Image, cv2.COLOR_RGB2BGR) # re-enabling the color conversion.
        mp_drawing.draw_landmarks(Image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=8, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=4, circle_radius=2))
        cv2.imshow('Media pipe pose detected', Image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()



