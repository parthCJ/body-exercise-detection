import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture('body-detection2.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = mp_pose.process(frame)
    mp_drawing.draw_landmarks(
        frame,
        results.pose_landmarks,
        mp.solutions.pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=4, circle_radius=2),
        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=6, circle_radius=2)
    )
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) == ord('1'):
        break
cap.release()
cv2.destroyAllWindows()