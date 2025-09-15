import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    """Calculates angle between three points (a-b-c)."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


# --- Counter for Sit-Ups ---
situp_count = 0
# NEW: Simplified state machine. Can be "DOWN" or "UP".
state = "DOWN"

video_path = "media/situps.mp4"  # change to your sit-up video

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file at {video_path}")
    exit()

scale_factor = 0.7

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image = cv2.flip(image, 1)

        width = int(image.shape[1] * scale_factor)
        height = int(image.shape[0] * scale_factor)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        hip_angle = 0

        try:
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Extract keypoints
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]

                # Hip angle (shoulder-hip-knee)
                hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)

                # Detect positions
                is_up_position = hip_angle < 100  # Sitting up
                is_down_position = hip_angle > 150  # Lying down

                # --- REFINED STATE-MACHINE COUNTING LOGIC ---
                # If we are in the 'DOWN' state and detect an 'UP' position...
                if state == "DOWN" and is_up_position:
                    state = "UP"  # ...change the state to 'UP'.

                # If we are in the 'UP' state and detect a 'DOWN' position...
                elif state == "UP" and is_down_position:
                    state = "DOWN"  # ...change the state back to 'DOWN'...
                    situp_count += 1  # ...and count this as one completed rep.
                    print(f"REP COMPLETED! Count: {situp_count}")

        except Exception as e:
            print(f"Error: {e}")

        # --- Drawing ---
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        # Overlay panel
        cv2.rectangle(image, (0, 0), (350, 120), (245, 117, 16), -1)

        cv2.putText(image, 'SIT-UPS', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(image, str(situp_count), (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)

        cv2.putText(image, 'STATE', (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(image, state, (150, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        cv2.putText(image, f'HIP_ANGLE: {hip_angle:.1f}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

        cv2.imshow('Sit-Up Counter', image)

        if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
            break

cap.release()
cv2.destroyAllWindows()