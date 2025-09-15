import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose solution
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    """Calculates the angle of a joint given three landmark points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


# --- State and Counter Variables for Jumping Jacks ---
jj_count = 0
up_detected = False
down_detected = False
frame_count = 0

video_path = "media/jumping-jacks.mp4"  # Make sure this path is correct

# --- Video Capture ---
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

        frame_count += 1
        image = cv2.flip(image, 1)

        width = int(image.shape[1] * scale_factor)
        height = int(image.shape[0] * scale_factor)
        dim = (width, height)
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # Initialize debug values
        wrist_distance = 0
        ankle_distance = 0
        is_up_position = False
        is_down_position = False
        current_state = "UNKNOWN"

        try:
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Get key landmarks
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
                right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

                # Calculate distances
                wrist_distance = abs(left_wrist.x - right_wrist.x)
                ankle_distance = abs(left_ankle.x - right_ankle.x)

                # Check if arms are raised
                left_arm_raised = left_wrist.y < left_shoulder.y
                right_arm_raised = right_wrist.y < right_shoulder.y

                # CLEAR POSITION DETECTION with relaxed thresholds
                # UP POSITION: Arms wide AND at least one arm raised AND legs apart
                is_up_position = (wrist_distance > 0.25 and
                                  (left_arm_raised or right_arm_raised) and
                                  ankle_distance > 0.1)

                # DOWN POSITION: Arms close AND legs together
                is_down_position = (wrist_distance < 0.15 and ankle_distance < 0.08)

                # SIMPLE FLAG-BASED COUNTING SYSTEM
                if is_up_position:
                    current_state = "UP"
                    if not up_detected:  # First time detecting up
                        up_detected = True
                        down_detected = False  # Reset down flag
                        print(f"UP position detected! Frame {frame_count}")

                elif is_down_position:
                    current_state = "DOWN"
                    if not down_detected and up_detected:  # Down after up = complete rep
                        down_detected = True
                        up_detected = False  # Reset up flag
                        jj_count += 1
                        print(f"REP COMPLETED! Count: {jj_count} at frame {frame_count}")

                else:
                    current_state = "TRANSITION"

        except Exception as e:
            print(f"Error processing landmarks: {e}")

        # --- Drawing and Display ---
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        # --- CLEAR DEBUG PANEL ---
        cv2.rectangle(image, (0, 0), (550, 160), (245, 117, 16), -1)

        # Main display
        cv2.putText(image, 'REPS', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, str(jj_count), (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

        cv2.putText(image, 'STATE', (120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, current_state, (120, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

        # Position detection
        cv2.putText(image, f'UP_POS: {"YES" if is_up_position else "NO"}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if is_up_position else (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f'DOWN_POS: {"YES" if is_down_position else "NO"}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if is_down_position else (255, 255, 255), 2, cv2.LINE_AA)

        # Flags
        cv2.putText(image, f'UP_FLAG: {"SET" if up_detected else "CLEAR"}', (200, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if up_detected else (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f'DOWN_FLAG: {"SET" if down_detected else "CLEAR"}', (200, 110), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0) if down_detected else (255, 255, 255), 2, cv2.LINE_AA)

        # Distances
        cv2.putText(image, f'WRIST: {wrist_distance:.3f}', (380, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(image, f'ANKLE: {ankle_distance:.3f}', (380, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                    1, cv2.LINE_AA)
        cv2.putText(image, f'FRAME: {frame_count}', (380, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)

        # Thresholds for reference
        cv2.putText(image, 'Thresholds:', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(image, 'UP: W>0.25, A>0.1', (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1,
                    cv2.LINE_AA)
        cv2.putText(image, 'DOWN: W<0.15, A<0.08', (200, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1,
                    cv2.LINE_AA)

        cv2.imshow('Jumping Jack Counter', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()