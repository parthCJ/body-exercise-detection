import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose solution
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    """Calculates the angle of a joint given three landmark points."""
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


# --- State and Counter Variables ---
pushup_state = "up"
pushup_count = 0
is_in_pushup_position = False  # New state to track if the user is in position

# --- Video Capture ---
cap = cv2.VideoCapture(0)
cv2.namedWindow('Push-up Counter', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Push-up Counter', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.flip(image, 1)
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        image.flags.writeable = True

        try:
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                image_height, image_width, _ = image.shape

                # --- NEW LOGIC START: Check for Push-up Position ---

                # Get coordinates for position check
                shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

                # Check 1: Body is horizontal (shoulder and hip y-coordinates are similar)
                # We use pixel coordinates for this check
                shoulder_y = shoulder.y * image_height
                hip_y = hip.y * image_height
                is_horizontal = abs(shoulder_y - hip_y) < 100  # Threshold of 100 pixels

                # Check 2: Shoulders are positioned above wrists
                wrist_y = wrist.y * image_height
                is_supported = shoulder_y < wrist_y

                if is_horizontal and is_supported:
                    is_in_pushup_position = True

                    # --- ORIGINAL ANGLE COUNTING LOGIC (NOW NESTED) ---
                    # Get coordinates for angle calculation
                    elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]

                    # Convert to list for the function
                    shoulder_pt = [shoulder.x, shoulder.y]
                    elbow_pt = [elbow.x, elbow.y]
                    wrist_pt = [wrist.x, wrist.y]

                    # Calculate elbow angle
                    angle = calculate_angle(shoulder_pt, elbow_pt, wrist_pt)

                    # Visualize the angle on the elbow
                    elbow_coords = tuple(np.multiply(elbow_pt, [image_width, image_height]).astype(int))
                    cv2.putText(image, str(int(angle)), elbow_coords,
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    # Push-up counting state machine
                    if angle < 90:
                        pushup_state = "down"
                    if angle > 160 and pushup_state == "down":
                        pushup_state = "up"
                        pushup_count += 1
                else:
                    is_in_pushup_position = False
                    pushup_state = "up"  # Reset state if not in position

                # --- NEW LOGIC END ---

        except Exception as e:
            pass

        # --- Drawing and Display ---
        # Display status text
        status_text = "IN POSITION" if is_in_pushup_position else "GET IN POSITION"
        color = (0, 255, 0) if is_in_pushup_position else (0, 0, 255)
        cv2.putText(image, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # Display push-up count
        cv2.putText(image, "Push-ups: " + str(pushup_count), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3,
                    cv2.LINE_AA)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

        cv2.imshow('Push-up Counter', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()