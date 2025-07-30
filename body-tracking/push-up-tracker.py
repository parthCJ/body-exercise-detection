import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose solution
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    """Calculates the angle of a joint given three landmark points."""
    a = np.array(a)  # First point
    b = np.array(b)  # Mid pointg
    c = np.array(c)  # End point

    # converting the angles to the radians.
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


# --- State and Counter Variables ---
pushup_state = "up"
pushup_count = 0

# --- Video Capture ---
# THE ONLY CHANGE NEEDED IS HERE: from file path to 0
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a selfie-view display
        image = cv2.flip(image, 1)

        # Convert the BGR image to RGB before processing
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # Make image writeable again to draw on it
        image.flags.writeable = True

        # --- Landmark Extraction and Angle Calculation ---
        try:
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates for the right arm
                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
                elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
                wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]

                # Calculate elbow angle
                angle = calculate_angle(shoulder, elbow, wrist)

                # Visualize the angle on the elbow
                image_height, image_width, _ = image.shape
                elbow_coords = tuple(np.multiply(elbow, [image_width, image_height]).astype(int))
                cv2.putText(image, str(int(angle)),
                            elbow_coords,
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # --- Push-up counting logic ---
                if angle < 90:
                    pushup_state = "down"
                if angle > 160 and pushup_state == "down":
                    pushup_state = "up"
                    pushup_count += 1

        except Exception as e:
            pass

            # --- Drawing and Display ---
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

        cv2.putText(image, "Push-ups: " + str(pushup_count),
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (255, 0, 0), 3, cv2.LINE_AA)

        cv2.imshow('Push-up Counter', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()