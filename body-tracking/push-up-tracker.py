import cv2
import mediapipe as mp
import numpy as np
import csv

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
video_path = "media/psuh-ups.mp4"  # <--- IMPORTANT: SET YOUR VIDEO PATH HERE

# --- ADD THIS: Set a scale factor for the screen size ---
scale_factor = 0.7 # 70% of original size. Adjust as needed.

# --- Video Capture ---
cap = cv2.VideoCapture(video_path)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a selfie-view display
        image = cv2.flip(image, 1)

        # --- ADD THIS BLOCK TO RESIZE THE FRAME ---
        width = int(image.shape[1] * scale_factor)
        height = int(image.shape[0] * scale_factor)
        image = cv2.resize(image, (width, height))
        # --- END OF RESIZING BLOCK ---

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
from datetime import datetime

today_date = datetime.now().strftime("%Y-%m-%d")
reps_count = pushup_count
csv_filename = "pushup_log.csv"

data_row = [today_date, reps_count]

file_exists = False
try:
    with open(csv_filename, 'r') as file:
        if file.read(1):
            file_exists = True
except FileNotFoundError:
    file_exists = False

with open(csv_filename, 'a', newline='') as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow(['date', 'reps'])
    writer.writerow(data_row)

print(f"Push-up workout saved! Date: {today_date}, Reps: {reps_count}")


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the data from your push-up CSV file
try:
    df = pd.read_csv(csv_filename)
except FileNotFoundError:
    print("Error: pushup_log.csv not found. Run the push-up counter script first to create it.")
    exit()

# Convert the 'date' column to actual datetime objects
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by='date')

# --- ML PART: Prepare data for the model ---
# The model needs numerical data, so we convert dates to "days since first workout"
df['days_since_start'] = (df['date'] - df['date'].min()).dt.days

# X is our feature (days), y is our target (reps)
X = df[['days_since_start']]
y = df['reps']

# --- ML PART: Train the Linear Regression model ---
model = LinearRegression()
model.fit(X, y)

# --- ML PART: Get the trend line from the model's predictions ---
y_pred = model.predict(X)


# --- Create the Plot using Matplotlib ---
plt.style.use('seaborn-v0_8-darkgrid') # Use a nice-looking style
fig, ax = plt.subplots(figsize=(10, 6)) # Set the figure size

# Plot the ACTUAL data as scatter points
ax.scatter(df['date'], df['reps'], color='mediumspringgreen', label='Actual Reps')

# Plot the ML PREDICTION as a trend line
ax.plot(df['date'], y_pred, color='cyan', linestyle='--', linewidth=2, label='ML Trend Line')

# Format the plot to make it look good
ax.set_title("Push-up Progress and ML Trend 💪", fontsize=16)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Number of Reps", fontsize=12)
ax.tick_params(axis='x', labelrotation=45) # Rotate date labels for readability
ax.legend()
plt.tight_layout() # Adjust layout to make room for labels

# Display the plot in a new window
plt.show()