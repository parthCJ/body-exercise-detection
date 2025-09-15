import cv2
import mediapipe as mp
import numpy as np
import csv
from datetime import datetime

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

# --- ADD THIS BLOCK TO SAVE YOUR DATA ---

# Get the current date in a simple format (YYYY-MM-DD)
today_date = datetime.now().strftime("%Y-%m-%d")
reps_count = jj_count
csv_filename = "umping_jack_log.csv"

# The data we want to save
data_row = [today_date, reps_count]

# Check if the file already exists to decide whether to write the header
file_exists = False
try:
    with open(csv_filename, 'r') as file:
        # If the file is not empty, it exists and has content
        if file.read(1):
            file_exists = True
except FileNotFoundError:
    file_exists = False

# Open the file in 'append' mode ('a') to add a new row
# newline='' prevents extra blank rows
with open(csv_filename, 'a', newline='') as file:
    writer = csv.writer(file)
    # If the file is new, write the header first
    if not file_exists:
        writer.writerow(['date', 'reps'])

    # Write the actual data
    writer.writerow(data_row)

print(f"Workout saved! Date: {today_date}, Reps: {reps_count}")
# --- END OF SAVING BLOCK ---

import pandas as pd
import matplotlib.pyplot as plt

# Load the data from your CSV file
try:
    df = pd.read_csv(csv_filename)
except FileNotFoundError:
    print("Error: workout_log.csv not found. Run the jumping jack script first to create it.")
    exit()

# Convert the 'date' column to actual datetime objects
df['date'] = pd.to_datetime(df['date'])

# Sort the data by date just in case it's out of order
df = df.sort_values(by='date')

# --- Create the Plot ---
plt.style.use('seaborn-v0_8-darkgrid') # Use a nice-looking style
fig, ax = plt.subplots(figsize=(10, 6)) # Set the figure size

# Plot the data: date on the x-axis, reps on the y-axis
ax.plot(df['date'], df['reps'], marker='o', linestyle='-', color='b', label='Jumping Jack Reps')

# Format the plot to make it look good
ax.set_title("Jumping Jack Progress Over Time", fontsize=16)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Number of Reps", fontsize=12)
ax.tick_params(axis='x', labelrotation=45) # Rotate date labels for readability
ax.legend()
plt.tight_layout() # Adjust layout to make room for labels

# Display the plot
plt.show()

## Step 3: Add Machine Learning for Trend Analysis 🤖

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the data from your CSV file
try:
    df = pd.read_csv(csv_filename)
except FileNotFoundError:
    print("Error: workout_log.csv not found. Run the jumping jack script first to create it.")
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


# --- Create the Plot ---
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the ACTUAL data as scatter points
ax.scatter(df['date'], df['reps'], color='b', label='Actual Reps')

# Plot the ML PREDICTION as a trend line
ax.plot(df['date'], y_pred, color='r', linestyle='--', linewidth=2, label='ML Trend Line')

# Format the plot
ax.set_title("Jumping Jack Progress and ML Trend", fontsize=16)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Number of Reps", fontsize=12)
ax.tick_params(axis='x', labelrotation=45)
ax.legend()
plt.tight_layout()

# Display the plot
plt.show()