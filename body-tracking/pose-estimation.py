import cv2 #to read and process images
import matplotlib.pyplot as plt #to show resultant images
import mediapipe as mp

# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose
# Setting up the Pose model for images.
pose_img = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=1)
# Setting up the Pose model for videos.
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5,
                          min_tracking_confidence=0.5, model_complexity=1)

# Initializing mediapipe drawing class to draw landmarks on specified image.
mp_drawing = mp.solutions.drawing_utils


def estimPose_img(input_file, pose=pose_img, landmarks_c=(234, 63, 247), connection_c=(117, 249, 77),
                  thickness=20, circle_r=10, display=True):
    # Read the input image
    if isinstance(input_file, str):
        input_img = cv2.imread(input_file)
    else:
        input_img = input_file

    # Create a copy of the input image
    output_img = input_img.copy()

    # Convert the image from BGR into RGB format.
    RGB_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)

    # Perform the Pose Detection.
    results = pose.process(RGB_img)

    # Retrieve the height and width of the input image.
    height, width, _ = input_img.shape

    # Initialize a list to store the detected landmarks.
    landmarks = []

    # Check if any landmarks are detected.
    if results.pose_landmarks:

        # Draw Pose landmarks on the output image.
        mp_drawing.draw_landmarks(output_img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(landmarks_c, thickness, circle_r),
                                  mp_drawing.DrawingSpec(connection_c, thickness, circle_r))

        # Iterate over the detected landmarks.
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                              (landmark.z * width)))

    # Check if we want to display.
    if display:
        # Display the original input image and the resulting image.
        plt.figure(figsize=[15, 15])
        plt.subplot(121);
        plt.imshow(input_img[:, :, ::-1]);
        plt.title("Original image");
        plt.axis('off');
        plt.subplot(122);
        plt.imshow(output_img[:, :, ::-1]);
        plt.title("Output image");
        plt.axis('off');

        # Plot the Pose landmarks in 3D.
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

    # Just get output_img and landmarks
    else:
        # Return the output image and the found landmarks.
        return output_img, landmarks

estimPose_img('male-image.jpg')