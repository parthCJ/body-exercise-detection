import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode = False, maxHands = 2, detection_confidence = 0.5, track_confidence = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConfidence = detection_confidence
        self.tracking_confidence = track_confidence

        self.mpHands = mp.solutions.hands
        self.hands = mpHands.Hands()  # there are 21 points of the hands in the pretrained model.
        self.mpDraw = mp.solutions.drawing_utils  # drawing the landmarks.

imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # converting the image into RBG
results = hands.process(imgRGB) # process function processing the hands
# print(results.multi_hand_landmarks)

if results.multi_hand_landmarks: # if there is some value in the landmarks
    for handLms in results.multi_hand_landmarks: # looping over the hand_landmarks
        for id, lm in enumerate(handLms.landmark): # Looping over id and landmark in the handsLm.landmarks.
                print(id, lm) # Printing the landmarks and the ID when there is a hand detected
                h, w, c = img.shape # heighst, width, center
                cx, cy = int(lm.x*w), int(lm.y*h) # center X, center Y
                print(id, cx, cy) # printing the id, center(X), center(Y)
                if id == 0: # checking if the landmark ID is correct and if drawing a circle on that landmark ID.
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                if id == 4: # thumb tip landmark.
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
        mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) # drawing landmarks on each.

# calculating the FPS using the formula fps = 1/(current_time-previous_time)


def main():
    pTime = 0  # previous time
    cTime = 0  # current time
    cap = cv2.VideoCapture(0)  # Or the correct camera index
    while True:
        success, img = cap.read()  # reading the image

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int((fps))), (10, 70), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0),
                    1)  # putting texts on the screen(FPS)

        cv2.imshow('Image', img)  # getting the web cam image
        if cv2.waitKey(1) == ord('q'):
            break

if __name__ == "__main__":
    main()

cap.release()
cv2.destroyAllWindows()