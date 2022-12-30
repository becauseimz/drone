import cv2 as cv
import mediapipe as mp
import numpy as np

# mediapipe drawing utils makes it easy to render the landmarks on the video frame image
mp_drawing = mp.solutions.drawing_utils
# The mediapipe hand pose model
hands_model = mp.solutions.hands
# Use built-in webcam as the video source via OpenCV
src = cv.VideoCapture(0)

# initialise mediapipe hands model with minimum hand detection confidence of 70% and minimum hand landmarks confidence of 50%
with hands_model.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while src.isOpened():
        ## ===== Reading the video frame and Pre-processing ===== ##
        # Get the video frame (image) from the source (webcam)
        _, frame = src.read()
        # flip the frame horizontally (along the y-axis)
        frame = cv.flip(frame, 1)
        # OpenCV uses BGR format for images. Convert this to RBG  
        # format so that it is compatible with mediapipe model.
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        ## == Applying the model on pre-processed video frame == ##
        # Set the image to be non-writable, thus forcing it to be 
        # passed by reference to the model
        image.flags.writeable = False
        # Use configured mediapipe hands model to make detections on 
        # the converted frame from video source
        results = hands.process(image)

        # If there are any results (i.e. detected hands) then...
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                # Render results: Draw the landmarks and connections on the
                # original video frame
                mp_drawing.draw_landmarks(frame, hand, hands_model.HAND_CONNECTIONS)
                # get the x and y coordinates of the wrist for this hand
                x = int(hand.landmark[hands_model.HandLandmark.WRIST].x * frame.shape[1])
                y = int(hand.landmark[hands_model.HandLandmark.WRIST].y * frame.shape[0])
                # Get the handedness classification label (left/right) and
                # render it as text at the coordinates of the wrist
                # landmark, in green
                cv.putText(frame, results.multi_handedness[num].classification[0].label, (x, y), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1, cv.LINE_AA)
        # Display the frame with the renderings of the result and hand
        # classification
        cv.imshow("Hand Pose", frame)
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break

# Releases the webcam
src.release()
# Closes all the windows opened by the code (such as "Hand Pose")
cv.destroyAllWindows()
