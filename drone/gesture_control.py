import cv2 as cv
import mediapipe as mp
import numpy as np
from sympy import Plane
from tello import Tello
import time

# mediapipe drawing utils makes it easy to render the landmarks on the video frame image
mp_drawing = mp.solutions.drawing_utils
# The mediapipe hand pose model
hands_model = mp.solutions.hands
# Use built-in webcam as the video source via OpenCV
src = cv.VideoCapture(0)
# Create a drone object (connection to the drone via UDP socket is automatically done by the Tello class)
drone = Tello()
# Put drone in command mode so that it accepts and follows commands via UDP socket, from the computer.
drone.send_command("command")
# Send command to auto takeoff
drone.send_command("takeoff")
# Timestamp of last command sent to drone
command_timer = time.time()

# Variables to store the normal vectors of 2 planes that will be used to represent the initial orientation of the right hand
rightPlane_N, rightPlane_Orth_N = None, None

def calcAngle(l1, l2, l3):
    # Store the (y, z) coordinates of each landmark as points
    a = np.array([l1.y, l1.z])
    b = np.array([l2.y, l2.z])
    c = np.array([l3.y, l3.z])
    # Calculate the 2 vectors (ab and bc) between the points
    ab = b - a
    bc = c - b
    # Apply formula to calculate the cosine of the angle (The advantages of numpy and how easily it
    # allows us to do matrices/vector based computation can be seen here)
    cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    # use arccos to get the angle, convert to deg and round to 0 d.p
    return np.round(np.degrees(np.arccos(cosine_angle)), decimals = 0)

def initialPose(img, left, right):
    # If either hand has not been detected yet, return False
    if not right or not left:
        return False
    # Initialise flag to True - default assumption is it is in initial pose
    pose = True

    # Iterate through landmarks of the base of each of our fingers
    # (i.e. 5, 9, 13 and 17. Refer to the landmarks image in part 1 of the guide)
    for i in range(5, 18, 4):
        # Calculate the angle between the finger's base, first digit and tip
        angle = calcAngle(right.landmark[0], right.landmark[i], right.landmark[i+3])
        # Render the angle as text at the tip of the finger on the video frame (img) - this is just to aid the user via the UI
        x = int(right.landmark[i+3].x * img.shape[1])
        y = int(right.landmark[i+3].y * img.shape[0])
        cv.putText(img, str(angle), (x, y), cv.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 1, cv.LINE_AA)
        
        # If angle is more than 15 degrees, consider the finger to be bent and thus out of
        # initial pose (set flag to False)
        if angle > 15:
            pose = False

        ## --- Do the same for the corresponding finger on left hand --- ##
        angle = calcAngle(left.landmark[0], left.landmark[i], left.landmark[i+3])
        x = int(left.landmark[i+3].x * img.shape[1])
        y = int(left.landmark[i+3].y * img.shape[0])
        cv.putText(img, str(angle), (x, y), cv.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 1, cv.LINE_AA)

        if angle > 15:
            pose = False
    
    # Return the pose, which would be false if any finger on either hand is perceived to be bent by more than 15 degrees
    return pose

def angleToPlane(vec, plane_N):
    # Calculate the cosine of the angle between the vector and the normal to the plane
    cosine_angle = np.dot(vec, plane_N) / (np.linalg.norm(vec) * np.linalg.norm(plane_N))
    # Use arccos to get angle, convert it to degrees, subtract from
    # 90 to get angle to plane instead of its normal and then return it
    return 90 - np.degrees(np.arccos(cosine_angle))

def scaleValue(val, in_min, in_max, out_min, out_max):
    return (val - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def constrain(val, min, max):
    if val < min:
        return min
    elif val > max:
        return max
    else:
        return val

def controller(left, right):
    # variables for storing the command values for each axis, to control the drone
    tilt_RL, tilt_FwdBck, throttle = 0, 0, 0

    # Check to ensure right hand landmarks and normal vector for right palm plane exist
    if right and type(rightPlane_N) == np.ndarray:
        # Create vector from wrist to base of index finger as a numpy array
        wrist_index_vec = np.array([right.landmark[5].x, right.landmark[5].y, right.landmark[5].z]) - np.array([right.landmark[0].x, right.landmark[0].y, right.landmark[0].z])
        # Calculate angle between this vector and the plane representing the initial orientation of the palm - this angle represents the forwards/backwards tilt
        tilt_FwdBck = -1 * angleToPlane(wrist_index_vec, rightPlane_N)
        # Create vector from base to the tip of the middle finger as a numpy array
        midFingerVec = np.array([right.landmark[12].x, right.landmark[12].y, right.landmark[12].z]) - np.array([right.landmark[9].x, right.landmark[9].y, right.landmark[9].z])
        # Calculate the angle between this vector and the plane orthogonal to the initial orientation of the palm - this angle represents the right/left tilt
        tilt_RL = -1 * angleToPlane(midFingerVec, rightPlane_Orth_N)

        # Correct error in fowards/backwards tilt angle caused by hand rotations that negatively affects the accuracy of 3D coordinates returned by hand pose model
        tilt_FwdBck = tilt_FwdBck + scaleValue(tilt_RL, -40, 40, -10, 10)

        # uncomment to see tilt angle debug output for both axis.
        # print(f"fwd/bck angle: {tilt_FwdBck}, right/left angle: {tilt_RL}")

        # Define deadband region for forwards/backwards tilt. If tilt is +- 10 degrees or less then tilt is set to 0 (i.e. control ignored)
        if abs(tilt_FwdBck) <= 10:
            tilt_FwdBck = 0
        else:
            # Raw tilt angle is more than 10 degrees in either direction, scale this value to a controller value between -90% to 90% and constrain within the same range
            tilt_FwdBck = constrain(int(scaleValue(tilt_FwdBck, -25, 25, -90, 90)), -90, 90)
        # Define deadband region for right/left tilt. If tilt is +- 15 degrees or less then tilt is set to 0 (i.e. control ignored)
        if abs(tilt_RL) <= 15:
            tilt_RL = 0
        else:
            # Raw tilt angle is more than 15 degrees in either direction, scale this value to a controller value between -90% to 90% and constrain within the same range
            tilt_RL = constrain(int(scaleValue(tilt_RL, -40, 40, -90, 90)), -90, 90)
    
    # Check to ensure left hand landmarks exist
    if left:
        # Vector from Tip of thumb to tip of index finger
        thumb_index_vec = np.array([left.landmark[8].x, left.landmark[8].y, left.landmark[8].z]) - np.array([left.landmark[4].x, left.landmark[4].y, left.landmark[4].z])
        # Vector from wrist to base of index finger
        wrist_indexBase_vec = np.array([left.landmark[5].x, left.landmark[5].y, left.landmark[5].z]) - np.array([left.landmark[0].x, left.landmark[0].y, left.landmark[0].z])
        # Ratio between the vectors taken to make the control value invariant to distance of the hand from the camera
        altRatio = round(np.linalg.norm(thumb_index_vec) / np.linalg.norm(wrist_indexBase_vec), 3)
        # Scale the ratio to find the current target altitude between 350mm and 2200mm
        target_altitude = constrain(scaleValue(altRatio, 0.1, 1.5, 350, 2200), 350, 2200)
        # Get the current altitude of the drone
        current_alt = drone.send_command("tof?").rstrip()[:-2]
        # If response is valid (i.e. an integer)
        if current_alt.isdecimal():
            # Convert the response string containing the current altitude, into an actual integer
            current_alt = int(current_alt)
            # Calculate error to be the difference between the target and current altitude
            error = target_altitude - current_alt
            # Deadband for altitude change. Difference must be greater than 15mm for throttle to be given by controller
            if abs(error) > 15:
                # Error is scaled to produce throttle control value between -90% and 90% and constrained in the same range
                throttle = constrain(int(scaleValue(error, -1750, 1750, -90, 90)), -90, 90)

        # uncomment to see altitude related debug outputs
        # print(f"target: {target_altitude}, current: {cur_alt}, throttle: {throttle}")
    
    # Send command to drone with all 3 controller values. Last value is yaw (rotation of the drone) and is always set to 0 as no hand control is mapped to this axis.
    drone.send_command(f"rc {tilt_RL} {tilt_FwdBck} {throttle} 0")


# initialise mediapipe hands model with minimum hand detection confidence of 70% and minimum hand landmarks confidence of 50%
with hands_model.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    # Initialisation countdown timestamp
    init_timer = time.time()
    # Initialised state flag
    initialised = False

    while src.isOpened():
        # If it has been more than 0.5 seconds since last command, then send a
        # default command to make the drone hover in place
        if time.time() - command_timer > 0.5:
            # Command to hover in place
            drone.send_command("rc 0 0 0 0")
            # Reset the timer by setting timestamp to current time
            command_timer = time.time()

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

        left, right = None, None

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

                # Store the hand landmarks in either the `left` or `right` variable depending on which hand it is
                if results.multi_handedness[num].classification[0].label == "Right":
                    right = hand
                else:
                    left = hand
        
        # Check if the program has already completed initialisation before
        if initialised:
            # Initialisation has been completed - pass the set of landmarks for left and right hand from the current video frame to the controller() function
            controller(left, right)
        else:
            # Initialisation not yet completed - Check if left and right hands are in their initial pose
            if not initialPose(frame, left, right):
                # Hands not in intial pose so reset initialisation timer to current time
                init_timer = time.time()
                # Render message onto the video frame to inform the user
                cv.putText(frame, "Bad initial pose. Timer reset.", (int(0.2*frame.shape[1]), frame.shape[0] - 50), cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1, cv.LINE_AA)
            else:
                # Hands are in initial pose, render initialisation message with countdown onto the video frame to inform the user
                cv.putText(frame, "Initialising... " + str(round(5 - (time.time() - init_timer), 2)) + "s", (int(0.3*frame.shape[1]), frame.shape[0] - 50), cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1, cv.LINE_AA)
            
            # If it has been 5s or more since hands have been in initial position then set initialisation values
            if time.time() - init_timer >= 5:
                # Set flag to true to indicate program has completed initialisation phase
                initialised = True
                # Store x, y, and z coordinates of the wrist, base of index finger (landmark 5) and base of pinky finger (landmark 17) of the right hand
                wrist = (right.landmark[0].x, right.landmark[0].y, right.landmark[0].z)
                indexBase = (right.landmark[5].x, right.landmark[5].y, right.landmark[5].z)
                pinkyBase = (right.landmark[17].x, right.landmark[17].y, right.landmark[17].z)
                # Construct a plane using these 3 landmark points - this will be the plane representing the initial orientation of the palm
                rightPlane = Plane(wrist, indexBase, pinkyBase)
                # Store the normal to the plane
                rightPlane_N = np.array(list(map(float, rightPlane.normal_vector)))
                # Construct the perpendicular plane of the right hand palm plane, going through the base (landmark 9) and tip (landmark 12) of the middle finger
                rightPlane_Orth = rightPlane.perpendicular_plane((right.landmark[9].x, right.landmark[9].y, right.landmark[9].z), (right.landmark[12].x, right.landmark[12].y, right.landmark[12].z))
                # Store the normal of the perpendicular plane
                rightPlane_Orth_N = np.array(list(map(float, rightPlane_Orth.normal_vector)))

        # Display the frame with the renderings of the result and hand
        # classification
        cv.imshow("Hand Pose", frame)
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            drone.send_command("rc 0 0 0 0")
            drone.send_command("land")
            break

# Releases the webcam
src.release()
# Closes all the windows opened by the code (such as "Hand Pose")
cv.destroyAllWindows()
