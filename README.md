import cv2
import mediapipe as mp
import numpy as np
from mediapipe import Image, ImageFormat 
from mediapipe.tasks import python 
from mediapipe.tasks.python import vision

import urllib.request
urllib.request.urlretrieve(
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "hand_landmarker.task"
)
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions( base_options=base_options, num_hands=1, min_hand_detection_confidence=0.7, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5)
detector = vision.HandLandmarker.create_from_options(options)

def recognise_gesture(landmarks):
    thumb = landmarks[4]
    index = landmarks[8]
    middle = landmarks[12]
    ring = landmarks[16]
    pinky = landmarks[20]
    wrist = landmarks[0]

    def finger_extended(tip, wrist_y):
        return tip.y < wrist.y

    if (finger_extended(thumb, wrist.y) and
           finger_extended(index, wrist.y) and
           finger_extended(middle, wrist.y) and
           finger_extended(ring, wrist.y) and
           finger_extended(pinky, wrist.y)):
           return "Open Palm"
    elif (not finger_extended(thumb, wrist.y) and
             not finger_extended(index, wrist.y) and
             not finger_extended(middle, wrist.y) and
             not finger_extended(ring, wrist.y) and
             not finger_extended(pinky, wrist.y)):
        return "Fist"
        
    elif (finger_extended(thumb, wrist.y) and
             not finger_extended(index, wrist.y) and
             not finger_extended(middle, wrist.y) and
             not finger_extended(ring, wrist.y) and
             not finger_extended(pinky, wrist.y)):
        return "Thumbs up"

    elif (not finger_extended(thumb, wrist.y) and
             finger_extended(index, wrist.y) and
             finger_extended(middle, wrist.y) and
             not finger_extended(ring, wrist.y) and
             not finger_extended(pinky, wrist.y)):
        return "Peace Sign"

    return "Unknown"

    cap = cv2.VideoCapture(0)

    while True:
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = vision.Image(image_format=vision.ImageFormat.SRGB, data=frame_rgb)
    results = detector.detect(mp_image)

    if results.multi_hand_landmatrks:
        for hand_landmark in multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmark.landmark
            gesture = recognize_gesture(landmarks)
            cv2.putText(frame, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 0), CV2.LINE_AA)

    cv2.imshow("HAND GESTURE RECOGNITION", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
