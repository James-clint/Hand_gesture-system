import cv2
import mediapipe as mp
import numpy as np
from mediapipe import Image, ImageFormat 
from mediapipe.tasks import python 
from mediapipe.tasks.python import vision


base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions( base_options=base_options, num_hands=1, min_hand_detection_confidence=0.7, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5)
detector = vision.HandLandmarker.create_from_options(options)

def recognise_gesture(landmarks):
    def finger_extended(tip_id, pip_id):
        return landmarks[tip_id].y < landmarks[pip_id].y
        
    def thumb_extended():
        return landmarks[4].x < landmarks[3].x
            
    thumb = thumb_extended()
    index = finger_extended(8, 6)
    middle = finger_extended(12, 10)
    ring = finger_extended(16, 14)
    pinky = finger_extended(20, 18)

    if all([thumb, index, middle, ring, pinky]):
        return "Open Palm"
    elif not any([thumb, index, middle, ring, pinky]):
        return "Fist"
    elif thumb and not any([index, middle, ring, pinky]):
        return "Thumbs Up"
    elif index and middle and not ring and not pinky:
        return "Peace Sign"
    elif pinky and not any([thumb, index, middle, ring]):
        return "Pinky"
    else:
        return "Unknown"

    cap = cv2.VideoCapture(0)

    while True:
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = vision.Image(image_format=vision.ImageFormat.SRGB, data=frame_rgb)
    results = detector.detect(mp_image)

   if results.hand_landmarks:
        for hand_landmark in results.hand_landmarks:
            points = []
            for landmark in hand_landmark:
                cx, cy = int(landmark.x * width), int(landmark.y * height) 
                points.append((cx, cy)) 
                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
            if len(points) >= 9:
                cv2.line(frame, points[0], points[8], (255, 0, 0), 2)
            #landmark = hand_landmark.landmark
            gesture = recognise_gesture(hand_landmark)
            cv2.putText(frame, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow("HAND GESTURE RECOGNITION", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()
