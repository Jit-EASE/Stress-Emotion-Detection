#Stress and Emotion Detection using Deepface and Tomography
#Device Used for Testing -- iPhone 13 Pro Max with LiDAR Sensor
#Prototype Developed and Designed by Shubhojit Bagchi -- 124107294@umail.ucc.ie - University College Cork, Ireland
#Prototype Developed as part of MSc. Business Economics Report - "Evaluating the Effectiveness of Corporate Mental Health Interventions on Employee Well-being and Productivity in Ireland"

import cv2
import dlib
import numpy as np
from deepface import DeepFace
from skimage.transform import radon, iradon

# --- Camera setup (Continuity Camera via iPhone 13 Pro Max Triple Camera & LiDAR Sensor) ---
cap_color = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
# If AVFoundation fails for depth, you can switch to GStreamer pipeline (uncomment below):
# depth_pipeline = (
#     'avfvideosrc device-index=1 ! '
#     'video/x-raw,format=GRAY8,width=1280,height=720 ! appsink drop=true'
# )
cap_depth = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

print("Color camera open:", cap_color.isOpened())
print("Depth camera open:", cap_depth.isOpened())

# --- Window setup ---
cv2.namedWindow("Combined Feed", cv2.WINDOW_NORMAL)    
cv2.resizeWindow("Combined Feed", 1280, 720)
cv2.namedWindow("Tomogram Slice", cv2.WINDOW_NORMAL)   
cv2.resizeWindow("Tomogram Slice", 640, 480)

# --- Dlib face & landmark detector ---
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# --- Tomography simulation routines ---
def simulate_tomogram(depth_img_2d, angles=None):
    """
    Perform Radon transform and inverse to simulate tomographic slice.
    """
    if angles is None:
        angles = np.linspace(0., 180., max(depth_img_2d.shape), endpoint=False)
    sinogram = radon(depth_img_2d, theta=angles, circle=True)
    reconstruction = iradon(sinogram, theta=angles, circle=True)
    return reconstruction, sinogram

# --- Eye aspect ratio computation ---
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# --- Main loop ---
while True:
    ret_c, frame = cap_color.read()
    ret_d, depth = cap_depth.read()
    if not ret_c:
        print("Failed to grab color frame")
        break
    if not ret_d:
        print("Failed to grab depth frame")
        break

    # Ensure depth is single-channel 2D
    if depth.ndim == 3:
        depth_gray = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
    else:
        depth_gray = depth

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        # Extract landmarks
        shape = predictor(gray, face)
        coords = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
        left_eye = coords[36:42]
        right_eye = coords[42:48]
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

        # DeepFace analysis (emotion, age, gender)
        raw_analysis = DeepFace.analyze(frame, actions=['emotion', 'age', 'gender'], enforce_detection=False)
        # DeepFace.analyze may return a list if multiple faces; ensure dict
        analysis = raw_analysis[0] if isinstance(raw_analysis, list) else raw_analysis

        # Tomographic simulation on depth ROI
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        depth_roi = depth_gray[y1:y2, x1:x2]
        if depth_roi.size and depth_roi.ndim == 2:
            recon, _ = simulate_tomogram(depth_roi)
            recon_disp = cv2.normalize(recon, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imshow('Tomogram Slice', recon_disp)

        # Overlay results on color frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Emotion: {analysis['dominant_emotion']}", (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"Age: {analysis['age']}", (x1, y2+40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"Gender: {analysis['gender']}", (x1, y2+60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"EAR: {ear:.2f}", (x1, y2+80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow("Combined Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_color.release()
cap_depth.release()
cv2.destroyAllWindows()
