Stress, Emotion & Tomography Eye Tracking

-- Developed and Designed by Shubhojit Bagchi -- 124107294@umail.ucc.ie - University College Cork, Ireland | LinkedIn: https://linkedin.com/in/shubhojitbagchi

-- Prototype Developed as part of MSc. Business Economics Report - "Evaluating the Effectiveness of Corporate Mental Health Interventions on Employee Well-being and Productivity in Ireland"

This project fuses facial analytics, stress/emotion tracking, and a simulated tomographic imaging layer using LiDAR depth data from iPhone 13 Pro Max (Continuity Camera).

It integrates DeepFace for emotion/age/gender detection, Eye Aspect Ratio (EAR) for stress estimation, and Radon transform-based tomography to reconstruct depth slices in real-time.
Features
-- Real-time Video Feed: Powered by macOS Continuity Camera (iPhone LiDAR-enabled).

-- Face Detection & Eye Tracking using dlib’s 68 facial landmarks.

-- Emotion, Age, Gender Analysis via DeepFace.

-- Stress Estimation: Uses Eye Aspect Ratio (EAR) for blink/stress correlation.

Tomographic Simulation:

-- Extracts LiDAR depth ROI (Region of Interest) of the face.

-- Applies Radon transform to generate projection sinograms.

-- Performs inverse Radon reconstruction to create real-time tomographic slices.

Dual-window interface: Live facial overlay + separate tomography visualization.

Requirements
Python 3.10+

macOS with Continuity Camera (iPhone LiDAR capable) / It can be ported to real-time medical sensors.  

OpenCV (with AVFoundation backend)

Dlib (with shape_predictor_68_face_landmarks.dat)

DeepFace (TensorFlow backend)

scikit-image (for Radon/Inverse Radon transforms)

numpy
