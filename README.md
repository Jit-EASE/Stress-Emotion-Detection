Stress, Emotion & Tomography Eye Tracking — Econometric Integration Framework
Developer: Shubhojit Bagchi
Email: 124107294@umail.ucc.ie
Institution: University College Cork, Ireland

Project Context:
Prototype developed as part of the MSc Business Economics Report 2025 — "Evaluating the Effectiveness of Corporate Mental Health Interventions on Employee Well-being and Productivity in Ireland".

This system fuses facial analytics, stress/emotion tracking, and a simulated tomographic imaging layer using LiDAR depth data from the iPhone 13 Pro Max (Continuity Camera). It forms part of a broader econometric research infrastructure integrating real-time behavioural metrics into corporate productivity and well-being analysis.

Methodological Integration
Econometric Context:
This application operates as an auxiliary data collection and validation tool for Difference-in-Differences (DiD) and panel regression models assessing workplace mental health programme effectiveness.
Processed stress/emotion data can be linked to productivity KPIs and absenteeism rates, enabling sector-specific econometric modelling.

Core Computational Modules:

DeepFace — Emotion, age, and gender classification.

Eye Aspect Ratio (EAR) — Stress estimation via blink frequency/stress correlation.

Radon Transform Tomography — Simulated imaging slices from LiDAR ROI depth data, providing a layered visual proxy for cognitive load analysis.

System Features
Real-time Video Feed via macOS Continuity Camera (LiDAR-enabled iPhone).

Facial Landmark Detection — dlib’s 68-point predictor for high-precision tracking.

Emotion & Demographic Analysis — DeepFace (TensorFlow backend).

Stress Estimation — EAR metric for blink pattern analysis.

Tomographic Simulation —

LiDAR ROI extraction.

Projection sinogram generation (Radon transform).

Inverse Radon reconstruction for real-time slice rendering.

Dual-Window Interface — Live overlay + tomographic visualisation.

Econometric Applications
Validate behavioural response variables in corporate well-being studies.

Generate additional explanatory variables for sector-size stratified panel models.

Facilitate experimental modules for real-time intervention monitoring.

Technical Requirements
Python 3.10+

macOS with Continuity Camera (LiDAR-enabled iPhone) — portable to real-time medical/biometric sensors.

Libraries:

opencv-python (AVFoundation backend)

dlib (with shape_predictor_68_face_landmarks.dat)

deepface (TensorFlow backend)

scikit-image (Radon/Inverse Radon transforms)

numpy

Repository Scope
GitHub Repository for validation if required:

Stress & Emotion Detection Tool & Econometric Application Codebase

Imputed and Cleaned Datasets (Baseline, Enhanced, RealWorld)

Full Imputation, Transformation, and Cleaning Strategy Documentation

Python/Pandas implementation scripts with open-science reproducibility compliance.

