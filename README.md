Stress, Emotion & Tomography Eye Tracking

-- Developed and Designed by Shubhojit Bagchi -- 124107294@umail.ucc.ie - University College Cork, Ireland 

-- Prototype Developed as part of MSc. Business Economics Report 2025 - "Evaluating the Effectiveness of Corporate Mental Health Interventions on Employee Well-being and Productivity in Ireland"

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

-- Python 3.10+

-- macOS with Continuity Camera (iPhone LiDAR capable) / It can be ported to real-time medical sensors.  

-- OpenCV (with AVFoundation backend)

-- Dlib (with shape_predictor_68_face_landmarks.dat)

-- DeepFace (TensorFlow backend)

-- scikit-image (for Radon/Inverse Radon transforms - Tomography)

-- numpy

BER Overview

The project employs a Difference-in-Differences (DiD) econometric model, integrating real-world datasets (Enterprise Ireland, CSO, BEEPS, SilverCloud, Laya Healthcare) and AI-enhanced simulations (DeepFaceit, Tomographit) to measure the causal impact of structured workplace mental health interventions.

---

Repository Features**
- Data transformations: Standardisation, log scaling, and NACE sector aggregation (Pandas).
- Imputation routines: Median sectoral imputation, hot-deck (KNN), and regression-based completion (Scikit-learn & Statsmodels).
- AI simulation modules: Monte Carlo adoption scenarios calibrated with Lenze et al. (2022) and Hawrilenko (2025).
- Diagnostics: VIF checks, PCA decomposition, placebo falsification, and sector-specific sub-analyses.
- Econometric modelling: Fully robust DiD regression with clustered HC3 errors and sectoral fixed effects.

---

Dependencies**
This project is built with:
- Python 3.11
- Pandas (v2.0)
- Statsmodels (v0.14)
- Scikit-learn (v1.4)
- Seaborn & Matplotlib (visualization)
- NumPy & SciPy (scientific computation)

