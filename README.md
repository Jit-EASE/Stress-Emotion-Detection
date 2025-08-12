
Stress, Emotion & Tomography Eye Tracking — Econometric Integration Framework
Designed and Developed by: Shubhojit Bagchi
Email: 124107294@umail.ucc.ie
Institution: University College Cork, Ireland

Project Context
This prototype forms part of the MSc Business Economics Report 2025:
“Evaluating the Effectiveness of Corporate Mental Health Interventions on Employee Well-being and Productivity in Ireland.”

Developed as an econometric data augmentation and validation system, the framework integrates:

Facial analytics (emotion, demographic, blink-frequency analysis)

Stress and cognitive load estimation

Simulated tomographic imaging using LiDAR depth data from the iPhone 13 Pro Max (Continuity Camera)

The system’s purpose is to feed real-time behavioural metrics into corporate productivity and well-being econometric models, enabling more precise causal inference.

Platform: Built on Dash (Plotly) for real-time multi-component econometric visualisation and statistical feedback, ensuring modular deployment and reproducible analysis workflows.

Econometric Context & Methodological Integration
Primary Role: Auxiliary data capture and variable validation for Difference-in-Differences (DiD) and panel regression models.

Analytical Linkage: Processed stress/emotion metrics are matched to firm-level KPIs (productivity, absenteeism) in sector-size stratified datasets.

Objective: Improve internal validity by triangulating traditional survey/administrative data with behavioural indicators, reducing measurement error.

Core Computational Modules
DeepFace — Emotion, age, and gender classification (TensorFlow backend)

Eye Aspect Ratio (EAR) — Blink pattern-based stress estimation

Radon Transform Tomography — LiDAR ROI depth capture → projection sinogram → inverse Radon reconstruction for real-time layered imaging of cognitive load proxies

System Features
Real-Time Video Capture: macOS Continuity Camera with LiDAR-enabled iPhone

Facial Landmark Detection: dlib 68-point predictor for sub-millimetre precision

Dual-Window Dash Interface: Live overlay with real-time tomographic rendering and econometric output panel

Stress & Emotion Analysis: Integrated EAR and DeepFace metrics with dynamic R² and p-value updates

Econometric Applications
Variable Validation: Behavioural response variables for DiD analysis of corporate well-being interventions

Model Enhancement: Generation of additional explanatory variables for multi-sector panel regressions

Experimental Monitoring: Real-time intervention effect tracking for behavioural economics field studies

Technical Requirements
Python 3.10+

macOS with Continuity Camera (LiDAR-enabled iPhone)

Portable to compatible biometric or medical sensor feeds

Repository Scope
GitHub Repository Contents:

Full Python/Pandas implementation scripts (open-science compliant)

Baseline, Enhanced, and RealWorld datasets (imputed and cleaned)

Complete data transformation, imputation, and cleaning strategy documentation

Stress & Emotion Detection Tool source code and econometric integration modules
