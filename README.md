
Stress, Emotion & Tomography Eye Tracking — Econometric Integration Framework

Designed and Developed by: Shubhojit Bagchi

Email: 124107294@umail.ucc.ie

Institution: University College Cork, Ireland


Project Context
This prototype forms part of the MSc Business Economics Report 2025:
“Evaluating the Effectiveness of Corporate Mental Health Interventions on Employee Well-being and Productivity in Ireland.”

Developed as an econometric data augmentation and validation system, the framework integrates:

-- Facial analytics (emotion, demographic, blink-frequency analysis)

-- Stress and cognitive load estimation

-- Simulated tomographic imaging using LiDAR depth data from the iPhone 13 Pro Max (Continuity Camera)

The system’s purpose is to feed real-time sensor-based behavioural metrics into econometric models, enabling more precise causal inference.

Platform: Built on Dash (Plotly) for real-time multi-component econometric visualisation and statistical feedback, ensuring modular deployment and reproducible analysis workflows.

Econometric Context & Methodological Integration

-- Primary Role: Auxiliary data capture and variable validation in econometric models.

-- Analytical Linkage: Processed stress/emotion metrics matched to KPIs of official datasets.

-- Objective: Improve internal validity by triangulating traditional survey/administrative data with behavioural indicators, reducing measurement error.


-- Tool Portability with full potential and low risk under EU AI ACT - Econometric modelling in Agri-Food Systems Research, Agri-Tech, Supply Chain/Logistics and Environment/Climate Research

-- Tool Portability with limited potential with moderate to high risk according to EU AI ACT - Econometric Modeling in Medical Research, Workplace and Education (Live Human Monitoring)


Core Computational Modules

-- DeepFace — Emotion, age, and gender classification 

-- Eye Aspect Ratio (EAR) — Blink pattern-based stress estimation

-- Radon Transform Tomography — LiDAR ROI depth capture → projection sinogram → inverse Radon reconstruction for real-time layered imaging of cognitive load proxies

-- Agentic AI - Contextual Reference (powered by Open AI GPT 4o-mini - text based model to stay compliant with EU AI ACT 2024 and GDPR)

System Features
-- Real-Time Video Capture: macOS Continuity Camera with LiDAR-enabled iPhone

-- Facial Landmark Detection: dlib 68-point predictor for sub-millimetre precision

-- Dual-Window Dash Interface: Live overlay with real-time tomographic rendering and econometric output panel

-- Stress & Emotion Analysis: Integrated EAR and DeepFace metrics with dynamic R² and p-value updates (live econometric modelling)

Econometric Application

Experimental Monitoring: Real-time intervention effect tracking for behavioural economics field studies

Portability with full potential and low risk under EU AI ACT - Agri Food Systems Research, Agri-Tech, Supply Chain/Logistics and Environment/Climate Research

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
