import cv2
import dlib
import numpy as np
from deepface import DeepFace
import os

def get_predictor_path():
    # Absolute or relative path to the .dat file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    predictor_path = os.path.join(current_dir, "shape_predictor_68_face_landmarks.dat")

    if not os.path.exists(predictor_path):
        raise FileNotFoundError(f"The file {predictor_path} was not found. Please ensure the correct path.")
    return predictor_path

def detect_eyes_and_stress(frame, landmarks):
    left_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                (landmarks.part(37).x, landmarks.part(37).y),
                                (landmarks.part(38).x, landmarks.part(38).y),
                                (landmarks.part(39).x, landmarks.part(39).y),
                                (landmarks.part(40).x, landmarks.part(40).y),
                                (landmarks.part(41).x, landmarks.part(41).y)], np.int32)

    right_eye_region = np.array([(landmarks.part(42).x, landmarks.part(42).y),
                                 (landmarks.part(43).x, landmarks.part(43).y),
                                 (landmarks.part(44).x, landmarks.part(44).y),
                                 (landmarks.part(45).x, landmarks.part(45).y),
                                 (landmarks.part(46).x, landmarks.part(46).y),
                                 (landmarks.part(47).x, landmarks.part(47).y)], np.int32)

    # Calculate eye aspect ratio (EAR) for stress detection
    def eye_aspect_ratio(eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C)

    left_EAR = eye_aspect_ratio(left_eye_region)
    right_EAR = eye_aspect_ratio(right_eye_region)
    EAR = (left_EAR + right_EAR) / 2

    stress_detected = EAR < 0.2  # Threshold for stress detection based on EAR
    return stress_detected

def analyze_face(frame, face_coords):
    x, y, w, h = face_coords
    face_img = frame[y:y + h, x:x + w]

    try:
        # Use DeepFace to analyze the face
        analysis = DeepFace.analyze(face_img, actions=['emotion', 'age', 'gender'], enforce_detection=False)
        return analysis[0]  # DeepFace returns a list of results; we use the first result
    except Exception as e:
        print(f"Error analyzing face: {e}")
        return None

def main():
    predictor_path = get_predictor_path()

    cap = cv2.VideoCapture(0)
    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor(predictor_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray)

        for face in faces:
            landmarks = landmark_predictor(gray, face)
            x, y, w, h = face.left(), face.top(), face.width(), face.height()

            # Analyze emotion, age, and gender with DeepFace
            analysis = analyze_face(frame, (x, y, w, h))

            # Detect stress based on eye landmarks
            stress_detected = detect_eyes_and_stress(frame, landmarks)
            stress_status = "Stressed" if stress_detected else "Relaxed"

            if analysis:
                emotion = analysis['dominant_emotion']
                age = int(analysis['age'])
                gender = analysis['gender']

                # Display results
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f"Emotion: {emotion}, Age: {age}, Gender: {gender}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"Stress: {stress_status}", (x, y - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow('Stress and Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
