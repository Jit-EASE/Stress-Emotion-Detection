import cv2
import dlib
import numpy as np
import threading
import time
import os
import urllib.request
import bz2
from collections import deque
from deepface import DeepFace
from skimage.transform import radon, iradon
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

# --- Robust Env Loader ---
def load_env_variables():
    """
    Loads .env or Stress_Agentic_AI.env from:
    1. Script directory
    2. Current working directory
    3. Home directory (~)
    """
    script_dir = Path(__file__).parent
    cwd_dir = Path.cwd()
    home_dir = Path.home()

    candidates = [
        script_dir / ".env",
        script_dir / "Stress_Agentic_AI.env",
        cwd_dir / ".env",
        cwd_dir / "Stress_Agentic_AI.env",
        home_dir / ".env",
        home_dir / "Stress_Agentic_AI.env",
    ]

    for env_path in candidates:
        if env_path.exists():
            print(f"[INFO] Loading credentials from: {env_path}")
            load_dotenv(env_path)
            break
    else:
        print("[ERROR] No .env or Stress_Agentic_AI.env file found in script, cwd, or home directory.")
        exit(1)

    api_key = os.getenv("OPENAI_API_KEY")
    project_id = os.getenv("OPENAI_PROJECT")

    if not api_key:
        print("[ERROR] OPENAI_API_KEY missing in env file.")
        exit(1)
    if not project_id:
        print("[ERROR] OPENAI_PROJECT missing in env file.")
        exit(1)

    print(f"[INFO] Loaded API Key: {api_key[:12]}... and Project ID: {project_id}")
    return api_key, project_id

# --- Load Env Variables ---
OPENAI_KEY, OPENAI_PROJECT = load_env_variables()

# --- Initialize OpenAI Client ---
client = OpenAI(api_key=OPENAI_KEY, project=OPENAI_PROJECT)

# --- Test Authentication ---
try:
    models = client.models.list()
    print(f"[INFO] OpenAI Authenticated. Models available: {len(models.data)}")
except Exception as e:
    print(f"[ERROR] Authentication failed: {e}")
    exit(1)

# --- Dlib Landmark Model ---
predictor_path = "/Users/jit/Documents/Forensic_Data_test/shape_predictor_68_face_landmarks.dat"
if not os.path.exists(predictor_path):
    print("[INFO] Downloading Dlib landmark predictor...")
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    compressed_path = predictor_path + ".bz2"
    urllib.request.urlretrieve(url, compressed_path)
    with bz2.BZ2File(compressed_path) as fr, open(predictor_path, 'wb') as fw:
        fw.write(fr.read())
    os.remove(compressed_path)
    print("[INFO] Landmark predictor ready.")

predictor = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()

# --- Camera Setup ---
cap_color = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
cap_depth = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
print("Color camera open:", cap_color.isOpened())
print("Depth camera open:", cap_depth.isOpened())

cv2.namedWindow("Combined Feed", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Combined Feed", 1280, 720)
cv2.namedWindow("Tomogram Slice", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tomogram Slice", 640, 480)

# --- Tomography ---
def simulate_tomogram(depth_img_2d, angles=None):
    if angles is None:
        angles = np.linspace(0., 180., max(depth_img_2d.shape), endpoint=False)
    sinogram = radon(depth_img_2d, theta=angles, circle=True)
    reconstruction = iradon(sinogram, theta=angles, circle=True)
    return reconstruction, sinogram

# --- Eye Aspect Ratio ---
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# --- Memory Buffer ---
memory_window = deque(maxlen=30)
ai_response_text = "Initializing Agentic AI..."
lock = threading.Lock()

# --- Agentic AI Call ---
def fetch_ai_response(emotion, age, gender, ear, memory_list):
    global ai_response_text
    emotions = [m['emotion'] for m in memory_list]
    ears = [m['ear'] for m in memory_list]
    trend_emotion = f"Recent emotions: {', '.join(emotions[-5:])}" if emotions else "No trend yet"
    avg_ear = np.mean(ears) if ears else ear

    prompt = f"""
    You are an Agentic AI analyzing live stress and emotion feeds.
    Current frame: Emotion={emotion}, Age={age}, Gender={gender}, EAR={ear:.2f}
    Emotional trend: {trend_emotion}
    Avg EAR (fatigue indicator): {avg_ear:.2f}

    Provide a short, real-time supportive mental health insight (1 line).
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an Agentic AI providing real-time behavioral insights."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=60,
            temperature=0.6
        )
        with lock:
            ai_response_text = response.choices[0].message.content.strip()
    except Exception as e:
        with lock:
            ai_response_text = f"[AI Error: {e}]"

def run_ai_in_background(emotion, age, gender, ear, memory_list):
    thread = threading.Thread(target=fetch_ai_response, args=(emotion, age, gender, ear, memory_list))
    thread.daemon = True
    thread.start()

# --- Main Loop ---
last_ai_call = 0
last_emotion = None

while True:
    ret_c, frame = cap_color.read()
    ret_d, depth = cap_depth.read()
    if not ret_c or not ret_d:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        coords = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
        left_eye, right_eye = coords[36:42], coords[42:48]
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

        raw_analysis = DeepFace.analyze(frame, actions=['emotion', 'age', 'gender'], enforce_detection=False)
        analysis = raw_analysis[0] if isinstance(raw_analysis, list) else raw_analysis
        emotion, age, gender = analysis['dominant_emotion'], analysis['age'], analysis['gender']

        memory_window.append({"emotion": emotion, "ear": ear, "timestamp": time.time()})

        if emotion != last_emotion or (time.time() - last_ai_call > 5):
            run_ai_in_background(emotion, age, gender, ear, list(memory_window))
            last_ai_call = time.time()
            last_emotion = emotion

        # Tomography
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        depth_gray = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        depth_roi = depth_gray[y1:y2, x1:x2]
        if depth_roi.size:
            recon, _ = simulate_tomogram(depth_roi)
            recon_disp = cv2.normalize(recon, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imshow('Tomogram Slice', recon_disp)

        # Overlay frame info
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Emotion: {emotion}", (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"Age: {age}", (x1, y2+40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"Gender: {gender}", (x1, y2+60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"EAR: {ear:.2f}", (x1, y2+80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # Display AI Insight
    with lock:
        cv2.putText(frame, f"AI: {ai_response_text}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.imshow("Combined Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_color.release()
cap_depth.release()
cv2.destroyAllWindows()
