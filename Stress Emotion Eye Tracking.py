import os
import time
import base64
import threading

import cv2
import numpy as np
import pandas as pd
import statsmodels.api as sm
import plotly.graph_objs as go
import openai

from dash import Dash, dcc, html, Input, Output, State
import dash
import flask

# ─────────────────────────────────────────────────────────────────────────────
# Configuration: pick up your GPT key from env
openai.api_key = os.getenv("OPENAI_API_KEY", "")
if not openai.api_key:
    raise RuntimeError("Set your OPENAI_API_KEY in the environment")

# ─────────────────────────────────────────────────────────────────────────────
# Global data storage
df = pd.DataFrame(columns=["t", "face_count", "sensor"])
csv_path = "live_data.csv"
df.to_csv(csv_path, index=False)

# ─────────────────────────────────────────────────────────────────────────────
# Camera capture thread
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
latest_frame = None
frame_lock = threading.Lock()

def capture_frames():
    global latest_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        with frame_lock:
            latest_frame = frame.copy()
        time.sleep(0.03)

threading.Thread(target=capture_frames, daemon=True).start()

# ─────────────────────────────────────────────────────────────────────────────
# Create Dash app (light mode only)
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div(
    style={"backgroundColor": "#ffffff", "color": "#000000", "fontFamily": "Arial"},
    children=[
        html.H1("Live Face & Sensor Dashboard"),

        # Video + Tomogram side by side
        html.Div(
            style={"display": "flex", "gap": "20px"},
            children=[
                html.Div([
                    html.Img(
                        id="live-camera",
                        style={"border": "2px solid green", "width": "320px", "height": "240px"},
                    ),
                    dcc.Interval(id="interval-frame", interval=100, n_intervals=0),
                ]),
                html.Div([
                    html.Img(
                        id="tomogram",
                        style={"border": "2px solid blue", "width": "320px", "height": "240px"},
                    ),
                    dcc.Interval(id="interval-tomo", interval=200, n_intervals=0),
                ]),
            ],
        ),

        # Sensor and regression plots
        html.Div([
            dcc.Graph(id="sensor-waveform"),
            dcc.Graph(id="face-regression"),
            dcc.Interval(id="interval-data", interval=500, n_intervals=0),
        ]),

        html.Hr(),
        # OpenAI chat
        html.Div([
            html.H3("Ask the Agent:"),
            dcc.Input(id="agent-prompt", type="text", style={"width": "80%"}),
            html.Button("Send", id="agent-send"),
            html.Div(
                id="agent-response",
                style={
                    "whiteSpace": "pre-wrap",
                    "border": "1px solid #ccc",
                    "padding": "10px",
                    "marginTop": "10px"
                },
            ),
        ]),
    ]
)

# ─────────────────────────────────────────────────────────────────────────────
# 1) DeepFace video callback
@app.callback(
    Output("live-camera", "src"),
    Input("interval-frame", "n_intervals")
)
def update_frame(n):
    with frame_lock:
        frame = latest_frame.copy() if latest_frame is not None else None
    if frame is None:
        return dash.no_update

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    _, buffer = cv2.imencode(".png", frame)
    encoded = base64.b64encode(buffer).decode()
    return f"data:image/png;base64,{encoded}"

# ─────────────────────────────────────────────────────────────────────────────
# 2) Tomogram slice callback
@app.callback(
    Output("tomogram", "src"),
    Input("interval-tomo", "n_intervals")
)
def update_tomogram(n):
    # === Replace this with your tomogram-generation logic ===
    tomo_path = "current_slice.png"
    if os.path.exists(tomo_path):
        with open(tomo_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        return f"data:image/png;base64,{b64}"

    # fallback: blank image
    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".png", blank)
    b64 = base64.b64encode(buf).decode()
    return f"data:image/png;base64,{b64}"

# ─────────────────────────────────────────────────────────────────────────────
# 3) Sensor waveform & face-vs-sensor regression
@app.callback(
    Output("sensor-waveform", "figure"),
    Output("face-regression", "figure"),
    Input("interval-data", "n_intervals"),
)
def update_data(n):
    global df
    t = time.time()
    sensor_val = float(1.0 + 0.5 * np.sin(t) + 0.1 * np.random.randn())

    with frame_lock:
        frame = latest_frame.copy() if latest_frame is not None else None
    face_count = 0
    if frame is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_count = len(face_cascade.detectMultiScale(gray, 1.3, 5))

    # append new row
    df.loc[len(df)] = {"t": t, "face_count": face_count, "sensor": sensor_val}
    df.to_csv(csv_path, index=False)

    # Sensor plot
    fig1 = go.Figure(go.Scatter(x=df["t"], y=df["sensor"], mode="lines", name="Sensor"))
    fig1.update_layout(title="Live Sensor Waveform", xaxis_title="Time", yaxis_title="Sensor")

    # Regression plot
    fig2 = go.Figure()
    if len(df) >= 2:
        X = sm.add_constant(df["sensor"])
        model = sm.OLS(df["face_count"], X).fit()
        intercept, slope = model.params
        line_y = intercept + slope * df["sensor"]
        fig2.add_traces([
            go.Scatter(x=df["sensor"], y=df["face_count"], mode="markers", name="Data"),
            go.Scatter(x=df["sensor"], y=line_y, mode="lines",
                       name=f"y={slope:.2f}x+{intercept:.2f}")
        ])
        stats = f"R²={model.rsquared:.3f}, p={model.pvalues['sensor']:.3g}"
    else:
        stats = "Collecting data..."
    fig2.update_layout(
        title=f"Face Count vs. Sensor ({stats})",
        xaxis_title="Sensor",
        yaxis_title="Face Count",
    )

    return fig1, fig2

# ─────────────────────────────────────────────────────────────────────────────
# 4) OpenAI chat callback
@app.callback(
    Output("agent-response", "children"),
    Input("agent-send", "n_clicks"),
    State("agent-prompt", "value"),
    prevent_initial_call=True,
)
def ask_agent(n, prompt):
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
