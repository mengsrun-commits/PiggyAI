from flask import Flask, render_template, Response, request, jsonify
import cv2 as cv
from cv2 import aruco
import numpy as np
from datetime import datetime
import math
import pygame
import os
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
import librosa
import threading
import time
import random
app = Flask(__name__, template_folder='.', static_folder='static')
# ==========================================
# 1. GLOBAL CONFIGURATION & STATE
# ==========================================
# Video State
output_frame = None
video_lock = threading.Lock()
MAX_SEQ_LENGTH = 10
label_names = ["Pneumonia", "Background Noise", "PRRS", "Swine Fever"]
AUDIO_FOLDER = "test_audio"
# State variables shared between Video Loop, AI Thread, and Flask Routes
current_pred_text = "System Ready"
current_pred_color = (255, 255, 255)
prediction_timestamp = 0
ai_lock = threading.Lock()
selected_id = -1 # -1: None, None: All, 0-9: Specific
flip_mode = -1 # None: No flip, 0: Vertical, 1: Horizontal, -1: Both
# Drawing Config
box_margin = 30         # pixels to expand the marker bounding box
box_thickness = 3       # thickness of the red box
DELAY_SECONDS = 1.5     
CENTER_TOLERANCE = 0.6  # How close to (6.5, 6.5) counts as "center" (in grid units)
camera_index = 0     # Default camera index, 1 for webcam

# Initialize Audio
pygame.mixer.init()
# Markers & Grid
marker_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)
param_markers = aruco.DetectorParameters()
detector = aruco.ArucoDetector(marker_dict, param_markers)
grid_rows = 12
grid_cols = 12
show_grid = True
marker_data = {} # Stores tracking history

ID_TO_FILENAME = {
    1: "pnu_1_noisy.wav", 2: "prrs_17_noisy.wav", 3: "sf_17_noisy.wav",
    4: "pnu_1_slower.wav", 5: "prrs_17_pitch_shifted.wav", 6: "sf_17_shifted.wav",
    7: "pnu_1_stretched.wav", 8: "prrs_17_stretched.wav", 9: "sf_17_slower.wav",
}
# Mic Physics State
last_static_mic_values = None
last_static_active = None
physics_active = False
physics_mic_values = None
physics_target_values = None
physics_last_time = time.time()
physics_tau = 0.35
# Mic Regions
mic_bases = {
    1: [(6, 1), (1, 7)], 2: [(1, 7), (6, 12)],
    3: [(6, 12), (12, 7)], 4: [(6, 1), (12, 7)],
}
region_mics = {1: (1, 2), 2: (2, 3), 3: (3, 4), 4: (1, 4)}
mic_positions = {
    1: (6.0, 1.0),
    2: (1.0, 7.0),
    3: (6.0, 12.0),
    4: (12.0, 7.0)
}
# ==========================================
# 2. AI MODEL LOADING & WARMUP
# ==========================================
print("Loading AI Models... (Please wait)")
# 1. Load YAMNet
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
# 2. Load Custom Model
try:
    model = load_model("best_rnn_model.keras")
    print("Custom RNN loaded.")
except:
    print("WARNING: 'best_rnn_model.keras' not found.")
    model = None
# --- 3. AGGRESSIVE WARMUP ROUTINE ---
if model:
    print("WARMING UP: Optimizing TensorFlow Graph & Audio Libs...")
   
    # A. Warm up Librosa (Audio processing)
    # Librosa caches FFT kernels on the first run. We force it now.
    dummy_audio = np.zeros(16000, dtype=np.float32) # 1 second of silence
    librosa.feature.melspectrogram(y=dummy_audio, sr=16000)
   
    # B. Warm up YAMNet (Feature Extractor)
    # This builds the graph for the specific input shape
    _, dummy_embeddings, _ = yamnet_model(dummy_audio)
   
    # C. Warm up Custom RNN (Classifier)
    # We must match the exact shape (1, 10, 1024)
    dummy_input = np.zeros((1, MAX_SEQ_LENGTH, 1024), dtype=np.float32)
   
    # Run it TWICE.
    # The first run builds the graph. The second run confirms memory allocation.
    model.predict(dummy_input, verbose=0)
    model.predict(dummy_input, verbose=0)
   
    print("WARMUP COMPLETE: System is hot and ready.")
# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def _init_physics_from_snapshot(snapshot_vals):
    global physics_active, physics_mic_values, physics_target_values, physics_last_time
    if not snapshot_vals:
        physics_active = False; return
    physics_target_values = list(snapshot_vals)
    physics_mic_values = list(snapshot_vals)
    physics_last_time = time.time()
    physics_active = True

def _update_physics():
    global physics_mic_values, physics_last_time
    if not physics_active or physics_mic_values is None: return physics_mic_values
    now = time.time()
    dt = max(1e-6, now - physics_last_time)
    alpha = 1 - math.exp(-dt / physics_tau)
    for i in range(len(physics_mic_values)):
        physics_mic_values[i] += (physics_target_values[i] - physics_mic_values[i]) * alpha
        physics_mic_values[i] += random.uniform(-0.08, 0.08)
        physics_mic_values[i] = max(0.0, min(120.0, physics_mic_values[i]))
    physics_last_time = now
    return physics_mic_values

def compute_static_mic_values_from_entry(entry):
    if not entry or entry.get('region') is None:
        return [random.uniform(30, 45) for _ in range(4)]
    region = entry.get('region')
    # Special handling for dead center
    if region == "center":
        base = random.uniform(30, 40)
        return [
            base + random.uniform(-5, 5),
            base + random.uniform(-5, 5),
            base + random.uniform(-5, 5),
            base + random.uniform(-5, 5)
        ]
    
    region_pair = region_mics.get(region, (1, 2))
    active_indices = [m - 1 for m in region_pair]
    mic_values = []
    
    if 'pos_x' in entry and 'pos_y' in entry:
        pos_x = entry['pos_x']
        pos_y = entry['pos_y']
        dists = []
        for m in region_pair:
            mx, my = mic_positions[m]
            dist = math.sqrt((pos_x - mx)**2 + (pos_y - my)**2)
            dists.append(dist)
        min_idx = np.argmin(dists)
        closer_m = region_pair[min_idx]
        farther_m = region_pair[1 - min_idx]
        for i in range(4):
            if i + 1 == closer_m:
                val = random.uniform(70, 80)
            elif i + 1 == farther_m:
                val = random.uniform(50, 69)
            else:
                val = random.uniform(10, 30)
            mic_values.append(float(val))
    else:
        for i in range(4):
            if i in active_indices:
                val = random.uniform(70, 80) if i == active_indices[0] else random.uniform(50, 69)
            else:
                val = random.uniform(10, 30)
            mic_values.append(float(val))
    return mic_values

def draw_sidebar(sidebar, mic_values):
    """
    Draws the mic levels onto a dedicated black sidebar image.
    """
    H, W = sidebar.shape[:2]
    padding = 20
   
    # Title
    cv.putText(sidebar, "LIVE STATS", (padding, 40), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv.line(sidebar, (padding, 55), (W - padding, 55), (100, 100, 100), 1)
    # Draw Bars
    bar_h = 30
    gap = 20
    start_y = 100
   
    for i, val in enumerate(mic_values):
        y = start_y + i * (bar_h + gap)
       
        # Label (Mic 1, Mic 2...)
        cv.putText(sidebar, f"Mic {i+1}", (padding, y + 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
       
        # Bar Background (Gray)
        bar_x = 90
        max_bar_w = W - bar_x - padding
        cv.rectangle(sidebar, (bar_x, y), (bar_x + max_bar_w, y + bar_h), (50, 50, 50), cv.FILLED)
       
        # Active Bar (Color changes based on volume)
        fill_w = int((val / 120.0) * max_bar_w) # Assuming 120 is max dB
        fill_w = min(fill_w, max_bar_w)
       
        if val < 50: color = (50, 255, 50)   # Green
        elif val < 80: color = (0, 255, 255) # Yellow
        else: color = (0, 0, 255)            # Red
           
        cv.rectangle(sidebar, (bar_x, y), (bar_x + fill_w, y + bar_h), color, cv.FILLED)
       
        # Text Value
        cv.putText(sidebar, f"{val:.0f}", (bar_x + 5, y + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    # AI Prediction Area (Bottom of sidebar)
    cv.line(sidebar, (padding, H - 150), (W - padding, H - 150), (100, 100, 100), 1)
    cv.putText(sidebar, "AI DIAGNOSIS:", (padding, H - 120), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
   
    # Use the global variable for the text
    with ai_lock:
        text = current_pred_text
        color = current_pred_color
       
    # Wrap text if it's too long
    font_scale = 0.7
    cv.putText(sidebar, text, (padding, H - 80), cv.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

def classify_point(x, y):
    # Handle dead center first
    if abs(x - 6.5) <= CENTER_TOLERANCE and abs(y - 6.5) <= CENTER_TOLERANCE:
        return "center"  # Special case: equidistant to all 4 mics
    
    if 1 <= x <= 6 and 1 <= y <= 6: return 1
    elif 1 <= x <= 6 and 7 <= y <= 12: return 2
    elif 7 <= x <= 12 and 7 <= y <= 12: return 3
    elif 7 <= x <= 12 and 1 <= y <= 6: return 4
    return None

def draw_grid(frame, size, cell_size):
    """
    Draws grid on a square frame.
    Start coordinates are (0,0) because the frame is already cropped.
    """
    # Vertical lines
    for i in range(1, grid_cols):
        x = int(i * cell_size)
        cv.line(frame, (x, 0), (x, size), (150, 150, 150), 1)
    # Horizontal lines
    for i in range(1, grid_rows):
        y = int(i * cell_size)
        cv.line(frame, (0, y), (size, y), (150, 150, 150), 1)
    # Draw X labels (centered in the top cell for each column)
    for i in range(grid_cols):
        center_x = int((i + 0.5) * cell_size)
        center_y = int(0.5 * cell_size)  # center vertically in the first/top cell
        label = str(i + 1)
        (tw, th), baseline = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        tx = int(center_x - tw / 2)
        ty = int(center_y + th / 2)  # baseline-adjusted vertical center
        # clamp so text doesn't go above the top or below the image
        ty = max(ty, th + 2)
        ty = min(ty, size - 2)
        tx = max(2, min(tx, size - tw - 2))
        cv.putText(frame, label, (tx, ty), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    # Draw Y labels (centered in each row cell)
    for i in range(grid_rows):
        center_y = int((i + 0.5) * cell_size)
        center_x = int(0.5 * cell_size)  # center horizontally in the first/left cell
        label = str(i + 1)
        (tw, th), baseline = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        tx = int(center_x - tw / 2)
        ty = int(center_y + th / 2)
        # clamp to avoid top/bottom clipping
        ty = max(ty, th + 2)
        ty = min(ty, size - 2)
        tx = max(2, min(tx, size - tw - 2))
        cv.putText(frame, label, (tx, ty), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

# ==========================================
# 4. CORE AI & VIDEO LOGIC
# ==========================================
def run_ai_prediction(filename, triggered_id):
    global current_pred_text, current_pred_color, prediction_timestamp, last_static_mic_values
   
    with ai_lock:
        current_pred_text = "Analyzing..."
        current_pred_color = (255, 255, 0)
        prediction_timestamp = 0
    try:
        filepath = os.path.join(AUDIO_FOLDER, filename)
        if not os.path.exists(filepath): raise Exception("File not found")
        waveform, sr = librosa.load(filepath, sr=16000, mono=True)
        _, embeddings, _ = yamnet_model(waveform)
        padded_embeddings = tf.keras.preprocessing.sequence.pad_sequences(
            [embeddings.numpy()], maxlen=MAX_SEQ_LENGTH, dtype='float32', padding='post', truncating='post'
        )
       
        if model:
            pred = model.predict(padded_embeddings, verbose=0)[0]
            best_index = np.argmax(pred)
            best_label = label_names[best_index]
            confidence = pred[best_index] * 100
           
            with ai_lock:
                current_pred_text = f"{best_label} ({confidence:.1f}%)"
                current_pred_color = (0, 255, 0) if best_label == "Background Noise" else (0, 0, 255)
                prediction_timestamp = time.time()
               
                # Snap physics to this marker
                chosen_entry = marker_data.get(triggered_id) if triggered_id in marker_data else None
                mic_vals = compute_static_mic_values_from_entry(chosen_entry)
                last_static_mic_values = mic_vals
                _init_physics_from_snapshot(mic_vals)
               
    except Exception as e:
        print(f"AI Error: {e}")
        with ai_lock:
            current_pred_text = "Error"
            current_pred_color = (100, 100, 100)

def process_video_feed():
    """
    Background thread: Reads camera, runs CV logic, updates global 'output_frame'
    """
    global output_frame, marker_data, last_static_mic_values, camera_index
    
    # Initialize Camera ONCE
    cap = cv.VideoCapture(camera_index) # Change to 0 if needed
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    
    while True:
        success, frame = cap.read()
        if not success: 
            time.sleep(0.1)
            continue
            
        if flip_mode is not None: frame = cv.flip(frame, flip_mode)
        success, frame = cap.read()
        if not success: break
        if flip_mode is not None: frame = cv.flip(frame, flip_mode)
        H, W = frame.shape[:2]
       
        # --- 1. PERFORM CROP (Make it Square) ---
        square_len = min(H, W)
        start_x = (W - square_len) // 2
        start_y = (H - square_len) // 2
        end_x = start_x + square_len
        end_y = start_y + square_len
       
        # Crop the image first
        frame = frame[start_y:end_y, start_x:end_x]
       
        # Calculate cell size based on the square crop
        cell_size = square_len / 12
        with ai_lock:
            ai_ts = prediction_timestamp
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        marker_corners, marker_IDs, _ = detector.detectMarkers(gray)
       
        if marker_IDs is not None:
            for ids, corners in zip(marker_IDs, marker_corners):
                current_id = int(ids[0])
                corners_data = corners.reshape(4, 2).astype(int)
                mx, my = int(np.mean(corners_data[:, 0])), int(np.mean(corners_data[:, 1]))
               
                # --- 2. SIMPLIFIED TRACKING MATH ---
                # mx, my are already relative to the cropped frame (0,0 is top-left)
                pos_x = mx / cell_size + 1
                pos_y = my / cell_size + 1
                grid_x = int(mx / cell_size) + 1
                grid_y = int(my / cell_size) + 1
               
                region = classify_point(grid_x, grid_y)
                marker_data[current_id] = {'region': region, 'time': datetime.now().isoformat(), 'pos_x': pos_x, 'pos_y': pos_y}
                # --- 3. DRAWING LOGIC ---
                if current_id == selected_id:
                    should_show_box = False
                    if ai_ts > 0:
                        elapsed = time.time() - ai_ts
                        if elapsed > DELAY_SECONDS:
                            should_show_box = True
                    if should_show_box:
                        # Red Box
                        pts = corners.reshape(4, 2).astype(np.float32)
                        rect = cv.minAreaRect(pts)
                        (rcx, rcy), (rw, rh), angle = rect
                        ew = rw + 2 * box_margin
                        eh = rh + 2 * box_margin
                        box_pts = cv.boxPoints(((rcx, rcy), (ew, eh), angle)).astype(int)
                        cv.polylines(frame, [box_pts], True, (0, 0, 255), box_thickness)
                        # Green ID
                        cv.polylines(frame, [corners.astype(int)], True, (0, 255, 0), 2)
                        cv.putText(frame, f"ID: {current_id}", (mx, my - 10), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
       
        # --- FIXED CALL HERE ---
        if show_grid:
            draw_grid(frame, square_len, cell_size)
        # Sidebar logic
        sidebar_width = 350
        sidebar = np.zeros((square_len, sidebar_width, 3), dtype=np.uint8)
        if physics_active:
            vals = _update_physics()
            draw_sidebar(sidebar, vals)
        else:
            draw_sidebar(sidebar, [0,0,0,0])
        dashboard = np.hstack((frame, sidebar))

        with video_lock:
            # We encode it here to save CPU for the web threads
            ret, buffer = cv.imencode('.jpg', dashboard)
            if ret:
                output_frame = buffer.tobytes()
    
        # Limit framerate slightly to save CPU (optional)
        time.sleep(0.01)

def generate_frames():
    """
    Flask Stream: Just returns the latest global frame
    """
    global output_frame
    while True:
        with video_lock:
            if output_frame is None:
                continue
            current_bytes = output_frame
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + current_bytes + b'\r\n')
       
# ==========================================
# 5. FLASK ROUTES
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# API to handle clicks from the website (Keyboard replacement)
@app.route('/control', methods=['POST'])
def control():
    global selected_id, prediction_timestamp
    data = request.json
    command = data.get('command')
    val = data.get('value')
    if command == 'select_id':
        marker_id = int(val)
       
        # Logic: Toggle Off if same ID, otherwise Switch
        if selected_id == marker_id:
            selected_id = -1
            return jsonify({"status": "disabled", "msg": "Tracking Disabled"})
       
        selected_id = marker_id
       
        # Trigger Audio & AI if file exists
        if marker_id in ID_TO_FILENAME:
            filename = ID_TO_FILENAME[marker_id]
            path = os.path.join(AUDIO_FOLDER, filename)
            if os.path.exists(path):
                pygame.mixer.music.stop()
                pygame.mixer.music.load(path)
                pygame.mixer.music.play()
                # --- ADD THESE 3 LINES ---
                with ai_lock:
                    prediction_timestamp = 0  # Reset timer so box hides
               
                threading.Thread(target=run_ai_prediction, args=(filename, marker_id), daemon=True).start()
                return jsonify({"status": "active", "msg": f"Tracking ID {marker_id}"})
           
        return jsonify({"status": "active", "msg": f"Tracking ID {marker_id} (No Audio)"})
   
    if command == 'reset_audio':
        # 1. Stop the Sound
        pygame.mixer.music.stop()
       
        # 2. Clear the UI Bars & Text (But keep selected_id!)
        global physics_active, physics_mic_values, last_static_mic_values
        global current_pred_text, current_pred_color
       
        with ai_lock:
            current_pred_text = "System Ready"
            current_pred_color = (255, 255, 255)
            prediction_timestamp = 0
           
            physics_active = False
            physics_mic_values = None
            last_static_mic_values = None
        return jsonify({"status": "audio_reset", "msg": "Bars Reset, Tracker Active"})
   
    elif command == 'reset':
        selected_id = -1
        pygame.mixer.music.stop()
        return jsonify({"status": "reset", "msg": "Tracking Reset"})
    return jsonify({"status": "error"})

@app.route('/get_status')
def get_status():
    global current_pred_text, prediction_timestamp, selected_id
   
    with ai_lock:
        return jsonify({
            "text": current_pred_text,
            "timestamp": prediction_timestamp,
            "id": selected_id
        })
   
if __name__ == '__main__':
    # --- FIX: Start the background thread ---
    t = threading.Thread(target=process_video_feed, daemon=True)
    t.start()
    
    # --- Run Flask (Turn off debug to prevent duplicate threads) ---
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)