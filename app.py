from flask import Flask, request, jsonify, render_template, Response, session, redirect, url_for
from flask_cors import CORS
from functools import wraps
import cv2
import numpy as np
import os
import json
import sqlite3
import re
import datetime
import threading
import time
from collections import defaultdict

app = Flask(__name__)
app.secret_key = "ec1ec19e2a66ee6873a56064de01b1d421da661ed9f8bc97"
CORS(app)

# ── ADMIN CREDENTIALS (change these!) ─────────────────────────────────────────
ADMIN_ID       = "admin"
ADMIN_PASSWORD = "admin"

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('admin_logged_in'):
            if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({"error": "Unauthorized"}), 401
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

DB_PATH = "toll.db"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

TOLL_RATES = {
    "car":        30,
    "motorcycle": 15,
    "bus":        80,
    "truck":      100,
}

# COCO class IDs for vehicles
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# ── LIVE VIDEO FEED STATE ────────────────────────────────────────────────────
latest_frame = None          # stores the latest annotated JPEG bytes
feed_active = False          # whether a video is being processed
feed_lock = threading.Lock()

BOX_COLORS = {
    "car":        (250, 166, 59),   # blue-ish (BGR)
    "motorcycle": (247, 85, 168),   # purple
    "bus":        (11, 158, 245),   # amber
    "truck":      (68, 68, 239),    # red
}

def draw_detections(frame, results):
    """Draw bounding boxes and labels on frame for all detected vehicles."""
    annotated = frame.copy()
    if results[0].boxes is not None and results[0].boxes.cls is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy() if hasattr(results[0].boxes.xyxy, 'cpu') else results[0].boxes.xyxy
        classes = results[0].boxes.cls.cpu().numpy() if hasattr(results[0].boxes.cls, 'cpu') else results[0].boxes.cls
        ids = None
        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy() if hasattr(results[0].boxes.id, 'cpu') else results[0].boxes.id
        confs = results[0].boxes.conf.cpu().numpy() if hasattr(results[0].boxes.conf, 'cpu') else results[0].boxes.conf

        for i, (box, cls) in enumerate(zip(boxes, classes)):
            x1, y1, x2, y2 = map(int, box)
            vtype = VEHICLE_CLASSES.get(int(cls), "car")
            color = BOX_COLORS.get(vtype, (255, 255, 255))
            
            # Draw filled rectangle behind text
            label = vtype.upper()
            if ids is not None:
                label += f" #{int(ids[i])}"
            if i < len(confs):
                label += f" {confs[i]:.0%}"

            # Bounding box with thickness
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 4, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Corner accents
            corner_len = min(20, (x2-x1)//4, (y2-y1)//4)
            cv2.line(annotated, (x1, y1), (x1+corner_len, y1), color, 3)
            cv2.line(annotated, (x1, y1), (x1, y1+corner_len), color, 3)
            cv2.line(annotated, (x2, y1), (x2-corner_len, y1), color, 3)
            cv2.line(annotated, (x2, y1), (x2, y1+corner_len), color, 3)
            cv2.line(annotated, (x1, y2), (x1+corner_len, y2), color, 3)
            cv2.line(annotated, (x1, y2), (x1, y2-corner_len), color, 3)
            cv2.line(annotated, (x2, y2), (x2-corner_len, y2), color, 3)
            cv2.line(annotated, (x2, y2), (x2, y2-corner_len), color, 3)

    return annotated

# ── DATABASE ─────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS vehicles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plate TEXT UNIQUE NOT NULL,
        owner TEXT NOT NULL,
        vehicle_type TEXT NOT NULL,
        balance REAL NOT NULL DEFAULT 500.0,
        registered_at TEXT NOT NULL
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plate TEXT NOT NULL,
        vehicle_type TEXT NOT NULL,
        toll_amount REAL NOT NULL,
        balance_before REAL NOT NULL,
        balance_after REAL NOT NULL,
        status TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )''')
    # Seed demo vehicles
    demo = [
        ("MH12AB1234", "Rahul Sharma",   "car",        800.0),
        ("MH14CD5678", "Priya Patel",    "car",        250.0),
        ("MH01EF9012", "Suresh Kumar",   "truck",     1200.0),
        ("MH20GH3456", "Anita Desai",    "motorcycle",  50.0),
        ("MH04IJ7890", "Vikram Singh",   "bus",        600.0),
        ("KA05KL2345", "Deepa Nair",     "car",        900.0),
        ("DL08MN6789", "Amit Gupta",     "truck",       10.0),
        ("GJ01OP1234", "Meena Shah",     "motorcycle", 300.0),
    ]
    for plate, owner, vtype, bal in demo:
        c.execute('''INSERT OR IGNORE INTO vehicles (plate, owner, vehicle_type, balance, registered_at)
                     VALUES (?, ?, ?, ?, ?)''',
                  (plate, owner, vtype, bal, datetime.datetime.now().isoformat()))
    conn.commit()
    conn.close()

# ── PLATE OCR ─────────────────────────────────────────────────────────────────
def extract_plate_text(frame):
    """Try EasyOCR, fall back to basic contour-based detection."""
    try:
        import easyocr
        if not hasattr(extract_plate_text, '_reader'):
            extract_plate_text._reader = easyocr.Reader(['en'], verbose=False)
        results = extract_plate_text._reader.readtext(frame)
        for (_, text, conf) in results:
            cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
            if 6 <= len(cleaned) <= 10 and conf > 0.4:
                return cleaned
    except ImportError:
        pass
    return None

def lookup_plate(raw_plate, vehicle_type):
    """Try exact match, then fuzzy match against registered plates."""
    if not raw_plate:
        return None
    conn = get_db()
    c = conn.cursor()
    # Exact match
    row = c.execute("SELECT * FROM vehicles WHERE REPLACE(plate,'-','') = ?",
                    (raw_plate.replace("-", ""),)).fetchone()
    if not row:
        # Fuzzy: find plate with most matching chars
        all_plates = c.execute("SELECT * FROM vehicles WHERE vehicle_type=?", (vehicle_type,)).fetchall()
        best, best_score = None, 0
        for p in all_plates:
            clean = re.sub(r'[^A-Z0-9]', '', p['plate'].upper())
            score = sum(a == b for a, b in zip(clean, raw_plate[:len(clean)]))
            if score > best_score and score >= len(clean) * 0.7:
                best, best_score = p, score
        row = best
    conn.close()
    return dict(row) if row else None

def process_toll(plate_data, vehicle_type):
    """Deduct toll from account. Returns transaction dict."""
    toll = TOLL_RATES.get(vehicle_type, 30)
    conn = get_db()
    c = conn.cursor()
    bal_before = plate_data['balance']
    if bal_before >= toll:
        bal_after = bal_before - toll
        status = "success"
        c.execute("UPDATE vehicles SET balance=? WHERE plate=?", (bal_after, plate_data['plate']))
    else:
        bal_after = bal_before
        status = "insufficient"
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''INSERT INTO transactions (plate, vehicle_type, toll_amount, balance_before, balance_after, status, timestamp)
                 VALUES (?,?,?,?,?,?,?)''',
              (plate_data['plate'], vehicle_type, toll, bal_before, bal_after, status, ts))
    conn.commit()
    conn.close()
    return {
        "plate": plate_data['plate'],
        "owner": plate_data['owner'],
        "vehicle_type": vehicle_type,
        "toll": toll,
        "balance_before": bal_before,
        "balance_after": bal_after,
        "status": status,
        "timestamp": ts
    }

# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login", methods=["GET"])
def login():
    if session.get('admin_logged_in'):
        return redirect(url_for('admin'))
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def do_login():
    data = request.json
    if data.get('username') == ADMIN_ID and data.get('password') == ADMIN_PASSWORD:
        session['admin_logged_in'] = True
        session.permanent = False
        return jsonify({"success": True})
    return jsonify({"error": "Invalid credentials"}), 401

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route("/admin")
@admin_required
def admin():
    return render_template("admin.html")

@app.route("/api/vehicles", methods=["GET"])
@admin_required
def get_vehicles():
    conn = get_db()
    rows = conn.execute("SELECT * FROM vehicles ORDER BY plate").fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route("/api/vehicles", methods=["POST"])
@admin_required
def add_vehicle():
    data = request.json
    try:
        conn = get_db()
        conn.execute('''INSERT INTO vehicles (plate, owner, vehicle_type, balance, registered_at)
                        VALUES (?,?,?,?,?)''',
                     (data['plate'].upper(), data['owner'], data['vehicle_type'],
                      float(data.get('balance', 500)),
                      datetime.datetime.now().isoformat()))
        conn.commit()
        conn.close()
        return jsonify({"success": True})
    except sqlite3.IntegrityError:
        return jsonify({"error": "Plate already exists"}), 400

@app.route("/api/vehicles/<plate>", methods=["PUT"])
@admin_required
def update_vehicle(plate):
    data = request.json
    conn = get_db()
    conn.execute("UPDATE vehicles SET balance=balance+? WHERE plate=?",
                 (float(data['topup']), plate))
    conn.commit()
    conn.close()
    return jsonify({"success": True})

@app.route("/api/vehicles/<plate>", methods=["DELETE"])
@admin_required
def delete_vehicle(plate):
    conn = get_db()
    conn.execute("DELETE FROM vehicles WHERE plate=?", (plate,))
    conn.commit()
    conn.close()
    return jsonify({"success": True})

@app.route("/api/transactions", methods=["GET"])
@admin_required
def get_transactions():
    conn = get_db()
    rows = conn.execute("SELECT * FROM transactions ORDER BY id DESC LIMIT 100").fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route("/api/stats", methods=["GET"])
def get_stats():
    conn = get_db()
    total_collected = conn.execute(
        "SELECT COALESCE(SUM(toll_amount),0) FROM transactions WHERE status='success'").fetchone()[0]
    total_vehicles  = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    insufficient    = conn.execute(
        "SELECT COUNT(*) FROM transactions WHERE status='insufficient'").fetchone()[0]
    by_type = conn.execute(
        "SELECT vehicle_type, COUNT(*) as cnt, SUM(CASE WHEN status='success' THEN toll_amount ELSE 0 END) as rev "
        "FROM transactions GROUP BY vehicle_type").fetchall()
    conn.close()
    return jsonify({
        "total_collected": total_collected,
        "total_vehicles": total_vehicles,
        "insufficient": insufficient,
        "by_type": [dict(r) for r in by_type]
    })

@app.route("/process_video", methods=["POST"])
def process_video():
    global latest_frame, feed_active
    # Stop any existing loop playback before starting a new video
    feed_active = False
    time.sleep(0.15)  # give the old loop thread time to exit
    with feed_lock:
        latest_frame = None

    if "video" not in request.files:
        return jsonify({"error": "No video"}), 400

    video_file = request.files["video"]
    path = os.path.join(UPLOAD_FOLDER, video_file.filename)
    video_file.save(path)

    def generate():
        global latest_frame, feed_active
        try:
            from ultralytics import YOLO
            model = YOLO("yolov8n.pt")
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return

        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_count = 0
        processed_ids = set()   # track IDs already tolled
        events = []
        feed_active = True

        # Store all annotated frames for looping playback
        annotated_frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % 5 != 0:
                continue

            results = model.track(frame, persist=True,
                                  classes=list(VEHICLE_CLASSES.keys()), verbose=False)

            # Draw bounding boxes on frame and update the live feed
            annotated = draw_detections(frame, results)
            # Resize for streaming efficiency
            h, w = annotated.shape[:2]
            max_w = 720
            if w > max_w:
                scale = max_w / w
                annotated = cv2.resize(annotated, (max_w, int(h * scale)))

            _, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            jpeg_bytes = jpeg.tobytes()
            with feed_lock:
                latest_frame = jpeg_bytes
            annotated_frames.append(jpeg_bytes)

            if results[0].boxes is not None and results[0].boxes.id is not None:
                for box, tid, cls in zip(results[0].boxes.xyxy,
                                         results[0].boxes.id,
                                         results[0].boxes.cls):
                    tid = int(tid)
                    vtype = VEHICLE_CLASSES.get(int(cls), "car")

                    if tid not in processed_ids:
                        processed_ids.add(tid)

                        # Crop the detected vehicle and try OCR
                        x1,y1,x2,y2 = map(int, box)
                        crop = frame[max(0,y1):y2, max(0,x1):x2]
                        plate_text = extract_plate_text(crop) if crop.size > 0 else None

                        if plate_text:
                            vehicle_data = lookup_plate(plate_text, vtype)
                        else:
                            vehicle_data = None

                        if vehicle_data:
                            txn = process_toll(vehicle_data, vtype)
                            txn['detected_plate'] = plate_text
                            txn['track_id'] = tid
                            events.append(txn)
                        else:
                            # Unknown plate — log as unregistered
                            events.append({
                                "plate": plate_text or f"UNKNOWN-{tid}",
                                "owner": "Unregistered",
                                "vehicle_type": vtype,
                                "toll": TOLL_RATES.get(vtype, 30),
                                "status": "unregistered",
                                "track_id": tid,
                                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })

            progress = int(frame_count / total_frames * 100)
            payload = {
                "progress": progress,
                "frame": frame_count,
                "total_frames": total_frames,
                "events": events,
                "vehicles_processed": len(processed_ids)
            }
            yield f"data: {json.dumps(payload)}\n\n"

        cap.release()

        # Start looping playback in background thread
        def loop_playback():
            global latest_frame, feed_active
            if not annotated_frames:
                feed_active = False
                return
            frame_delay = (5 / fps) if fps > 0 else 0.15  # match the 1-in-5 frame skip
            while feed_active:
                for f in annotated_frames:
                    if not feed_active:
                        return
                    with feed_lock:
                        latest_frame = f
                    time.sleep(frame_delay)

        playback_thread = threading.Thread(target=loop_playback, daemon=True)
        playback_thread.start()

        try:
            os.remove(path)
        except:
            pass

        yield f"data: {json.dumps({**payload, 'done': True, 'progress': 100})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/video_feed")
def video_feed():
    """MJPEG stream of annotated detection frames."""
    def gen():
        while True:
            with feed_lock:
                frame = latest_frame
            if frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                # Send a blank dark frame as placeholder
                blank = np.zeros((405, 720, 3), dtype=np.uint8)
                cv2.putText(blank, "Waiting for video...", (180, 210),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 80), 2)
                _, jpeg = cv2.imencode('.jpg', blank)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.04)  # ~25fps
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/stop_feed", methods=["POST"])
def stop_feed():
    global feed_active, latest_frame
    feed_active = False
    latest_frame = None
    return jsonify({"success": True})


if __name__ == "__main__":
    init_db()
    print("\n✅ Toll System running at http://127.0.0.1:5000")
    print("📊 Admin panel at  http://127.0.0.1:5000/admin\n")
    app.run(debug=False, port=5000, threaded=True)
