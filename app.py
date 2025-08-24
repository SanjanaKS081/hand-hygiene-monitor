from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO
import mediapipe as mp
import time
import json
from datetime import datetime

app = Flask(__name__)

# ---- SINGLE POSE SETUP ----
POSE_NAME = "palm_to_palm"
MODEL_PATH = r"C:\\Users\\IR\\Desktop\\palm_to_palm\\runs\\detect\\train2\\weights\\best.pt"
MODEL = YOLO(MODEL_PATH)

POSE_THRESHOLD = 0.45

# ---- progress and status ----
is_complete = False
leftProgress = 0
rightProgress = 0
maxProgress = 100
progress_increment = 7
start_time = None

# ---- logging ----
detections_log = []
frame_id = 0

# ---- camera setup ----
camera = cv2.VideoCapture(0)
frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('handwashing_output.avi', fourcc, 20.0, (frame_width, frame_height))

# ---- Mediapipe hands ----
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ---- hand detection helper ----
def classify_hands_mediapipe(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    hands_info = []
    h, w, _ = frame.shape

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm = hand_landmarks.landmark
            # simple student-style rule for "palm_to_palm"
            wrist_x = lm[0].x * w
            pinky_tip_x = lm[20].x * w
            label = "Right" if pinky_tip_x > wrist_x else "Left"

            # bounding box
            x_coords = [l.x for l in lm]
            y_coords = [l.y for l in lm]
            x_min = int(min(x_coords) * w)
            y_min = int(min(y_coords) * h)
            x_max = int(max(x_coords) * w)
            y_max = int(max(y_coords) * h)
            hands_info.append((label, (x_min, y_min, x_max, y_max)))

    return hands_info

# ---- pose detection ----
def detect_pose(frame):
    global frame_id, detections_log

    results = MODEL.predict(source=frame, conf=POSE_THRESHOLD, verbose=False)
    pose_detected = False

    log_entry = {
        "frame_id": frame_id,
        "timestamp": datetime.now().isoformat(),
        "pose": POSE_NAME,
        "detections": []
    }

    for result in results:
        boxes = result.boxes
        if boxes:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                log_entry["detections"].append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class": cls_id
                })
                if cls_id == 0 and conf >= POSE_THRESHOLD:
                    pose_detected = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    hands_info = classify_hands_mediapipe(frame) if pose_detected else []
    log_entry["hands"] = [{"label": lbl, "bbox": box} for lbl, box in hands_info]
    detections_log.append(log_entry)
    frame_id += 1

    return pose_detected, hands_info, frame

# ---- video stream generator ----
def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        _, _, processed_frame = detect_pose(frame)
        out.write(processed_frame)
        _, buffer = cv2.imencode('.jpg', processed_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ---- Flask routes ----
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    global is_complete, leftProgress, rightProgress, start_time

    success, frame = camera.read()
    if not success:
        return jsonify({"status": "error", "message": "Camera read failed"})

    pose_detected, hands_info, _ = detect_pose(frame.copy())
    left_detected = right_detected = False

    if start_time is None and pose_detected:
        start_time = time.time()

    if pose_detected:
        for hand_label, _ in hands_info:
            if hand_label == "Left":
                left_detected = True
            elif hand_label == "Right":
                right_detected = True

        if left_detected:
            leftProgress = min(leftProgress + progress_increment, maxProgress)
        if right_detected:
            rightProgress = min(rightProgress + progress_increment, maxProgress)

    status_text = "positive" if pose_detected else "negative"

    if leftProgress == maxProgress and rightProgress == maxProgress:
        is_complete = True

    elapsed_time = round(time.time() - start_time, 2) if start_time else 0

    return jsonify({
        "status": status_text,
        "detected": pose_detected,
        "pose_info": {
            "current_pose": POSE_NAME,
            "current_pose_index": 0,
            "total_poses": 1,
            "is_complete": is_complete,
            "leftProgress": leftProgress,
            "rightProgress": rightProgress,
            "leftHandDetected": int(left_detected),
            "rightHandDetected": int(right_detected),
            "totalTimeSeconds": elapsed_time
        }
    })

@app.route('/reset', methods=['POST'])
def reset():
    global is_complete, leftProgress, rightProgress, start_time
    is_complete = False
    leftProgress = rightProgress = 0
    start_time = None
    return jsonify({"message": "Reset completed"})

if __name__ == '__main__':
    app.run(debug=True)
