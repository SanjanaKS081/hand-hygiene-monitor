from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO
import mediapipe as mp
import time
import json
from datetime import datetime

app = Flask(__name__)

POSES = [
    "palm_to_palm", "Backofhands", "interlacedfingers", "Back_of_fingers",
    "Thumb_rotation", "Fingertips", "Wrist"
]

MODEL_PATHS = {
    "palm_to_palm": r"C:\\Users\\IR\\Desktop\\palm_to_palm\\runs\\detect\\train2\\weights\\best.pt",
    "Backofhands": r"C:\\Users\\IR\\Desktop\\Backofhands\\runs\\detect\\train2\\weights\\best.pt",
    "interlacedfingers": r"C:\\Users\\IR\\Desktop\\interlacedfingers\\runs\\detect\\train2\\weights\\best.pt",
    "Back_of_fingers": r"C:\\Users\\IR\\Desktop\\Back_of_fingers\\runs\\detect\\train2\\weights\\best.pt",
    "Thumb_rotation": r"C:\\Users\\IR\\Desktop\\Thumb_rotation\\runs\\detect\\train2\\weights\\best.pt",
    "Fingertips": r"C:\\Users\\IR\\Desktop\\Fingertips\\runs\\detect\\train3\\weights\\best.pt",
    "Wrist": r"C:\\Users\\IR\\Desktop\\Wrist\\runs\\detect\\train2\\weights\\best.pt",
}

POSE_THRESHOLDS = {
    "palm_to_palm": 0.45,
    "Backofhands": 0.4,
    "interlacedfingers": 0.5,
    "Back_of_fingers": 0.4,
    "Thumb_rotation": 0.45,
    "Fingertips": 0.5,
    "Wrist": 0.4
}

pose_index = 0
is_complete = False
leftProgress = 0
rightProgress = 0
maxProgress = 100
progress_increment = 7
start_time = None

detections_log = []
frame_id = 0

# NEW: globals for fps logging
fps_log = []
last_time = time.time()

camera = cv2.VideoCapture(0)
frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('handwashing_output.avi', fourcc, 20.0, (frame_width, frame_height))

MODEL = YOLO(MODEL_PATHS[POSES[pose_index]])

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def load_model(pose_name):
    global MODEL
    MODEL = YOLO(MODEL_PATHS[pose_name])

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def classify_hands_mediapipe_with_boxes(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    hands_info = []
    h, w, _ = frame.shape

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm = hand_landmarks.landmark
            current_pose = POSES[pose_index]

            if current_pose == "palm_to_palm":
                wrist_x = lm[0].x * w
                pinky_tip_x = lm[20].x * w
                label = "Right" if pinky_tip_x > wrist_x else "Left"
            elif current_pose == "Backofhands":
                index_mcp = lm[5].x * w
                pinky_mcp = lm[17].x * w
                label = "Left" if index_mcp < pinky_mcp else "Right"
            elif current_pose == "interlacedfingers":
                center_x = (lm[0].x + lm[9].x) / 2 * w
                label = "Left" if center_x < w / 2 else "Right"
            elif current_pose == "Back_of_fingers":
                thumb_base = lm[1].x * w
                pinky_tip = lm[20].x * w
                label = "Left" if pinky_tip < thumb_base else "Right"
            elif current_pose == "Thumb_rotation":
                thumb_tip = lm[4].x * w
                wrist_x = lm[0].x * w
                label = "Right" if thumb_tip > wrist_x else "Left"
            elif current_pose == "Fingertips":
                thumb_tip = lm[4].x * w
                pinky_tip = lm[20].x * w
                label = "Left" if thumb_tip < pinky_tip else "Right"
            elif current_pose == "Wrist":
                wrist = lm[0].x * w
                middle_tip = lm[12].x * w
                label = "Left" if middle_tip < wrist else "Right"
            else:
                label = "Unknown"

            x_coords = [l.x for l in lm]
            y_coords = [l.y for l in lm]
            x_min = int(min(x_coords) * w)
            y_min = int(min(y_coords) * h)
            x_max = int(max(x_coords) * w)
            y_max = int(max(y_coords) * h)
            hands_info.append((label, (x_min, y_min, x_max, y_max)))

    return hands_info

def detect_pose(frame):
    global frame_id, detections_log, fps_log, last_time

    # --- FPS logging ---
    current_time = time.time()
    fps = 1.0 / (current_time - last_time) if last_time else 0
    last_time = current_time
    fps_log.append(fps)
    with open("fps_log.txt", "w") as f:
        for value in fps_log:
            f.write(str(round(value, 2)) + "\n")

    # --- Pose detection ---
    conf_thresh = POSE_THRESHOLDS.get(POSES[pose_index], 0.1)
    results = MODEL.predict(source=frame, conf=conf_thresh, verbose=False)

    detected_poses = []
    expected_pose_detected = False

    log_entry = {
        "frame_id": frame_id,
        "timestamp": datetime.now().isoformat(),
        "pose": POSES[pose_index],
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

                if cls_id == 0 and conf >= conf_thresh:
                    expected_pose_detected = True
                    detected_poses.append((x1, y1, x2, y2))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    hands_info = classify_hands_mediapipe_with_boxes(frame) if expected_pose_detected else []

    log_entry["hands"] = [{"label": lbl, "bbox": box} for lbl, box in hands_info]

    detections_log.append(log_entry)
    with open("detections.json", "w") as f:
        json.dump(detections_log, f, indent=2)

    frame_id += 1
    return detected_poses, hands_info, frame

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

@app.route('/')
def index():
    return render_template('file.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    global is_complete, leftProgress, rightProgress, pose_index, start_time

    success, frame = camera.read()
    if not success:
        return jsonify({"status": "error", "message": "Camera read failed"})

    pose_detected, hands_info, _ = detect_pose(frame.copy())
    left_detected = right_detected = False

    if start_time is None and pose_detected:
        start_time = time.time()

    if pose_detected:
        for hand_label, hand_box in hands_info:
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
        if pose_index < len(POSES) - 1:
            pose_index += 1
            load_model(POSES[pose_index])
            leftProgress = rightProgress = 0
            start_time = time.time()
        else:
            is_complete = True

    elapsed_time = round(time.time() - start_time, 2) if start_time else 0

    return jsonify({
        "status": status_text,
        "detected": pose_detected,
        "pose_info": {
            "current_pose": POSES[pose_index],
            "current_pose_index": pose_index,
            "total_poses": len(POSES),
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
    global is_complete, leftProgress, rightProgress, pose_index, start_time
    is_complete = False
    leftProgress = rightProgress = 0
    pose_index = 0
    start_time = None
    load_model(POSES[pose_index])
    return jsonify({"message": "Reset completed"})

@app.route('/save_log')
def save_log():
    with open("detections.json", "w") as f:
        json.dump(detections_log, f, indent=2)
    return jsonify({"message": f"Saved {len(detections_log)} frames"})

if __name__ == '__main__':
    app.run(debug=True)
