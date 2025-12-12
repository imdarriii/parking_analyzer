import os
import time
import base64
import logging
import threading
from datetime import datetime


import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory, render_template
from flask_cors import CORS
from ultralytics import YOLO


# ---------------------------------------------------------
# 1) APP + LOGGING
# ---------------------------------------------------------
app = Flask(__name__)
CORS(app)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------
# 2) YOLO MODEL
# ---------------------------------------------------------
try:
    model = YOLO("yolo11s.pt")
    logging.info("✓ YOLO v11 model loaded")
except Exception as e:
    logging.error(f"✗ YOLO load error: {e}")
    model = None


# ---------------------------------------------------------
# 3) CONSTANTS
# ---------------------------------------------------------
CAR_CLASSES = {2, 3, 5, 7}          # COCO: car, motorcycle, bus, truck
PROCESS_EVERY_N_FRAMES = 3


FRAME_WIDTH = 960
FRAME_HEIGHT = 720


# IMPORTANT: video must be inside repo
TECHNOPARK_VIDEO_RELATIVE = os.path.join("videos", "technopark.mp4")


AUTO_LEARNING_FRAMES = 60
MIN_SPOT_CONFIDENCE = 0.5
IOU_THRESHOLD = 0.5
MOVEMENT_THRESHOLD = 15
MIN_BOX_AREA = 800


parking_systems = {}
_initialized = False
_init_lock = threading.Lock()


# ---------------------------------------------------------
# 4) HELPERS
# ---------------------------------------------------------
def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2


    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)


    if x2_i < x1_i or y2_i < y1_i:
        return 0.0


    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0




def box_distance(box1, box2):
    cx1 = (box1[0] + box1[2]) / 2
    cy1 = (box1[1] + box1[3]) / 2
    cx2 = (box2[0] + box2[2]) / 2
    cy2 = (box2[1] + box2[3]) / 2
    return float(np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2))




def expand_box(box, expand_ratio=0.1):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    expand_w = w * expand_ratio
    expand_h = h * expand_ratio
    return [
        max(0, int(x1 - expand_w)),
        max(0, int(y1 - expand_h)),
        int(x2 + expand_w),
        int(y2 + expand_h),
    ]




def normalize_location(loc_raw: str) -> str:
    if not loc_raw:
        return "ala-too"
    loc = loc_raw.strip().lower()


    # Accept many variants
    if "tech" in loc or "техно" in loc:
        return "technopark"
    if "ala" in loc or "ала" in loc:
        return "ala-too"


    # fallback
    return "ala-too"




def get_system_from_request():
    loc_raw = request.args.get("location", "")
    loc = normalize_location(loc_raw)


    system = parking_systems.get(loc)
    if system is None:
        raise ValueError(f"Unknown or not initialized location: {loc}")
    return system




# ---------------------------------------------------------
# 5) PARKING SYSTEM CLASS
# ---------------------------------------------------------
class ParkingSystem:
    def __init__(self, location_id: str, source: str, auto_learn: bool):
        self.location_id = location_id
        self.source = source
        self.auto_learn = auto_learn


        lower_src = (source or "").lower()
        self.is_file = lower_src.endswith((".mp4", ".mov", ".avi", ".mkv"))


        logging.info(f"🔧 Init ParkingSystem({self.location_id}) auto_learn={self.auto_learn}")


        self.learning_phase = auto_learn
        self.learning_frames = 0
        self.detected_spots_history = []
        self.previous_detections = []


        if not auto_learn:
            self._init_alatoo_spots()
        else:
            self.parking_spots = []


        self.current_frame = None
        self.is_running = False
        self.free_count = 0
        self.last_update = None
        self.detected_cars = []
        self.connection_status = "Connecting..."


    def _init_alatoo_spots(self):
        spots = []
        x_start = 10
        spot_size = 22
        num_spots = 35
        Y_OFFSET_DOWN = 13
        Y_BASE_START = 530 + Y_OFFSET_DOWN


        for i in range(num_spots):
            x_left = x_start + i * (spot_size + 6)
            center = (num_spots - 1) / 2.0
            distance_from_center = abs(i - center)
            vertical_offset = 0.2 * (distance_from_center ** 2)


            additional_offset = 3 if i < 18 else 5
            y_base = Y_BASE_START + additional_offset - int(vertical_offset)


            if 24 <= i <= 33:
                y_base += 8
            if 26 <= i <= 33:
                y_base += 10


            skew_offset = spot_size
            spot = np.array(
                [
                    [x_left + skew_offset, y_base],
                    [x_left + skew_offset + spot_size, y_base],
                    [x_left + spot_size, y_base + spot_size],
                    [x_left, y_base + spot_size],
                ],
                dtype=np.int32,
            )
            spots.append(spot)


        self.parking_spots = [
            {"id": i, "coords": spot, "occupied": False, "confidence": 1.0, "type": "alatoo"}
            for i, spot in enumerate(spots)
        ]


    def process_learning_frame(self, detections):
        current_frame_boxes = []


        for box in detections:
            try:
                cls = int(box.cls[0])
                conf = float(box.conf[0])


                if cls not in CAR_CLASSES or conf < 0.4:
                    continue


                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox_area = (x2 - x1) * (y2 - y1)
                if bbox_area < MIN_BOX_AREA:
                    continue


                current_frame_boxes.append([x1, y1, x2, y2, conf])
            except Exception:
                continue


        if len(self.previous_detections) > 0:
            filtered_boxes = []
            for box in current_frame_boxes:
                is_moving = False
                for prev_box in self.previous_detections:
                    if calculate_iou(box[:4], prev_box[:4]) > 0.3:
                        dist = box_distance(box[:4], prev_box[:4])
                        if dist > MOVEMENT_THRESHOLD:
                            is_moving = True
                            break
                if not is_moving:
                    filtered_boxes.append(box)
            current_frame_boxes = filtered_boxes


        self.previous_detections = current_frame_boxes
        self.detected_spots_history.append(current_frame_boxes)
        self.learning_frames += 1


        if self.learning_frames >= AUTO_LEARNING_FRAMES:
            self._finalize_learning()


    def _finalize_learning(self):
        logging.info(f"🎓 ({self.location_id}) Finalizing learning...")


        all_boxes = []
        for frame_boxes in self.detected_spots_history:
            all_boxes.extend(frame_boxes)


        if len(all_boxes) == 0:
            logging.warning(f"⚠ ({self.location_id}) No cars detected during learning")
            self.learning_phase = False
            return


        clusters = []
        for box in all_boxes:
            matched = False
            for cluster in clusters:
                avg_box = np.array([b[:4] for b in cluster["boxes"]]).mean(axis=0)
                iou = calculate_iou(box[:4], avg_box)
                dist = box_distance(box[:4], avg_box)
                if iou > IOU_THRESHOLD or dist < 30:
                    cluster["boxes"].append(box)
                    matched = True
                    break
            if not matched:
                clusters.append({"boxes": [box]})


        self.parking_spots = []
        min_frames_required = int(AUTO_LEARNING_FRAMES * MIN_SPOT_CONFIDENCE)


        for i, cluster in enumerate(clusters):
            if len(cluster["boxes"]) < min_frames_required:
                continue


            boxes_array = np.array([b[:4] for b in cluster["boxes"]])
            avg_box = boxes_array.mean(axis=0).astype(int)


            expanded_box = expand_box(avg_box, expand_ratio=0.05)
            x1, y1, x2, y2 = expanded_box


            coords = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
            confidence = len(cluster["boxes"]) / AUTO_LEARNING_FRAMES


            self.parking_spots.append(
                {
                    "id": i,
                    "coords": coords,
                    "occupied": False,
                    "confidence": confidence,
                    "type": "technopark",
                    "area": (x2 - x1) * (y2 - y1),
                }
            )


        self.parking_spots.sort(key=lambda s: (s["coords"][0][1], s["coords"][0][0]))
        for idx, spot in enumerate(self.parking_spots):
            spot["id"] = idx


        self.learning_phase = False
        self.detected_spots_history = []
        self.previous_detections = []
        logging.info(f"✅ ({self.location_id}) Learning done! Spots: {len(self.parking_spots)}")


    def check_spot_occupancy(self, spot, detections):
        if spot.get("type") == "alatoo":
            spot_coords_list = spot["coords"].tolist()
            x_min = min(c[0] for c in spot_coords_list)
            y_min = min(c[1] for c in spot_coords_list)
            x_max = max(c[0] for c in spot_coords_list)
            y_max = max(c[1] for c in spot_coords_list)
            spot_box = [x_min, y_min, x_max, y_max]
        elif spot.get("type") == "technopark":
            spot_coords = spot["coords"]
            x1_s, y1_s = spot_coords[0]
            x2_s, y2_s = spot_coords[2]
            spot_box = [int(x1_s), int(y1_s), int(x2_s), int(y2_s)]
        else:
            return False


        spot_area = (spot_box[2] - spot_box[0]) * (spot_box[3] - spot_box[1])
        if spot_area <= 0:
            return False


        for box in detections:
            try:
                cls = int(box.cls[0])
                conf = float(box.conf[0])


                if cls not in CAR_CLASSES or conf < 0.25:
                    continue


                x1, y1, x2, y2 = map(int, box.xyxy[0])
                det_box = [x1, y1, x2, y2]


                x1_i = max(spot_box[0], det_box[0])
                y1_i = max(spot_box[1], det_box[1])
                x2_i = min(spot_box[2], det_box[2])
                y2_i = min(spot_box[3], det_box[3])


                if x2_i < x1_i or y2_i < y1_i:
                    continue


                intersection = (x2_i - x1_i) * (y2_i - y1_i)
                overlap_with_spot = intersection / spot_area


                det_area = (det_box[2] - det_box[0]) * (det_box[3] - det_box[1])
                overlap_with_car = intersection / det_area if det_area > 0 else 0


                if overlap_with_spot > 0.30 or overlap_with_car > 0.40:
                    return True
            except Exception:
                continue


        return False




# ---------------------------------------------------------
# 6) VIDEO LOOP
# ---------------------------------------------------------
def process_video(system: ParkingSystem):
    # IMPORTANT: make file path absolute if it's a file
    src = system.source
    if system.is_file:
        src = os.path.join(BASE_DIR, system.source)


    cap = cv2.VideoCapture(src)


    if not cap.isOpened():
        logging.error(f"✗ ({system.location_id}) Failed to open source: {system.source}")
        system.connection_status = "Failed to connect"
        system.is_running = False
        return


    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


    logging.info(f"✓ ({system.location_id}) Connected")
    system.connection_status = "Connected"


    frame_count = 0
    last_results = None


    while system.is_running:
        ret, frame = cap.read()


        if not ret:
            if system.is_file:
                # loop file
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.03)
                continue


            system.connection_status = "Reconnecting..."
            time.sleep(1.0)
            cap.release()
            cap = cv2.VideoCapture(src)
            if cap.isOpened():
                system.connection_status = "Connected"
                continue
            else:
                system.connection_status = "Connection failed"
                time.sleep(2.0)
                continue


        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame_count += 1


        # YOLO
        results = None
        if model is not None:
            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                try:
                    results = model(
                        frame,
                        verbose=False,
                        conf=0.25,
                        iou=0.45,
                        imgsz=640,
                        half=False,
                        device="cpu",
                    )[0]
                    last_results = results
                except Exception as e:
                    logging.error(f"YOLO error ({system.location_id}): {e}")
                    results = last_results
            else:
                results = last_results


        if results is None:
            system.current_frame = frame.copy()
            system.last_update = datetime.now().isoformat()
            time.sleep(0.01)
            continue


        det_boxes = results.boxes


        # Learning phase (Technopark)
        if system.learning_phase:
            system.process_learning_frame(det_boxes)


            progress = int((system.learning_frames / max(1, AUTO_LEARNING_FRAMES)) * 100)
            cv2.putText(frame, f"LEARNING: {progress}%", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)


            for box in det_boxes:
                try:
                    cls = int(box.cls[0])
                    if cls in CAR_CLASSES:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                except Exception:
                    pass


            system.current_frame = frame.copy()
            system.last_update = datetime.now().isoformat()
            time.sleep(0.01)
            continue


        # Normal mode
        system.detected_cars = []
        free = 0
        total_spots = len(system.parking_spots)


        for i, spot in enumerate(system.parking_spots):
            occupied = system.check_spot_occupancy(spot, det_boxes)
            system.parking_spots[i]["occupied"] = occupied
            if not occupied:
                free += 1


            color = (0, 0, 255) if occupied else (0, 255, 0)
            thickness = 3 if occupied else 2
            coords = spot["coords"]
            cv2.polylines(frame, [coords], True, color, thickness)


            text_x = int(coords[0][0])
            text_y = int(coords[0][1]) - 5
            cv2.putText(frame, f"{i + 1}", (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


        for box in det_boxes:
            try:
                cls = int(box.cls[0])
                if cls in CAR_CLASSES:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    label = f"{results.names[cls]} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)


                    system.detected_cars.append({
                        "bbox": [x1, y1, x2, y2],
                        "conf": conf,
                        "class": results.names[cls],
                    })
            except Exception:
                pass


        cv2.putText(frame, f"FREE: {free}/{total_spots}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)


        system.free_count = free
        system.current_frame = frame.copy()
        system.last_update = datetime.now().isoformat()


        time.sleep(0.01)


    cap.release()
    logging.info(f"✓ ({system.location_id}) stopped")




# ---------------------------------------------------------
# 7) INIT SYSTEMS (IMPORTANT FOR GUNICORN!)
# ---------------------------------------------------------
def start_background_workers():
    global _initialized


    with _init_lock:
        if _initialized:
            return


        logging.info("🚀 Initializing parking systems...")


        # Ala-Too (stream)
        ala_too_source = "https://webcam.elcat.kg/Bishkek_Ala-Too_Square/tracks-v1/mono.m3u8"
        ala = ParkingSystem(location_id="ala-too", source=ala_too_source, auto_learn=False)
        ala.is_running = True
        parking_systems["ala-too"] = ala


        t1 = threading.Thread(target=process_video, args=(ala,), daemon=True)
        t1.start()


        # Technopark (file inside repo)
        tech = ParkingSystem(location_id="technopark", source=TECHNOPARK_VIDEO_RELATIVE, auto_learn=True)
        tech.is_running = True
        parking_systems["technopark"] = tech


        # if file missing -> thread will mark failed, but system exists and API won't crash
        abs_path = os.path.join(BASE_DIR, TECHNOPARK_VIDEO_RELATIVE)
        if not os.path.exists(abs_path):
            logging.warning(f"⚠ Technopark video not found at: {abs_path}")
            tech.connection_status = "Video file missing"
            tech.is_running = False
        else:
            t2 = threading.Thread(target=process_video, args=(tech,), daemon=True)
            t2.start()


        logging.info("✓ Active locations: " + ", ".join(parking_systems.keys()))
        _initialized = True




@app.before_request
def _ensure_started():
    # start threads at first request under gunicorn
    start_background_workers()




# ---------------------------------------------------------
# 8) ROUTES
# ---------------------------------------------------------
@app.route("/")
def index():
    # Works if index.html is in templates/index.html OR in same folder as app.py
    try:
        return render_template("index.html")
    except Exception:
        # fallback: serve ./index.html if not using templates folder
        try:
            return send_from_directory(BASE_DIR, "index.html")
        except Exception:
            return "<h1>Parking Analyzer Backend</h1>", 200




@app.route("/api/auth", methods=["POST"])
def auth():
    data = request.json or {}
    username = (data.get("username") or "").strip()
    if len(username) < 2:
        return jsonify({"error": "Name must be at least 2 characters"}), 400
    return jsonify({
        "success": True,
        "message": f"Welcome, {username}!",
        "user": username,
        "timestamp": datetime.now().isoformat(),
    })




@app.route("/api/status")
def get_status():
    try:
        system = get_system_from_request()
        total_spots = len(system.parking_spots)
        occ = max(0, total_spots - int(system.free_count))
        occupancy_rate = round((occ / total_spots * 100), 1) if total_spots > 0 else 0.0


        return jsonify({
            "location": system.location_id,
            "is_running": system.is_running,
            "learning_phase": system.learning_phase,
            "free_spots": int(system.free_count),
            "total_spots": int(total_spots),
            "spots_status": [{"id": s["id"], "occupied": bool(s["occupied"])} for s in system.parking_spots],
            "detected_cars_count": int(len(system.detected_cars)),
            "detected_cars": system.detected_cars,
            "last_update": system.last_update,
            "occupancy_rate": occupancy_rate,
            "connection_status": system.connection_status,
            "model_version": "YOLO v11",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500




@app.route("/api/video-feed")
def video_feed():
    try:
        system = get_system_from_request()


        if system.current_frame is None:
            # do not crash, return clear reason
            return jsonify({
                "success": False,
                "error": "No frame yet",
                "connection_status": system.connection_status,
                "location": system.location_id,
            }), 200


        ret, buffer = cv2.imencode(".jpg", system.current_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            return jsonify({"success": False, "error": "Encoding error"}), 500


        frame_b64 = base64.b64encode(buffer).decode("utf-8")
        return jsonify({
            "success": True,
            "frame": f"data:image/jpeg;base64,{frame_b64}",
            "connection_status": system.connection_status,
            "location": system.location_id,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500




@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "initialized": _initialized,
        "locations": list(parking_systems.keys()),
        "time": datetime.now().isoformat()
    })




# ---------------------------------------------------------
# 9) LOCAL RUN (not used by Render gunicorn)
# ---------------------------------------------------------
if __name__ == "__main__":
    start_background_workers()
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)





