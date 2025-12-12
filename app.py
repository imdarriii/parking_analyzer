from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import cv2
import threading
from ultralytics import YOLO
from datetime import datetime
import base64
import logging
import numpy as np
import time
import os

# ---------------------------------------------------------
# 1. НАСТРОЙКА ПРИЛОЖЕНИЯ И ЛОГГИРОВАНИЯ
# ---------------------------------------------------------
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------
# 2. ЗАГРУЗКА YOLO МОДЕЛИ
# ---------------------------------------------------------
try:
    model = YOLO("yolo11s.pt")
    logging.info("✓ YOLO v11 модель загружена")
except Exception as e:
    logging.error(f"✗ Ошибка загрузки модели: {e}")
    model = None

# ---------------------------------------------------------
# 3. КОНСТАНТЫ
# ---------------------------------------------------------
CAR_CLASSES = {2, 3, 5, 7}          # классы машин в YOLO
PROCESS_EVERY_N_FRAMES = 3          # обрабатывать каждый N-й кадр

FRAME_WIDTH = 960
FRAME_HEIGHT = 720

TECHNOPARK_VIDEO_PATH = "C:/Users/RedmiBook/Downloads/Технопарк .mp4"

AUTO_LEARNING_FRAMES = 60          # сколько кадров копим для обучения
MIN_SPOT_CONFIDENCE = 0.5          # доля кадров, в которых место должно появиться
IOU_THRESHOLD = 0.5
MOVEMENT_THRESHOLD = 15
MIN_BOX_AREA = 800                 # минимальная площадь бокса машины

parking_systems = {}

# ---------------------------------------------------------
# 4. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
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
    return np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)


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
        int(y2 + expand_h)
    ]

# ---------------------------------------------------------
# 5. КЛАСС СИСТЕМЫ ПАРКОВКИ
# ---------------------------------------------------------
class ParkingSystem:
    def __init__(self, location_id: str, source: str, auto_learn=False):
        self.location_id = location_id
        self.source = source
        self.auto_learn = auto_learn

        lower_src = source.lower()
        if lower_src.startswith("http"):
            self.is_file = False
        else:
            self.is_file = lower_src.endswith((".mp4", ".mov", ".avi", ".mkv"))

        logging.info(f"🔧 Инициализация ParkingSystem ({self.location_id}), auto_learn={auto_learn}")

        # состояние обучения
        self.learning_phase = auto_learn
        self.learning_frames = 0
        self.detected_spots_history = []
        self.previous_detections = []

        # инициализация парковочных мест
        if not auto_learn:
            # только для Ала-Тоо
            self._init_alatoo_spots()
        else:
            # для авто-обучения (Технопарк) начнём с пустого списка
            self.parking_spots = []

        # прочие поля
        self.current_frame = None
        self.is_running = False
        self.free_count = 0
        self.last_update = None
        self.processing_thread = None
        self.detected_cars = []
        self.connection_status = "Connecting..."

    # ---------- Ала-Тоо: фиксированные места ----------
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

            spot = np.array([
                [x_left + skew_offset, y_base],
                [x_left + skew_offset + spot_size, y_base],
                [x_left + spot_size, y_base + spot_size],
                [x_left, y_base + spot_size]
            ], dtype=np.int32)

            spots.append(spot)

        self.parking_spots = [
            {"id": i, "coords": spot, "occupied": False, "confidence": 1.0, "type": "alatoo"}
            for i, spot in enumerate(spots)
        ]

    # ---------- ОБУЧЕНИЕ СПОТОВ (используется для Технопарка) ----------
    def process_learning_frame(self, detections):
        """Обработка кадра во время обучения спотов (Технопарк)"""
        current_frame_boxes = []

        for box in detections:
            try:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                # фильтр по классу и уверенности
                if cls not in CAR_CLASSES or conf < 0.4:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox_area = (x2 - x1) * (y2 - y1)

                if bbox_area < MIN_BOX_AREA:
                    continue

                current_frame_boxes.append([x1, y1, x2, y2, conf])
            except Exception:
                continue

        # фильтрация движущихся машин
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
        """Финальное построение парковочных мест по истории детекций (Технопарк)"""
        logging.info(f"🎓 ({self.location_id}) Завершение обучения...")

        all_boxes = []
        for frame_boxes in self.detected_spots_history:
            all_boxes.extend(frame_boxes)

        if len(all_boxes) == 0:
            logging.warning(f"⚠ ({self.location_id}) Не обнаружено машин во время обучения")
            self.learning_phase = False
            return

        clusters = []
        for box in all_boxes:
            matched = False
            for cluster in clusters:
                avg_box = np.array([b[:4] for b in cluster['boxes']]).mean(axis=0)
                iou = calculate_iou(box[:4], avg_box)
                dist = box_distance(box[:4], avg_box)

                if iou > IOU_THRESHOLD or dist < 30:
                    cluster['boxes'].append(box)
                    matched = True
                    break

            if not matched:
                clusters.append({'boxes': [box]})

        self.parking_spots = []
        min_frames_required = int(AUTO_LEARNING_FRAMES * MIN_SPOT_CONFIDENCE)

        for i, cluster in enumerate(clusters):
            if len(cluster['boxes']) < min_frames_required:
                continue

            boxes_array = np.array([b[:4] for b in cluster['boxes']])
            avg_box = boxes_array.mean(axis=0).astype(int)

            expanded_box = expand_box(avg_box, expand_ratio=0.05)
            x1, y1, x2, y2 = expanded_box

            coords = np.array([
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2]
            ], dtype=np.int32)

            confidence = len(cluster['boxes']) / AUTO_LEARNING_FRAMES

            self.parking_spots.append({
                "id": i,
                "coords": coords,
                "occupied": False,
                "confidence": confidence,
                "type": "technopark",
                "area": (x2 - x1) * (y2 - y1)
            })

        # сортируем слева-направо, сверху-вниз
        self.parking_spots.sort(key=lambda s: (s["coords"][0][1], s["coords"][0][0]))
        for idx, spot in enumerate(self.parking_spots):
            spot["id"] = idx

        self.learning_phase = False
        self.detected_spots_history = []
        self.previous_detections = []

        logging.info(f"✅ ({self.location_id}) Обучение завершено! Создано {len(self.parking_spots)} парковочных мест")

    # ---------- ПРОВЕРКА ЗАНЯТОСТИ ----------
    def check_spot_occupancy(self, spot, detections):
        """Проверяет, занято ли место машиной"""

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
            spot_box = [x1_s, y1_s, x2_s, y2_s]
        else:
            return False

        spot_area = (spot_box[2] - spot_box[0]) * (spot_box[3] - spot_box[1])
        if spot_area == 0:
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
# 6. ПОТОК ОБРАБОТКИ ВИДЕО
# ---------------------------------------------------------
def process_video(system: ParkingSystem):
    cap = cv2.VideoCapture(system.source, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        logging.error(f"✗ ({system.location_id}) Не удалось открыть источник")
        system.connection_status = "Failed to connect"
        system.is_running = False
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)

    logging.info(f"✓ ({system.location_id}) Подключение установлено")
    system.connection_status = "Connected"

    frame_count = 0
    last_results = None
    reconnect_attempts = 0
    max_reconnect_attempts = 5

    while system.is_running:
        ret, frame = cap.read()

        if not ret:
            if system.is_file:
                logging.info(f"🔁 ({system.location_id}) Перематываем видео")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            logging.warning(f"⚠ ({system.location_id}) Потеря кадра")
            system.connection_status = "Reconnecting..."
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(system.source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            reconnect_attempts += 1

            if reconnect_attempts >= max_reconnect_attempts:
                logging.error(f"✗ ({system.location_id}) Превышено количество попыток")
                system.connection_status = "Connection failed"
                break

            if cap.isOpened():
                reconnect_attempts = 0
                system.connection_status = "Connected"
            continue

        reconnect_attempts = 0
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame_count += 1

        # YOLO-детекция
        if model is None:
            results = None
        elif frame_count % PROCESS_EVERY_N_FRAMES == 0:
            try:
                results = model(
                    frame,
                    verbose=False,
                    conf=0.25,
                    iou=0.45,
                    imgsz=640,
                    half=False,
                    device='cpu'
                )[0]
                last_results = results
            except Exception as e:
                logging.error(f"Ошибка YOLO ({system.location_id}): {e}")
                results = last_results
        else:
            results = last_results

        if results is None:
            time.sleep(0.01)
            continue

        det_boxes = results.boxes

        # ФАЗА ОБУЧЕНИЯ (Технопарк)
        if system.learning_phase:
            system.process_learning_frame(det_boxes)

            progress = int((system.learning_frames / AUTO_LEARNING_FRAMES) * 100)
            cv2.putText(frame, f"ОБУЧЕНИЕ: {progress}%",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
            cv2.putText(frame, f"Кадр: {system.learning_frames}/{AUTO_LEARNING_FRAMES}",
                        (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(frame, "Анализ парковочных мест...",
                        (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # рисуем только боксы машин
            for box in det_boxes:
                try:
                    cls = int(box.cls[0])
                    if cls in CAR_CLASSES:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                except Exception:
                    pass

            system.current_frame = frame.copy()
            time.sleep(0.01)
            continue

        # ОБЫЧНЫЙ РЕЖИМ
        system.detected_cars = []
        free = 0
        total_spots = len(system.parking_spots)

        # Проверяем занятость мест
        for i, spot in enumerate(system.parking_spots):
            occupied = system.check_spot_occupancy(spot, det_boxes)
            system.parking_spots[i]["occupied"] = occupied

            if not occupied:
                free += 1

            color = (0, 0, 255) if occupied else (0, 255, 0)
            thickness = 3 if occupied else 2

            coords = spot["coords"]
            cv2.polylines(frame, [coords], True, color, thickness)


            text_x = coords[0][0]
            text_y = coords[0][1] - 5
            cv2.putText(frame, f"{i + 1}", (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Рисуем сами машины
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
                        "class": results.names[cls]
                    })
            except Exception:
                pass

        # Текстовая инфа
        cv2.putText(frame, f"СВОБОДНО: {free}/{total_spots}",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
        cv2.putText(frame, f"ЗАНЯТО: {total_spots - free}",
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(frame, f"Машин: {len(system.detected_cars)}",
                    (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "YOLO v11", (840, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        system.free_count = free
        system.current_frame = frame.copy()
        system.last_update = datetime.now().isoformat()
        time.sleep(0.01)

    cap.release()
    logging.info(f"✓ ({system.location_id}) Источник отключен")

# ---------------------------------------------------------
# 7. API
# ---------------------------------------------------------
def normalize_location(loc_raw: str) -> str:
    if not loc_raw:
        return "ala-too"
    loc = loc_raw.strip().lower()
    if "tech" in loc or "техно" in loc:
        return "technopark"
    return "ala-too"


def get_system_from_request():
    loc_raw = request.args.get("location", "")
    loc = normalize_location(loc_raw)
    if loc not in parking_systems:
        loc = "ala-too"
    return parking_systems[loc]


@app.route("/")
def index():
    try:
        return render_template("index.html")
    except Exception:
        return "<h1>Parking Analyzer Backend</h1>"


@app.route("/api/auth", methods=["POST"])
def auth():
    try:
        data = request.json or {}
        username = data.get("username", "").strip()
        if len(username) < 2:
            return jsonify({"error": "Имя должно быть минимум 2 символа"}), 400
        return jsonify({
            "success": True,
            "message": f"Добро пожаловать, {username}! 👋",
            "user": username,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/status")
def get_status():
    try:
        system = get_system_from_request()
        total_spots = len(system.parking_spots)
        occ = total_spots - system.free_count
        occupancy_rate = round((occ / total_spots * 100), 1) if total_spots > 0 else 0.0

        return jsonify({
            "location": system.location_id,
            "is_running": system.is_running,
            "learning_phase": system.learning_phase,
            "free_spots": system.free_count,
            "total_spots": total_spots,
            "spots_status": [{"id": s["id"], "occupied": s["occupied"]} for s in system.parking_spots],
            "detected_cars_count": len(system.detected_cars),
            "detected_cars": system.detected_cars,
            "last_update": system.last_update,
            "occupancy_rate": occupancy_rate,
            "connection_status": system.connection_status
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/video-feed")
def video_feed():
    try:
        system = get_system_from_request()
        if system.current_frame is None:
            return jsonify({"error": "Нет кадра"}), 404

        ret, buffer = cv2.imencode(".jpg", system.current_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            return jsonify({"error": "Ошибка кодирования"}), 500

        frame_b64 = base64.b64encode(buffer).decode()
        return jsonify({
            "success": True,
            "frame": f"data:image/jpeg;base64,{frame_b64}",
            "connection_status": system.connection_status,
            "location": system.location_id
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------------------------------------------------
# 8. ЗАПУСК
# ---------------------------------------------------------
if __name__ == "__main__":
    logging.info("🚀 Parking Analyzer (YOLO v11) запускается...")

    # Ала-Тоо — фиксированные места
    ala_too_source = "https://webcam.elcat.kg/Bishkek_Ala-Too_Square/tracks-v1/mono.m3u8"
    ala_too_system = ParkingSystem(
        location_id="ala-too",
        source=ala_too_source,
        auto_learn=False
    )
    ala_too_system.is_running = True
    parking_systems["ala-too"] = ala_too_system

    ala_thread = threading.Thread(target=process_video, args=(ala_too_system,), daemon=True)
    ala_thread.start()

    # Технопарк — авто-обучение парковочных мест
    if os.path.exists(TECHNOPARK_VIDEO_PATH):
        technopark_system = ParkingSystem(
            location_id="technopark",
            source=TECHNOPARK_VIDEO_PATH,
            auto_learn=True
        )
        technopark_system.is_running = True
        parking_systems["technopark"] = technopark_system

        tech_thread = threading.Thread(target=process_video, args=(technopark_system,), daemon=True)
        tech_thread.start()

        logging.info("✓ Технопарк: режим авто-обучения активирован")
    else:
        logging.warning(f"⚠ Видео не найдено: {TECHNOPARK_VIDEO_PATH}")

    logging.info("✓ Активные локации: " + ", ".join(parking_systems.keys()))
    app.run(host="0.0.0.0", port=5000, debug=False)



