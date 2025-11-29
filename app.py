from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import cv2
import threading
from ultralytics import YOLO
from datetime import datetime
import base64
import logging
import numpy as np

# --- 1. НАСТРОЙКА ПРИЛОЖЕНИЯ И ЛОГГИРОВАНИЯ ---
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# --- 2. ГЛОБАЛЬНЫЕ КОНСТАНТЫ ---
try:
    model = YOLO("yolov8n.pt")
    logging.info("✓ YOLO модель загружена")
except Exception as e:
    logging.error(f"✗ Ошибка загрузки модели: {e}")

CAR_CLASSES = {2, 3, 5, 7}


# --- 3. КЛАСС СИСТЕМЫ ПАРКОВКИ ---
class ParkingSystem:
    def __init__(self):
        self.camera_url = "https://webcam.elcat.kg/Bishkek_Ala-Too_Square/tracks-v1/mono.m3u8"

        logging.info("🔧 Генерация парковочных мест...")
        spots = []
        x_start = 10
        spot_size = 22
        num_spots = 35

        # Смещение базовой линии
        Y_OFFSET_DOWN = 13
        Y_BASE_START = 530 + Y_OFFSET_DOWN

        for i in range(num_spots):
            x_left = x_start + i * (spot_size + 6)


            center = (num_spots - 1) / 2.0
            distance_from_center = abs(i - center)
            vertical_offset = 0.2 * (distance_from_center ** 2)

            if i < 18:
                additional_offset = 3
            else:
                additional_offset = 5

            # Базовая Y-координата
            y_base = Y_BASE_START + additional_offset - int(vertical_offset)

            # ✔ ДОБАВЛЕНО: смещение вниз для мест 24–33 (+8 пикселей)
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

        logging.info(f"✅ Создано {len(spots)} мест. (Базовая Y-координата: {Y_BASE_START})")

        self.parking_spots = [
            {"id": i, "coords": spot, "occupied": False}
            for i, spot in enumerate(spots)
        ]

        self.current_frame = None
        self.is_running = False
        self.free_count = len(self.parking_spots)
        self.last_update = None
        self.processing_thread = None
        self.detected_cars = []


parking_system = None


# --- 4. ПРОВЕРКА ЗАНЯТОСТИ ---
def spot_occupied(spot, detections):
    for box in detections:
        try:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls not in CAR_CLASSES or conf < 0.3:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox_area = (x2 - x1) * (y2 - y1)

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            if cv2.pointPolygonTest(spot, (center_x, center_y), False) >= 0:
                return True

            mask_bbox = np.zeros((720, 960), dtype=np.uint8)
            cv2.rectangle(mask_bbox, (x1, y1), (x2, y2), 255, -1)

            mask_spot = np.zeros((720, 960), dtype=np.uint8)
            cv2.fillPoly(mask_spot, [spot], 255)

            intersection = cv2.bitwise_and(mask_bbox, mask_spot)
            intersection_area = cv2.countNonZero(intersection)

            if bbox_area > 0:
                if intersection_area / bbox_area > 0.1:
                    return True
        except:
            continue
    return False


# --- 5. ПОТОК ДЛЯ ВИДЕО ---
def process_video():
    global parking_system
    cap = cv2.VideoCapture(parking_system.camera_url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        logging.error("✗ Не удалось подключиться к камере")
        parking_system.is_running = False
        return

    logging.info("✓ Подключение к камере установлено")

    while parking_system.is_running:
        ret, frame = cap.read()

        if not ret:
            logging.warning("⚠ Потеря кадра, попытка переподключения...")
            cap.release()
            cv2.waitKey(5000)
            cap = cv2.VideoCapture(parking_system.camera_url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                logging.error("✗ Переподключение не удалось")
                break
            continue

        frame = cv2.resize(frame, (960, 720))

        try:
            results = model(frame, verbose=False, conf=0.3)[0]
        except Exception as e:
            logging.error(f"Ошибка YOLO: {e}")
            continue

        parking_system.detected_cars = []

        for box in results.boxes:
            try:
                cls = int(box.cls[0])
                if cls in CAR_CLASSES:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    parking_system.detected_cars.append({
                        "bbox": [x1, y1, x2, y2],
                        "conf": conf
                    })
            except:
                pass

        free = 0

        for i, spot_obj in enumerate(parking_system.parking_spots):
            coords = spot_obj["coords"]
            occupied = spot_occupied(coords, results.boxes)
            parking_system.parking_spots[i]["occupied"] = occupied

            if not occupied:
                free += 1

            color = (0, 0, 255) if occupied else (0, 255, 0)
            thickness = 3 if occupied else 2

            cv2.polylines(frame, [coords], True, color, thickness)

            text_x = coords[0][0]
            text_y = coords[0][1] - 5
            cv2.putText(frame, f"{i+1}", (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        cv2.putText(frame, f"СВОБОДНО: {free}/{len(parking_system.parking_spots)}",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)

        cv2.putText(frame, f"ЗАНЯТО: {len(parking_system.parking_spots) - free}",
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.putText(frame, f"Машин видно: {len(parking_system.detected_cars)}",
                    (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        parking_system.free_count = free
        parking_system.current_frame = frame.copy()
        parking_system.last_update = datetime.now().isoformat()

    cap.release()
    logging.info("✓ Камера отключена")


# --- 6. API ---
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/auth", methods=["POST"])
def auth():
    try:
        data = request.json
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
    global parking_system
    try:
        total_spots = len(parking_system.parking_spots)
        occ = total_spots - parking_system.free_count
        occupancy_rate = round((occ / total_spots * 100), 1)

        spots_json = [{"id": s["id"], "occupied": s["occupied"]} for s in parking_system.parking_spots]

        return jsonify({
            "is_running": parking_system.is_running,
            "free_spots": parking_system.free_count,
            "total_spots": total_spots,
            "spots_status": spots_json,
            "detected_cars_count": len(parking_system.detected_cars),
            "last_update": parking_system.last_update,
            "occupancy_rate": occupancy_rate
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/video-feed")
def video_feed():
    global parking_system
    try:
        if parking_system.current_frame is None:
            return jsonify({"error": "Нет кадра"}), 404

        ret, buffer = cv2.imencode(".jpg", parking_system.current_frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, 85])

        if not ret:
            return jsonify({"error": "Ошибка кодирования"}), 500

        frame_b64 = base64.b64encode(buffer).decode()

        return jsonify({
            "success": True,
            "frame": f"data:image/jpeg;base64,{frame_b64}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- 7. ЗАПУСК ---
if __name__ == "__main__":
    logging.info("🚀 Parking Analyzer запускается...")

    parking_system = ParkingSystem()
    parking_system.is_running = True

    parking_system.processing_thread = threading.Thread(
        target=process_video, daemon=True
    )
    parking_system.processing_thread.start()

    logging.info("✓ Анализ парковки запущен")
    app.run(host="0.0.0.0", port=5000, debug=False)
