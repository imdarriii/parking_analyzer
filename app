from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import cv2
import threading
from ultralytics import YOLO
from datetime import datetime
import base64
import logging

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# --- 1. Конфигурация проекта ---

# Загружаем модель YOLO
try:
    # Используем 'n' (nano) версию для быстрой работы
    model = YOLO("yolov8n.pt") 
    logging.info("✓ YOLO модель загружена")
except Exception as e:
    logging.error(f"✗ Ошибка загрузки модели: {e}")

# Классы машин в YOLO: car (2), bus (5), truck (7)
CAR_CLASSES = {2, 5, 7}  

# Координаты парковочного места "Площадь Ала-Тоо" (одно место)
# NOTE: Координаты рассчитаны для кадра 960x720, взяты из общего списка
PARKING_SPOTS = [
    (40, 650, 110, 720), 
]

class ParkingSystem:
    def __init__(self):
        # URL камеры Бишкека
        self.camera_url = "https://cam.kt.kg/cam17/stream.m3u8" 
        self.parking_spots = [
            # ID 0 соответствует "Площади Ала-Тоо"
            {"id": i, "coords": spot, "occupied": False}
            for i, spot in enumerate(PARKING_SPOTS)
        ]
        self.current_frame = None
        self.is_running = False
        self.free_count = len(self.parking_spots)
        self.last_update = None
        self.processing_thread = None

parking_system = ParkingSystem()

def spot_occupied(spot, detections):
    """Проверяет, занято ли парковочное место на основании обнаружений"""
    sx1, sy1, sx2, sy2 = spot
    for box in detections:
        try:
            cls = int(box.cls[0])
            if cls not in CAR_CLASSES:
                continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Проверяем пересечение бокса машины с зоной парковки
            if not (x2 < sx1 or x1 > sx2 or y2 < sy1 or y1 > sy2):
                return True
        except:
            continue
    return False

def process_video():
    """Поток обработки видео с камеры"""
    # ... (Остальной код process_video остается прежним) ...
    cap = cv2.VideoCapture(parking_system.camera_url)
    
    if not cap.isOpened():
        logging.error("✗ Не удалось подключиться к камере")
        parking_system.is_running = False 
        return
    
    logging.info("✓ Подключение к камере установлено")
    
    frame_count = 0
    while parking_system.is_running:
        ret, frame = cap.read()
        if not ret:
            logging.warning("⚠ Потеря кадра. Пауза 5 сек, затем повтор...")
            cv2.waitKey(5000)
            continue
        
        frame_count += 1
        
        # Уменьшаем размер для быстрой обработки
        frame = cv2.resize(frame, (960, 720))
        
        # Обнаружение объектов
        try:
            results = model(frame, verbose=False, conf=0.4)[0] 
        except Exception as e:
            logging.error(f"Ошибка YOLO: {e}")
            continue
        
        free = 0
        
        # Проверяем каждое место
        for i, spot_obj in enumerate(parking_system.parking_spots):
            coords = spot_obj["coords"]
            occupied = spot_occupied(coords, results.boxes)
            parking_system.parking_spots[i]["occupied"] = occupied
            
            if not occupied:
                free += 1
            
            # Рисуем на кадре
            x1, y1, x2, y2 = coords
            color = (0, 0, 255) if occupied else (0, 255, 0)
            label = "ЗАНЯТО" if occupied else "СВОБОДНО"
            thickness = 3
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            # Изменяем надпись на "Ала-Тоо Свободно/Занято" для единственного места
            cv2.putText(frame, f"АЛА-ТОО: {label}", (x1 - 10, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Информация на кадре
        cv2.putText(frame, f"Свободно: {free} / {len(parking_system.parking_spots)}",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        cv2.putText(frame, f"Камера Бишкек | Кадр: {frame_count}",
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        parking_system.free_count = free
        parking_system.current_frame = frame.copy()
        parking_system.last_update = datetime.now().isoformat()
    
    cap.release()
    logging.info("✓ Камера отключена")

# --- 2. Маршруты (API) ---

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/auth", methods=["POST"])
def auth():
    # ... (функция auth остается прежней) ...
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
        return jsonify({"error": f"Ошибка: {str(e)}"}), 500

# Маршруты /api/start и /api/stop УДАЛЕНЫ, т.к. анализ запускается автоматически

@app.route("/api/status")
def get_status():
    # ... (функция get_status остается прежней) ...
    try:
        total_spots = len(parking_system.parking_spots)
        occupancy_rate = round(
            ((total_spots - parking_system.free_count) / total_spots * 100), 1
        ) if total_spots > 0 else 0
        
        return jsonify({
            "is_running": parking_system.is_running,
            "free_spots": parking_system.free_count,
            "total_spots": total_spots,
            "spots": parking_system.parking_spots, 
            "last_update": parking_system.last_update,
            "occupancy_rate": occupancy_rate
        })
    except Exception as e:
        return jsonify({"error": f"Ошибка получения статуса: {str(e)}"}), 500

@app.route("/api/video-feed")
def video_feed():
    # ... (функция video_feed остается прежней) ...
    try:
        if parking_system.current_frame is None:
            return jsonify({"error": "Нет кадра"}), 404
        
        ret, buffer = cv2.imencode(".jpg", parking_system.current_frame, 
                                 [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            return jsonify({"error": "Ошибка кодирования"}), 500
        
        frame_base64 = base64.b64encode(buffer).decode()
        return jsonify({
            "success": True,
            "frame": f"data:image/jpeg;base64,{frame_base64}"
        })
    except Exception as e:
        return jsonify({"error": f"Ошибка видео: {str(e)}"}), 500

# --- 3. Запуск приложения с АВТОЗАПУСКОМ ---

if __name__ == "__main__":
    logging.info("🚀 Parking Analyzer запускается...")
    
    # --- АВТОМАТИЧЕСКИЙ ЗАПУСК АНАЛИЗА ПРИ ЗАПУСКЕ СЕРВЕРА ---
    parking_system.is_running = True
    parking_system.processing_thread = threading.Thread(
        target=process_video, 
        daemon=True
    )
    parking_system.processing_thread.start()
    logging.info("✓ Автоматический анализ запущен")
    # --------------------------------------------------------
    
    app.run(host="0.0.0.0", port=5000, debug=False)