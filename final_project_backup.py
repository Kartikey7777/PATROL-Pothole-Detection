import cv2
from ultralytics import YOLO
import math
import time
import os
import csv
import threading
import winsound 
from datetime import datetime

# --- CONFIGURATION ---
model_path = 'best.pt'
save_folder = 'pothole_captures'
csv_file = 'pothole_report.csv'
cooldown_seconds = 3  # Wait 3 seconds between saving photos

# --- GLOBAL VARIABLES ---
current_danger_level = 0.0 
program_running = True

# --- 1. THE BEEPER SYSTEM (Runs in Background) ---
def proximity_beeper():
    global current_danger_level
    while program_running:
        danger = current_danger_level
        
        # Only beep if pothole is significant
        if danger > 0.01:
            # Pitch gets higher as danger increases
            pitch = int(500 + (danger * 2000))
            if pitch > 3000: pitch = 3000
            
            # Speed gets faster as danger increases
            wait_time = 1.0 - (danger * 3)
            if wait_time < 0.05: wait_time = 0.05
            
            try:
                winsound.Beep(pitch, 100)
            except:
                pass
            
            time.sleep(wait_time)
        else:
            time.sleep(0.1)

# --- 2. GPS SIMULATOR ---
def get_live_coordinates():
    return "23.0775 N", "76.8513 E"

# --- SETUP ---
# Start the Beeper Thread
beep_thread = threading.Thread(target=proximity_beeper)
beep_thread.daemon = True
beep_thread.start()

# Create Folders
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Create CSV
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Latitude", "Longitude", "Confidence", "Image Name"])

# Load AI
print("Loading Final Project System...")
model = YOLO(model_path)

# Start Camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

last_save_time = 0

print("------------------------------------------------")
print("SYSTEM READY: [Green > 80%] [Red < 80%]")
print("Features: [Audio] + [Visual] + [Logging]")
print("Press 'q' to quit.")
print("------------------------------------------------")

while True:
    success, img = cap.read()
    if not success:
        break

    # AI Detection
    results = model.track(img, augment=True, stream=True, verbose=False, conf=0.25, iou=0.45, tracker="bytetrack.yaml", persist=True)
    
    max_box_area = 0
    pothole_detected = False
    best_conf = 0
    
    height, width, _ = img.shape
    screen_area = width * height

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil((box.conf[0] * 100)) / 100
            
            if conf > 0.5:
                pothole_detected = True
                if conf > best_conf: best_conf = conf
                
                # --- NEW COLOR LOGIC ---
                if conf > 0.80:
                    # GREEN (High Confidence)
                    # OpenCV uses BGR format: (Blue, Green, Red)
                    box_color = (0, 255, 0) 
                else:
                    # RED (Low Confidence)
                    box_color = (0, 0, 255)
                # -----------------------

                # Calculate Area (for Beep)
                box_w = x2 - x1
                box_h = y2 - y1
                area = box_w * box_h
                if area > max_box_area:
                    max_box_area = area
                
                # Draw Box with Dynamic Color
                cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 3)
                cv2.putText(img, f"POTHOLE {int(conf*100)}%", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    # --- COMBINING THE FEATURES ---
    
    # Feature 1: Update Beeper
    if pothole_detected:
        current_danger_level = max_box_area / screen_area
    else:
        current_danger_level = 0.0
        
    # Feature 2: Auto-Save Photo & Log
    current_time = time.time()
    if pothole_detected and (current_time - last_save_time > cooldown_seconds):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        img_name = f"pothole_{timestamp}.jpg"
        lat, long = get_live_coordinates()
        
        # Save Image
        cv2.imwrite(os.path.join(save_folder, img_name), img)
        
        # Log CSV
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, lat, long, best_conf, img_name])
            
        print(f"[SAVED] {img_name} ({int(best_conf*100)}%)")
        last_save_time = current_time
        
        # Flash "CAPTURED" in Green
        cv2.putText(img, "CAPTURED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("Final Project", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        program_running = False
        break

cap.release()
cv2.destroyAllWindows()