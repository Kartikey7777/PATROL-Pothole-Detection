import cv2
from ultralytics import YOLO
import math
import time
import os
import csv
import threading
import winsound
from datetime import datetime
import numpy as np

# --- CONFIGURATION ---
model_path = 'best.pt'
save_folder = 'pothole_captures'
csv_file = 'pothole_report.csv'
cooldown_seconds = 3

# --- GLOBAL VARIABLES ---
current_danger_level = 0.0
program_running = True
session_start_time = time.time()
session_saved_count = 0
flash_timer = 0
last_fps_time = time.time()
fps = 0
frame_count = 0
danger_history = []

# --- COLORS (BGR) ---
COLOR_GREEN    = (0, 255, 120)
COLOR_YELLOW   = (0, 220, 255)
COLOR_RED      = (0, 60, 255)
COLOR_WHITE    = (255, 255, 255)
COLOR_BLACK    = (0, 0, 0)
COLOR_DARK_BG  = (15, 15, 25)
COLOR_ACCENT   = (0, 200, 255)
COLOR_PANEL_BG = (20, 20, 35)

FONT       = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX

# ─────────────────────────────────────────────
# BEEPER SYSTEM
# ─────────────────────────────────────────────
def proximity_beeper():
    global current_danger_level
    while program_running:
        danger = current_danger_level
        if danger > 0.01:
            pitch = int(500 + (danger * 2000))
            if pitch > 3000: pitch = 3000
            wait_time = 1.0 - (danger * 3)
            if wait_time < 0.05: wait_time = 0.05
            try:
                winsound.Beep(pitch, 100)
            except:
                pass
            time.sleep(wait_time)
        else:
            time.sleep(0.1)

# ─────────────────────────────────────────────
# GPS SIMULATOR
# ─────────────────────────────────────────────
def get_live_coordinates():
    return "23.0775 N", "76.8513 E"

# ─────────────────────────────────────────────
# DRAW HELPERS
# ─────────────────────────────────────────────
def draw_corner_box(img, x1, y1, x2, y2, color, thickness=2, corner_len=22):
    """Stylish corner brackets instead of a full rectangle"""
    cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, thickness)
    cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, thickness)
    cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, thickness)
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, thickness)

def get_severity(conf, area_ratio):
    if area_ratio > 0.08 or conf > 0.85:
        return "SEVERE",   COLOR_RED,    3
    elif area_ratio > 0.03 or conf > 0.65:
        return "MODERATE", COLOR_YELLOW, 2
    else:
        return "MINOR",    COLOR_GREEN,  1

def draw_hud_panel(img, x, y, w, h, alpha=0.65):
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), COLOR_PANEL_BG, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x, y), (x + w, y + h), COLOR_ACCENT, 1)

def draw_danger_bar(img, danger, x, y, w, h):
    cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 60), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), COLOR_ACCENT, 1)
    fill_w = int(w * min(danger * 5, 1.0))
    if fill_w > 0:
        bar_color = COLOR_GREEN if danger < 0.1 else COLOR_YELLOW if danger < 0.2 else COLOR_RED
        cv2.rectangle(img, (x, y), (x + fill_w, y + h), bar_color, -1)
    for i in range(1, 5):
        tx = x + int(w * i / 5)
        cv2.line(img, (tx, y), (tx, y + h), (80, 80, 100), 1)

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
beep_thread = threading.Thread(target=proximity_beeper)
beep_thread.daemon = True
beep_thread.start()

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Latitude", "Longitude", "Confidence", "Severity", "Image Name"])

print("Loading Pothole Detection System...")
model = YOLO(model_path)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

last_save_time = 0

print("=" * 50)
print("  POTHOLE DETECTION SYSTEM - READY")
print("  Press Q to quit | S to save manually")
print("=" * 50)

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
while True:
    success, img = cap.read()
    if not success:
        break

    frame_count += 1
    current_time_loop = time.time()

    # FPS Calculation
    if current_time_loop - last_fps_time >= 1.0:
        fps = frame_count
        frame_count = 0
        last_fps_time = current_time_loop

    height, width, _ = img.shape
    screen_area = width * height

    # ── AI Detection ──
    results = model.track(img, augment=True, stream=True, verbose=False, 
                      conf=0.25, iou=0.45, tracker="bytetrack.yaml", 
                      persist=True)

    max_box_area   = 0
    pothole_detected = False
    best_conf      = 0
    detected_potholes = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil((box.conf[0] * 100)) / 100

            if conf > 0.5:
                pothole_detected = True
                if conf > best_conf:
                    best_conf = conf
                area = (x2 - x1) * (y2 - y1)
                area_ratio = area / screen_area
                if area > max_box_area:
                    max_box_area = area
                severity, box_color, sev_level = get_severity(conf, area_ratio)
                detected_potholes.append((x1, y1, x2, y2, conf, severity, box_color, area_ratio))

    # ── Draw Detections ──
    for (x1, y1, x2, y2, conf, severity, box_color, area_ratio) in detected_potholes:

        # Outer glow
        glow = tuple(max(0, c - 130) for c in box_color)
        draw_corner_box(img, x1 - 5, y1 - 5, x2 + 5, y2 + 5, glow, thickness=4, corner_len=26)
        # Main bracket
        draw_corner_box(img, x1, y1, x2, y2, box_color, thickness=2, corner_len=22)

        # Label pill
        label = f" {severity}  {int(conf * 100)}% "
        lsz = cv2.getTextSize(label, FONT_SMALL, 0.52, 1)[0]
        lx1, ly1 = x1, max(0, y1 - lsz[1] - 12)
        lx2, ly2 = x1 + lsz[0] + 6, y1

        overlay = img.copy()
        cv2.rectangle(overlay, (lx1, ly1), (lx2, ly2), box_color, -1)
        cv2.addWeighted(overlay, 0.82, img, 0.18, 0, img)
        cv2.putText(img, label, (lx1 + 3, ly2 - 4), FONT_SMALL, 0.52, COLOR_BLACK, 1, cv2.LINE_AA)

        # Center crosshair
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.drawMarker(img, (cx, cy), box_color, cv2.MARKER_CROSS, 14, 1)

    # ── Danger Smoothing ──
    if pothole_detected:
        current_danger_level = max_box_area / screen_area
    else:
        current_danger_level = max(0.0, current_danger_level - 0.02)

    danger_history.append(current_danger_level)
    if len(danger_history) > 10:
        danger_history.pop(0)
    smooth_danger = sum(danger_history) / len(danger_history)

    # ── TOP BAR ──
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (width, 50), COLOR_DARK_BG, -1)
    cv2.addWeighted(overlay, 0.78, img, 0.22, 0, img)
    cv2.line(img, (0, 50), (width, 50), COLOR_ACCENT, 1)

    cv2.putText(img, "POTHOLE DETECTION SYSTEM", (12, 33), FONT, 0.7, COLOR_ACCENT, 1, cv2.LINE_AA)

    now_str  = datetime.now().strftime("%H:%M:%S")
    date_str = datetime.now().strftime("%d %b %Y")
    cv2.putText(img, now_str,  (width - 155, 22), FONT,       0.6,  COLOR_WHITE,        1, cv2.LINE_AA)
    cv2.putText(img, date_str, (width - 155, 42), FONT_SMALL, 0.42, (160, 160, 180),    1, cv2.LINE_AA)

    # ── CENTER STATUS INDICATOR ──
    sx = width // 2
    if pothole_detected:
        pulse = int(time.time() * 4) % 2
        rc = COLOR_RED if pulse == 0 else COLOR_YELLOW
        cv2.circle(img, (sx - 90, 27), 8, rc, -1)
        cv2.circle(img, (sx - 90, 27), 10, rc, 1)
        cv2.putText(img, "POTHOLE DETECTED", (sx - 78, 33), FONT_SMALL, 0.52, COLOR_RED, 1, cv2.LINE_AA)
    else:
        cv2.circle(img, (sx - 50, 27), 7, COLOR_GREEN, -1)
        cv2.putText(img, "ALL CLEAR", (sx - 38, 33), FONT_SMALL, 0.52, COLOR_GREEN, 1, cv2.LINE_AA)

    # ── LEFT PANEL ──
    pw, px, py = 205, 10, 60
    draw_hud_panel(img, px, py, pw, 215)

    elapsed = int(time.time() - session_start_time)
    mins, secs = divmod(elapsed, 60)

    cv2.putText(img, "SESSION TIME",        (px+10, py+20),  FONT_SMALL, 0.38, COLOR_ACCENT, 1, cv2.LINE_AA)
    cv2.putText(img, f"{mins:02d}:{secs:02d}", (px+10, py+48), FONT,       0.9,  COLOR_WHITE,  1, cv2.LINE_AA)
    cv2.line(img, (px+8, py+56), (px+pw-8, py+56), (50,50,80), 1)

    cv2.putText(img, "POTHOLES SAVED",   (px+10, py+76),  FONT_SMALL, 0.38, COLOR_ACCENT, 1, cv2.LINE_AA)
    cv2.putText(img, str(session_saved_count), (px+10, py+106), FONT, 1.1, COLOR_WHITE, 1, cv2.LINE_AA)
    cv2.line(img, (px+8, py+116), (px+pw-8, py+116), (50,50,80), 1)

    cv2.putText(img, "DANGER LEVEL",     (px+10, py+136), FONT_SMALL, 0.38, COLOR_ACCENT, 1, cv2.LINE_AA)
    draw_danger_bar(img, smooth_danger, px+10, py+148, pw-20, 13)

    dlabel, dcol = ("CLEAR", COLOR_GREEN) if smooth_danger < 0.05 else \
                   ("CAUTION", COLOR_YELLOW) if smooth_danger < 0.15 else \
                   ("DANGER!", COLOR_RED)
    cv2.putText(img, dlabel, (px+10, py+182), FONT, 0.62, dcol, 1, cv2.LINE_AA)
    cv2.line(img, (px+8, py+190), (px+pw-8, py+190), (50,50,80), 1)

    fps_col = COLOR_GREEN if fps > 20 else COLOR_YELLOW if fps > 10 else COLOR_RED
    cv2.putText(img, "FPS", (px+10, py+210), FONT_SMALL, 0.38, COLOR_ACCENT, 1, cv2.LINE_AA)
    cv2.putText(img, str(fps), (px+45, py+210), FONT_SMALL, 0.5, fps_col, 1, cv2.LINE_AA)

    # ── BOTTOM BAR ──
    overlay = img.copy()
    cv2.rectangle(overlay, (0, height-38), (width, height), COLOR_DARK_BG, -1)
    cv2.addWeighted(overlay, 0.78, img, 0.22, 0, img)
    cv2.line(img, (0, height-38), (width, height-38), COLOR_ACCENT, 1)

    lat, lon = get_live_coordinates()
    cv2.putText(img, f"GPS  {lat}  {lon}",    (12, height-14),         FONT_SMALL, 0.44, (160,200,255), 1, cv2.LINE_AA)
    cv2.putText(img, "YOLOv8  |  ByteTrack", (width//2-85, height-14), FONT_SMALL, 0.42, (100,100,130), 1, cv2.LINE_AA)
    cv2.putText(img, "[Q] Quit   [S] Save",   (width-195, height-14),   FONT_SMALL, 0.42, (120,120,150), 1, cv2.LINE_AA)

    # ── AUTO-SAVE ──
    current_time = time.time()
    if pothole_detected and (current_time - last_save_time > cooldown_seconds):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        img_name  = f"pothole_{timestamp}.jpg"
        lat, lon  = get_live_coordinates()
        _, _, _, _, conf_s, severity_s, _, _ = detected_potholes[0]

        cv2.imwrite(os.path.join(save_folder, img_name), img)
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, lat, lon, best_conf, severity_s, img_name])

        session_saved_count += 1
        flash_timer = 20
        print(f"[SAVED] {img_name} | {severity_s} | {int(best_conf*100)}%")
        last_save_time = current_time

    # ── CAPTURE FLASH ──
    if flash_timer > 0:
        alpha_f = flash_timer / 20.0 * 0.3
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), (0, 255, 120), -1)
        cv2.addWeighted(overlay, alpha_f, img, 1 - alpha_f, 0, img)
        cv2.putText(img, "CAPTURED!", (width//2 - 90, height//2),
                    FONT, 1.2, COLOR_WHITE, 2, cv2.LINE_AA)
        flash_timer -= 1

    cv2.imshow("Pothole Detection System", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        program_running = False
        break
    elif key == ord('s') and pothole_detected:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        img_name  = f"manual_{timestamp}.jpg"
        cv2.imwrite(os.path.join(save_folder, img_name), img)
        session_saved_count += 1
        flash_timer = 20
        print(f"[MANUAL SAVE] {img_name}")

# ── END OF SESSION ──
elapsed = int(time.time() - session_start_time)
mins, secs = divmod(elapsed, 60)
print("\n" + "=" * 50)
print(f"  SESSION COMPLETE")
print(f"  Duration : {mins:02d}:{secs:02d}")
print(f"  Saved    : {session_saved_count} potholes")
print(f"  CSV Log  : {csv_file}")
print("=" * 50)

cap.release()
cv2.destroyAllWindows()