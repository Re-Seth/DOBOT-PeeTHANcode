import numpy as np
import serial.tools.list_ports
from pydobot import Dobot
import time
import os
import matplotlib
matplotlib.use('Agg') # ⭐️ เพิ่ม: ตั้งค่า Matplotlib ไม่ให้ใช้ GUI
import matplotlib.pyplot as plt
import glob
import sys
import cv2  
import re  
from PIL import Image
import threading # ⭐️ เพิ่ม: สำหรับการวาดแบบไม่บล็อก Server
import shutil # ⭐️ เพิ่ม: สำหรับการจัดการไฟล์

# ⭐️ เพิ่ม: Import ที่จำเป็นสำหรับ Flask Server
from flask import Flask, request, jsonify, send_from_directory, render_template

try:
    from PIL import Image
    RESAMPLE_FILTER = Image.Resampling.LANCZOS
except ImportError:
    RESAMPLE_FILTER = Image.LANCZOS 

# ================== CONFIG (Base Settings - Combined) ==================
REMOVE_BG_FOLDER = 'thaan_code/remove_background' 
OUTPUT_DIR_BASE = 'drawing_experiments_combined' 
EXP_PREFIX = 'exp_'

CANNY_THRESHOLD_LOWER = 30 
CANNY_THRESHOLD_UPPER = 90  
MAX_OUTPUT_DIMENSION = 400  
GAUSSIAN_BLUR_SIZE = (3, 3) 
CONTOUR_RETRIEVAL_MODE = cv2.RETR_LIST
CONTOUR_APPROX_METHOD = cv2.CHAIN_APPROX_NONE
IMAGE_EXTENSIONS = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')

MIN_CONTOUR_AREA = 5 
MAX_CONNECT_DISTANCE = 1.0 # [mm] 
CONTOUR_APPROX_EPSILON = 0.0005 

TRIPLE_COLLAGE_HEIGHT_PX = 600

# Dobot Settings
PEN_DOWN_Z = -39
PEN_UP_Z = 20
DOBOT_SPEED = 1500
DOBOT_ACCELERATION = 1000
RETRY_ATTEMPTS = 3
JUMP_HEIGHT_OFFSET = 10 # mm

# Fixed Paper Corners (MM) - ⭐️ นี่คือค่าเริ่มต้น
DEFAULT_PAPER_CORNERS = np.float32([
    [88.06, 31.66], [223.18, 38.10], [223.18, -73.39], [88.06, -54.85]
])

# ================== ⭐️ Global State Variables (สำหรับ Web Server) ==================
app = Flask(__name__)
g_bot = None # Global Dobot object
g_drawing_state = {
    'status': 'idle', # idle, drawing, paused, error
    'progress': 0,
    'message': 'Waiting',
    'progress_image_url': None # ⭐️ นี่คือส่วนที่เพิ่มสำหรับ Request ที่ 2
}
g_stop_drawing_flag = False # ⭐️ Flag สำหรับสั่งหยุด
g_drawing_thread = None # ⭐️ Thread ที่ใช้ในการวาด

# ⭐️ Global paths: เราจะใช้ค่าคงที่แทนการสร้างแบบไดนามิกใน Server
# เพื่อให้ URL ที่ส่งไปหน้าเว็บคงที่
OUTPUT_UPLOAD_DIR = os.path.join(OUTPUT_DIR_BASE, 'uploads')
OUTPUT_CURRENT_RUN_PATH = os.path.join(OUTPUT_DIR_BASE, 'current_run_web') 
os.makedirs(OUTPUT_UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_CURRENT_RUN_PATH, exist_ok=True)

# ⭐️ Global settings ที่จะถูกอัปเดตจาก Web UI
g_current_paper_corners = DEFAULT_PAPER_CORNERS.copy()
g_pen_settings = {
    'speed_percent': 50,
    'pen_offset': 0.0,
    'safety_height': 10.0
}
g_generated_paths = {
    'sorted_paths_mm': [],
    'sorted_lengths_mm': [],
    'bg_img': None,
    'm_inv': None
}
# ==============================================================================

# ----------------- Utility Functions (Dobot & Progress) -----------------
# (ฟังก์ชันส่วนใหญ่จากไฟล์เดิมของคุณ: find_dobot_port, safe_move, safe_jump, ฯลฯ)

def find_dobot_port():
    ports = serial.tools.list_ports.comports()
    dobot_port = None
    for p in ports:
        if not hasattr(p, 'description') or not hasattr(p, 'device'):
            continue
        is_dobot = "USB" in p.description.upper() or \
                   "SERIAL" in p.description.upper() or \
                   "CH340" in p.description.upper() or \
                   "CP210" in p.description.upper()
        is_dobot = is_dobot or \
                   "MODEM" in p.device.upper() or \
                   "USB" in p.device.upper() 
        if is_dobot:
            print(f"✅ พบพอร์ตที่น่าจะเป็น Dobot: {p.device} ({p.description})")
            dobot_port = p.device
            break
    if not dobot_port:
        print("⚠️ ไม่พบ Dobot")
    return dobot_port

def safe_move(bot, x, y, z, r=0, wait=True):
    if bot is None: return True
    for i in range(RETRY_ATTEMPTS):
        try:
            bot.move_to(x, y, z, r, wait=wait)
            return True
        except Exception as e:
            print(f"Error in safe_move: {e}")
            if i < RETRY_ATTEMPTS - 1: time.sleep(0.1)
    return False

def safe_jump(bot, x, y, z, r=0, wait=True):
    if bot is None: return True
    current_pos = [0,0,0,0]
    if bot:
        try:
            current_pos = bot.pose()
        except Exception:
            pass 
    for i in range(RETRY_ATTEMPTS):
        try:
            if bot: 
                 bot.move_to(current_pos[0], current_pos[1], z + JUMP_HEIGHT_OFFSET, r, wait=True)
            bot.move_j(x, y, z, r, wait=wait)
            return True
        except Exception as e:
            print(f"Error in safe_jump: {e}")
            if i < RETRY_ATTEMPTS - 1: time.sleep(0.1)
    return False

def create_progress_image(base_img_bgr, sorted_paths_mm, current_set_index, M_inv, path_prefix, is_final=False):
    """
    สร้างและบันทึกภาพ Progress (CV2)
    ⭐️ แก้ไข: บันทึกทับไฟล์เดิมเสมอเพื่อให้ URL คงที่
    """
    global g_drawing_state
    
    h, w, _ = base_img_bgr.shape
    preview = np.full((h, w, 3), 255, dtype=np.uint8) 
    
    # วาดขอบกระดาษ (ใช้ g_current_paper_corners)
    paper_corners_px = cv2.perspectiveTransform(g_current_paper_corners.reshape(-1, 1, 2), M_inv).reshape(-1, 2)
    cv2.polylines(preview, [paper_corners_px[[0,1,2,3,0]].astype(np.int32)], isClosed=True, color=(0, 165, 0), thickness=1, lineType=cv2.LINE_AA) 

    # เส้นที่วาดเสร็จแล้ว (สีฟ้า)
    if current_set_index > 0:
        for path_mm in sorted_paths_mm[:current_set_index]:
            path_px = cv2.perspectiveTransform(path_mm, M_inv).reshape(-1, 2)
            cv2.polylines(preview, [path_px.astype(np.int32)], isClosed=False, color=(255, 0, 0), thickness=2)

    # เส้นวาดปัจจุบัน (สีเขียว)
    if not is_final and current_set_index < len(sorted_paths_mm):
        path_mm_current = sorted_paths_mm[current_set_index]
        path_px_current = cv2.perspectiveTransform(path_mm_current, M_inv).reshape(-1, 2)
        cv2.polylines(preview, [path_px_current.astype(np.int32)], isClosed=False, color=(0, 255, 0), thickness=3) 

    status_name = 'done' if is_final else 'drawing'
    # ⭐️ บันทึกทับไฟล์เดิมเสมอ
    filename_current = os.path.join(path_prefix, f"current_progress.png")
    cv2.imwrite(filename_current, preview)
    
    # ⭐️ อัปเดต Global State ให้หน้าเว็บดึงไปแสดงผล
    if not is_final:
        g_drawing_state['progress_image_url'] = f"/output/current_progress.png"

# --- (ข้ามฟังก์ชันที่ไม่จำเป็นสำหรับ Server เช่น PDF, numerical_sort, collage) ---

# ----------------- Core Function: Optimized Sorting -----------------
# (ยกฟังก์ชัน `euclidean_distance` และ `sort_contours_for_efficiency` มาไว้ที่นี่)

def euclidean_distance(p1, p2):
    if p1 is None or p2 is None:
        return float('inf')
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def sort_contours_for_efficiency(initial_paths_data, max_connect_dist):
    valid_paths_data = [item for item in initial_paths_data if item['path'] is not None and len(item['path']) > 1]
    if not valid_paths_data: return [], []
    valid_paths_data.sort(key=lambda p: (p['centroid_mm'][1], p['centroid_mm'][0])) 
    remaining_paths_data = list(valid_paths_data) 
    all_path_sets = [] 
    def find_best_next_path(current_end_point_mm, current_remaining_data):
        best_i_in_data = -1
        min_connection_distance = float('inf')
        reverse_flag = False
        for i, item in enumerate(current_remaining_data):
            next_path = item['path'] 
            dist_to_start = euclidean_distance(current_end_point_mm, next_path[0])
            dist_to_end = euclidean_distance(current_end_point_mm, next_path[-1])
            if dist_to_start < min_connection_distance and dist_to_start <= max_connect_dist:
                min_connection_distance = dist_to_start
                best_i_in_data = i
                reverse_flag = False
            if dist_to_end < min_connection_distance and dist_to_end <= max_connect_dist:
                min_connection_distance = dist_to_end
                best_i_in_data = i
                reverse_flag = True
        if best_i_in_data != -1:
            return best_i_in_data, reverse_flag, min_connection_distance
        else:
            return -1, False, float('inf')
    while remaining_paths_data:
        current_item = remaining_paths_data.pop(0) 
        current_path_segment = current_item['path'].copy()
        while True:
            current_end_point_mm = current_path_segment[-1] 
            best_i, reverse_flag, dist = find_best_next_path(current_end_point_mm, remaining_paths_data)
            if best_i != -1:
                next_item_to_add = remaining_paths_data.pop(best_i) 
                next_path_to_add = next_item_to_add['path']
                if reverse_flag:
                    next_path_to_add = next_path_to_add[::-1]
                current_path_segment = np.vstack((current_path_segment, next_path_to_add))
            else:
                length = np.sum(np.sqrt(np.sum(np.diff(current_path_segment, axis=0) ** 2, axis=1)))
                all_path_sets.append({'path': current_path_segment, 'length': length, 'centroid_mm': current_item['centroid_mm']})
                break
    all_path_sets.sort(key=lambda p: p['length'], reverse=True)
    main_outline_paths = all_path_sets[:5] 
    detail_paths = all_path_sets[5:]
    detail_paths.sort(key=lambda p: (p['centroid_mm'][1], p['centroid_mm'][0])) 
    final_sorted_paths = main_outline_paths + detail_paths
    sorted_lengths = [p['length'] for p in final_sorted_paths]
    sorted_paths_final = [p['path'].reshape(-1, 1, 2) for p in final_sorted_paths]
    return sorted_paths_final, sorted_lengths

# ----------------- Overlap Filter Function -----------------
def filter_overlapping_contours(all_contours_px):
    contours_with_area = [(cnt, cv2.contourArea(cnt)) for cnt in all_contours_px]
    contours_with_area.sort(key=lambda x: x[1])
    final_contours = []
    for i, (small_cnt, small_area) in enumerate(contours_with_area):
        if small_area < MIN_CONTOUR_AREA:
            continue
        is_redundant = False
        for j in range(i + 1, len(contours_with_area)):
            large_cnt, large_area = contours_with_area[j]
            M = cv2.moments(small_cnt)
            if M["m00"] == 0: continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            distance = cv2.pointPolygonTest(large_cnt, (cx, cy), False) 
            if distance > 0:
                is_redundant = True
                break
        if is_redundant: continue 
        for approved_cnt in final_contours:
             M = cv2.moments(small_cnt)
             if M["m00"] == 0: continue
             cx = int(M["m10"] / M["m00"])
             cy = int(M["m01"] / M["m00"])
             distance = cv2.pointPolygonTest(approved_cnt, (cx, cy), False)
             if distance > 0 and cv2.contourArea(approved_cnt) > small_area:
                 is_redundant = True
                 break
        if not is_redundant:
            final_contours.append(small_cnt)
    return final_contours

# ----------------- Main Experiment Runner (Path Planning) -----------------
# ⭐️ แก้ไข: ให้รับ paper_corners_mm มาเป็น argument
def run_experiment(exp_name, contour_approx_epsilon, max_connect_distance, base_paths_px, base_img_bgr, paper_corners_mm):
    
    # ⭐️ ใช้ Path ที่กำหนดใน Global
    CURRENT_RUN_TEST_PATH = OUTPUT_CURRENT_RUN_PATH 
    
    h, w, _ = base_img_bgr.shape
    base_img_contour_bg = np.full((h, w, 3), 255, dtype=np.uint8)

    reprocessed_paths_data_mm = [] 
    
    # ⭐️ คำนวณ M, M_inv โดยใช้ g_current_paper_corners
    img_h, img_w = base_img_bgr.shape[:2]
    img_corners = np.float32([[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]])
    M = cv2.getPerspectiveTransform(img_corners, paper_corners_mm)
    M_inv = cv2.getPerspectiveTransform(paper_corners_mm, img_corners) 

    for cnt_px in base_paths_px: 
        perimeter = cv2.arcLength(cnt_px, True)
        approx_px = cv2.approxPolyDP(cnt_px, contour_approx_epsilon * perimeter, True) 
        if len(approx_px) > 1:
            pts_transformed_mm = cv2.perspectiveTransform(approx_px.astype(np.float32), M).reshape(-1, 2)
            M_cnt = cv2.moments(approx_px) 
            if M_cnt["m00"] != 0:
                cx_px = int(M_cnt["m10"] / M_cnt["m00"])
                cy_px = int(M_cnt["m01"] / M_cnt["m00"])
                centroid_px = np.array([[[cx_px, cy_px]]], dtype=np.float32)
                centroid_mm = cv2.perspectiveTransform(centroid_px, M).reshape(-1, 2)[0]
            else:
                centroid_mm = pts_transformed_mm[0] 
            reprocessed_paths_data_mm.append({'path': pts_transformed_mm, 'centroid_mm': centroid_mm})

    all_final_paths_mm, all_final_lengths = sort_contours_for_efficiency(reprocessed_paths_data_mm, max_connect_distance)
    total_drawing_length = sum(all_final_lengths)
    num_sets = len(all_final_paths_mm)
    print(f"  -> ผลลัพธ์: ชุดเส้นรวม (Sets): {num_sets}, ความยาวรวม: {total_drawing_length:.1f} mm")

    if num_sets > 0:
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0) 
        ax.set_aspect('equal')
        
        paper_corners_px = cv2.perspectiveTransform(paper_corners_mm.reshape(-1, 1, 2), M_inv).reshape(-1, 2)
        ax.plot(paper_corners_px[[0,1,2,3,0], 0], paper_corners_px[[0,1,2,3,0], 1], 'g--', linewidth=1, label='Paper Area')

        prev_end_point = None 
        for i, path_mm in enumerate(all_final_paths_mm):
            path_px = cv2.perspectiveTransform(path_mm, M_inv).reshape(-1, 2)
            ax.plot(path_px[:, 0], path_px[:, 1], 'b-', linewidth=2, label='Drawing Path' if i == 0 else "")
            if prev_end_point is not None:
                prev_end_point_px = cv2.perspectiveTransform(np.array([[prev_end_point]]), M_inv).reshape(-1, 2)[0]
                current_start_point_px = path_px[0]
                ax.plot([prev_end_point_px[0], current_start_point_px[0]], 
                        [prev_end_point_px[1], current_start_point_px[1]], 
                        'r--', linewidth=1, label='Jump Path' if i == 0 else "")
            prev_end_point = path_mm[-1][0] 
            
        ax.set_title(f"{exp_name} (Sets: {num_sets}, Len: {total_drawing_length:.0f} mm, Connect: {max_connect_distance:.1f} mm)")
        ax.axis("off")
        ax.legend()
        plt.tight_layout()
        
        plan_path = os.path.join(CURRENT_RUN_TEST_PATH, 'drawing_plan.png')
        try:
            plt.savefig(plan_path)
        except Exception as e:
            print(f"Error saving plot: {e}")
        plt.close(fig) 
        
        # ⭐️ สร้างภาพ Progress ภาพสุดท้าย (ภาพที่เสร็จสมบูรณ์)
        create_progress_image(base_img_contour_bg, all_final_paths_mm, num_sets, M_inv, CURRENT_RUN_TEST_PATH, is_final=True)
    else:
        print("  -> ⚠️ ไม่พบชุดเส้นทางวาด")
    
    return num_sets, total_drawing_length, all_final_paths_mm, all_final_lengths, base_img_contour_bg, M_inv

# ==============================================================================
# ⬇️⬇️⬇️ ฟังก์ชันสำหรับ Thread การวาด (แทนที่ run_dobot_drawing เดิม) ⬇️⬇️⬇️
# ==============================================================================

def drawing_thread_function():
    """
    ฟังก์ชันนี้จะรันใน Thread แยก เพื่อวาดโดยไม่บล็อก Web Server
    - ไม่มีการ `input()`
    - อ่านค่าจาก Global State (g_bot, g_generated_paths, g_pen_settings)
    - เขียนสถานะไปยัง Global State (g_drawing_state)
    - ตรวจสอบ g_stop_drawing_flag
    """
    global g_bot, g_drawing_state, g_generated_paths, g_pen_settings, g_stop_drawing_flag

    if g_bot is None:
        g_drawing_state = {'status': 'error', 'progress': 0, 'message': 'Dobot not connected'}
        return
    
    if not g_generated_paths['sorted_paths_mm']:
        g_drawing_state = {'status': 'error', 'progress': 0, 'message': 'No drawing paths generated'}
        return

    # 1. ดึงข้อมูลที่จำเป็นจาก Global State
    sorted_paths_final_mm = g_generated_paths['sorted_paths_mm']
    sorted_lengths = g_generated_paths['sorted_lengths_mm']
    preview_img_bgr = g_generated_paths['bg_img']
    M_inv = g_generated_paths['m_inv']
    
    # ⭐️ ใช้ค่า Z, Speed จาก g_pen_settings
    pen_z_down = PEN_DOWN_Z + g_pen_settings['pen_offset']
    pen_z_up = pen_z_down + g_pen_settings['safety_height'] # ⭐️ ใช้ Safety Height เทียบกับจุดกด
    
    # ⭐️ คำนวณ Speed (แปลง % เป็นค่าที่ pydobot ใช้)
    # pydobot speed: 1000 = ช้า, 8000 = เร็วมาก (อ้างอิงจากโค้ด pydobot)
    # เราจะ map 1-100% ไปยัง 1000-8000
    speed_percent = g_pen_settings['speed_percent']
    dobot_speed_val = 1000 + (speed_percent / 100.0) * (7000) 
    dobot_accel_val = dobot_speed_val # ให้ Accel สัมพันธ์กับ Speed
    
    g_drawing_state = {'status': 'drawing', 'progress': 0, 'message': 'Starting...'}
    g_stop_drawing_flag = False

    try:
        g_bot.speed(dobot_speed_val, dobot_accel_val)
        
        # 2. เริ่มลูปวาด (คล้ายของเดิม แต่ไม่มี input)
        start_index = 0 # ⭐️ เริ่มจาก 0 เสมอ
        offset_vector = np.array([0.0, 0.0]) # ⭐️ ไม่มีการย้ายตำแหน่ง

        # บันทึกภาพสถานะเริ่มต้น
        create_progress_image(preview_img_bgr, sorted_paths_final_mm, start_index, M_inv, OUTPUT_CURRENT_RUN_PATH, is_final=False)

        # Jump ไปมุมกระดาษ Top-Left ก่อนเริ่ม (ใช้ g_current_paper_corners)
        safe_jump(g_bot, g_current_paper_corners[0][0], g_current_paper_corners[0][1], pen_z_up, wait=True) 
        time.sleep(0.5)

        start_time = time.time()
        total_length_to_draw = sum(sorted_lengths[start_index:])
        current_length_drawn = 0
        
        last_pen_down_x, last_pen_down_y = g_current_paper_corners[0][0], g_current_paper_corners[0][1] 

        for i in range(start_index, len(sorted_paths_final_mm)):
            
            # ⭐️⭐️⭐️ ตรวจสอบ Flag การหยุด ⭐️⭐️⭐️
            if g_stop_drawing_flag:
                print("🛑 Stop flag detected. Halting drawing.")
                g_drawing_state['message'] = 'Stopped by user'
                break
                
            # ⭐️⭐️⭐️ ตรวจสอบการ Pause ⭐️⭐️⭐️
            while g_drawing_state['status'] == 'paused':
                if g_stop_drawing_flag: # ตรวจสอบอีกครั้งเผื่อกดยกเลิกขณะ Pause
                    break
                time.sleep(0.5) # รอขณะ Pause

            if g_stop_drawing_flag:
                break # ออกจากลูปหลักถ้าโดนสั่งหยุดขณะ Pause

            ci = i + 1
            pts_original = sorted_paths_final_mm[i] 
            if pts_original is None or len(pts_original) < 2:
                continue
            
            # ⭐️ ไม่มีการใช้ offset_vector (หรือคือ [0,0])
            pts_transformed = (pts_original.reshape(-1, 2) + offset_vector).reshape(-1, 1, 2)
            
            # สร้างภาพ Progress
            create_progress_image(preview_img_bgr, sorted_paths_final_mm, i, M_inv, OUTPUT_CURRENT_RUN_PATH, is_final=False)

            # 1. Jump
            sx, sy = pts_transformed[0][0]
            safe_jump(g_bot, sx, sy, pen_z_up, wait=True) 
            
            # 2. ลดปากกา
            safe_move(g_bot, sx, sy, pen_z_down, wait=True)
            
            # 3. วาด
            for p in pts_transformed[1:]:
                x, y = p[0]
                safe_move(g_bot, x, y, pen_z_down, wait=False) 
                last_pen_down_x, last_pen_down_y = x, y

            # 4. รอและยกปากกา
            if g_bot is not None:
                g_bot.wait(1) 
            safe_move(g_bot, last_pen_down_x, last_pen_down_y, pen_z_up, wait=True) 

            # 6. คำนวณความคืบหน้า
            current_length_drawn += sorted_lengths[i]
            percent_done = (current_length_drawn / total_length_to_draw) * 100
            
            g_drawing_state['progress'] = round(percent_done, 1)
            g_drawing_state['message'] = f'Drawing path {ci}/{len(sorted_paths_final_mm)}'
            
            print(f"✅ วาดชุดเส้นที่ {ci}/{len(sorted_paths_final_mm)} เสร็จ | Progress: {percent_done:.1f}%")
        
        # 3. สิ้นสุดการวาด
        safe_move(g_bot, last_pen_down_x, last_pen_down_y, pen_z_up, wait=True)
        
        if g_stop_drawing_flag:
            g_drawing_state['status'] = 'idle'
            g_drawing_state['progress'] = 0
            g_drawing_state['message'] = 'Stopped'
        else:
            # บันทึกภาพเสร็จสมบูรณ์
            create_progress_image(preview_img_bgr, sorted_paths_final_mm, len(sorted_paths_final_mm), M_inv, OUTPUT_CURRENT_RUN_PATH, is_final=True)
            g_drawing_state['status'] = 'idle'
            g_drawing_state['progress'] = 100
            g_drawing_state['message'] = 'Drawing complete'
            g_drawing_state['progress_image_url'] = f"/output/current_progress.png" # ⭐️ ภาพสุดท้าย

        print(f"🎉 Thread การวาดสิ้นสุด: {g_drawing_state['message']}")

    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดใน Thread การวาด: {e}")
        g_drawing_state = {'status': 'error', 'progress': 0, 'message': str(e)}
    
    finally:
        # ⭐️ เคลียร์ Flag และ Thread
        g_stop_drawing_flag = False
        g_drawing_thread = None
        # ⭐️ ไม่ต้องปิด g_bot ที่นี่ ให้ /disconnect จัดการ

# ==============================================================================
# ⬇️⬇️⬇️ API Endpoints (Flask) ⬇️⬇️⬇️
# ==============================================================================

@app.route("/")
def index():
    """เสิร์ฟไฟล์ index.html"""
    # ⭐️ ใช้ render_template เพื่อให้แน่ใจว่าไฟล์ถูกอ่านใหม่ทุกครั้ง (ดีสำหรับ debug)
    return render_template("index.html")

@app.route('/connect', methods=['POST'])
def connect_dobot():
    global g_bot
    if g_bot is not None:
        return jsonify({'status': 'success', 'message': 'Already connected'})
    
    port = find_dobot_port()
    if not port:
        return jsonify({'status': 'error', 'message': 'Dobot not found on any port.'})
    
    try:
        g_bot = Dobot(port=port, verbose=False)
        g_bot.speed(DOBOT_SPEED, DOBOT_ACCELERATION)
        print(f"✅ Dobot connected on {port}")
        return jsonify({'status': 'success', 'message': 'Connected', 'model': 'Dobot Magician', 'port': port})
    except Exception as e:
        print(f"❌ Dobot connection failed: {e}")
        g_bot = None
        return jsonify({'status': 'error', 'message': f'Connection failed: {e}'})

@app.route('/disconnect', methods=['POST'])
def disconnect_dobot():
    global g_bot, g_drawing_state
    if g_drawing_state['status'] == 'drawing':
        return jsonify({'status': 'error', 'message': 'Cannot disconnect while drawing'})
        
    if g_bot:
        try:
            g_bot.close()
            print("✅ Dobot disconnected.")
        except Exception as e:
            print(f"Error during disconnect: {e}")
    g_bot = None
    return jsonify({'status': 'success', 'message': 'Disconnected'})

@app.route('/get_position', methods=['GET'])
def get_position():
    """⭐️ API: สำหรับปุ่ม Set (4 Corners)"""
    global g_bot
    if g_bot is None:
        return jsonify({'status': 'error', 'message': 'Not connected'})
    
    try:
        pos = g_bot.pose()
        # pos = (x, y, z, r, j1, j2, j3, j4)
        return jsonify({'status': 'success', 'x': round(pos[0], 2), 'y': round(pos[1], 2), 'z': round(pos[2], 2)})
    except Exception as e:
        print(f"Error getting position: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/set_paper_corners', methods=['POST'])
def set_paper_corners():
    """⭐️ API: (FIX 1) ตั้งค่า 4 มุมกระดาษ"""
    global g_current_paper_corners
    data = request.json
    try:
        corners = data['corners']
        # แปลงจาก dict {tl, tr, ...} เป็น array ที่ CV2 ใช้
        g_current_paper_corners = np.float32([
            corners['tl'], corners['tr'], corners['br'], corners['bl']
        ])
        print(f"✅ New paper corners set: {g_current_paper_corners.tolist()}")
        return jsonify({'status': 'success', 'message': '4 corners applied'})
    except Exception as e:
        print(f"Error setting corners: {e}")
        return jsonify({'status': 'error', 'message': 'Invalid corner data'})

@app.route('/set_default_area', methods=['POST'])
def set_default_area():
    """⭐️ API: (FIX 2) รีเซ็ตกลับไปใช้ค่าเริ่มต้น"""
    global g_current_paper_corners
    g_current_paper_corners = DEFAULT_PAPER_CORNERS.copy()
    print("✅ Paper corners reset to default.")
    return jsonify({'status': 'success', 'message': 'Default center area set'})

@app.route('/process_image', methods=['POST'])
def process_image():
    """⭐️ API: อัปโหลดและประมวลผลภาพ"""
    global g_generated_paths, g_current_paper_corners
    
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'No image file provided'})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No image selected'})

    # 1. บันทึกไฟล์ที่อัปโหลด
    upload_path = os.path.join(OUTPUT_UPLOAD_DIR, file.filename)
    file.save(upload_path)
    
    # 2. ล้างค่าเก่า
    g_generated_paths = { 'sorted_paths_mm': [], 'sorted_lengths_mm': [], 'bg_img': None, 'm_inv': None }
    
    try:
        # 3. เริ่มขั้นตอนการประมวลผล (จาก __main__ เดิม)
        img = cv2.imread(upload_path)
        if img is None: 
            return jsonify({'status': 'error', 'message': 'Could not read image file'})
            
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        h, w = img.shape[:2]
        ratio = MAX_OUTPUT_DIMENSION / max(h, w)
        new_w, new_h = int(w * ratio), int(h * ratio)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # ... (สร้าง Canny)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY) 
        blurred = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_SIZE, 0)
        edges = cv2.Canny(blurred, CANNY_THRESHOLD_LOWER, CANNY_THRESHOLD_UPPER)
        edges_inverted = cv2.bitwise_not(edges) 
        canny_output_bgr = cv2.cvtColor(edges_inverted, cv2.COLOR_GRAY2BGR) 
        
        # ⭐️ บันทึก Canny (Line Art)
        canny_path = os.path.join(OUTPUT_CURRENT_RUN_PATH, 'canny_edge_output.png')
        cv2.imwrite(canny_path, canny_output_bgr)
        
        contours, hierarchy = cv2.findContours(edges, CONTOUR_RETRIEVAL_MODE, CONTOUR_APPROX_METHOD)
        base_paths_px_unfiltered = []
        for cnt in contours:
            if cv2.contourArea(cnt) >= MIN_CONTOUR_AREA and len(cnt) > 1:
                base_paths_px_unfiltered.append(cnt)

        # ⭐️ กรอง
        base_paths_px = filter_overlapping_contours(base_paths_px_unfiltered)
        print(f"🔥 Final Contours: {len(base_paths_px)} (from {len(base_paths_px_unfiltered)})")

        # 4. Run Experiment (Path Planning)
        # ⭐️ ส่ง g_current_paper_corners เข้าไป
        num_sets, total_length, sorted_paths_mm, sorted_lengths_mm, bg_img, M_inv_final = run_experiment(
            "WebApp_Run", 
            CONTOUR_APPROX_EPSILON, 
            MAX_CONNECT_DISTANCE, 
            base_paths_px, 
            img_resized, 
            g_current_paper_corners 
        )
        
        if num_sets == 0:
            return jsonify({'status': 'error', 'message': 'No drawing paths found after processing.'})

        # 5. บันทึกผลลัพธ์ลง Global State
        g_generated_paths['sorted_paths_mm'] = sorted_paths_mm
        g_generated_paths['sorted_lengths_mm'] = sorted_lengths_mm
        g_generated_paths['bg_img'] = bg_img
        g_generated_paths['m_inv'] = M_inv_final

        # 6. ส่ง URL ของภาพผลลัพธ์กลับไป
        return jsonify({
            'status': 'success',
            'message': f'Processed. Found {num_sets} paths.',
            # ⭐️ ใช้ URL ที่ชี้ไปที่ /output/
            'lineart_url': '/output/canny_edge_output.png',
            'vector_url': '/output/drawing_plan.png',
            'lineart_filename': 'canny_edge_output.png', # JS อาจจะยังใช้
            'vector_filename': 'drawing_plan.png'       # JS อาจจะยังใช้
        })
        
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': f'Processing error: {e}'})

@app.route('/output/<path:filename>')
def serve_output_file(filename):
    """⭐️ API: เสิร์ฟไฟล์ภาพที่ประมวลผล (Canny, Plan, Progress)"""
    return send_from_directory(OUTPUT_CURRENT_RUN_PATH, filename)

@app.route('/start_drawing', methods=['POST'])
def start_drawing():
    """⭐️ API: เริ่มการวาดใน Thread ใหม่"""
    global g_drawing_thread, g_drawing_state, g_pen_settings
    
    if g_drawing_thread is not None:
        return jsonify({'status': 'error', 'message': 'Already drawing'})
    
    if g_bot is None:
        return jsonify({'status': 'error', 'message': 'Dobot not connected'})

    # 1. อัปเดตการตั้งค่าจากหน้าเว็บ
    try:
        data = request.json
        g_pen_settings['speed_percent'] = float(data.get('speed', 50))
        g_pen_settings['pen_offset'] = float(data.get('pen_offset', 0))
        g_pen_settings['safety_height'] = float(data.get('safety_height', 10))
        print(f"Starting drawing with settings: {g_pen_settings}")
    except Exception as e:
        print(f"Error reading drawing settings: {e}")
        return jsonify({'status': 'error', 'message': 'Invalid drawing settings'})
        
    # 2. เริ่ม Thread
    g_drawing_state = {'status': 'drawing', 'progress': 0, 'message': 'Initializing...'}
    g_drawing_thread = threading.Thread(target=drawing_thread_function)
    g_drawing_thread.start()
    
    return jsonify({'status': 'success', 'message': 'Drawing started'})

@app.route('/progress', methods=['GET'])
def get_progress():
    """⭐️ API: สำหรับหน้าเว็บ Polling สถานะ"""
    global g_drawing_state
    # ⭐️ เพิ่ม timestamp เพื่อกัน Cache ของภาพ
    if g_drawing_state.get('progress_image_url'):
        state_copy = g_drawing_state.copy()
        state_copy['progress_image_url'] = f"{g_drawing_state['progress_image_url']}?t={time.time()}"
        return jsonify(state_copy)
        
    return jsonify(g_drawing_state)

@app.route('/pause', methods=['POST'])
def pause_drawing():
    """⭐️ API: สั่ง Pause"""
    global g_bot, g_drawing_state
    if g_drawing_state['status'] == 'drawing':
        try:
            if g_bot: g_bot.pause() # ⭐️ ใช้คำสั่ง pause ของ pydobot
            g_drawing_state['status'] = 'paused'
            g_drawing_state['message'] = 'Paused'
            print("Drawing Paused")
            return jsonify({'status': 'success', 'message': 'Paused'})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)})
    return jsonify({'status': 'error', 'message': 'Not drawing'})

@app.route('/resume', methods=['POST'])
def resume_drawing():
    """⭐️ API: สั่ง Resume"""
    global g_bot, g_drawing_state
    if g_drawing_state['status'] == 'paused':
        try:
            if g_bot: g_bot.resume() # ⭐️ ใช้คำสั่ง resume ของ pydobot
            g_drawing_state['status'] = 'drawing'
            g_drawing_state['message'] = 'Resumed' # Thread จะรับช่วงต่อ
            print("Drawing Resumed")
            return jsonify({'status': 'success', 'message': 'Resumed'})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)})
    return jsonify({'status': 'error', 'message': 'Not paused'})

@app.route('/stop', methods=['POST'])
def stop_drawing():
    """⭐️ API: สั่ง Stop"""
    global g_bot, g_drawing_state, g_stop_drawing_flag
    
    if g_drawing_state['status'] == 'drawing' or g_drawing_state['status'] == 'paused':
        print("🛑 Stop command received.")
        g_stop_drawing_flag = True # ⭐️ ตั้ง Flag ให้ Thread ตรวจสอบ
        
        try:
            if g_bot:
                # ⭐️ สั่งล้างคิวและหยุดทันที
                g_bot._set_queued_cmd_stop_exec()
                g_bot._set_queued_cmd_clear()
                # ยกปากกาขึ้นทันที
                pos = g_bot.pose()
                g_bot.move_to(pos[0], pos[1], pos[2] + 20, pos[3], wait=True)

            # Thread จะจัดการเปลี่ยน status เป็น idle เมื่อจบ
            return jsonify({'status': 'success', 'message': 'Stop signal sent'})
        except Exception as e:
            print(f"Error during stop: {e}")
            return jsonify({'status': 'error', 'message': str(e)})
            
    return jsonify({'status': 'success', 'message': 'Already stopped'})


# ==============================================================================
# ⬇️⬇️⬇️ Main Execution ⬇️⬇️⬇️
# ==============================================================================

if __name__ == "__main__":
    print("============================================")
    print("   Dobot Drawing Web Server")
    print(f"   Serving files from: {OUTPUT_CURRENT_RUN_PATH}")
    print("   Access at: http://127.0.0.1:5000")
    print("============================================")
    # ⭐️ รัน Flask Server
    # host='0.0.0.0' ทำให้เข้าถึงได้จากเครื่องอื่นใน Network เดียวกัน
    app.run(debug=True, host='0.0.0.0', port=5002, use_reloader=False)