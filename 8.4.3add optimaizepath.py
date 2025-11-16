import cv2
import numpy as np
import serial.tools.list_ports
from pydobot import Dobot
import time
import os
import matplotlib.pyplot as plt
import shutil 
import glob 
import sys 
import subprocess 
import math 
import json 

# ================== CONFIG ==================
IMAGE_PATH = '/Users/student/Desktop/dfcall/stitched_cartoon_512x512_4_auto_crop.jpg'
OUTPUT_DIR_BASE = 'drawing_experiments' 
EXP_PREFIX = 'exp_' 
IMAGE_MAX_SIZE = 1000
PEN_DOWN_Z = -39
PEN_UP_Z = 20
RETRY_ATTEMPTS = 3

# ❗️ MODIFIED 1 (SPEED): เพิ่มความเร็วและลดความละเอียดลงเล็กน้อย
DOBOT_SPEED = 3200        # เพิ่มความเร็ว (ค่าสูงสุดที่ Dobot Magician รับได้)
DOBOT_ACCELERATION = 2000 # เพิ่มอัตราเร่ง (ช่วยให้เคลื่อนที่เร็วขึ้น)
EPSILON = 0.0015          # (ค่าเดิม 0.0005) 
                          # ❗️ นี่คือตัวแปรสำคัญ:
                          # ค่าที่ 'สูงขึ้น' -> เส้นจะมีจุดน้อยลง -> วาดเร็วขึ้นมาก
                          # (แต่ถ้าสูงไป เส้นอาจจะดูหยาบ/เป็นเหลี่ยม)
MIN_CONTOUR_AREA = 1

# ❗️ ปรับค่า EPSILON ในนี้เพื่อทดสอบ
# ❗️ แนะนำ: ลองตั้งค่าหนึ่งให้ EPSILON ต่ำๆ (เช่น 0.0005) เพื่อเน้นรายละเอียดดวงตา
# ❗️ MODIFIED (User Request): เพิ่มเป็น 9 รายการสำหรับ Grid 3x3
TEST_PARAMS = [
    # (Name, Blur, ThreshBlock, ThreshC, Epsilon, MinArea)
    # --- แถวที่ 1 ---
    ("Default (Fine)", 5, 11, 7, 0.0015, 1),
    ("High Detail (Slower)", 3, 9, 5, 0.00075, 3),
    ("Smooth Lines (High E)", 9, 15, 10, 0.002, 5),
    
    # --- แถวที่ 2 ---
    ("Coarse Detail (Low E)", 5, 21, 5, 0.0002, 10),
    ("Aggressive Thresh (Low C)", 5, 11, 2, 0.0005, 1),
    ("Very Smooth (High Blur)", 11, 15, 7, 0.003, 5),

    # --- แถวที่ 3 ---
    ("No Blur (Detail)", 3, 9, 5, 0.001, 1),
    ("Large Blocksize", 7, 31, 7, 0.0015, 3),
    ("Fine Detail (High C)", 3, 9, 10, 0.00075, 2)
]


CALIBRATION_FILE = 'dobot_calibration.json'

PAPER_CORNERS_DEFAULT = np.float32([
    [1.69, 96.04],      # top-left
    [134.10, 215.25],   # top-right
    [264.16, 28.42],    # bottom-right
    [106.29, -51.89]    # bottom-left
])
# =============================================

# Global variables
OUTPUT_ALL_STEPS_PATH = ""
OUTPUT_CURRENT_RUN_PATH = ""
OUTPUT_PROCESSED_PATH = "" # OUTPUT_PROCESSED_PATH จะไม่ถูกใช้แล้วในการแสดงผลขั้นสุดท้าย
OUTPUT_SQUARE_PATH = "" 

def load_calibration():
    """โหลดค่า PAPER_CORNERS จากไฟล์ JSON ถ้ามี"""
    if os.path.exists(CALIBRATION_FILE):
        try:
            with open(CALIBRATION_FILE, 'r') as f:
                corners_list = json.load(f)
                if len(corners_list) == 4 and all(len(c) == 2 for c in corners_list):
                    print(f"✅ โหลดค่า Calibration ล่าสุดจาก {CALIBRATION_FILE}")
                    return np.float32(corners_list)
                else:
                    print(f"⚠️ ไฟล์ {CALIBRATION_FILE} มีรูปแบบไม่ถูกต้อง, ใช้ค่า Default")
                    return PAPER_CORNERS_DEFAULT
        except Exception as e:
            print(f"⚠️ ไม่สามารถโหลด {CALIBRATION_FILE}: {e}. ใช้ค่า Default แทน")
            return PAPER_CORNERS_DEFAULT
    else:
        print(f"ℹ️ ไม่พบไฟล์ {CALIBRATION_FILE}, ใช้ค่า Default ในโค้ด")
        return PAPER_CORNERS_DEFAULT

PAPER_CORNERS = load_calibration()


# ----------------- Utility Functions -----------------

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
                   "USB" in p.device.upper() or \
                   "WCHUSB" in p.device.upper()
        if is_dobot:
            print(f"✅ พบพอร์ตที่น่าจะเป็น Dobot: {p.device} ({p.description})")
            dobot_port = p.device
            break
    if not dobot_port:
        print("\n⚠️ ไม่พบ Dobot โดยอัตโนมัติ ลองดูรายการพอร์ตทั้งหมดที่พบ:")
        all_ports = [f"  - {p.device} ({getattr(p, 'description', 'N/A')})" for p in ports if hasattr(p, 'device')]
        if all_ports:
            print("\n".join(all_ports))
        else:
            print("❌ ไม่พบพอร์ต Serial ใด ๆ เลย")
        return None  
    return dobot_port

def safe_move(bot, x, y, z, r=0, wait=True):
    for i in range(RETRY_ATTEMPTS):
        try:
            bot.move_to(x, y, z, r, wait=wait)
            return True
        except Exception as e:
            if i < RETRY_ATTEMPTS - 1:
                time.sleep(0.1) 
    return False

def get_next_experiment_dir():
    os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)
    existing_dirs = glob.glob(os.path.join(OUTPUT_DIR_BASE, f'{EXP_PREFIX}[0-9]*'))
    max_num = 0
    for dir_path in existing_dirs:
        try:
            num_str = os.path.basename(dir_path).replace(EXP_PREFIX, '')
            max_num = max(max_num, int(num_str))
        except ValueError:
            continue
    next_num = max_num + 1
    new_exp_dir = os.path.join(OUTPUT_DIR_BASE, f'{EXP_PREFIX}{next_num}')
    global OUTPUT_ALL_STEPS_PATH, OUTPUT_CURRENT_RUN_PATH # OUTPUT_PROCESSED_PATH ถูกลบออก
    OUTPUT_ALL_STEPS_PATH = os.path.join(new_exp_dir, 'all_steps')
    OUTPUT_CURRENT_RUN_PATH = os.path.join(new_exp_dir, 'current_run')
    # OUTPUT_PROCESSED_PATH จะไม่ถูกใช้ในลักษณะ Global อีกต่อไป
    os.makedirs(OUTPUT_ALL_STEPS_PATH, exist_ok=True)
    os.makedirs(OUTPUT_CURRENT_RUN_PATH, exist_ok=True)
    print(f"✅ สร้างโฟลเดอร์ทดลองใหม่: {new_exp_dir}/")
    return new_exp_dir

def create_progress_image(base_img_bgr, filtered_contours, current_contour_index, is_final=False):
    preview = base_img_bgr.copy()
    if current_contour_index > 1:
        cv2.drawContours(preview, filtered_contours[:current_contour_index-1], -1, (255, 0, 0), 1) 
    if not is_final and current_contour_index <= len(filtered_contours):
        cv2.drawContours(preview, [filtered_contours[current_contour_index-1]], -1, (0, 255, 0), 2)
    if not is_final:
        filename_all = os.path.join(OUTPUT_ALL_STEPS_PATH, f"step_{current_contour_index:04d}_drawing.jpg")
        cv2.imwrite(filename_all, preview)
    filename_current = os.path.join(OUTPUT_CURRENT_RUN_PATH, f"current_progress_{'done' if is_final else 'drawing'}.jpg")
    cv2.imwrite(filename_current, preview)

def update_current_progress_image(base_img_bgr, filtered_contours, current_contour_index, is_final=False):
    preview = base_img_bgr.copy()
    if current_contour_index > 1:
        cv2.drawContours(preview, filtered_contours[:current_contour_index-1], -1, (255, 0, 0), 1) 
    if not is_final and current_contour_index <= len(filtered_contours):
        cv2.drawContours(preview, [filtered_contours[current_contour_index-1]], -1, (0, 255, 0), 2)
    filename_current = os.path.join(OUTPUT_CURRENT_RUN_PATH, f"current_progress_{'done' if is_final else 'drawing'}.jpg")
    cv2.imwrite(filename_current, preview)


# ⭐️⭐️⭐️ NEW HELPER FUNCTION ⭐️⭐️⭐️
def get_aspect_ratio_corrected_img_corners(img_shape_hw, paper_corners_quad):
    """
    คำนวณพิกัดมุมของ 'ภาพต้นทาง' (img_corners) ใหม่
    โดยเพิ่ม "padding" เสมือน เพื่อให้อัตราส่วนของภาพตรงกับอัตราส่วนของกระดาษ
    ซึ่งจะป้องกันไม่ให้ภาพถูกบิดเบี้ยว (ยืด/หด) ตอนวาด
    """
    
    # 1. รับขนาดของภาพ (pixel)
    img_h, img_w = img_shape_hw
    if img_h == 0: img_ratio = 1.0
    else: img_ratio = img_w / img_h

    # 2. คำนวณขนาด "เฉลี่ย" ของกระดาษ (mm)
    tl, tr, br, bl = paper_corners_quad
    paper_top_width = np.linalg.norm(tr - tl)
    paper_bottom_width = np.linalg.norm(br - bl)
    paper_left_height = np.linalg.norm(bl - tl)
    paper_right_height = np.linalg.norm(br - tr)
    
    paper_width = (paper_top_width + paper_bottom_width) / 2
    paper_height = (paper_left_height + paper_right_height) / 2
    
    if paper_height == 0: # ป้องกันการหารด้วยศูนย์
        paper_ratio = 1.0
    else:
        paper_ratio = paper_width / paper_height

    # 3. สร้าง "กรอบ" (padding)
    padding_w = 0.0
    padding_h = 0.0

    if img_ratio > paper_ratio:
        # 🖼️ ภาพกว้างกว่ากระดาษ (ต้องเพิ่ม padding บน/ล่าง)
        # เราจะยึด 'ความกว้าง' ของภาพเป็นหลัก
        new_total_height_px = img_w / paper_ratio
        padding_h = (new_total_height_px - img_h) / 2.0
    
    elif img_ratio < paper_ratio:
        # 📱 ภาพสูงกว่ากระดาษ (ต้องเพิ่ม padding ซ้าย/ขวา)
        # เราจะยึด 'ความสูง' ของภาพเป็นหลัก
        new_total_width_px = img_h * paper_ratio
        padding_w = (new_total_width_px - img_w) / 2.0
    
    # 4. กำหนดพิกัดมุมใหม่ของ "ภาพ" (รวม padding)
    img_corners = np.float32([
        [-padding_w, -padding_h],                                 # top-left
        [img_w - 1 + padding_w, -padding_h],                      # top-right
        [img_w - 1 + padding_w, img_h - 1 + padding_h],           # bottom-right
        [-padding_w, img_h - 1 + padding_h]                       # bottom-left
    ])
    
    # ส่งข้อมูลสำหรับแสดง Log กลับไปด้วย
    return img_corners, (paper_width, paper_height, paper_ratio), (img_w, img_h, img_ratio)


# ❗️ MODIFIED (User Request): อัปเดตฟังก์ชันนี้ให้ใช้ Aspect Ratio ที่ถูกต้อง
# ❗️❗️❗️ [MODIFIED] - แก้ไขฟังก์ชันนี้ให้ return 5 ค่า (เพิ่ม img, thresh) ❗️❗️❗️
def process_and_draw_contours(img_gray, blur_ksize, thresh_blocksize, thresh_c, epsilon_factor, min_contour_area):
    if blur_ksize % 2 == 0: blur_ksize += 1
    if blur_ksize < 3: blur_ksize = 3
    img = cv2.GaussianBlur(img_gray, (blur_ksize, blur_ksize), 0)
    
    if thresh_blocksize % 2 == 0: thresh_blocksize += 1
    if thresh_blocksize < 3: thresh_blocksize = 3
    thresh = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, thresh_blocksize, thresh_c
    )
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    filtered_contours = []
    total_length_mm = 0.0
    
    # ⭐️ MODIFIED: คำนวณ img_corners ใหม่เพื่อรักษาอัตราส่วน
    # (ใช้ global PAPER_CORNERS)
    img_corners, _, _ = get_aspect_ratio_corrected_img_corners(img_gray.shape, PAPER_CORNERS)
    M = cv2.getPerspectiveTransform(img_corners, PAPER_CORNERS) 
    # (จบส่วนแก้ไข)

    for cnt in contours:
        if cv2.contourArea(cnt) < min_contour_area:
            continue
        arc_length = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon_factor * arc_length, True)
        if len(approx) >= 2:
            filtered_contours.append(approx)
            pts = np.array(approx, dtype=np.float32).reshape(-1, 1, 2)
            # ใช้ M ที่ถูกต้องในการคำนวณความยาว (mm)
            pts_transformed = cv2.perspectiveTransform(pts, M)
            length = np.sum(np.sqrt(np.sum(np.diff(pts_transformed.reshape(-1, 2), axis=0)**2, axis=1)))
            total_length_mm += length
            
    preview_img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(preview_img_bgr, filtered_contours, -1, (0, 0, 255), 1) 
    
    # ❗️❗️❗️ [MODIFIED] - ส่งคืน img (เบลอ) และ thresh กลับไปด้วย
    return preview_img_bgr, filtered_contours, total_length_mm, img, thresh

# ❗️ MODIFIED (User Request): แก้ไขฟังก์ชันนี้ให้แสดงผลแบบ 3x3 (9 รายการ)
# (ฟังก์ชันนี้ถูกแก้ไขให้แสดงผลแบบ 3x3 และลบภาพ Original ออก)
def visualize_parameters(original_img_color, original_img_gray, test_params, output_dir):
    
    # ❗️ MODIFIED 2: เปลี่ยนเป็น 3x3 (A4 Portrait)
    fig, axs = plt.subplots(3, 3, figsize=(8.27, 11.69)) 
    axs = axs.flatten()
    
    # ❗️ MODIFIED 3: ลบการแสดงภาพ Original (axs[0]) ออก
    #    (Grid นี้จะแสดงเฉพาะผลลัพธ์ 9 แบบ)
    
    all_test_params = TEST_PARAMS
    
    if len(all_test_params) < 9:
        print(f"⚠️ คำเตือน: TEST_PARAMS มีเพียง {len(all_test_params)} รายการ แต่ต้องการ 9 รายการสำหรับ Grid 3x3")

    # ❗️ MODIFIED 4: แก้ไข Loop ให้วน 9 รอบ (หรือน้อยกว่า) โดยเริ่มจาก i=0
    for i, (name, blur, block, c, eps, min_area) in enumerate(all_test_params):
        if i >= len(axs): # (กันไว้เผื่อ TEST_PARAMS มีมากกว่า 9)
            break 
            
        # ❗️❗️❗️ [MODIFIED] - รับ 5 ค่า (ใช้ _ ละเว้นค่าที่ไม่ต้องการ) ❗️❗️❗️
        processed_img_bgr, _, length_mm, _, _ = process_and_draw_contours(
            original_img_gray.copy(), 
            blur_ksize=blur, 
            thresh_blocksize=block, 
            thresh_c=c, 
            epsilon_factor=eps, 
            min_contour_area=min_area
        )
        
        # ❗️ MODIFIED 5: ใช้ 'i' เป็น index ของ axs
        axs[i].imshow(cv2.cvtColor(processed_img_bgr, cv2.COLOR_BGR2RGB))
        params_text = f"B={blur}, T={block}, C={c}, E={eps*1000:.2f}e-3, MinA={min_area}"
        
        # ❗️ MODIFIED 6: ใช้ 'i+1' สำหรับการนับเลข
        axs[i].set_title(
            f"{i+1}. {name}\n({params_text})", 
            fontsize=8
        )
        axs[i].axis("off")
        
    # ❗️ MODIFIED 7: ลบแกนที่เหลือ (ถ้า TEST_PARAMS < 9)
    for j in range(len(all_test_params), len(axs)):
        fig.delaxes(axs[j])
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # ❗️ MODIFIED 8: อัปเดต Title
    plt.suptitle("Dobot Drawing Parameter Comparison (3x3 Grid)", fontsize=16, fontweight='bold')
    output_filename = os.path.join(output_dir, "parameter_comparison.jpg")
    plt.savefig(output_filename, dpi=200) 
    print(f"✅ บันทึกภาพเปรียบเทียบที่: {output_filename}")
    
    return output_filename # (ส่งคืนแค่ชื่อไฟล์ภาพรวม)

def run_calibration_mode(bot):
    """
    โหมดสำหรับให้ผู้ใช้ขยับ Dobot ไปยังมุมกระดาษ 4 มุม (Teach Mode)
    และ "บันทึก" ค่าลงไฟล์ JSON อัตโนมัติ
    """
    print("\n--- 🤖 โหมดตั้งค่ารูปทรง/ขนาดกระดาษ ---")
    print("1. กดปุ่มบนแขน Dobot ค้างไว้ (Teach Mode)")
    print("2. ขยับหัวปากกาไปยังมุมกระดาษตามลำดับ")
    print("3. กด Enter ที่คอมพิวเตอร์เพื่อบันทึกพิกัดในแต่ละมุม")
    
    corners = [] 
    corner_names = [
        "มุมบนซ้าย (Top-Left)", 
        "มุมบนขวา (Top-Right)", 
        "มุมล่างขวา (Bottom-Right)", 
        "มุมล่างซ้าย (Bottom-Left)"
    ]
    
    try:
        current_pose = bot.pose()
        safe_move(bot, current_pose[0], current_pose[1], PEN_UP_Z, 0, wait=True) 
        
        for name in corner_names:
            input(f"\n👉 กรุณาขยับหัวปากกาไปที่ '{name}' แล้วกด [Enter] เพื่อบันทึก...")
            pose = bot.pose()
            x, y = round(pose[0], 2), round(pose[1], 2) 
            corners.append([x, y])
            print(f"✅ บันทึกแล้ว {name}: (X={x:.2f}, Y={y:.2f})")
            time.sleep(0.2)
    
        print("\n--- ✅ ตั้งค่าเสร็จสิ้น ---")
        
        try:
            with open(CALIBRATION_FILE, 'w') as f:
                json.dump(corners, f, indent=4)
            print(f"✅ บันทึกค่า Calibration ใหม่ลงใน {CALIBRATION_FILE} เรียบร้อย")
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดในการบันทึกไฟล์: {e}")

        global PAPER_CORNERS
        PAPER_CORNERS = np.float32(corners)
        print("✅ อัปเดตค่าในหน่วยความจำแล้ว")
        
        print("👉 คุณสามารถเลือก 'โหมด 1' เพื่อเริ่มวาดด้วยค่าใหม่นี้ได้เลย")
        print("กำลังกลับไปที่เมนูหลัก...")
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดระหว่างตั้งค่า: {e}")
    
    current_pose = bot.pose()
    safe_move(bot, current_pose[0], current_pose[1], PEN_UP_Z + 10, 0, wait=True) 


# ----------------- ❗️❗️❗️ [REMOVED] ลบฟังก์ชัน Optimize -----------------
# (ฟังก์ชัน optimize_path_nearest_neighbor ถูกลบออกจากที่นี่)
# -----------------

# ----------------- ❗️❗️❗️ [MODIFIED] แก้ไขฟังก์ชัน Visualize -----------------
# (เปลี่ยนชื่อจาก visualize_optimized_path เป็น visualize_drawing_path)
def visualize_drawing_path(base_img_gray, drawing_data, M_inv, output_dir):
    """
    สร้างภาพ Visualization ที่แสดงลำดับการวาด (Travel Path)
    - สีน้ำเงิน: คือเส้นที่จะวาด (Contours)
    - สีแดง: คือเส้นทางการเคลื่อนที่ตอนยกปากกา (Travel)
    """
    try:
        vis_img = cv2.cvtColor(base_img_gray, cv2.COLOR_GRAY2BGR)
        
        # 1. วาดเส้น Contour ทั้งหมด (สีน้ำเงิน)
        all_contours_px = [item['contour_px'] for item in drawing_data]
        cv2.drawContours(vis_img, all_contours_px, -1, (255, 0, 0), 1) # (Blue)

        # 2. วาดเส้นเดินทาง (สีแดง)
        for i in range(len(drawing_data) - 1):
            # 2a. หาจุดสิ้นสุด (mm) ของเส้นนี้ และ จุดเริ่มต้น (mm) ของเส้นถัดไป
            end_point_mm = drawing_data[i]['end_point_mm']
            start_point_mm = drawing_data[i+1]['start_point_mm']
            
            # 2b. แปลงพิกัดจาก mm (Dobot) กลับไปเป็น px (Image)
            end_mm_np = np.float32([[end_point_mm]]).reshape(-1, 1, 2)
            start_mm_np = np.float32([[start_point_mm]]).reshape(-1, 1, 2)
            
            end_px_np = cv2.perspectiveTransform(end_mm_np, M_inv)
            start_px_np = cv2.perspectiveTransform(start_mm_np, M_inv)
            
            p1 = (int(end_px_np[0][0][0]), int(end_px_np[0][0][1]))
            p2 = (int(start_px_np[0][0][0]), int(start_px_np[0][0][1]))
            
            # 2c. วาดเส้นสีแดงเชื่อม
            cv2.line(vis_img, p1, p2, (0, 0, 255), 1) # (Red)

        # 3. บันทึกภาพ
        # ❗️❗️❗️ [MODIFIED] - เปลี่ยนชื่อไฟล์ ❗️❗️❗️
        save_path = os.path.join(output_dir, "05_drawing_path_visualization.jpg")
        cv2.imwrite(save_path, vis_img)
        print(f"✅ บันทึกภาพ '05_drawing_path_visualization.jpg' สำเร็จ!")

    except Exception as e:
        print(f"⚠️ เกิดข้อผิดพลาดในการสร้างภาพ Visualization: {e}")

# ----------------- Drawing Mode Function -----------------
# ❗️❗️❗️ [MODIFIED] - แก้ไขฟังก์ชัน run_drawing_mode ทั้งหมด ❗️❗️❗️
def run_drawing_mode(bot): 
    new_exp_dir = get_next_experiment_dir()
    
    print("⏳ กำลังโหลดและประมวลผลรูปภาพ...")
    img_color = cv2.imread(IMAGE_PATH) 
    if img_color is None:
        print(f"❌ ไม่พบรูปภาพที่ {IMAGE_PATH}")
        print("❗️ โปรดตรวจสอบว่าได้แก้ไขตัวแปร 'IMAGE_PATH' ให้ถูกต้อง")
        return

    original_h, original_w = img_color.shape[:2]
    scale_factor = IMAGE_MAX_SIZE / max(original_h, original_w)
    target_w = int(original_w * scale_factor)
    target_h = int(original_h * scale_factor)
    
    img_color_resized = cv2.resize(img_color, (target_w, target_h), interpolation=cv2.INTER_AREA)
    img_gray_resized = cv2.cvtColor(img_color_resized, cv2.COLOR_BGR2GRAY)
    
    # (รับแค่ชื่อไฟล์ภาพรวม)
    comparison_image_path = visualize_parameters(
        img_color_resized, 
        img_gray_resized.copy(), 
        TEST_PARAMS, 
        OUTPUT_CURRENT_RUN_PATH
    )

    # (เปิดเฉพาะภาพเปรียบเทียบแบบตาราง)
    print(f"🖼️ กำลังเปิดภาพเปรียบเทียบที่: {comparison_image_path}")
    try:
        if sys.platform == "win32": os.startfile(comparison_image_path)
        elif sys.platform == "darwin": subprocess.call(["open", comparison_image_path])
        else: subprocess.call(["xdg-open", comparison_image_path])
        time.sleep(0.5) 

    except Exception as e:
        print(f"⚠️ ไม่สามารถเปิดภาพอัตโนมัติได้: {e}")

    # (แก้ไขข้อความเมนู)
    print(f"\n👉 กรุณาดูภาพเปรียบเทียบที่เปิดขึ้นมาบนหน้าจอ")
    print("\n" + "="*40)
    print("  🖼️ เลือก Parameters ที่ต้องการใช้วาด 🖼️")
    print("="*40)

    for i, (name, blur, block, c, eps, min_area) in enumerate(TEST_PARAMS):
        # (อัปเดตการแสดงผลให้รองรับ 9 รายการ)
        print(f" {i+1}. {name} (B={blur}, T={block}, C={c}, E={eps}, MinA={min_area})")
    
    choice = 0
    while True:
        try:
            user_input = input(f"\n👉 กรุณาเลือกหมายเลข (1-{len(TEST_PARAMS)}) [ค่าเริ่มต้นคือ 1]: ")
            if not user_input:
                choice = 1
                break
            choice = int(user_input)
            if 1 <= choice <= len(TEST_PARAMS):
                break
            else:
                print(f"⚠️ หมายเลขต้องอยู่ระหว่าง 1 ถึง {len(TEST_PARAMS)}")
        except ValueError:
            print("❌ กรุณาใส่ตัวเลขเท่านั้น")

    selected_params = TEST_PARAMS[choice - 1]
    name, blur_ksize, thresh_blocksize, thresh_c, epsilon_factor, min_contour_area = selected_params
    
    print(f"\n✅ คุณเลือก: {name}")
    print(f"⏳ กำลังประมวลผลภาพด้วยค่าที่เลือก (B={blur_ksize}, T={thresh_blocksize}, C={thresh_c}, E={epsilon_factor}, MinA={min_contour_area})...")
    
    img = img_gray_resized.copy() 
    
    # ❗️❗️❗️ [MODIFIED] - รับ 5 ค่า และเพิ่มบล็อกสำหรับบันทึกภาพกลับเข้ามา ❗️❗️❗️
    
    # รับ 5 ค่าที่ return มาจากฟังก์ชัน
    # (filtered_contours ที่ได้จากที่นี่ จะอยู่ใน "ลำดับดั้งเดิม")
    preview_img_bgr, filtered_contours, total_drawing_length, img_blurred, img_thresh = process_and_draw_contours(
        img, 
        blur_ksize, 
        thresh_blocksize, 
        thresh_c, 
        epsilon_factor,  
        min_contour_area 
    )

    # --- (ส่วนที่เพิ่มใหม่) บันทึกภาพขั้นตอนต่างๆ ---
    print(f"✅ กำลังบันทึกภาพขั้นตอนการประมวลผลลงใน: {OUTPUT_CURRENT_RUN_PATH}/")
    try:
        # 0. บันทึกภาพ Grayscale ต้นฉบับ (ที่ Resize แล้ว)
        save_path_gray = os.path.join(OUTPUT_CURRENT_RUN_PATH, "01_selected_grayscale.jpg")
        cv2.imwrite(save_path_gray, img) 
        
        # 1. บันทึกภาพเบลอ (ผลจาก GaussianBlur)
        save_path_blur = os.path.join(OUTPUT_CURRENT_RUN_PATH, "02_selected_blurred.jpg")
        cv2.imwrite(save_path_blur, img_blurred)
        
        # 2. บันทึกภาพ Threshold (ผลจาก AdaptiveThreshold)
        save_path_thresh = os.path.join(OUTPUT_CURRENT_RUN_PATH, "03_selected_threshold.jpg")
        cv2.imwrite(save_path_thresh, img_thresh)
        
        # 3. บันทึกภาพ Contours (ภาพเส้นเวกเตอร์สีแดงทับบน Grayscale)
        save_path_contours = os.path.join(OUTPUT_CURRENT_RUN_PATH, "04_selected_final_contours.jpg")
        cv2.imwrite(save_path_contours, preview_img_bgr)
        
        print("✅ บันทึกภาพขั้นตอน 01, 02, 03, 04 สำเร็จ!")
    except Exception as e:
        print(f"⚠️ เกิดข้อผิดพลาดในการบันทึกภาพขั้นตอน: {e}")
    # --- (จบส่วนที่เพิ่มใหม่) ---


    if not filtered_contours:
        print(f"❌ ไม่พบ Contours ที่มีขนาดเหมาะสม โปรดปรับค่าใน TEST_PARAMS")
        return
        
    print(f"✅ พบ {len(filtered_contours)} contours ที่จะถูกวาด")

    # ⭐️ MODIFIED: คำนวณ img_corners ใหม่เพื่อรักษาอัตราส่วน
    print("ℹ️ กำลังคำนวณอัตราส่วนภาพเพื่อให้พอดีกับกระดาษ...")
    
    # คำนวณ img_corners ใหม่เพื่อรักษาอัตราส่วน
    img_corners, paper_dims, img_dims = get_aspect_ratio_corrected_img_corners(img.shape, PAPER_CORNERS)
    
    # (แสดง Log)
    p_w, p_h, p_r = paper_dims
    i_w, i_h, i_r = img_dims
    print(f"   - อัตราส่วนภาพ (W/H): {i_r:.3f} ({i_w}x{i_h} px)")
    print(f"   - อัตราส่วนกระดาษ (W/H): {p_r:.3f} ({p_w:.1f}x{p_h:.1f} mm)")
    if i_r > p_r:
        print(f"   - ⚖️ ผลลัพธ์: ภาพ 'กว้างกว่า' กระดาษ, จะถูกบีบให้พอดี (Letterbox)")
    elif i_r < p_r:
        print(f"   - ⚖️ ผลลัพธ์: ภาพ 'สูงกว่า' กระดาษ, จะถูกบีบให้พอดี (Pillarbox)")
    else:
        print("   - ✅ อัตราส่วนภาพและกระดาษตรงกัน")

    M = cv2.getPerspectiveTransform(img_corners, PAPER_CORNERS)
    # ❗️❗️❗️ [MODIFIED] - คำนวณ Matrix (M_inv) สำหรับแปลง mm กลับไปเป็น px
    M_inv = cv2.getPerspectiveTransform(PAPER_CORNERS, img_corners)
    # (จบส่วนแก้ไข)


    print("⏳ กำลังเตรียมพิกัด Dobot...")
    
    # ❗️❗️❗️ [MODIFIED] - กลับไปใช้การสร้าง List แบบ 3 List ขนานกัน ❗️❗️❗️
    processed_paths = []
    contour_lengths = []
    # (เราใช้ 'filtered_contours' (px) ที่ได้มาจาก process_and_draw_contours)
    
    for cnt_px in filtered_contours: # (วน Loop 'filtered_contours' ในลำดับดั้งเดิม)
        if len(cnt_px) < 2:
            processed_paths.append(None)
            contour_lengths.append(0)
            continue
            
        pts_px = np.array(cnt_px, dtype=np.float32).reshape(-1, 1, 2)
        
        # ⭐️ MODIFIED: ใช้ M ที่ถูกต้อง
        pts_transformed_mm = cv2.perspectiveTransform(pts_px, M) 
        processed_paths.append(pts_transformed_mm)
        
        length_mm = np.sum(np.sqrt(np.sum(np.diff(pts_transformed_mm.reshape(-1, 2), axis=0)**2, axis=1)))
        contour_lengths.append(length_mm)
    
    print(f"✅ ตรวจสอบความยาวรวม: {total_drawing_length:.2f} mm")


    START_CONTOUR_INDEX = 1
    start_index_original = 0 # (Index ที่ 0)
    
    while True: 
        print("\n--- 🎯 วิธีเลือก Contour เริ่มต้น ---")
        print("1. ⌨️  พิมพ์หมายเลข Contour (เช่น 1, 50, 150)")
        print("2. 👆 ชี้จุดที่ต้องการเริ่มวาด (ค้นหาเส้นที่ใกล้ที่สุด)")
        print("[Enter] 💨 เริ่มจากเส้นแรก (ค่าเริ่มต้น = 1)")
        
        method_choice = input("กรุณาเลือก (1, 2, หรือ Enter): ").strip()

        if method_choice == '1':
            while True:
                try:
                    user_input = input(f"👉 กรุณาใส่หมายเลข Contour เริ่มต้น (1 ถึง {len(processed_paths)}) หรือ Enter เพื่อใช้ 1: ")
                    if not user_input:
                        START_CONTOUR_INDEX = 1
                    else:
                        START_CONTOUR_INDEX = int(user_input)
                        if not (1 <= START_CONTOUR_INDEX <= len(processed_paths)):
                            print(f"⚠️ หมายเลขต้องอยู่ระหว่าง 1 ถึง {len(processed_paths)}")
                            continue
                    start_index_original = START_CONTOUR_INDEX - 1
                    print(f"✅ เลือก Contour ที่ {START_CONTOUR_INDEX}")
                    break 
                except ValueError:
                    print("❌ กรุณาใส่ตัวเลขเท่านั้น")
            break 

        elif method_choice == '2':
            print("\n--- 👆 กำหนดจุดเริ่มวาด (โดยการชี้) ---")
            print("1. กดปุ่มบนแขน Dobot ค้างไว้ (Teach Mode)")
            print("2. ขยับหัวปากกาไปยัง 'เส้น' ที่คุณต้องการเริ่มวาด")
            x_target, y_target = 0, 0
            try:
                input(f"👉 กรุณาขยับหัวปากกาไปที่ 'จุดเริ่มวาด' ที่ต้องการ แล้วกด [Enter]...")
                start_pose = bot.pose()
                x_target, y_target = start_pose[0], start_pose[1]
                print(f"✅ บันทึกจุดเป้าหมายแล้ว: (X={x_target:.2f}, Y={y_target:.2f})")
                
                print("⏳ กำลังค้นหา Contour ที่ใกล้ที่สุดกับจุดที่คุณชี้...")
                min_dist = float('inf')
                closest_index = 0
                for i, path_mm in enumerate(processed_paths):
                    if path_mm is None or len(path_mm) == 0:
                        continue
                    start_point_x, start_point_y = path_mm[0][0]
                    dist = math.sqrt((x_target - start_point_x)**2 + (y_target - start_point_y)**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_index = i
                
                START_CONTOUR_INDEX = closest_index + 1
                start_index_original = closest_index
                print(f"✅ พบ! จะเริ่มวาดจาก Contour ที่: {START_CONTOUR_INDEX} (ใกล้สุด {min_dist:.2f} mm)")
                break 
            except Exception as e:
                print(f"❌ ไม่สามารถอ่านค่าได้: {e}.")
        elif method_choice == '':
            START_CONTOUR_INDEX = 1
            start_index_original = 0
            print("✅ เลือก Contour ที่ 1 (ค่าเริ่มต้น)")
            break
        else:
            print("⚠️ ไม่พบคำสั่ง กรุณาเลือก 1, 2, หรือ Enter")

    plt.close() 

    # ❗️❗️❗️ [MODIFIED] - นำตรรกะ "Wrap-around" (จัดลำดับ List) แบบเก่ากลับมา ❗️❗️❗️
    start_index = start_index_original 

    if start_index != 0:
        print(f"✅ กำลังจัดลำดับการวาดใหม่: จะเริ่มที่เส้น {START_CONTOUR_INDEX} แล้ววนกลับมาวาดที่เหลือ")
        
        # จัดเรียง List ใหม่ทั้งหมด 3 รายการ
        processed_paths = processed_paths[start_index:] + processed_paths[:start_index]
        filtered_contours = filtered_contours[start_index:] + filtered_contours[:start_index]
        contour_lengths = contour_lengths[start_index:] + contour_lengths[:start_index]
        
        # สร้าง List ของ "ดัชนีเดิม" เพื่อใช้ในการแสดงผล Log
        original_indices_logging = [i+1 for i in range(start_index, len(processed_paths))] + [i+1 for i in range(0, start_index)]
    else:
        # ถ้าเริ่มจาก 1 ก็ไม่ต้องทำอะไร
        print("✅ เริ่มวาดจากเส้นแรกตามปกติ")
        original_indices_logging = [i+1 for i in range(len(processed_paths))]
    
    # ⭐️⭐️⭐️ (จบส่วนที่นำกลับมา) ⭐️⭐️⭐️

    # (ส่วนการสร้าง Checkpoint จะทำงานได้ทันที
    #  เพราะ 'filtered_contours' ถูกจัดลำดับใหม่ (Wrap-around) แล้ว)
    print(f"⏳ กำลังสร้างภาพลำดับการวาด (แบบ Wrap-around) ลงใน {OUTPUT_ALL_STEPS_PATH}/ ...")
    for ci in range(1, len(filtered_contours) + 1):
        create_progress_image(preview_img_bgr, filtered_contours, ci, is_final=False)
    create_progress_image(preview_img_bgr, filtered_contours, len(filtered_contours) + 1, is_final=True)
    
    print(f"✅ ความยาวเส้นวาดทั้งหมด (ตามอัตราส่วนจริง): {total_drawing_length:.2f} mm") 
    print(f"✅ สร้างภาพ Checkpoint ทั้งหมด {len(filtered_contours)} ภาพเสร็จสิ้น!")
    
    # ❗️❗️❗️ [MODIFIED] - สร้างภาพ Visualization (เส้นสีแดง) ❗️❗️❗️
    print(f"⏳ กำลังสร้างภาพแสดงเส้นทาง (05_drawing_path_visualization.jpg)...")
    
    # 1. ต้อง "แพ็ค" ข้อมูลที่จัดลำดับใหม่ (Wrap-around)
    #    กลับไปเป็น List of Dictionaries เพื่อส่งให้ฟังก์ชัน visualize
    drawing_data_for_vis = []
    for i in range(len(processed_paths)):
        if processed_paths[i] is None:
            continue
        drawing_data_for_vis.append({
            'contour_px': filtered_contours[i],
            'start_point_mm': processed_paths[i][0][0],
            'end_point_mm': processed_paths[i][-1][0]
        })
    
    # 2. เรียกใช้ฟังก์ชัน Visualize
    visualize_drawing_path(img, drawing_data_for_vis, M_inv, OUTPUT_CURRENT_RUN_PATH)
    
    
    print("\n" + "="*50)
    print(f"⭐️ คุณสามารถตรวจสอบ 'ลำดับการวาด' ทั้งหมดได้แล้ว")
    print(f"   ในโฟลเดอร์: {OUTPUT_ALL_STEPS_PATH}/")
    print(f"⭐️ และดู 'เส้นทาง' (เส้นสีแดง) ได้ที่:") # (ลบคำว่า Optimize ออก)
    print(f"   ในโฟลเดอร์: {os.path.join(OUTPUT_CURRENT_RUN_PATH, '05_drawing_path_visualization.jpg')}")
    print("="*50 + "\n")
    # ⭐️⭐️⭐️ (จบส่วนที่ย้ายมา) ⭐️⭐️⭐️

    # (แสดงภาพ Contour แรก "ของ List ใหม่" ที่กำลังจะวาด)
    update_current_progress_image(preview_img_bgr, filtered_contours, 1, is_final=False) 
    print(f"🖼️ ภาพ Progress ปัจจุบันถูกบันทึกที่: {os.path.join(OUTPUT_CURRENT_RUN_PATH, 'current_progress_drawing.jpg')}")

    print("\n--- 📌 กำหนดจุดเริ่มต้น (Home) ---")
    print("1. กดปุ่มบนแขน Dobot ค้างไว้เพื่อเปิด 'Teach Mode'")
    print("2. ขยับหัวปากกาไปยังจุดที่คุณต้องการให้เป็น 'จุดพัก' หรือ 'จุดเริ่มต้น'")
    
    start_x, start_y = 0, 0
    try:
        input(f"👉 กรุณาขยับหัวปากกาไปที่ 'จุด Home' ที่ต้องการ แล้วกด [Enter]...")
        start_pose = bot.pose()
        start_x, start_y = start_pose[0], start_pose[1]
        print(f"✅ บันทึกจุด Home แล้ว: (X={start_x:.2f}, Y={start_y:.2f})")
    except Exception as e:
        print(f"❌ ไม่สามารถอ่านค่าได้: {e}. ใช้ค่าเริ่มต้นแทน")
        start_x, start_y = PAPER_CORNERS[0][0]


    print("✏️ เริ่มวาด...")

    safe_move(bot, start_x, start_y, PEN_UP_Z, wait=True) 
    time.sleep(0.5)

    start_time = time.time()
    
    # (คำนวณความยาวใหม่)
    total_length_to_draw = sum(contour_lengths) # (ความยาวรวมคือผลรวมของ List ที่จัดลำดับใหม่ (ซึ่งก็คือทั้งหมด))
    current_length_drawn = 0 
    avg_speed_mm_per_sec = 0.0 

    # (แสดง Log เริ่มต้น)
    # ❗️❗️❗️ [MODIFIED] - อัปเดต Log ให้ใช้ original_indices_logging[0] ❗️❗️❗️
    print(f"▶️ เริ่มวาดจาก contour ดั้งเดิมที่ {original_indices_logging[0]} / {len(processed_paths)} (ความยาวรวมที่ต้องวาด: {total_length_to_draw:.2f} mm)")
    
    x, y = start_x, start_y 

    # (ลูปการวาด (Loop ที่ List ที่จัดลำดับใหม่))
    for i in range(len(processed_paths)):
        # ❗️❗️❗️ [MODIFIED] - ใช้ List ที่เราแตกข้อมูลมาใหม่ ❗️❗️❗️
        ci_original = original_indices_logging[i] # (นี่คือหมายเลข Contour ดั้งเดิม (มี +1 แล้ว))
        ci_loop = i + 1                           # (นี่คือลำดับการวาด (1, 2, 3, ...))
        
        pts_transformed = processed_paths[i]

        if pts_transformed is None:
            continue
        
        # (อัปเดตภาพ Progress โดยใช้ "ลำดับการวาด" (ci_loop))
        update_current_progress_image(preview_img_bgr, filtered_contours, ci_loop, is_final=False) 
        
        sx, sy = pts_transformed[0][0]

        safe_move(bot, sx, sy, PEN_UP_Z, wait=False) 
        safe_move(bot, sx, sy, PEN_DOWN_Z, wait=True) 

        x_last, y_last = sx, sy 
        
        for p in pts_transformed[1:]:
            x_last, y_last = p[0] 
            safe_move(bot, x_last, y_last, PEN_DOWN_Z, wait=False) 
        
        safe_move(bot, x_last, y_last, PEN_DOWN_Z, wait=True)
        
        safe_move(bot, x_last, y_last, PEN_UP_Z, wait=False) 
        
        current_length_drawn += contour_lengths[i]
        
        percent_done = (current_length_drawn / total_length_to_draw) * 100 if total_length_to_draw > 0 else 100
        elapsed_time = time.time() - start_time
        
        eta_display = "ETA: Calculating..."
        
        if elapsed_time > 5 and current_length_drawn > 10 and current_length_drawn < total_length_to_draw: 
            avg_speed_mm_per_sec = current_length_drawn / elapsed_time 
            remaining_length = total_length_to_draw - current_length_drawn
            eta_seconds = remaining_length / avg_speed_mm_per_sec
            eta_minutes = eta_seconds / 60
            eta_display = f"ETA: {eta_minutes:.1f} min"
        
        # (แสดง Log การวาด)
        print(f"✅ (วาดเส้นที่ {ci_loop}/{len(processed_paths)}) [Contour ดั้งเดิม #{ci_original}] เสร็จ | Progress: {percent_done:.1f}% | {eta_display}")
        
    # --- จบการทำงาน ---
    
    safe_move(bot, x, y, PEN_UP_Z, wait=True) 
    update_current_progress_image(preview_img_bgr, filtered_contours, len(processed_paths) + 1, is_final=True)

    elapsed_seconds = time.time() - start_time
    hours = int(elapsed_seconds // 3600)
    minutes = int((elapsed_seconds % 3600) // 60)
    seconds = int(elapsed_seconds % 60)
    
    time_display = ""
    if hours > 0: time_display += f"{hours} ชม. "
    if minutes > 0 or hours > 0: time_display += f"{minutes} นาที "
    time_display += f"{seconds} วินาที"

    print(f"\n🎉 วาดเสร็จสมบูรณ์!")
    print(f"⏱️ ใช้เวลาวาดทั้งหมด: {time_display} (รวม {elapsed_seconds:.2f} วินาที)")
    
# ----------------- Main Menu Function -----------------

def main():
    """
    เมนูหลักสำหรับเชื่อมต่อ Dobot และเลือกว่าจะวาดหรือตั้งค่า
    """
    port = find_dobot_port()
    if not port:
        print("❌ ไม่พบ Dobot! โปรดตรวจสอบการเชื่อมต่อ และลองติดตั้งไดรเวอร์ (CH340/CP210) หากจำเป็น")
        sys.exit(1)

    bot = None
    try:
        print(f"✅ กำลังเชื่อมต่อกับ Dobot ที่ {port}...")
        bot = Dobot(port=port, verbose=False)
        bot.speed(DOBOT_SPEED, DOBOT_ACCELERATION) 
        print("✅ เชื่อมต่อสำเร็จ!")
    except Exception as e:
        print(f"❌ ไม่สามารถเชื่อมต่อ Dobot ได้: {e}")
        print("💡 ตรวจสอบว่าไม่มีโปรแกรมอื่น (เช่น DobotStudio) ใช้งานพอร์ตนี้อยู่")
        sys.exit(1)

    while True:
        print("\n" + "="*30)
        print("  🤖 เมนูหลัก Dobot Drawing (macOS) 🤖")
        print("="*30)
        print("1. 🎨 เริ่มวาดภาพ (Drawing Mode)")
        print("2. 📐 ตั้งค่ารูปทรง/ขนาดกระดาษ (Calibration)")
        print("Q. ❌ ออกจากโปรแกรม (Quit)")
        
        choice = input("\nกรุณาเลือกโหมด (1, 2, หรือ Q): ").strip().upper()

        if choice == '1':
            print("\n--- เริ่มโหมดวาดภาพ ---")
            run_drawing_mode(bot)
            print("--- โหมดวาดภาพเสร็จสิ้น ---")
            break 

        elif choice == '2':
            print("\n--- เริ่มโหมดตั้งค่ารูปทรง/ขนาดกระดาษ ---")
            run_calibration_mode(bot)
            
        elif choice == 'Q':
            print("กำลังออกจากโปรแกรม...")
            break
            
        else:
            print("⚠️ ไม่พบคำสั่งนี้ กรุณาเลือก 1, 2, หรือ Q")

    if bot:
        try:
            current_pose = bot.pose()
            safe_move(bot, current_pose[0], current_pose[1], PEN_UP_Z + 20, wait=True) 
        except Exception:
            pass 
        bot.close()
        print("✅ ปิดการเชื่อมต่อ Dobot เรียบร้อย")

if __name__ == "__main__":
    main()