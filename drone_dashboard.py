import os
import time
import shutil
import cv2
import numpy as np
import streamlit as st
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from scipy.io import savemat # --- THÊM MỚI: Thư viện xuất file .mat ---
from scipy.io import loadmat

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="Drone Monitoring System", layout="wide")
st.title("🛰️ Hệ thống Giám sát & Matching Hạt từ Drone")

# --- KHỞI TẠO MODEL ---
@st.cache_resource
def load_predictor():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    base_path = os.path.dirname(__file__)
    cfg.MODEL.WEIGHTS = os.path.join(base_path, "model_final.pth")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    return DefaultPredictor(cfg), cfg

predictor, cfg = load_predictor()

# --- CẤU HÌNH THƯ MỤC ---
input_path = "./input_images/"
output_path = "./processed_images/"
os.makedirs(input_path, exist_ok=True)
os.makedirs(output_path, exist_ok=True)

# --- GIAO DIỆN CỘT ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("🖼️ Ảnh gốc & Dự đoán")
    placeholder_img = st.empty()
with col2:
    st.subheader("🌑 Mask nhị phân & Tâm hạt")
    placeholder_mask = st.empty()

log_area = st.sidebar.header("📜 Nhật ký hệ thống")
log_text = st.sidebar.empty()

# --- VÒNG LẶP XỬ LÝ ---
st.info("Hệ thống đang chạy... Hãy thả ảnh vào thư mục 'input_images'.")

while True:
    files = [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not files:
        log_text.write("Chờ ảnh mới từ drone...")
        time.sleep(2)
        continue

    for file_name in files:
        full_path = os.path.join(input_path, file_name)
        log_text.write(f"🔄 Đang xử lý AI: {file_name}")
        
        im = cv2.imread(full_path)
        if im is None: continue
        
        # 1. Chạy AI Segment
        outputs = predictor(im)
        
        # 2. Hiển thị ảnh dự đoán
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        placeholder_img.image(out.get_image()[:, :, ::-1], caption=f"File: {file_name}")

        # 3. MATCHING: Trích xuất tọa độ và Điểm số cho MATLAB
        if len(outputs["instances"]) > 0:
            instances = outputs["instances"].to("cpu")
            masks = instances.pred_masks.numpy() 
            scores = instances.scores.numpy() # Đây là alpha_m
            
            u_m_list = []
            alpha_m_list = []

            # Duyệt qua từng hạt tìm thấy để tính trọng tâm (Centroid)
            for i in range(len(masks)):
                mask_uint8 = masks[i].astype(np.uint8)
                M = cv2.moments(mask_uint8)
                
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"]) # Tọa độ x của hạt
                    cY = int(M["m01"] / M["m00"]) # Tọa độ y của hạt
                    
                    # MATLAB cần định dạng u_m = [x1, y1, x2, y2...]
                    u_m_list.append(cX)
                    u_m_list.append(cY)
                    alpha_m_list.append(scores[i])

            # Chuyển thành định dạng ma trận mà MATLAB yêu cầu
            u_m_final = np.array([u_m_list], dtype=float)
            alpha_m_final = np.array([alpha_m_list], dtype=float)

            # Xuất file .mat (Ghi đè mỗi khi có ảnh mới để MATLAB load)
            savemat('u_m.mat', {'u_m': u_m_final})
            savemat('alpha_m.mat', {'alpha_m': alpha_m_final})

            # Hiển thị Mask nhị phân lên Dashboard
            img_seg = np.any(masks, axis=0).astype(np.uint8) * 255
            placeholder_mask.image(img_seg, caption=f"Đã tìm thấy {len(masks)} hạt - Đã cập nhật .mat")
            
            # Lưu mask ảnh (tùy chọn)
            cv2.imwrite(os.path.join(output_path, f"mask_{file_name}"), img_seg)
        
        else:
            log_text.write(f"⚠️ Không tìm thấy hạt nào trong {file_name}")

        # 4. Dọn dẹp
        shutil.move(full_path, os.path.join(output_path, file_name))
        log_text.write(f"✅ Hoàn thành Matching cho: {file_name}")

        # --- CHÈN VÀO ĐÂY: Đọc kết quả từ MATLAB trả về ---
        try:
            if os.path.exists('localization-code/ParticleFilter_ver2.m'):
                mat_data = loadmat('localization-code/ParticleFilter_ver2.m')
                # Lấy tọa độ [x, y]
                pos = mat_data['current_drone_pos'][0] 
                
                # Hiển thị lên Dashboard bằng ô số (Metric)
                st.sidebar.divider()
                st.sidebar.subheader("📍 Vị trí Drone hiện tại")
                st.sidebar.metric("Kinh độ (East/X)", f"{pos[0]:.2f} m")
                st.sidebar.metric("Vĩ độ (North/Y)", f"{pos[1]:.2f} m")
        except:
            pass # Tránh lỗi nếu MATLAB đang ghi file mà Python lại đọc
            
    time.sleep(1)