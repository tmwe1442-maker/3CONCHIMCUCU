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

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="Drone Monitoring System", layout="wide")
st.title("🛰️ Hệ thống Giám sát & Phân tách Hạt từ Drone")

# --- KHỞI TẠO MODEL (Dùng cache để không load lại mỗi khi trang refresh) ---
@st.cache_resource
def load_predictor():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    # Sử dụng đường dẫn tuyệt đối để tránh lỗi "Checkpoint not found"
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
    st.subheader("🌑 Mask nhị phân (Dữ liệu cho MATLAB)")
    placeholder_mask = st.empty()

log_area = st.sidebar.header("📜 Nhật ký hệ thống")
log_text = st.sidebar.empty()

# --- VÒNG LẶP XỬ LÝ ---
st.info("Hệ thống đang chạy... Hãy thả ảnh vào thư mục 'input_images'.")

while True:
    # Quét danh sách file
    files = [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not files:
        log_text.write("Chờ ảnh mới từ drone...")
        time.sleep(2)
        continue

    for file_name in files:
        full_path = os.path.join(input_path, file_name)
        log_text.write(f"🔄 Đang xử lý: {file_name}")
        
        im = cv2.imread(full_path)
        if im is None: continue
        
        # 1. Chạy AI
        outputs = predictor(im)
        
        # 2. Tạo ảnh dự đoán (Overlay)
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        placeholder_img.image(out.get_image()[:, :, ::-1], caption=f"File: {file_name}")

        # 3. Tạo Mask nhị phân
        if len(outputs["instances"]) > 0:
            masks = outputs["instances"].to("cpu").pred_masks.numpy()
            img_seg = np.any(masks, axis=0).astype(np.uint8) * 255
            placeholder_mask.image(img_seg, caption="Binary Mask")
            
            # Lưu mask để dùng cho bước Matching sau này
            cv2.imwrite(os.path.join(output_path, f"mask_{file_name}"), img_seg)

        # 4. Dọn dẹp: Di chuyển ảnh gốc sang folder 'processed'
        shutil.move(full_path, os.path.join(output_path, file_name))
        log_text.write(f"✅ Đã xong: {file_name}")
        
    time.sleep(1)