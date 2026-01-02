# import some common libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode

if __name__ == "__main__":
    cfg = get_cfg()
    
    # 1. SỬA DÒNG NÀY: Lấy config trực tiếp từ thư viện thay vì tìm thư mục ./configs
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    
    # 2. ĐẢM BẢO CHẠY TRÊN CPU (Dành cho Mac)
    cfg.MODEL.DEVICE = "cpu"
    
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Ngưỡng tin cậy
    cfg.MODEL.WEIGHTS = "model_final.pth"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    predictor = DefaultPredictor(cfg)
    
    # 3. KIỂM TRA FILE ẢNH: Đảm bảo bạn có file tên là '1.png' trong cùng thư mục
    im = cv2.imread('2.png')
    
    if im is None:
        print("Lỗi: Không tìm thấy file ảnh '1.png'. Bạn hãy kiểm tra lại tên file ảnh nhé!")
    else:
        outputs = predictor(im)

        # Hiển thị kết quả dự đoán
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.figure(figsize=(10, 7))
        plt.imshow(v.get_image()[:, :, ::-1])
        plt.title("Dự đoán vật thể")

        # Hiển thị kết quả Mask (Binary)
        height = im.shape[0]   
        width = im.shape[1] 
        img_seg = np.zeros((height, width), dtype=bool)
        
        if len(outputs["instances"]) > 0:
            masks = outputs["instances"].to("cpu").pred_masks.numpy()
            # Gộp tất cả các mask lại thành một
            img_seg = np.any(masks, axis=0)
        
        plt.figure(figsize=(10, 7))
        plt.imshow(img_seg, cmap="gray")
        plt.title("Kết quả nhị phân (Binary Mask)")
        plt.show()