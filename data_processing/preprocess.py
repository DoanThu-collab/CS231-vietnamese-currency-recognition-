import os
import cv2
import glob

def preprocess_images(input_dir, output_dir, target_size=(128, 128)):
    """
    Đọc ảnh gốc, thay đổi kích thước và lưu vào thư mục mới đã qua xử lý.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for label_name in os.listdir(input_dir):
        input_label_dir = os.path.join(input_dir, label_name)
        output_label_dir = os.path.join(output_dir, label_name)
        
        if os.path.isdir(input_label_dir):
            if not os.path.exists(output_label_dir):
                os.makedirs(output_label_dir)
                
            for img_path in glob.glob(os.path.join(input_label_dir, '*.*')):
                img = cv2.imread(img_path)
                if img is not None:
                    # Đưa ảnh về kích thước chuẩn
                    img_resized = cv2.resize(img, target_size)
                    
                    filename = os.path.basename(img_path)
                    save_path = os.path.join(output_label_dir, filename)
                    cv2.imwrite(save_path, img_resized)

if __name__ == "__main__":
    RAW_TRAIN_DIR = '../DATASET/train'
    PROCESSED_TRAIN_DIR = '../DATASET/train_processed'
    
    print("Bắt đầu tiền xử lý dữ liệu...")
    preprocess_images(RAW_TRAIN_DIR, PROCESSED_TRAIN_DIR, target_size=(128, 128))
    print("Tiền xử lý hoàn tất!")