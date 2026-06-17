import os
import cv2
import glob
import numpy as np

def augment_image(img):
    aug_images = []
    # Lật ngang ảnh
    aug_images.append(cv2.flip(img, 1))
    
    # Thay đổi độ sáng trên không gian HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Tăng độ sáng
    v_bright = np.clip(v + 30, 0, 255)
    img_bright = cv2.cvtColor(cv2.merge((h, s, v_bright)), cv2.COLOR_HSV2BGR)
    aug_images.append(img_bright)
    
    # Giảm độ sáng
    v_dark = np.clip(v - 30, 0, 255)
    img_dark = cv2.cvtColor(cv2.merge((h, s, v_dark)), cv2.COLOR_HSV2BGR)
    aug_images.append(img_dark)

    return aug_images

def balance_dataset(data_dir, target_count=300):
    for label_name in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label_name)
        if os.path.isdir(label_dir):
            images = glob.glob(os.path.join(label_dir, '*.*'))
            current_count = len(images)
            
            if current_count < target_count:
                print(f"Tăng cường dữ liệu cho lớp {label_name} (Hiện tại: {current_count})...")
                for img_path in images:
                    if current_count >= target_count:
                        break
                        
                    img = cv2.imread(img_path)
                    if img is not None:
                        augmented = augment_image(img)
                        for i, aug_img in enumerate(augmented):
                            if current_count >= target_count:
                                break
                            save_path = img_path.replace('.jpg', f'_aug_{i}.jpg').replace('.png', f'_aug_{i}.png')
                            cv2.imwrite(save_path, aug_img)
                            current_count += 1

if __name__ == "__main__":
    PROCESSED_TRAIN_DIR = '../DATASET/train_processed'
    balance_dataset(PROCESSED_TRAIN_DIR, target_count=300)
    print("Hoàn tất cân bằng dữ liệu!")