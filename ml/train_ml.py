import os
import cv2
import glob
import numpy as np
import joblib
from skimage.feature import hog
from sklearn.svm import SVC

def extract_features(img_path):
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (128, 128))
    
    # 1. Trích xuất đặc trưng HOG trên ảnh Grayscale
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    hog_features = hog(gray, orientations=9, pixels_per_cell=(8, 8), 
                       cells_per_block=(2, 2), block_norm='L2-Hys')
    
    # 2. Trích xuất đặc trưng Color Histogram trên không gian HSV
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    color_features = hist.flatten()
    
    # 3. Nối 2 vector
    return np.hstack((hog_features, color_features))

def load_dataset(data_dir):
    features, labels = [], []
    for label_name in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label_name)
        if os.path.isdir(label_dir):
            for img_file in glob.glob(os.path.join(label_dir, '*.*')):
                features.append(extract_features(img_file))
                labels.append(label_name)
    return np.array(features), np.array(labels)

if __name__ == "__main__":
    train_dir = '../DATASET/train_processed' 
    
    print("Trích xuất đặc trưng HOG + Color...")
    X_train, y_train = load_dataset(train_dir)
    
    print("Huấn luyện mô hình SVM...")
    # Cấu hình siêu tham số đã tối ưu
    svm_model = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced')
    svm_model.fit(X_train, y_train)
    
    # Lưu mô hình
    model_path = os.path.join(os.path.dirname(__file__), 'ml_model.pkl')
    joblib.dump(svm_model, model_path)
    print(f"Mô hình đã được lưu tại: {model_path}")