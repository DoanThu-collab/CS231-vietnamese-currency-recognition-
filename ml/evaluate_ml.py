import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

from train_ml import load_dataset 

if __name__ == "__main__":
    test_dir = '../DATASET/test'
    model_path = os.path.join(os.path.dirname(__file__), 'ml_model.pkl')
    report_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'report', 'assets')
    
    print("Tải mô hình SVM và trích xuất đặc trưng tập test...")
    svm_model = joblib.load(model_path)
    X_test, y_test = load_dataset(test_dir)
    
    print("Đang dự đoán...")
    y_pred = svm_model.predict(X_test)
    
    print("\n================ BÁO CÁO PHÂN LOẠI ================")
    print(classification_report(y_test, y_pred))
    
    # Khởi tạo và lưu Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    classes = np.unique(np.concatenate((y_test, y_pred)))
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix - HOG + Color + SVM')
    plt.xlabel('Nhãn dự đoán (Predicted)')
    plt.ylabel('Nhãn thực tế (True)')
    
    os.makedirs(report_dir, exist_ok=True)
    save_path = os.path.join(report_dir, 'ml_results.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Đã lưu Confusion Matrix tại: {save_path}")