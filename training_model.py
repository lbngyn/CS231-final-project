import numpy as np
import pickle
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
import cv2
import os


def compute_color_histogram(image, bins=(8, 8, 8)):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Compute the color histogram
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    # Normalize the histogram
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def load_images(folder_path, flower_types):
    features = []
    labels = []
    for flower in flower_types:
        flower_folder = os.path.join(folder_path, flower)
        for img in os.listdir(flower_folder):
            img_path = os.path.join(flower_folder, img)
            image = cv2.imread(img_path)
            if image is None:
                continue
            image = np.array(image).astype('uint8')
            image = cv2.resize(image, (64, 64))
            grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Extract HOG features
            hog_features = hog(grey_image, orientations=9, pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2")

            # Extract color histogram features
            color_hist = compute_color_histogram(image)

            # Concatenate HOG and color histogram features
            combined_features = np.hstack((hog_features, color_hist))

            features.append(combined_features)
            labels.append(flower)

    return features, labels

# Cấu hình đường dẫn và loại hoa
train_dir = 'F:\\Ky4\\CV\\cv\\flowers\\train'
val_dir ='F:\\Ky4\\CV\\cv\\flowers\\val'
test_dir='F:\\Ky4\\CV\\cv\\flowers\\test'

# Tải và xử lý dữ liệu
flower_types = ['lily', 'orchid', 'daisy', 'sunflower', 'tulip']
X_train, y_train = load_images(train_dir, flower_types)
X_val, y_val = load_images(val_dir, flower_types)
X_test, y_test = load_images(test_dir, flower_types)

# Mã hóa nhãn
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Áp dụng PCA
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Huấn luyện mô hình SVM
svm_model = SVC(kernel='rbf', C=15)
svm_model.fit(X_train_pca, y_train_encoded)

# Huấn luyện mô hình RF
rf_model = RandomForestClassifier(n_estimators=1000, random_state=42)
rf_model.fit(X_train_pca, y_train_encoded)

# Huấn luyện mô hình LightGBM
lgbm_model = lgb.LGBMClassifier(
    boosting_type='gbdt', 
    objective='multiclass', 
    num_class=5, 
    metric='multi_logloss', 
    learning_rate=0.35, 
    n_estimators=350, 
    verbosity=-1  # Tắt các cảnh báo
)
lgbm_model.fit(X_train_pca, y_train_encoded)

# Đánh giá mô hình
svm_predictions = svm_model.predict(X_test_pca)
rf_predictions = rf_model.predict(X_test_pca)
lgbm_predictions = lgbm_model.predict(X_test_pca)
print("SVM Accuracy:", accuracy_score(y_test_encoded, svm_predictions))
print("LightGBM Accuracy:", accuracy_score(y_test_encoded, lgbm_predictions))
print("RF Accuracy:", accuracy_score(y_test_encoded, rf_predictions))

# Lưu mô hình và các đối tượng biến đổi
with open('svm_model_4.pkl', 'wb') as f:
    pickle.dump(svm_model, f)
with open('lightgbm_model_4.pkl', 'wb') as f:
    pickle.dump(lgbm_model, f)
with open('scaler_4.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('pca_4.pkl', 'wb') as f:
    pickle.dump(pca, f)
with open('encoder_4.pkl', 'wb') as f:
    pickle.dump(encoder, f)
with open('rf_model_4.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
