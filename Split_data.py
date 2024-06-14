
import os
import shutil
from sklearn.model_selection import train_test_split

# Đường dẫn đến dataset gốc
dataset_dir = 'flower_images'

# Đường dẫn đến các thư mục train, val và test
base_dir = 'Data'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Tạo các thư mục train, val và test
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Các loại hoa
flower_types = ['Lilly', 'Lotus', 'Orchid', 'Sunflower', 'Tulip']
for flower in flower_types:
    # Tạo các thư mục con cho từng loại hoa trong train, val và test
    os.makedirs(os.path.join(train_dir, flower), exist_ok=True)
    os.makedirs(os.path.join(val_dir, flower), exist_ok=True)
    os.makedirs(os.path.join(test_dir, flower), exist_ok=True)

    # Lấy danh sách tất cả các tệp ảnh
    all_images = os.listdir(os.path.join(dataset_dir, flower))
    train_images, test_val_images = train_test_split(all_images, test_size=0.3, random_state=42)
    test_images, val_images = train_test_split(test_val_images, test_size=0.5, random_state=42)  

    # Di chuyển các tệp ảnh vào thư mục train
    for image in train_images:
        shutil.copy(os.path.join(dataset_dir, flower, image), os.path.join(train_dir, flower, image))

    # Di chuyển các tệp ảnh vào thư mục val
    for image in val_images:
        shutil.copy(os.path.join(dataset_dir, flower, image), os.path.join(val_dir, flower, image))

    # Di chuyển các tệp ảnh vào thư mục test
    for image in test_images:
        shutil.copy(os.path.join(dataset_dir, flower, image), os.path.join(test_dir, flower, image))

print("Data has been successfully split into train, val, and test directories.")