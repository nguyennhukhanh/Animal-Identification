import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
from torch import nn
from torchvision import models, transforms

# Tạo một mô hình với cấu trúc giống như mô hình đã được huấn luyện
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)

# Tải trạng thái của mô hình từ tệp .pth
state_dict = torch.load('animal_classifier.pth')
model.load_state_dict(state_dict)

# Chuyển mô hình sang chế độ đánh giá
model.eval()

# Định nghĩa các phép biến đổi cho hình ảnh
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Xác định danh sách các loài động vật
class_names = ['Capybara', 'Chim cánh cụt', 'Chó', 'Gà', 'Mèo']


def predict_image():
    # Mở cửa sổ chọn tệp để người dùng chọn hình ảnh
    file_path = filedialog.askopenfilename()

    # Tải hình ảnh từ đường dẫn
    img = Image.open(file_path)

    # Hiển thị hình ảnh lên giao diện người dùng
    img_tk = ImageTk.PhotoImage(img)
    img_label.configure(image=img_tk)
    img_label.image = img_tk

    # Áp dụng các phép biến đổi cho hình ảnh
    img_t = transform(img)
    img_t = img_t.unsqueeze(0)

    # Dự đoán nhãn cho hình ảnh
    with torch.no_grad():
        output = model(img_t)
        predicted_class = output.argmax(dim=1).item()

    # Lấy nhãn được dự đoán
    predicted_label = class_names[predicted_class]

    # Hiển thị nhãn được dự đoán lên giao diện người dùng
    result_label.configure(text=f'Predicted label: {predicted_label}')
    print(predicted_label)


# Tạo cửa sổ giao diện người dùng
root = tk.Tk()
root.title('Animal Classifier')

# Tạo nút chọn hình ảnh
select_button = tk.Button(root, text='Select Image', command=predict_image)
select_button.pack()

# Tạo nhãn hiển thị hình ảnh
img_label = tk.Label(root)
img_label.pack()

# Tạo nhãn hiển thị kết quả dự đoán
result_label = tk.Label(root)
result_label.pack()

# Chạy vòng lặp sự kiện của giao diện người dùng
root.mainloop()
