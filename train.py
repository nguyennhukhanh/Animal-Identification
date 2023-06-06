import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models

# Định nghĩa các phép biến đổi cho hình ảnh
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Tải dữ liệu huấn luyện và kiểm tra
data_dir = 'data/animals'
image_datasets = {x: datasets.ImageFolder(os.path.join(
    data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(
    image_datasets[x], batch_size=4, shuffle=True) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# Sử dụng GPU nếu có thể
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Định nghĩa mô hình
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model = model.to(device)

# Định nghĩa hàm mất mát và thuật toán tối ưu
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Huấn luyện mô hình
num_epochs = 25
for epoch in range(num_epochs):
    print(f'Epoch {epoch}/{num_epochs - 1}')
    print('-' * 10)

    # Mỗi epoch có hai giai đoạn: huấn luyện và kiểm tra
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        # Duyệt qua dữ liệu
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Xóa gradient của các tham số
            optimizer.zero_grad()

            # Tính toán đầu ra và hàm mất mát
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # Cập nhật tham số nếu đang trong giai đoạn huấn luyện
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # Thống kê kết quả
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    print()

# Lưu mô hình lại dưới dạng tệp .pth
torch.save(model.state_dict(), 'animal_classifier.pth')
