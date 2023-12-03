import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder
from efficientnet_pytorch import EfficientNet
from vit_improved import ImprovedViT
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import timm
from typing import *
import os


class FilterableImageFolder(ImageFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = datasets.folder.default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        valid_classes: List = None
    ):
        self.valid_classes = valid_classes
        super(FilterableImageFolder, self).__init__(root, transform, target_transform, loader, is_valid_file)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        classes = [valid_class for valid_class in classes if valid_class in self.valid_classes]
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


# Set the training parameters
batch_size = 32
num_classes = 3
# Move the model to the specified GPUs
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# Define transform for test data
test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Load the training and validation datasets
data_dir = '/data6/wsb/archive'
image_datasets = FilterableImageFolder(root=f'{data_dir}/cross', transform=test_transform,
                                       valid_classes=['1_ulcerative_colitis', '2_polyps', '3_esophagitis'])
dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size, shuffle=True, num_workers=4)
dataset_sizes = len(image_datasets)
class_names = image_datasets.classes
print(dataset_sizes)
print(class_names)


# Load the ViT-improved model
model = ImprovedViT(2, num_classes)
# Load the saved checkpoint
checkpoint_path = '/data6/wsb/ckpt/final/resvit_final.pth'
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model = model.to(device)


if __name__ == '__main__':

    # Evaluation mode
    model.eval()

    true_labels = []  # 存储真实标签
    predicted_labels = []  # 存储模型预测标签
    matrix = [[0,0,0],[0,0,0],[0,0,0]]

    # 测试分类效果
    with torch.no_grad():
        for images, labels in dataloaders:

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            output1, outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            # 收集真实标签和模型预测标签
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

            matrix[0][0] += ((predicted == 0) & (labels == 0)).sum().item()
            matrix[0][1] += ((predicted == 1) & (labels == 0)).sum().item()
            matrix[0][2] += ((predicted == 2) & (labels == 0)).sum().item()
            matrix[1][0] += ((predicted == 0) & (labels == 1)).sum().item()
            matrix[1][1] += ((predicted == 1) & (labels == 1)).sum().item()
            matrix[1][2] += ((predicted == 2) & (labels == 1)).sum().item()
            matrix[2][0] += ((predicted == 0) & (labels == 2)).sum().item()
            matrix[2][1] += ((predicted == 1) & (labels == 2)).sum().item()
            matrix[2][2] += ((predicted == 2) & (labels == 2)).sum().item()

    # 计算分类指标
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='macro')
    recall = recall_score(true_labels, predicted_labels, average='macro')
    f1_score = f1_score(true_labels, predicted_labels, average='macro')

    # 输出分类指标
    print("Accuracy: {:.4f}".format(accuracy))
    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("F1-Score: {:.4f}".format(f1_score))
    print(matrix)


