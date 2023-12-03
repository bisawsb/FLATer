import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from vit_improved import ImprovedViT
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from typing import *
import numpy as np
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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Define transform for test data
test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Load the training and validation datasets
data_dir = '/data6/wsb/archive'
image_datasets = FilterableImageFolder(root=f'{data_dir}/val', transform=test_transform,
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

    # 存储每个类别的真阳性率、假阳性率和AUC值
    fprs = []
    tprs = []
    aucs = []

    # 存储所有预测概率和真实标签
    probs = []
    labels = []

    # 在测试集上进行预测
    with torch.no_grad():
        for images, true_labels in dataloaders:

            # Forward pass
            images = images.to(device)
            output1, outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)

            probs.extend(probabilities.cpu().numpy())
            labels.extend(true_labels.numpy())

    for class_idx in range(num_classes):

        y_true = np.array(labels) == class_idx
        y_score = np.array(probs)[:, class_idx]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        fprs.append(fpr)
        tprs.append(tpr)
        aucs.append(roc_auc)


    # 绘制每个类别的ROC曲线
    plt.figure()

    for class_idx in range(num_classes):
        plt.plot(fprs[class_idx], tprs[class_idx], label='ROC Curve of Class {} (AUC = {:.4f})'.format(class_idx, aucs[class_idx]))

    # 设置图例、标签、标题等图形属性
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='lower right')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')

    # 保存图形
    plt.savefig('./results/tri_roc_val.png')
