from vit_improved import ImprovedViT
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder
from efficientnet_pytorch import EfficientNet
import timm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import time


# Set the training parameters
batch_size = 32
num_classes = 2
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
image_datasets = ImageFolder(f'{data_dir}/val', test_transform)
dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size, shuffle=True, num_workers=4)
dataset_sizes = len(image_datasets)
class_names = image_datasets.classes

# Load the ViT-improved model
model = ImprovedViT(num_classes, 3)
# Load the saved checkpoint
checkpoint_path = '/data6/wsb/ckpt/final/resvit_final.pth'
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model = model.to(device)

# Modify the label of the category as required
class_mapping = {0: 0, 1: 1, 2: 1, 3: 1}


if __name__ == '__main__':

    # Evaluation mode
    model.eval()
    total_time = 0.0

    # 定义分类指标
    correct = 0
    total = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0

    # 存储预测概率和真实标签
    probs = []
    true_labels = []

    # 测试分类效果
    with torch.no_grad():
        for images, labels in dataloaders:

            images = images.to(device)
            # Maps labels to two-class labels
            labels_binary = torch.tensor([class_mapping[label.item()] for label in labels])
            labels_binary = labels_binary.to(device)

            # Forward pass
            start_time = time.time()
            outputs, output2 = model(images)
            _, predicted = torch.max(outputs.data, 1)
            probabilities = torch.softmax(outputs, dim=1)
            end_time = time.time()
            total_time += (end_time - start_time)

            total += labels_binary.size(0)
            correct += (predicted == labels_binary).sum().item()

            true_positive += ((predicted == labels_binary) & (predicted == 1)).sum().item()
            false_positive += ((predicted != labels_binary) & (predicted == 1)).sum().item()
            false_negative += ((predicted != labels_binary) & (predicted == 0)).sum().item()

            probs.extend(probabilities[:, 1].cpu().numpy())  # 仅取正例的概率值
            true_labels.extend(labels_binary.cpu().numpy())

    # 计算分类指标
    accuracy = correct / total
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1_score = 2 * precision * recall / (precision + recall)

    # 输出分类指标
    print("Accuracy: {:.4f}".format(accuracy))
    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("F1-Score: {:.4f}".format(f1_score))
    print(f"Total inference time: {total_time*1000:.3f} ms")

    # 计算ROC曲线和AUC值
    fpr, tpr, _ = roc_curve(true_labels, probs)
    roc_auc = auc(fpr, tpr)

    # 绘制ROC曲线
    plt.figure()
    plt.plot(fpr, tpr, label='FLATer (AUC = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # 保存ROC曲线图为PNG格式
    #plt.savefig('./results/ViT_no_pretrain_val.png')

    # 输出AUC值
    print("AUC: {:.4f}".format(roc_auc))
