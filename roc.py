import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder
from efficientnet_pytorch import EfficientNet
import timm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from vit_improved import ImprovedViT, ViTbackbone
import time

# Set the training parameters
batch_size = 32
num_classes = 2
# Move the model to the specified GPUs
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_ids = [0, 1, 3, 4]  # List of GPU device IDs to use


# Define transform for test data
test_transform = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

vit_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Load the training and validation datasets
data_dir = '/data6/wsb/archive'
image_datasets = ImageFolder(f'{data_dir}/test', test_transform)
dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size, shuffle=True, num_workers=4)
vit_datasets = ImageFolder(f'{data_dir}/test', vit_transform)
vit_dataloaders = torch.utils.data.DataLoader(vit_datasets, batch_size, shuffle=True, num_workers=4)

model_list = ['DenseNet', 'EfficientNet', 'GoogLeNet', 'ResNet18', 'ResNet50', 'ResNeXt', 'VGG16', 'MobileNet', 'Xception']

models = {
    'DenseNet': models.densenet121(pretrained=False),
    'EfficientNet': EfficientNet.from_pretrained('efficientnet-b0'),
    'GoogLeNet': models.googlenet(pretrained=True),
    'ResNet18': models.resnet18(pretrained=False),
    'ResNet50': models.resnet50(pretrained=False),
    'ResNeXt': models.resnext50_32x4d(pretrained=False),
    'VGG16': models.vgg16_bn(pretrained=False),
    'MobileNet': models.mobilenet_v2(pretrained=False),
    'Xception': timm.create_model('xception', pretrained=False),
}

models['DenseNet'].classifier = nn.Linear(models['DenseNet'].classifier.in_features, num_classes)
models['EfficientNet']._fc = nn.Linear(models['EfficientNet']._fc.in_features, num_classes)
models['GoogLeNet'].fc = nn.Linear(models['GoogLeNet'].fc.in_features, num_classes)
models['ResNet18'].fc = nn.Linear(models['ResNet18'].fc.in_features, num_classes)
models['ResNet50'].fc = nn.Linear(models['ResNet50'].fc.in_features, num_classes)
models['ResNeXt'].fc = nn.Linear(models['ResNeXt'].fc.in_features, num_classes)
models['VGG16'].classifier[6] = nn.Linear(models['VGG16'].classifier[6].in_features, num_classes)
models['MobileNet'].classifier[1] = nn.Linear(models['MobileNet'].last_channel, num_classes)
models['Xception'].fc = nn.Linear(models['Xception'].fc.in_features, num_classes)

# Load the saved checkpoint
densenet_ckpt = '/data6/wsb/ckpt/densenet/dich/densenet_0.9090.pth'
models['DenseNet'].load_state_dict(torch.load(densenet_ckpt, map_location=device))
efficientnet_ckpt = '/data6/wsb/ckpt/efficientnet/dich/efficientnet_0.8130.pth'
models['EfficientNet'].load_state_dict(torch.load(efficientnet_ckpt, map_location=device))
googlenet_ckpt = '/data6/wsb/ckpt/googlenet/dich/googlenet_0.8494.pth'
models['GoogLeNet'].load_state_dict(torch.load(googlenet_ckpt, map_location=device))
resnet18_ckpt = '/data6/wsb/ckpt/resnet18/dich/resnet18_0.8822.pth'
models['ResNet18'].load_state_dict(torch.load(resnet18_ckpt, map_location=device))
resnet50_ckpt = '/data6/wsb/ckpt/resnet50/dich/resnet50_0.9153.pth'
models['ResNet50'].load_state_dict(torch.load(resnet50_ckpt, map_location=device))
resnext_ckpt = '/data6/wsb/ckpt/resnext/dich/resnext_0.9244.pth'
models['ResNeXt'].load_state_dict(torch.load(resnext_ckpt, map_location=device))
vgg16_ckpt = '/data6/wsb/ckpt/vgg16/dich/vgg16_0.9001.pth'
models['VGG16'].load_state_dict(torch.load(vgg16_ckpt, map_location=device))
mobilenet_ckpt = '/data6/wsb/ckpt/mobilenet/dich/mobilenet_0.8844.pth'
models['MobileNet'].load_state_dict(torch.load(mobilenet_ckpt, map_location=device))
xception_ckpt = '/data6/wsb/ckpt/xception/dich/xception_0.8756.pth'
models['Xception'].load_state_dict(torch.load(xception_ckpt, map_location=device))

for x in model_list:
    models[x] = nn.DataParallel(models[x], device_ids=device_ids)
    models[x] = models[x].to(device)

# Load the ViT-improved model
model = ImprovedViT(num_classes, 3)
# Load the saved checkpoint
checkpoint_path = '/data6/wsb/ckpt/final/resvit_final.pth'
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model = model.to(device)

# Load the ViT-improved model
model_ = ViTbackbone(num_classes, 3)
# Load the saved checkpoint
checkpoint_path_ = '/data6/wsb/ckpt/vit/resvit_0.9206.pth'
model_.load_state_dict(torch.load(checkpoint_path_, map_location=device))
model_ = model_.to(device)

# Modify the label of the category as required
class_mapping = {0: 0, 1: 1, 2: 1, 3: 1}

plt.figure()

model.eval()
probs = []
true_labels = []

with torch.no_grad():
    for images, labels in vit_dataloaders:
        images = images.to(device)
        # Maps labels to two-class labels
        labels_binary = torch.tensor([class_mapping[label.item()] for label in labels])
        labels_binary = labels_binary.to(device)

        # Forward pass
        outputs, output2 = model(images)
        _, predicted = torch.max(outputs.data, 1)
        probabilities = torch.softmax(outputs, dim=1)

        probs.extend(probabilities[:, 1].cpu().numpy())  # 仅取正例的概率值
        true_labels.extend(labels_binary.cpu().numpy())

# 计算ROC曲线和AUC值,绘制曲线
fpr, tpr, _ = roc_curve(true_labels, probs)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='FLATer (AUC = %0.4f)' % roc_auc)


for x in model_list:

    models[x].eval()
    probs = []
    true_labels = []
    total_time = 0.0

    with torch.no_grad():
        for images, labels in dataloaders:

            images = images.to(device)
            # Maps labels to two-class labels
            labels_binary = torch.tensor([class_mapping[label.item()] for label in labels])
            # Forward pass
            start_time = time.time()
            outputs = models[x](images)
            _, predicted = torch.max(outputs.data, 1)
            probabilities = torch.softmax(outputs, dim=1)
            end_time = time.time()
            total_time += (end_time - start_time)

            probs.extend(probabilities[:, 1].cpu().numpy())  # 仅取正例的概率值
            true_labels.extend(labels_binary.numpy())

    # 计算ROC曲线和AUC值,绘制曲线
    fpr, tpr, _ = roc_curve(true_labels, probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=x+' (AUC = %0.4f)' % roc_auc)
    print(f"Inference time: {total_time:.3f} s")


model_.eval()
probs_ = []
true_labels_ = []

with torch.no_grad():
    for images, labels in vit_dataloaders:
        images = images.to(device)
        # Maps labels to two-class labels
        labels_binary = torch.tensor([class_mapping[label.item()] for label in labels])
        labels_binary = labels_binary.to(device)

        # Forward pass
        outputs, output2 = model_(images)
        _, predicted = torch.max(outputs.data, 1)
        probabilities = torch.softmax(outputs, dim=1)

        probs_.extend(probabilities[:, 1].cpu().numpy())  # 仅取正例的概率值
        true_labels_.extend(labels_binary.cpu().numpy())

# 计算ROC曲线和AUC值,绘制曲线
fpr_, tpr_, _ = roc_curve(true_labels_, probs_)
roc_auc_ = auc(fpr_, tpr_)
plt.plot(fpr_, tpr_, label='ViT (AUC = %0.4f)' % roc_auc_)


plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

# 保存ROC曲线图为PNG格式
plt.savefig('./results/all-test.png')
