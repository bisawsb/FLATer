from vit_improved import ImprovedViT
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder
from typing import *
import os
import wandb

# Set the training parameters
batch_size = 16
num_classes1 = 2
num_classes2 = 3
epochs = 300
base_lr = 1e-3
# Move the model to the specified GPUs
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device_ids = [1, ]  # List of GPU device IDs to use

# 设置WANDB API_KEY
os.environ["WANDB_API_KEY"] = 'fc50f6345e2212e580812573076d0dd503584bf0'
wandb.login()

# 设置TensorBoard
wandb.init(project='vit_image_classification', sync_tensorboard=True)

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

# Define the learning rate scheduler
def adjust_learning_rate(optimizer, epoch):
    # Linearly decrease learning rate
    lr = base_lr * (1 - epoch / epochs)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Define data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load the training and validation datasets
data_dir = '/data6/wsb/archive'
image_datasets = {x: FilterableImageFolder(root=f'{data_dir}/{x}', valid_classes=['1_ulcerative_colitis', '2_polyps', '3_esophagitis'],
                                           transform=data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
print(dataset_sizes)
print(class_names)

# Two-category datasets
image_datasets_ = {x: datasets.ImageFolder(f'{data_dir}/{x}', data_transforms[x]) for x in ['train', 'val']}
dataloaders_ = {x: torch.utils.data.DataLoader(image_datasets_[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}
dataset_sizes_ = {x: len(image_datasets_[x]) for x in ['train', 'val']}
# Modify the label of the category as required
class_mapping = {0: 0, 1: 1, 2: 1, 3: 1}

# Load the ViT-improved model
model = ImprovedViT(num_classes1, num_classes2)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-8)

# Move the model to the specified GPUs
if torch.cuda.is_available() and len(device_ids) >= 1:
    model = nn.DataParallel(model, device_ids=device_ids)
model = model.to(device)

# Training the model
def train_model(model, dataloaders, dataloaders_, criterion, optimizer, num_epochs=10):

    best_val_accuracies = []  # List to store the top three best validation accuracies
    ckpt_dir = '/data6/wsb/ckpt/final/16-bt'

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 30)

        # Adjust learning rate
        adjust_learning_rate(optimizer, epoch)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                if epoch % 10 != 0:
                    continue
                model.eval()

            tri_running_loss = 0.0
            tri_running_corrects = 0
            bi_running_loss = 0.0
            bi_running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Only track history during training phase
                with torch.set_grad_enabled(phase == 'train'):
                    output1, outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backpropagation and optimization only performed during training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                tri_running_loss += loss.item() * inputs.size(0)
                tri_running_corrects += torch.sum(preds == labels.data)

            for inputs, labels in dataloaders_[phase]:
                inputs = inputs.to(device)
                # Maps labels to two-class labels
                labels_binary = torch.tensor([class_mapping[label.item()] for label in labels])
                labels_binary = labels_binary.to(device)
                optimizer.zero_grad()

                # Only track history during training phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs, output2 = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels_binary)

                    # Backpropagation and optimization only performed during training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                bi_running_loss += loss.item() * inputs.size(0)
                bi_running_corrects += torch.sum(preds == labels_binary.data)

            tri_epoch_loss = tri_running_loss / dataset_sizes[phase]
            tri_epoch_acc = tri_running_corrects.double() / dataset_sizes[phase]
            bi_epoch_loss = bi_running_loss / dataset_sizes_[phase]
            bi_epoch_acc = bi_running_corrects.double() / dataset_sizes_[phase]

            if phase == 'val':
                # Save the top three best validation accuracies
                if len(best_val_accuracies) < 3:
                    best_val_accuracies.append(bi_epoch_acc)
                    best_val_accuracies.sort(reverse=True)
                    # Save the model checkpoint
                    torch.save(model.module.state_dict(), f'{ckpt_dir}/resvit_{bi_epoch_acc:.4f}.pth')
                elif bi_epoch_acc > best_val_accuracies[-1] and bi_epoch_acc not in best_val_accuracies:
                    # Romove the worst model checkpoint
                    os.remove(f'{ckpt_dir}/resvit_{best_val_accuracies[-1]:.4f}.pth')
                    best_val_accuracies.pop()
                    best_val_accuracies.append(bi_epoch_acc)
                    best_val_accuracies.sort(reverse=True)
                    # Save the model checkpoint
                    torch.save(model.module.state_dict(), f'{ckpt_dir}/resvit_{bi_epoch_acc:.4f}.pth')
            else:
                # 记录训练损失和准确率到WANDB
                wandb.log({'Binary Classification Loss': bi_epoch_loss, 'Binary Classification Acc': bi_epoch_acc})
                wandb.log({'Three-class Classification Loss': tri_epoch_loss, 'Three-class Classification Acc': tri_epoch_acc})

            print(f'{phase} Binary Classification Loss: {bi_epoch_loss:.4f} Acc: {bi_epoch_acc:.4f}')
            print(f'{phase} Three-class Classification Loss: {tri_epoch_loss:.4f} Acc: {tri_epoch_acc:.4f}')


if __name__ == '__main__':

    train_model(model, dataloaders, dataloaders_, criterion, optimizer, num_epochs=epochs)
    print('Training complete!')
