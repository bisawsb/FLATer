import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

# 加载完整的"ViT base"模型
model_full = ViTModel.from_pretrained('google/vit-base-patch16-224')

# 获取"ViT base"模型的配置
config_full = model_full.config

# 创建新的配置，包含6层Transformer Encoder层
config_half = ViTConfig(
    hidden_size=config_full.hidden_size,
    num_hidden_layers=4,
    num_attention_heads=config_full.num_attention_heads,
    intermediate_size=config_full.intermediate_size,
    hidden_act=config_full.hidden_act,
    hidden_dropout_prob=config_full.hidden_dropout_prob,
    attention_probs_dropout_prob=config_full.attention_probs_dropout_prob,
    initializer_range=config_full.initializer_range
)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.upsample(self.bn2(self.conv2(out)))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(6, 3, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        avg_out = torch.reshape(avg_out, (-1, 3, 16, 16))
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        max_out = torch.reshape(max_out, (-1, 3, 16, 16))
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ImprovedViT(nn.Module):
    def __init__(self, num_classes1=2, num_classes2=3):
        super(ImprovedViT, self).__init__()

        #self.base_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
        # 创建新的模型，加载部分预训练参数
        self.base_model = ViTModel(config=config_half)
        # 将完整模型的对应层参数加载到新模型中
        self.base_model.embeddings.load_state_dict(model_full.embeddings.state_dict())
        self.base_model.pooler.load_state_dict(model_full.pooler.state_dict())
        layers_to_load = [0, 5, 6, 11]
        for i, layer in enumerate(layers_to_load):
            self.base_model.encoder.layer[i].load_state_dict(model_full.encoder.layer[layer].state_dict())

        self.residual_block = ResidualBlock(in_channels=3, out_channels=3, stride=2)
        self.spatial_attention = SpatialAttention()
        self.fc1 = nn.Linear(768, num_classes1)  # 768 is the dimension of output from ViT model
        self.fc2 = nn.Linear(768, num_classes2)

    def forward(self, x):
        x = self.residual_block(x)
        x = self.base_model(x).last_hidden_state
        x = self.spatial_attention(x)
        x = x.view(x.size(0), -1)
        output1 = self.fc1(x)
        output2 = self.fc2(x)
        return output1, output2
