from vit_improved_ import ImprovedViT
import torch
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
import timm
from thop import profile

# 加载ResNet-50模型
model = ImprovedViT(2, 3)
checkpoint_path = '/data6/wsb/ckpt/ViT-improved/tiny/4_layer_nodrop/resvit_0.9250.pth'
model.load_state_dict(torch.load(checkpoint_path))
model.eval()  # 设置为评估模式

# 创建一个模拟的输入张量
input_tensor = torch.randn(1, 3, 224, 224)

# 使用pytorch-OpCounter来估算FLOPs
macs, params = profile(model, inputs=(input_tensor,))

# 获取FLOPs数值
flops = macs / 1e9  # 转换为GigaFLOPs
print(f"ResNet-50的总FLOPs: {flops:.2f} GigaFLOPs")


