import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


class ResNetBackbone(nn.Module):
    """ResNet骨干网络"""
    
    def __init__(self, layers=[3, 4, 6, 3], num_classes=1000):
        super(ResNetBackbone, self).__init__()
        self.inplanes = 64
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet层
        self.layer1 = self._make_layer(BasicBlock, 64, layers[0])
        self.layer2 = self._make_layer(BasicBlock, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, layers[3], stride=2)
        
        # 初始化权重
        self._initialize_weights()
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        return [x1, x2, x3, x4]


class BasicBlock(nn.Module):
    """ResNet基本块"""
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                              padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out


class FPN(nn.Module):
    """特征金字塔网络"""
    
    def __init__(self, in_channels_list, out_channels=256):
        super(FPN, self).__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        
        for in_channels in in_channels_list:
            inner_block = nn.Conv2d(in_channels, out_channels, 1)
            layer_block = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
    
    def forward(self, x):
        """
        Args:
            x: List of feature maps from backbone
        Returns:
            List of FPN feature maps
        """
        last_inner = self.inner_blocks[-1](x[-1])
        results = []
        results.append(self.layer_blocks[-1](last_inner))
        
        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.inner_blocks[idx](x[idx])
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, 
                                         mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.layer_blocks[idx](last_inner))
        
        return results


class MonoDISHead(nn.Module):
    """MonoDIS检测头"""
    
    def __init__(self, in_channels=256, num_classes=1):
        super(MonoDISHead, self).__init__()
        self.num_classes = num_classes
        
        # 分类分支
        self.cls_subnet = self._make_subnet(in_channels, num_classes)
        
        # 2D边界框回归分支
        self.bbox_subnet = self._make_subnet(in_channels, 4)
        
        # 3D属性回归分支
        self.depth_subnet = self._make_subnet(in_channels, 1)  # 深度
        self.dim_subnet = self._make_subnet(in_channels, 3)    # 尺寸 h,w,l
        self.angle_subnet = self._make_subnet(in_channels, 2)  # 角度 sin,cos
        self.offset_subnet = self._make_subnet(in_channels, 2) # 中心点偏移
        
        # 初始化权重
        self._initialize_weights()
    
    def _make_subnet(self, in_channels, out_channels):
        """创建子网络"""
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(in_channels, in_channels, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, features):
        """
        Args:
            features: List of FPN feature maps
        Returns:
            Dict of predictions
        """
        cls_logits = []
        bbox_preds = []
        depth_preds = []
        dim_preds = []
        angle_preds = []
        offset_preds = []
        
        for feature in features:
            cls_logits.append(self.cls_subnet(feature))
            bbox_preds.append(self.bbox_subnet(feature))
            depth_preds.append(self.depth_subnet(feature))
            dim_preds.append(self.dim_subnet(feature))
            angle_preds.append(self.angle_subnet(feature))
            offset_preds.append(self.offset_subnet(feature))
        
        return {
            'cls_logits': cls_logits,
            'bbox_preds': bbox_preds,
            'depth_preds': depth_preds,
            'dim_preds': dim_preds,
            'angle_preds': angle_preds,
            'offset_preds': offset_preds
        }


class MonoDIS(nn.Module):
    """MonoDIS单目3D目标检测模型"""
    
    def __init__(self, num_classes=1, backbone_layers=[3, 4, 6, 3]):
        super(MonoDIS, self).__init__()
        self.num_classes = num_classes
        
        # 骨干网络
        self.backbone = ResNetBackbone(backbone_layers)
        
        # 特征金字塔网络
        backbone_channels = [64, 128, 256, 512]  # ResNet输出通道数
        self.fpn = FPN(backbone_channels, out_channels=256)
        
        # 检测头
        self.head = MonoDISHead(in_channels=256, num_classes=num_classes)
        
    def forward(self, images, targets=None):
        """
        Args:
            images: 输入图像 [B, 3, H, W]
            targets: 训练时的标注信息
        Returns:
            预测结果或损失
        """
        # 特征提取
        backbone_features = self.backbone(images)
        fpn_features = self.fpn(backbone_features)
        
        # 预测
        predictions = self.head(fpn_features)
        
        if self.training and targets is not None:
            # 训练模式：计算损失
            losses = self.compute_losses(predictions, targets)
            return losses
        else:
            # 推理模式：返回预测结果
            return self.post_process(predictions, images.shape[-2:])
    
    def compute_losses(self, predictions, targets):
        """计算损失函数"""
        # 这里简化实现，实际需要更复杂的损失计算
        losses = {}
        
        # 分类损失
        cls_loss = 0
        for cls_logit in predictions['cls_logits']:
            cls_loss += F.binary_cross_entropy_with_logits(
                cls_logit, torch.zeros_like(cls_logit))
        losses['cls_loss'] = cls_loss / len(predictions['cls_logits'])
        
        # 回归损失
        reg_loss = 0
        for bbox_pred in predictions['bbox_preds']:
            reg_loss += F.smooth_l1_loss(
                bbox_pred, torch.zeros_like(bbox_pred))
        losses['reg_loss'] = reg_loss / len(predictions['bbox_preds'])
        
        # 3D损失
        depth_loss = 0
        for depth_pred in predictions['depth_preds']:
            depth_loss += F.smooth_l1_loss(
                depth_pred, torch.ones_like(depth_pred))
        losses['depth_loss'] = depth_loss / len(predictions['depth_preds'])
        
        # 总损失
        total_loss = losses['cls_loss'] + losses['reg_loss'] + losses['depth_loss']
        losses['total_loss'] = total_loss
        
        return losses
    
    def post_process(self, predictions, image_size):
        """后处理：将预测结果转换为最终检测结果"""
        # 简化实现，实际需要NMS等后处理步骤
        results = []
        
        for i, cls_logit in enumerate(predictions['cls_logits']):
            # 获取预测概率
            scores = torch.sigmoid(cls_logit)
            
            # 简单阈值过滤 - 只在第一个类别通道上应用阈值
            if scores.shape[1] > 0:  # 确保有类别通道
                valid_mask = scores[:, 0:1] > 0.5  # 只取第一个类别
                
                # 将mask扩展到所有空间位置
                B, C, H, W = valid_mask.shape
                
                if valid_mask.sum() > 0:
                    # 重新整形以便索引
                    valid_indices = valid_mask.squeeze(1).nonzero(as_tuple=False)
                    
                    if len(valid_indices) > 0:
                        result = {
                            'num_detections': len(valid_indices),
                            'feature_map_level': i,
                            'valid_positions': valid_indices,
                        }
                        results.append(result)
        
        return results


def create_model(num_classes=1, pretrained=False):
    """创建MonoDIS模型"""
    model = MonoDIS(num_classes=num_classes)
    
    if pretrained:
        # 这里可以加载预训练权重
        print("注意：预训练权重加载功能尚未实现")
    
    return model


if __name__ == "__main__":
    # 测试模型
    print("=== 测试MonoDIS模型 ===")
    
    # 创建模型
    model = create_model(num_classes=1)
    model.eval()
    
    # 创建测试输入
    batch_size = 2
    height, width = 375, 1242
    test_input = torch.randn(batch_size, 3, height, width)
    
    print(f"输入形状: {test_input.shape}")
    
    # 前向传播
    with torch.no_grad():
        outputs = model(test_input)
    
    print(f"输出结果数量: {len(outputs)}")
    for i, output in enumerate(outputs):
        if output:
            print(f"  层{i}: {len(output)} 个检测结果")
            for key, value in output.items():
                if hasattr(value, 'shape'):
                    print(f"    {key}: {value.shape}")
                else:
                    print(f"    {key}: {value}")
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型参数统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    print("\n✅ 模型测试完成！") 