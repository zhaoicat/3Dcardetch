import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


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


class Bottleneck(nn.Module):
    """ResNet瓶颈块"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetBackbone(nn.Module):
    """ResNet骨干网络"""
    
    def __init__(self, model_cfg, input_channels):
        super(ResNetBackbone, self).__init__()
        self.model_cfg = model_cfg
        
        # 获取配置参数
        layers = model_cfg.get('LAYERS', [2, 2, 2, 2])
        pretrained = model_cfg.get('PRETRAINED', True)
        freeze_bn = model_cfg.get('FREEZE_BN', False)
        
        # 确定使用的块类型
        if len(layers) == 4 and max(layers) <= 3:
            block = BasicBlock
        else:
            block = Bottleneck
        
        self.inplanes = 64
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, 
                              padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # 冻结BN层
        if freeze_bn:
            self._freeze_bn()
        
        # 预训练权重加载
        if pretrained:
            self._load_pretrained_weights(layers)
        
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
    
    def _freeze_bn(self):
        """冻结BatchNorm层"""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
    
    def _load_pretrained_weights(self, layers):
        """加载预训练权重"""
        try:
            # 根据层数选择对应的预训练模型
            if layers == [2, 2, 2, 2]:
                pretrained_model = models.resnet18(pretrained=True)
            elif layers == [3, 4, 6, 3]:
                pretrained_model = models.resnet34(pretrained=True)
            elif layers == [3, 4, 6, 3] and hasattr(self, 'bottleneck'):
                pretrained_model = models.resnet50(pretrained=True)
            else:
                pretrained_model = models.resnet18(pretrained=True)
            
            # 加载预训练权重
            pretrained_dict = pretrained_model.state_dict()
            model_dict = self.state_dict()
            
            # 过滤不匹配的键
            pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                             if k in model_dict and model_dict[k].shape == v.shape}
            
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            
            print(f"Loaded pretrained ResNet weights: {len(pretrained_dict)} layers")
            
        except Exception as e:
            print(f"Warning: Failed to load pretrained weights: {e}")
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, batch_dict):
        """前向传播"""
        # 获取图像
        images = batch_dict.get('images', None)
        if images is None:
            raise ValueError("Images not found in batch_dict")
        
        x = images
        
        # ResNet前向传播
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        # 返回多尺度特征
        batch_dict['spatial_features_2d'] = [x1, x2, x3, x4]
        
        return batch_dict 