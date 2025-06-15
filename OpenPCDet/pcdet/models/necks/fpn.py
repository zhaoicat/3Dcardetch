import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    """特征金字塔网络 (Feature Pyramid Network)"""
    
    def __init__(self, model_cfg):
        super(FPN, self).__init__()
        self.model_cfg = model_cfg
        
        # 获取配置参数
        in_channels_list = model_cfg.get('IN_CHANNELS', [64, 128, 256, 512])
        out_channels = model_cfg.get('OUT_CHANNELS', 256)
        num_outs = model_cfg.get('NUM_OUTS', 5)
        
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.num_outs = num_outs
        
        # 构建lateral连接（1x1卷积）
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            lateral_conv = nn.Conv2d(in_channels, out_channels, 1)
            self.lateral_convs.append(lateral_conv)
        
        # 构建输出卷积（3x3卷积）
        self.fpn_convs = nn.ModuleList()
        for _ in range(len(in_channels_list)):
            fpn_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.fpn_convs.append(fpn_conv)
        
        # 额外的层用于生成更多尺度
        self.extra_convs = nn.ModuleList()
        if num_outs > len(in_channels_list):
            for i in range(num_outs - len(in_channels_list)):
                if i == 0:
                    # 第一个额外层从最后一个输入特征开始
                    in_chs = in_channels_list[-1]
                else:
                    in_chs = out_channels
                extra_conv = nn.Conv2d(in_chs, out_channels, 3, stride=2, padding=1)
                self.extra_convs.append(extra_conv)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, batch_dict):
        """前向传播"""
        # 获取输入特征
        inputs = batch_dict.get('spatial_features_2d', None)
        if inputs is None:
            raise ValueError("spatial_features_2d not found in batch_dict")
        
        if len(inputs) != len(self.in_channels_list):
            raise ValueError(f"Expected {len(self.in_channels_list)} input features, "
                           f"got {len(inputs)}")
        
        # 构建FPN
        # Step 1: 构建lateral连接
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            laterals.append(lateral_conv(inputs[i]))
        
        # Step 2: 自顶向下路径
        # 从最高级别特征开始
        for i in range(len(laterals) - 1, 0, -1):
            # 上采样高级别特征
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode='nearest'
            )
        
        # Step 3: 应用输出卷积
        outs = []
        for i, fpn_conv in enumerate(self.fpn_convs):
            outs.append(fpn_conv(laterals[i]))
        
        # Step 4: 添加额外的层
        if self.num_outs > len(outs):
            # 使用最后一个输入特征来生成额外的层
            if len(self.extra_convs) > 0:
                source = inputs[-1]
                outs.append(self.extra_convs[0](source))
                
                for i in range(1, len(self.extra_convs)):
                    outs.append(self.extra_convs[i](F.relu(outs[-1])))
        
        # 只返回需要的输出数量
        outs = outs[:self.num_outs]
        
        # 更新batch_dict
        batch_dict['multi_scale_2d_features'] = outs
        
        return batch_dict 