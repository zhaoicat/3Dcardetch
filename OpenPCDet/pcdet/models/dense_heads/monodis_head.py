import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ...utils import loss_utils, common_utils


class MonoDISHead(nn.Module):
    """
    MonoDIS检测头
    用于单目3D目标检测的多任务检测头
    """
    
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, voxel_size=None, **kwargs):
        super().__init__()
        
        self.model_cfg = model_cfg
        self.input_channels = input_channels
        self.num_classes = num_class
        self.class_names = class_names
        self.predict_boxes_when_training = predict_boxes_when_training
        self.feat_channels = model_cfg.get('FEAT_CHANNELS', 256)
        
        # 构建子网络
        self._build_subnets()
        
        # 损失函数权重
        loss_cfg = model_cfg.get('LOSS_CONFIG', {})
        self.loss_weights = {
            'cls': loss_cfg.get('CLASSIFICATION_LOSS', {}).get('WEIGHT', 1.0),
            'reg_2d': loss_cfg.get('REGRESSION_2D_LOSS', {}).get('WEIGHT', 1.0),
            'depth': loss_cfg.get('DEPTH_LOSS', {}).get('WEIGHT', 1.0),
            'dim': loss_cfg.get('DIMENSION_LOSS', {}).get('WEIGHT', 1.0),
            'angle': loss_cfg.get('ORIENTATION_LOSS', {}).get('WEIGHT', 1.0)
        }
        
        # 损失函数
        self._build_losses(loss_cfg)
        
        # 初始化权重
        self.init_weights()
        
        self.forward_ret_dict = {}
    
    def _build_subnets(self):
        """构建各个子网络"""
        # 分类分支
        self.cls_subnet = self._make_conv_layers(self.input_channels, self.num_classes)
        
        # 2D边界框回归分支
        self.bbox_2d_subnet = self._make_conv_layers(self.input_channels, 4)
        
        # 深度回归分支
        self.depth_subnet = self._make_conv_layers(self.input_channels, 1)
        
        # 3D尺寸回归分支
        self.dim_subnet = self._make_conv_layers(self.input_channels, 3)
        
        # 角度回归分支（使用sin, cos表示）
        self.angle_subnet = self._make_conv_layers(self.input_channels, 2)
        
        # 中心点偏移分支
        self.offset_subnet = self._make_conv_layers(self.input_channels, 2)
    
    def _make_conv_layers(self, input_channels, output_channels):
        """创建卷积层"""
        layers = []
        
        # 4层3x3卷积 + ReLU
        for i in range(4):
            layers.extend([
                nn.Conv2d(input_channels, self.feat_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            ])
            input_channels = self.feat_channels
        
        # 输出层
        layers.append(nn.Conv2d(self.feat_channels, output_channels, 3, padding=1))
        
        return nn.Sequential(*layers)
    
    def _build_losses(self, loss_cfg):
        """构建损失函数"""
        # 分类损失 - Focal Loss
        cls_loss_cfg = loss_cfg.get('CLASSIFICATION_LOSS', {})
        if cls_loss_cfg.get('NAME', 'FocalLoss') == 'FocalLoss':
            self.cls_loss_func = loss_utils.SigmoidFocalClassificationLoss(
                alpha=cls_loss_cfg.get('ALPHA', 0.25),
                gamma=cls_loss_cfg.get('GAMMA', 2.0)
            )
        else:
            self.cls_loss_func = loss_utils.WeightedSmoothL1Loss()
        
        # 2D回归损失 - Smooth L1 Loss
        reg_2d_cfg = loss_cfg.get('REGRESSION_2D_LOSS', {})
        self.reg_2d_loss_func = loss_utils.WeightedSmoothL1Loss(
            beta=reg_2d_cfg.get('BETA', 1.0 / 9.0),
            code_weights=None
        )
        
        # 其他损失函数
        self.depth_loss_func = nn.L1Loss(reduction='none')
        self.dim_loss_func = nn.L1Loss(reduction='none')
        self.angle_loss_func = nn.L1Loss(reduction='none')
    
    def init_weights(self):
        """初始化权重"""
        for modules in [self.cls_subnet, self.bbox_2d_subnet, self.depth_subnet, 
                       self.dim_subnet, self.angle_subnet, self.offset_subnet]:
            for m in modules.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        
        # 分类分支的偏置特殊初始化
        prior_prob = 0.01
        bias_value = -np.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_subnet[-1].bias, bias_value)
    
    def forward(self, batch_dict):
        """前向传播"""
        # 获取多尺度特征
        multi_scale_features = batch_dict.get('multi_scale_2d_features', None)
        if multi_scale_features is None:
            # 如果没有多尺度特征，使用单一特征
            spatial_features_2d = batch_dict.get('spatial_features_2d', None)
            if spatial_features_2d is not None:
                multi_scale_features = [spatial_features_2d]
            else:
                raise ValueError("No 2D features found in batch_dict")
        
        # 对每个尺度的特征进行预测
        cls_preds = []
        box_preds_2d = []
        depth_preds = []
        dim_preds = []
        angle_preds = []
        offset_preds = []
        
        for feature in multi_scale_features:
            cls_preds.append(self.cls_subnet(feature))
            box_preds_2d.append(self.bbox_2d_subnet(feature))
            depth_preds.append(self.depth_subnet(feature))
            dim_preds.append(self.dim_subnet(feature))
            angle_preds.append(self.angle_subnet(feature))
            offset_preds.append(self.offset_subnet(feature))
        
        # 存储预测结果
        self.forward_ret_dict.update({
            'cls_preds': cls_preds,
            'box_preds_2d': box_preds_2d,
            'depth_preds': depth_preds,
            'dim_preds': dim_preds,
            'angle_preds': angle_preds,
            'offset_preds': offset_preds
        })
        
        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=batch_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)
        
        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'],
                cls_preds=cls_preds, 
                box_preds_2d=box_preds_2d,
                depth_preds=depth_preds,
                dim_preds=dim_preds,
                angle_preds=angle_preds,
                offset_preds=offset_preds
            )
            
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        
        return batch_dict
    
    def assign_targets(self, gt_boxes):
        """分配训练目标"""
        # 这里简化处理，实际需要根据特征图位置和GT boxes计算目标
        # 返回一个简单的目标字典
        targets_dict = {
            'box_cls_labels': None,
            'box_reg_targets': None,
            'reg_weights': None
        }
        return targets_dict
    
    def generate_predicted_boxes(self, batch_size, cls_preds, box_preds_2d, 
                                depth_preds, dim_preds, angle_preds, offset_preds):
        """生成预测框"""
        batch_cls_preds = []
        batch_box_preds = []
        
        for batch_idx in range(batch_size):
            # 简化处理，这里需要实现从2D框和深度等信息恢复3D框的逻辑
            # 暂时返回占位符
            cls_pred = torch.zeros((100, self.num_classes), device=cls_preds[0].device)
            box_pred = torch.zeros((100, 7), device=cls_preds[0].device)  # [x, y, z, dx, dy, dz, heading]
            
            batch_cls_preds.append(cls_pred)
            batch_box_preds.append(box_pred)
        
        return batch_cls_preds, batch_box_preds
    
    def get_loss(self):
        """计算损失"""
        tb_dict = {}
        
        # 简化的损失计算，实际需要更复杂的实现
        loss = torch.tensor(0.0, device=next(self.parameters()).device, requires_grad=True)
        
        tb_dict.update({
            'monodis_loss': loss.item()
        })
        
        return loss, tb_dict 