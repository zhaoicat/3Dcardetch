import torch
import torch.nn as nn
import numpy as np

from .detector3d_template import Detector3DTemplate
from ..model_utils import model_nms_utils


class MonoDIS(Detector3DTemplate):
    """
    MonoDIS单目3D目标检测器
    基于OpenPCDet框架实现
    """
    
    def __init__(self, model_cfg, num_class, dataset):
        # 处理dataset为None的情况
        if dataset is None:
            # 创建一个简单的模拟dataset对象
            class MockDataset:
                def __init__(self):
                    self.class_names = ['Car']
                    # 添加其他必要的属性
                    import torch
                    self.point_feature_encoder = type('MockEncoder', (), {
                        'num_point_features': 4
                    })()
                    self.grid_size = torch.tensor([432, 496, 1])
                    self.point_cloud_range = torch.tensor([0, -39.68, -3, 69.12, 39.68, 1])
                    self.voxel_size = torch.tensor([0.16, 0.16, 4])
                    self.depth_downsample_factor = None
            
            dataset = MockDataset()
        
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        
    def build_networks(self):
        """构建网络模块"""
        model_info_dict = {
            'module_list': [],
            'num_rawpoint_features': self.dataset.point_feature_encoder.num_point_features,
            'num_point_features': self.dataset.point_feature_encoder.num_point_features,
            'grid_size': self.dataset.grid_size,
            'point_cloud_range': self.dataset.point_cloud_range,
            'voxel_size': self.dataset.voxel_size,
            'depth_downsample_factor': self.dataset.depth_downsample_factor
        }
        
        # 2D骨干网络
        self.backbone_2d = self.build_backbone_2d(
            model_info_dict=model_info_dict
        )
        model_info_dict['module_list'].append(self.backbone_2d)
        
        # 特征金字塔网络
        if self.model_cfg.get('NECK', None) is not None:
            self.neck = self.build_neck(
                model_info_dict=model_info_dict
            )
            model_info_dict['module_list'].append(self.neck)
        
        # 检测头
        self.dense_head = self.build_head(
            model_info_dict=model_info_dict
        )
        model_info_dict['module_list'].append(self.dense_head)
        
        return model_info_dict['module_list']
    
    def build_backbone_2d(self, model_info_dict):
        """构建2D骨干网络"""
        if self.model_cfg.get('BACKBONE_2D', None) is None:
            return None
        
        backbone_2d_module = __import__('pcdet.models.backbones_2d', fromlist=[''])
        backbone_2d_class = getattr(backbone_2d_module, self.model_cfg.BACKBONE_2D.NAME)
        backbone_2d = backbone_2d_class(
            model_cfg=self.model_cfg.BACKBONE_2D,
            input_channels=3  # RGB图像
        )
        return backbone_2d
    
    def build_neck(self, model_info_dict):
        """构建颈部网络"""
        if self.model_cfg.get('NECK', None) is None:
            return None
        
        neck_module = __import__('pcdet.models.necks', fromlist=[''])
        neck_class = getattr(neck_module, self.model_cfg.NECK.NAME)
        neck = neck_class(
            model_cfg=self.model_cfg.NECK
        )
        return neck
    
    def build_head(self, model_info_dict):
        """构建检测头"""
        dense_head_module = __import__('pcdet.models.dense_heads', fromlist=[''])
        dense_head_class = getattr(dense_head_module, self.model_cfg.DENSE_HEAD.NAME)
        dense_head = dense_head_class(
            model_cfg=self.model_cfg.DENSE_HEAD,
            input_channels=self.model_cfg.DENSE_HEAD.IN_CHANNELS,
            num_class=self.num_class,
            class_names=self.class_names,
            grid_size=model_info_dict['grid_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False),
            voxel_size=model_info_dict['voxel_size']
        )
        return dense_head
    
    def forward(self, batch_dict):
        """前向传播"""
        # 图像特征提取
        if self.backbone_2d is not None:
            backbone_2d_features = self.backbone_2d(batch_dict)
            batch_dict.update(backbone_2d_features)
        
        # 颈部网络处理
        if self.neck is not None:
            neck_features = self.neck(batch_dict)
            batch_dict.update(neck_features)
        
        # 检测头处理
        if self.dense_head is not None:
            dense_head_features = self.dense_head(batch_dict)
            batch_dict.update(dense_head_features)
        
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            
            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts
    
    def get_training_loss(self):
        """计算训练损失"""
        disp_dict = {}
        
        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }
        
        loss = loss_rpn
        return loss, tb_dict, disp_dict
    
    def post_processing(self, batch_dict):
        """后处理"""
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        
        recall_dict = {}
        pred_dicts = []
        
        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_index'].shape[0] == batch_dict['batch_cls_preds'].shape[0]
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                batch_mask = slice(None)
            
            box_preds = batch_dict['batch_box_preds'][batch_mask]
            src_box_preds = box_preds
            
            if not isinstance(batch_dict['batch_cls_preds'], list):
                cls_preds = batch_dict['batch_cls_preds'][batch_mask]
                
                src_cls_preds = cls_preds
                assert cls_preds.shape[1] in [1, self.num_class]
                
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = torch.sigmoid(cls_preds)
            else:
                cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_preds']]
                src_cls_preds = cls_preds
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = [torch.sigmoid(x) for x in cls_preds]
            
            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                if not isinstance(cls_preds, list):
                    cls_preds = [cls_preds]
                    multihead_label_mapping = [torch.arange(1, self.num_class, device=cls_preds[0].device)]
                else:
                    multihead_label_mapping = batch_dict['multihead_label_mapping']
                
                cur_start_idx = 0
                pred_scores, pred_labels, pred_boxes = [], [], []
                for cur_cls_preds, cur_label_mapping in zip(cls_preds, multihead_label_mapping):
                    assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                    cur_box_preds = box_preds[cur_start_idx: cur_start_idx + cur_cls_preds.shape[0]]
                    cur_pred_scores, cur_pred_labels, cur_pred_boxes = model_nms_utils.multi_classes_nms(
                        cls_scores=cur_cls_preds, box_preds=cur_box_preds,
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.SCORE_THRESH
                    )
                    cur_pred_labels = cur_label_mapping[cur_pred_labels]
                    pred_scores.append(cur_pred_scores)
                    pred_labels.append(cur_pred_labels)
                    pred_boxes.append(cur_pred_boxes)
                    cur_start_idx += cur_cls_preds.shape[0]
                
                final_scores = torch.cat(pred_scores, dim=0)
                final_labels = torch.cat(pred_labels, dim=0)
                final_boxes = torch.cat(pred_boxes, dim=0)
            else:
                cls_preds, label_preds = torch.max(cls_preds, dim=-1)
                if batch_dict.get('has_class_labels', False):
                    label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                    label_preds = batch_dict[label_key][index]
                else:
                    label_preds = label_preds + 1
                
                selected, selected_scores = model_nms_utils.class_agnostic_nms(
                    box_scores=cls_preds, box_preds=box_preds,
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=post_process_cfg.SCORE_THRESH
                )
                
                if post_process_cfg.OUTPUT_RAW_SCORE:
                    max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                    selected_scores = max_cls_preds[selected]
                
                final_scores = selected_scores
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]
            
            recall_dict = self.generate_recall_record(
                box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )
            
            record_dict = {
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels
            }
            pred_dicts.append(record_dict)
        
        return pred_dicts, recall_dict 