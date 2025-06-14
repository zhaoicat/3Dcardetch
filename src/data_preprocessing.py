import os
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import yaml
from torch.utils.data import Dataset

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class KITTIDataset(Dataset):
    """KITTI 3D目标检测数据集"""
    
    def __init__(self,
                 data_root: str,
                 split: str = 'training',
                 classes: List[str] = ['Car'],
                 image_size: Tuple[int, int] = (1242, 375)):
        """
        初始化KITTI数据集
        
        Args:
            data_root: 数据根目录
            split: 数据集划分 ('training' or 'testing')
            classes: 目标类别列表
            image_size: 图像尺寸 (width, height)
        """
        self.data_root = data_root
        self.split = split
        self.classes = classes
        self.image_size = image_size
        
        # 数据路径
        self.image_dir = os.path.join(
            data_root, 'data_object_image_2', split, 'image_2')
        self.label_dir = os.path.join(data_root, split, 'label_2')
        self.calib_dir = os.path.join(
            data_root, 'data_object_calib', split, 'calib')
        
        # 获取所有样本ID
        self.sample_ids = self._get_sample_ids()
        
        print(f"加载 {split} 数据集: {len(self.sample_ids)} 个样本")
        
    def _get_sample_ids(self) -> List[str]:
        """获取所有样本ID"""
        if os.path.exists(self.label_dir):
            # 从标注文件获取样本ID
            label_files = [f for f in os.listdir(self.label_dir) 
                          if f.endswith('.txt')]
            sample_ids = [f.replace('.txt', '') for f in label_files]
        else:
            # 从图像文件获取样本ID
            image_files = [f for f in os.listdir(self.image_dir) 
                          if f.endswith('.png')]
            sample_ids = [f.replace('.png', '') for f in image_files]
        
        return sorted(sample_ids)
    
    def __len__(self) -> int:
        return len(self.sample_ids)
    
    def __getitem__(self, idx: int) -> Dict:
        """获取单个样本"""
        sample_id = self.sample_ids[idx]
        
        # 加载图像
        image = self._load_image(sample_id)
        
        # 加载标定参数
        calib = self._load_calibration(sample_id)
        
        # 加载标注（如果存在）
        annotations = (self._load_annotations(sample_id) 
                      if self.split == 'training' else None)
        
        sample = {
            'sample_id': sample_id,
            'image': image,
            'calib': calib,
            'annotations': annotations
        }
        
        return sample
    
    def _load_image(self, sample_id: str) -> np.ndarray:
        """加载图像"""
        image_path = os.path.join(self.image_dir, f"{sample_id}.png")
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 调整图像尺寸
        if image.shape[:2] != (self.image_size[1], self.image_size[0]):
            image = cv2.resize(image, self.image_size)
        
        return image
    
    def _load_calibration(self, sample_id: str) -> Dict:
        """加载相机标定参数"""
        calib_path = os.path.join(self.calib_dir, f"{sample_id}.txt")
        
        if not os.path.exists(calib_path):
            # 如果标定文件不存在，使用默认参数
            return self._get_default_calib()
        
        calib = {}
        with open(calib_path, 'r') as f:
            for line in f.readlines():
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    calib[key] = np.array([float(x) for x in value.split()])
        
        # 提取相机内参矩阵P2 (左彩色相机)
        P2 = calib['P2'].reshape(3, 4)
        calib['P2'] = P2
        calib['intrinsic'] = P2[:3, :3]  # 内参矩阵
        
        return calib
    
    def _get_default_calib(self) -> Dict:
        """获取默认标定参数"""
        # KITTI数据集的典型标定参数
        P2 = np.array([
            [7.070493e+02, 0.000000e+00, 6.040814e+02, 4.575831e+01],
            [0.000000e+00, 7.070493e+02, 1.805066e+02, -3.454157e-01],
            [0.000000e+00, 0.000000e+00, 1.000000e+00, 4.981016e-03]
        ])
        
        return {
            'P2': P2,
            'intrinsic': P2[:3, :3]
        }
    
    def _load_annotations(self, sample_id: str) -> Optional[List[Dict]]:
        """加载标注信息"""
        label_path = os.path.join(self.label_dir, f"{sample_id}.txt")
        
        if not os.path.exists(label_path):
            return None
        
        annotations = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(' ')
                if len(parts) < 15:
                    continue
                
                # 解析标注信息
                obj_type = parts[0]
                
                # 只保留指定类别
                if obj_type not in self.classes and obj_type != 'DontCare':
                    continue
                
                if obj_type == 'DontCare':
                    continue
                
                annotation = {
                    'type': obj_type,
                    'truncated': float(parts[1]),
                    'occluded': int(parts[2]),
                    'alpha': float(parts[3]),
                    'bbox': [float(parts[4]), float(parts[5]), 
                            float(parts[6]), float(parts[7])],  # x1,y1,x2,y2
                    'dimensions': [float(parts[8]), float(parts[9]), 
                                  float(parts[10])],  # h,w,l
                    'location': [float(parts[11]), float(parts[12]), 
                                float(parts[13])],  # x,y,z
                    'rotation_y': float(parts[14])
                }
                
                annotations.append(annotation)
        
        return annotations
    
    def visualize_sample(self, idx: int, save_path: Optional[str] = None):
        """可视化样本"""
        if not HAS_MATPLOTLIB:
            print("matplotlib未安装，无法可视化")
            return
            
        sample = self[idx]
        image = sample['image']
        annotations = sample['annotations']
        
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        
        if annotations:
            for ann in annotations:
                bbox = ann['bbox']
                # 绘制2D边界框
                rect = plt.Rectangle(
                    (bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], 
                    fill=False, color='red', linewidth=2)
                plt.gca().add_patch(rect)
                
                # 添加类别标签
                plt.text(bbox[0], bbox[1]-5, ann['type'], 
                        color='red', fontsize=12, weight='bold')
        
        plt.title(f"Sample {sample['sample_id']}")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        else:
            plt.show()
        
        plt.close()


def create_data_splits(data_root: str, 
                      train_ratio: float = 0.8,
                      val_ratio: float = 0.2,
                      save_path: str = 'data_splits.yaml'):
    """创建数据集划分"""
    
    # 获取所有样本ID
    label_dir = os.path.join(data_root, 'training', 'label_2')
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    sample_ids = [f.replace('.txt', '') for f in label_files]
    sample_ids = sorted(sample_ids)
    
    # 随机打乱
    np.random.seed(42)
    np.random.shuffle(sample_ids)
    
    # 划分数据集
    n_total = len(sample_ids)
    n_train = int(n_total * train_ratio)
    n_val = n_total - n_train
    
    train_ids = sample_ids[:n_train]
    val_ids = sample_ids[n_train:]
    
    splits = {
        'train': train_ids,
        'val': val_ids,
        'statistics': {
            'total': n_total,
            'train': n_train,
            'val': n_val,
            'train_ratio': train_ratio,
            'val_ratio': val_ratio
        }
    }
    
    # 保存划分结果
    with open(save_path, 'w') as f:
        yaml.dump(splits, f, default_flow_style=False)
    
    print(f"数据集划分完成:")
    print(f"  总样本数: {n_total}")
    print(f"  训练集: {n_train} ({train_ratio:.1%})")
    print(f"  验证集: {n_val} ({val_ratio:.1%})")
    print(f"  划分结果保存至: {save_path}")
    
    return splits


def analyze_dataset(data_root: str):
    """分析数据集统计信息"""
    dataset = KITTIDataset(
        data_root, 
        classes=['Car', 'Van', 'Truck', 'Pedestrian', 'Cyclist'])
    
    # 统计类别分布
    class_counts = {}
    total_objects = 0
    
    for i in range(len(dataset)):
        sample = dataset[i]
        if sample['annotations']:
            for ann in sample['annotations']:
                obj_type = ann['type']
                class_counts[obj_type] = class_counts.get(obj_type, 0) + 1
                total_objects += 1
    
    print("数据集统计信息:")
    print(f"  总样本数: {len(dataset)}")
    print(f"  总目标数: {total_objects}")
    print("  类别分布:")
    for cls, count in sorted(class_counts.items()):
        print(f"    {cls}: {count} ({count/total_objects:.1%})")
    
    return class_counts


if __name__ == "__main__":
    # 数据根目录
    data_root = "data"
    
    # 分析数据集
    print("=== 数据集分析 ===")
    analyze_dataset(data_root)
    
    # 创建数据集划分
    print("\n=== 创建数据集划分 ===")
    create_data_splits(data_root)
    
    # 测试数据加载
    print("\n=== 测试数据加载 ===")
    dataset = KITTIDataset(data_root, classes=['Car'])
    
    # 可视化几个样本
    print(f"可视化前3个样本...")
    for i in range(min(3, len(dataset))):
        dataset.visualize_sample(i, f"sample_{i}.png")
    
    print("数据预处理完成！") 