import os
import sys
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from data_preprocessing import KITTIDataset
from model import create_model


class Trainer:
    """训练器类"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 创建输出目录
        os.makedirs(config['output_dir'], exist_ok=True)
        os.makedirs(os.path.join(config['output_dir'], 'checkpoints'), exist_ok=True)
        
        # 初始化模型
        self.model = create_model(num_classes=config['num_classes'])
        self.model.to(self.device)
        
        # 初始化优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config['lr_step_size'],
            gamma=config['lr_gamma']
        )
        
        # 初始化数据加载器
        self._init_dataloaders()
        
        # TensorBoard
        self.writer = SummaryWriter(os.path.join(config['output_dir'], 'logs'))
        
        # 训练状态
        self.epoch = 0
        self.best_loss = float('inf')
        
    def _init_dataloaders(self):
        """初始化数据加载器"""
        # 加载数据集划分
        splits_path = '../data_splits.yaml'
        with open(splits_path, 'r') as f:
            splits = yaml.safe_load(f)
        
        # 训练集
        train_dataset = KITTIDataset(
            data_root=self.config['data_root'],
            split='training',
            classes=self.config['classes']
        )
        
        # 这里简化处理，实际应该根据splits划分数据
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            collate_fn=self._collate_fn
        )
        
        # 验证集（这里简化为使用相同数据）
        self.val_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            collate_fn=self._collate_fn
        )
        
        print(f"训练样本数: {len(train_dataset)}")
        print(f"训练批次数: {len(self.train_loader)}")
    
    def _collate_fn(self, batch):
        """数据批处理函数"""
        images = []
        targets = []
        
        for sample in batch:
            # 转换图像为tensor
            image = torch.from_numpy(sample['image']).permute(2, 0, 1).float() / 255.0
            images.append(image)
            
            # 简化目标处理
            target = {
                'sample_id': sample['sample_id'],
                'annotations': sample['annotations']
            }
            targets.append(target)
        
        images = torch.stack(images)
        return images, targets
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            losses = self.model(images, targets)
            
            # 反向传播
            loss = losses['total_loss']
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 打印进度
            if batch_idx % self.config['print_freq'] == 0:
                print(f'Epoch {self.epoch}, Batch {batch_idx}/{len(self.train_loader)}, '
                      f'Loss: {loss.item():.4f}')
                
                # 记录到TensorBoard
                step = self.epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss', loss.item(), step)
                for key, value in losses.items():
                    if key != 'total_loss':
                        self.writer.add_scalar(f'Train/{key}', value.item(), step)
            
            # 限制训练批次数（用于快速测试）
            if batch_idx >= self.config.get('max_batches_per_epoch', float('inf')):
                break
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss
    
    def validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(self.val_loader):
                images = images.to(self.device)
                
                # 前向传播（验证时也传入targets以计算损失）
                self.model.train()  # 临时设为训练模式以计算损失
                losses = self.model(images, targets)
                self.model.eval()   # 恢复验证模式
                loss = losses['total_loss']
                
                total_loss += loss.item()
                num_batches += 1
                
                # 限制验证批次数
                if batch_idx >= self.config.get('max_val_batches', 10):
                    break
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss
    
    def save_checkpoint(self, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        # 保存最新检查点
        checkpoint_path = os.path.join(
            self.config['output_dir'], 'checkpoints', 'latest.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(
                self.config['output_dir'], 'checkpoints', 'best.pth')
            torch.save(checkpoint, best_path)
            print(f"保存最佳模型: {best_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.epoch = checkpoint['epoch']
            self.best_loss = checkpoint['best_loss']
            print(f"加载检查点: {checkpoint_path}, epoch: {self.epoch}")
            return True
        return False
    
    def train(self):
        """主训练循环"""
        print("开始训练...")
        start_time = time.time()
        
        for epoch in range(self.epoch, self.config['num_epochs']):
            self.epoch = epoch
            
            # 训练
            train_loss = self.train_epoch()
            
            # 验证
            val_loss = self.validate()
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录
            self.writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
            self.writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
            self.writer.add_scalar('Epoch/Learning_Rate', 
                                 self.optimizer.param_groups[0]['lr'], epoch)
            
            print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}, LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # 保存检查点
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            
            self.save_checkpoint(is_best)
        
        total_time = time.time() - start_time
        print(f"训练完成! 总用时: {total_time:.2f}秒")
        self.writer.close()


def create_config():
    """创建训练配置"""
    config = {
        # 数据配置
        'data_root': '../data',
        'classes': ['Car'],
        'num_classes': 1,
        
        # 模型配置
        'backbone': 'resnet18',
        
        # 训练配置
        'num_epochs': 10,
        'batch_size': 2,  # 小批次用于CPU训练
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'lr_step_size': 5,
        'lr_gamma': 0.1,
        
        # 数据加载配置
        'num_workers': 0,  # CPU训练使用0
        
        # 输出配置
        'output_dir': '../outputs',
        'print_freq': 10,
        
        # 调试配置（限制批次数以加快训练）
        'max_batches_per_epoch': 20,
        'max_val_batches': 5,
    }
    
    return config


def main():
    """主函数"""
    # 创建配置
    config = create_config()
    
    # 保存配置
    os.makedirs(config['output_dir'], exist_ok=True)
    with open(os.path.join(config['output_dir'], 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # 创建训练器
    trainer = Trainer(config)
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main() 