import os
import sys
sys.path.append('src')

from data_preprocessing import KITTIDataset, create_data_splits, analyze_dataset

def test_data_loading():
    """测试数据加载功能"""
    print("=== 测试数据加载 ===")
    
    # 数据根目录
    data_root = "data"
    
    try:
        # 创建数据集实例
        dataset = KITTIDataset(data_root, classes=['Car'])
        print(f"数据集创建成功，共 {len(dataset)} 个样本")
        
        # 测试加载第一个样本
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"样本ID: {sample['sample_id']}")
            print(f"图像形状: {sample['image'].shape}")
            print(f"标定参数: {list(sample['calib'].keys())}")
            
            if sample['annotations']:
                print(f"标注数量: {len(sample['annotations'])}")
                for i, ann in enumerate(sample['annotations'][:3]):  # 只显示前3个
                    print(f"  目标{i+1}: {ann['type']}, bbox: {ann['bbox']}")
            else:
                print("无标注信息")
        
        return True
        
    except Exception as e:
        print(f"数据加载失败: {e}")
        return False

def test_data_splits():
    """测试数据集划分"""
    print("\n=== 测试数据集划分 ===")
    
    try:
        splits = create_data_splits("data")
        print("数据集划分成功")
        return True
    except Exception as e:
        print(f"数据集划分失败: {e}")
        return False

if __name__ == "__main__":
    # 测试数据加载
    success1 = test_data_loading()
        
    # 测试数据集划分
    success2 = test_data_splits()
    
    if success1 and success2:
        print("\n✅ 所有测试通过！")
    else:
        print("\n❌ 部分测试失败") 