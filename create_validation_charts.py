#!/usr/bin/env python3
"""
真实KITTI数据验证结果可视化脚本
生成500样本验证的各种图表和分析
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# 设置字体和样式
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('default')

def load_validation_results():
    """加载验证结果"""
    with open('real_kitti_validation_results.json', 'r') as f:
        return json.load(f)

def create_performance_comparison_chart(results):
    """创建性能对比图表"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 数据准备
    classes = ['Car', 'Pedestrian', 'Cyclist']
    metrics = results['evaluation_results']['simplified_metrics']
    
    ap_values = [metrics[cls]['AP'] * 100 for cls in classes]
    aos_values = [metrics[cls]['AOS'] * 100 for cls in classes]
    precision_values = [metrics[cls]['Precision'] * 100 for cls in classes]
    recall_values = [metrics[cls]['Recall'] * 100 for cls in classes]
    
    # AP对比
    bars1 = ax1.bar(classes, ap_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
    ax1.set_title('Average Precision (AP) by Class', fontsize=14, fontweight='bold')
    ax1.set_ylabel('AP (%)')
    ax1.set_ylim(0, 100)
    for i, v in enumerate(ap_values):
        ax1.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # AOS对比
    bars2 = ax2.bar(classes, aos_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
    ax2.set_title('Average Orientation Similarity (AOS) by Class', fontsize=14, fontweight='bold')
    ax2.set_ylabel('AOS (%)')
    ax2.set_ylim(0, 100)
    for i, v in enumerate(aos_values):
        ax2.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Precision vs Recall
    ax3.scatter(recall_values, precision_values, s=[200, 150, 100], 
               c=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.7)
    for i, cls in enumerate(classes):
        ax3.annotate(cls, (recall_values[i], precision_values[i]), 
                    xytext=(5, 5), textcoords='offset points', fontweight='bold')
    ax3.set_xlabel('Recall (%)')
    ax3.set_ylabel('Precision (%)')
    ax3.set_title('Precision vs Recall', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 100)
    ax3.set_ylim(0, 100)
    
    # F1 Score
    f1_values = [metrics[cls]['F1'] * 100 for cls in classes]
    bars4 = ax4.bar(classes, f1_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
    ax4.set_title('F1 Score by Class', fontsize=14, fontweight='bold')
    ax4.set_ylabel('F1 Score (%)')
    ax4.set_ylim(0, 100)
    for i, v in enumerate(f1_values):
        ax4.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('kitti_500_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_data_distribution_chart(results):
    """创建数据分布图表"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    stats = results['data_statistics']
    
    # 目标数量分布
    classes = list(stats.keys())
    counts = [stats[cls]['count'] for cls in classes]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    wedges, texts, autotexts = ax1.pie(counts, labels=classes, colors=colors, autopct='%1.1f%%', 
                                      startangle=90, textprops={'fontweight': 'bold'})
    ax1.set_title('Target Distribution (500 Samples)', fontsize=14, fontweight='bold')
    
    # 平均深度对比
    depths = [stats[cls]['avg_depth'] for cls in classes]
    bars2 = ax2.bar(classes, depths, color=colors, alpha=0.8)
    ax2.set_title('Average Depth by Class', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Depth (meters)')
    for i, v in enumerate(depths):
        ax2.text(i, v + 0.5, f'{v:.1f}m', ha='center', va='bottom', fontweight='bold')
    
    # 遮挡分布
    occlusion_data = []
    occlusion_labels = ['Fully Visible', 'Partly Occluded', 'Largely Occluded']
    
    for cls in classes:
        occ_dist = stats[cls]['occlusion_dist']
        occlusion_data.append([
            occ_dist['fully_visible'],
            occ_dist['partly_occluded'], 
            occ_dist['largely_occluded']
        ])
    
    x = np.arange(len(classes))
    width = 0.25
    
    for i, label in enumerate(occlusion_labels):
        values = [occlusion_data[j][i] for j in range(len(classes))]
        ax3.bar(x + i*width, values, width, label=label, alpha=0.8)
    
    ax3.set_xlabel('Class')
    ax3.set_ylabel('Count')
    ax3.set_title('Occlusion Distribution by Class', fontsize=14, fontweight='bold')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(classes)
    ax3.legend()
    
    # 平均边界框高度
    heights = [stats[cls]['avg_height'] for cls in classes]
    bars4 = ax4.bar(classes, heights, color=colors, alpha=0.8)
    ax4.set_title('Average Bounding Box Height', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Height (pixels)')
    for i, v in enumerate(heights):
        ax4.text(i, v + 1, f'{v:.1f}px', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('kitti_500_data_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_kitti_official_results_chart(results):
    """创建KITTI官方结果图表"""
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 12))
    
    # 从官方结果中提取数据
    official_results = results['evaluation_results']['kitti_official'][1]
    
    # Car类别结果
    car_2d = [official_results['Car_image/easy_R40'], 
              official_results['Car_image/moderate_R40'], 
              official_results['Car_image/hard_R40']]
    car_3d = [official_results['Car_3d/easy_R40'], 
              official_results['Car_3d/moderate_R40'], 
              official_results['Car_3d/hard_R40']]
    car_aos = [official_results['Car_aos/easy_R40'], 
               official_results['Car_aos/moderate_R40'], 
               official_results['Car_aos/hard_R40']]
    
    # Pedestrian类别结果
    ped_2d = [official_results['Pedestrian_image/easy_R40'], 
              official_results['Pedestrian_image/moderate_R40'], 
              official_results['Pedestrian_image/hard_R40']]
    ped_3d = [official_results['Pedestrian_3d/easy_R40'], 
              official_results['Pedestrian_3d/moderate_R40'], 
              official_results['Pedestrian_3d/hard_R40']]
    ped_aos = [official_results['Pedestrian_aos/easy_R40'], 
               official_results['Pedestrian_aos/moderate_R40'], 
               official_results['Pedestrian_aos/hard_R40']]
    
    # Cyclist类别结果
    cyc_2d = [official_results['Cyclist_image/easy_R40'], 
              official_results['Cyclist_image/moderate_R40'], 
              official_results['Cyclist_image/hard_R40']]
    cyc_3d = [official_results['Cyclist_3d/easy_R40'], 
              official_results['Cyclist_3d/moderate_R40'], 
              official_results['Cyclist_3d/hard_R40']]
    cyc_aos = [official_results['Cyclist_aos/easy_R40'], 
               official_results['Cyclist_aos/moderate_R40'], 
               official_results['Cyclist_aos/hard_R40']]
    
    difficulties = ['Easy', 'Moderate', 'Hard']
    x = np.arange(len(difficulties))
    width = 0.25
    
    # 2D AP结果
    ax1.bar(x - width, car_2d, width, label='Car', color='#FF6B6B', alpha=0.8)
    ax1.bar(x, ped_2d, width, label='Pedestrian', color='#4ECDC4', alpha=0.8)
    ax1.bar(x + width, cyc_2d, width, label='Cyclist', color='#45B7D1', alpha=0.8)
    ax1.set_xlabel('Difficulty')
    ax1.set_ylabel('2D AP (%)')
    ax1.set_title('KITTI Official 2D AP Results', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(difficulties)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 3D AP结果
    ax2.bar(x - width, car_3d, width, label='Car', color='#FF6B6B', alpha=0.8)
    ax2.bar(x, ped_3d, width, label='Pedestrian', color='#4ECDC4', alpha=0.8)
    ax2.bar(x + width, cyc_3d, width, label='Cyclist', color='#45B7D1', alpha=0.8)
    ax2.set_xlabel('Difficulty')
    ax2.set_ylabel('3D AP (%)')
    ax2.set_title('KITTI Official 3D AP Results', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(difficulties)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # AOS结果
    ax3.bar(x - width, car_aos, width, label='Car', color='#FF6B6B', alpha=0.8)
    ax3.bar(x, ped_aos, width, label='Pedestrian', color='#4ECDC4', alpha=0.8)
    ax3.bar(x + width, cyc_aos, width, label='Cyclist', color='#45B7D1', alpha=0.8)
    ax3.set_xlabel('Difficulty')
    ax3.set_ylabel('AOS (%)')
    ax3.set_title('KITTI Official AOS Results', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(difficulties)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # TP/FP/FN分析
    metrics = results['evaluation_results']['simplified_metrics']
    classes = ['Car', 'Pedestrian', 'Cyclist']
    
    tp_values = [metrics[cls]['TP'] for cls in classes]
    fp_values = [metrics[cls]['FP'] for cls in classes]
    fn_values = [metrics[cls]['FN'] for cls in classes]
    
    x_pos = np.arange(len(classes))
    
    ax4.bar(x_pos, tp_values, label='True Positive', color='#2ECC71', alpha=0.8)
    ax4.bar(x_pos, fp_values, bottom=tp_values, label='False Positive', color='#E74C3C', alpha=0.8)
    ax4.bar(x_pos, fn_values, bottom=np.array(tp_values) + np.array(fp_values), 
            label='False Negative', color='#F39C12', alpha=0.8)
    
    ax4.set_xlabel('Class')
    ax4.set_ylabel('Count')
    ax4.set_title('Detection Results Breakdown', fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(classes)
    ax4.legend()
    
    # IoU分布
    iou_values = [metrics[cls]['avg_IoU'] * 100 for cls in classes]
    bars5 = ax5.bar(classes, iou_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
    ax5.set_title('Average IoU by Class', fontweight='bold')
    ax5.set_ylabel('Average IoU (%)')
    ax5.set_ylim(0, 100)
    for i, v in enumerate(iou_values):
        ax5.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 综合性能雷达图
    categories = ['AP', 'AOS', 'Precision', 'Recall', 'IoU']
    
    # 归一化数据到0-1范围
    car_values = [metrics['Car']['AP'], metrics['Car']['AOS'], 
                  metrics['Car']['Precision'], metrics['Car']['Recall'], 
                  metrics['Car']['avg_IoU']]
    ped_values = [metrics['Pedestrian']['AP'], metrics['Pedestrian']['AOS'], 
                  metrics['Pedestrian']['Precision'], metrics['Pedestrian']['Recall'], 
                  metrics['Pedestrian']['avg_IoU']]
    cyc_values = [metrics['Cyclist']['AP'], metrics['Cyclist']['AOS'], 
                  metrics['Cyclist']['Precision'], metrics['Cyclist']['Recall'], 
                  metrics['Cyclist']['avg_IoU']]
    
    # 创建雷达图
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    car_values += car_values[:1]
    ped_values += ped_values[:1]
    cyc_values += cyc_values[:1]
    
    ax6 = plt.subplot(2, 3, 6, projection='polar')
    ax6.plot(angles, car_values, 'o-', linewidth=2, label='Car', color='#FF6B6B')
    ax6.fill(angles, car_values, alpha=0.25, color='#FF6B6B')
    ax6.plot(angles, ped_values, 'o-', linewidth=2, label='Pedestrian', color='#4ECDC4')
    ax6.fill(angles, ped_values, alpha=0.25, color='#4ECDC4')
    ax6.plot(angles, cyc_values, 'o-', linewidth=2, label='Cyclist', color='#45B7D1')
    ax6.fill(angles, cyc_values, alpha=0.25, color='#45B7D1')
    
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(categories)
    ax6.set_ylim(0, 1)
    ax6.set_title('Performance Radar Chart', fontweight='bold', pad=20)
    ax6.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    
    plt.tight_layout()
    plt.savefig('kitti_500_official_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_report():
    """创建总结报告"""
    results = load_validation_results()
    
    print("🎯 KITTI 500样本验证结果总结")
    print("=" * 60)
    
    # 数据统计
    stats = results['data_statistics']
    total_targets = sum(stats[cls]['count'] for cls in stats.keys())
    
    print(f"\n📊 数据统计:")
    print(f"   验证样本数: {results['sample_count']}")
    print(f"   目标总数: {total_targets}")
    for cls in stats.keys():
        count = stats[cls]['count']
        percentage = count / total_targets * 100
        print(f"   {cls}: {count}个 ({percentage:.1f}%)")
    
    # 性能总结
    metrics = results['evaluation_results']['simplified_metrics']
    print(f"\n🏆 性能总结:")
    
    avg_ap = np.mean([metrics[cls]['AP'] for cls in metrics.keys()]) * 100
    avg_aos = np.mean([metrics[cls]['AOS'] for cls in metrics.keys()]) * 100
    
    print(f"   整体平均AP: {avg_ap:.2f}%")
    print(f"   整体平均AOS: {avg_aos:.2f}%")
    
    for cls in metrics.keys():
        m = metrics[cls]
        print(f"   {cls}:")
        print(f"     AP: {m['AP']*100:.2f}%")
        print(f"     AOS: {m['AOS']*100:.2f}%")
        print(f"     Precision: {m['Precision']*100:.2f}%")
        print(f"     Recall: {m['Recall']*100:.2f}%")
    
    print(f"\n✅ 验证完成！所有图表已保存。")

def main():
    """主函数"""
    print("🎨 生成KITTI 500样本验证结果可视化图表...")
    
    # 检查结果文件是否存在
    if not Path('real_kitti_validation_results.json').exists():
        print("❌ 未找到验证结果文件，请先运行验证脚本")
        return
    
    # 加载结果
    results = load_validation_results()
    
    # 生成图表
    print("📈 生成性能对比图表...")
    create_performance_comparison_chart(results)
    
    print("📊 生成数据分布图表...")
    create_data_distribution_chart(results)
    
    print("🏆 生成KITTI官方结果图表...")
    create_kitti_official_results_chart(results)
    
    # 打印总结
    create_summary_report()

if __name__ == '__main__':
    main() 