import warnings
import os
import pandas as pd
from prettytable import PrettyTable
from datetime import datetime

# 设置CUDA设备
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"    # 使用CPU训练
# os.environ["CUDA_VISIBLE_DEVICES"]="0"     # 使用第一张GPU训练
warnings.filterwarnings('ignore')
from ultralytics import YOLO


def save_epoch_results(results, save_path):
    """将每个epoch的结果保存为CSV表格和格式化文本"""
    # 提取训练过程中的关键指标
    epochs = []
    train_box_loss = []
    train_cls_loss = []
    train_dfl_loss = []
    val_precision = []
    val_recall = []
    val_map50 = []
    val_map50_95 = []
    lr = []
    train_time = []

    # 收集每个epoch的数据
    for epoch in range(len(results.history['epoch'])):
        epochs.append(int(results.history['epoch'][epoch]))
        train_box_loss.append(float(results.history['train/box_loss'][epoch]))
        train_cls_loss.append(float(results.history['train/cls_loss'][epoch]))
        train_dfl_loss.append(float(results.history['train/dfl_loss'][epoch]))
        val_precision.append(float(results.history['metrics/precision(B)'][epoch]))
        val_recall.append(float(results.history['metrics/recall(B)'][epoch]))
        val_map50.append(float(results.history['metrics/mAP50(B)'][epoch]))
        val_map50_95.append(float(results.history['metrics/mAP50-95(B)'][epoch]))
        lr.append(float(results.history['lr/pg0'][epoch]))
        train_time.append(float(results.history['time'][epoch]))

    # 创建数据框
    data = {
        'Epoch': epochs,
        'Train Box Loss': train_box_loss,
        'Train Cls Loss': train_cls_loss,
        'Train DFL Loss': train_dfl_loss,
        'Val Precision': val_precision,
        'Val Recall': val_recall,
        'Val mAP50': val_map50,
        'Val mAP50-95': val_map50_95,
        'Learning Rate': lr,
        'Train Time (s)': train_time
    }

    df = pd.DataFrame(data)

    # 保存为CSV文件
    csv_path = os.path.join(save_path, 'epoch_results.csv')
    df.to_csv(csv_path, index=False)

    # 创建格式化表格并保存为文本文件
    table = PrettyTable()
    table.title = "Training Epoch Results"
    table.field_names = df.columns

    # 添加数据行
    for _, row in df.iterrows():
        table.add_row([
            int(row['Epoch']),
            f"{row['Train Box Loss']:.4f}",
            f"{row['Train Cls Loss']:.4f}",
            f"{row['Train DFL Loss']:.4f}",
            f"{row['Val Precision']:.4f}",
            f"{row['Val Recall']:.4f}",
            f"{row['Val mAP50']:.4f}",
            f"{row['Val mAP50-95']:.4f}",
            f"{row['Learning Rate']:.6f}",
            f"{row['Train Time (s)']:.2f}"
        ])

    # 保存为文本文件
    txt_path = os.path.join(save_path, 'epoch_results.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(str(table))

    print(f"\nEpoch results saved to:")
    print(f"CSV file: {csv_path}")
    print(f"Text table: {txt_path}")


if __name__ == '__main__':
    # 初始化模型
    model = YOLO('ultralytics/cfg/models/11/yolo11-C3k2-LGLB.yaml')  # YOLO11模型配置

    # 可以选择加载预训练权重
    # model.load('yolo11n.pt')  # 加载预训练权重

    # 设置训练参数
    project = 'runs/train'
    exp_name = 'exp'
    save_dir = os.path.join(project, exp_name)

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 开始训练并获取结果
    results = model.train(
        data='/root/dataset/dataset_visdrone/data.yaml',
        cache=False,
        imgsz=640,
        epochs=200,
        batch=32,
        close_mosaic=0,  # 最后多少个epoch关闭mosaic数据增强，0代表全程开启
        workers=4,  # Windows下可设为0避免卡顿
        device='0',  # 指定显卡
        optimizer='SGD',  # 使用SGD优化器
        # patience=0,  # 设置0关闭早停
        # resume=True,  # 断点续训，需加载last.pt
        # amp=False,  # 关闭自动混合精度训练，loss出现nan时可尝试
        # fraction=0.2,  # 使用部分数据集进行训练
        project=project,
        name=exp_name,
    )

    # 保存每个epoch的结果
    save_epoch_results(results, save_dir)

    print("\n训练完成！所有epoch数据已保存。")
