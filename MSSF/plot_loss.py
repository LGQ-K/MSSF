import json
import matplotlib.pyplot as plt
import glob
import os
import numpy as np

# 配置你最新的训练日志目录
WORK_DIR = './work_dirs/centerpoint_voxel005_second_secfpn_8xb4-cyclic-20e_tj4d_mm'


def plot_loss(work_dir):
    search_path = os.path.join(work_dir, '*/vis_data/scalars.json')
    log_files = glob.glob(search_path)
    if not log_files:
        print(f"致命错误：未找到日志文件，请检查路径！")
        return

    latest_log = max(log_files, key=os.path.getctime)
    print(f"正在读取最新的日志文件: {latest_log}")

    raw_steps = []
    raw_losses = []
    with open(latest_log, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                if 'loss' in data and 'step' in data:
                    raw_steps.append(data['step'])
                    raw_losses.append(data['loss'])
            except json.JSONDecodeError:
                continue

    if not raw_steps:
        print("致命错误：日志文件中没有找到 loss 记录。")
        return

    # ===== 终极修复：启用防抖模式，移除开头的异常点 =====
    # 将数据转换为 numpy 数组方便处理
    raw_steps = np.array(raw_steps)
    raw_losses = np.array(raw_losses)

    # 计算移除多少数据。这里设置移除前 500 个步骤（可以根据图表情况自行调整）
    # 只要移除了那个 ~2700 的点，Y轴刻度就会回归正常。
    skip_steps = 500

    if len(raw_losses) <= skip_steps:
        print(f"错误：日志数据量（{len(raw_losses)} 条）太少，不足以跳过前 {skip_steps} 步。尝试将 skip_steps 调小。")
        # 如果数据太少，就不跳过了
        steps, losses = raw_steps, raw_losses
    else:
        print(f"防抖模式已启用：已自动跳过训练开头的 {skip_steps} 个异常步骤，以便观察后续收敛。")
        # 只保留 skip_steps 之后的数据
        steps = raw_steps[skip_steps:]
        losses = raw_losses[skip_steps:]

    # 绘制图像
    plt.figure(figsize=(12, 7))
    plt.plot(steps, losses, label='Total Loss (Raw, Skipped Start)', color='#1f77b4', alpha=0.3, linewidth=1)

    # 移动平均线，使用新数据重新计算
    if len(losses) > 100:
        moving_avg = np.convolve(losses, np.ones(100) / 100, mode='valid')
        # 调整步骤以匹配移动平均线的长度
        ma_steps = steps[99:]
        plt.plot(ma_steps, moving_avg, label='Moving Average (100 pts)', color='#ff7f0e', linewidth=2.5)

    plt.title(f'Training Loss Convergence Curve (100 Epochs, Skipped first {skip_steps} steps)', fontsize=14)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Loss (Zoomed In)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)

    # 根据新数据自动调整纵轴范围，只显示 0.5 到 3.0，彻底排除 ~2700 的干扰
    plt.ylim(0.5, 3.0)

    save_path = os.path.join(work_dir, 'loss_curve_100e_fixed.png')
    plt.savefig(save_path, dpi=300)
    print(f"修复版图像已保存至: {save_path}")
    plt.show()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    plot_loss(WORK_DIR)