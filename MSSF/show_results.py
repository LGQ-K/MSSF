import os
from mmdet3d.apis import init_model, inference_detector
from mmdet3d.visualization import Det3DLocalVisualizer

# 定义路径
config_file = 'configs/centerpoint/centerpoint_voxel005_second_secfpn_8xb4-cyclic-20e_tj4d_mm.py'
checkpoint_file = 'work_dirs/mssf_tj4d/epoch_40.pth'
output_image = 'detected_result.png'

# 默认使用 training 里的第一帧数据
img_path = 'data/tj4d/training/image_2/000000.png'
radar_path = 'data/tj4d/training/velodyne/000000.bin'

if not (os.path.exists(img_path) and os.path.exists(radar_path)):
    print(f"🚨 找不到文件！请在左侧文件树确认 {img_path} 和 {radar_path} 是否存在。如果名字不是 000000，请手动修改 show_results.py 里的数字。")
    exit(1)

print("正在加载模型和权重...")
model = init_model(config_file, checkpoint_file, device='cuda:0')

print("正在推理单帧数据...")
data = dict(img=img_path, pts=radar_path)
result, data_samples = inference_detector(model, data)

print("正在绘制 3D 边界框...")
visualizer = Det3DLocalVisualizer()
visualizer.set_dataset_meta(model.dataset_meta)

visualizer.add_datasample(
    'predict',
    img_path,
    data_sample=data_samples[0],
    draw_gt=False,
    show=False,
    out_file=output_image,
    pred_score_thr=0.4  # 只显示置信度大于 0.4 的检测框
)

print(f"✅ 成功！图像已保存为 {output_image}")
