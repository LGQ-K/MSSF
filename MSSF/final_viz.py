import os
import torch
import mmcv
import numpy as np
from mmengine.config import Config
from mmengine.registry import DATASETS
from mmengine.dataset import pseudo_collate
from mmdet3d.apis import init_model
from mmdet3d.visualization import Det3DLocalVisualizer

print("1. 正在初始化模型...")
config_file = 'configs/centerpoint/centerpoint_voxel005_second_secfpn_8xb4-cyclic-20e_tj4d_mm.py'
checkpoint_file = 'work_dirs/mssf_tj4d/epoch_40.pth'
cfg = Config.fromfile(config_file)
if 'work_dir' not in cfg:
    cfg.work_dir = 'work_dirs/mssf_tj4d'
model = init_model(cfg, checkpoint_file, device='cuda:0')

print("2. 正在加载数据集并搜寻第 100 帧...")
dataset = DATASETS.build(cfg.val_dataloader.dataset)
frame_idx = 100
raw_info = dataset.get_data_info(frame_idx)
data_input = dataset[frame_idx]
data_batch = pseudo_collate([data_input])

print("3. 模型推理中...")
with torch.no_grad():
    results = model.test_step(data_batch)

print("4. 正在精确定位图像文件...")
img_path = raw_info.get('img_path', None)
if img_path is None:
    img_dir = dataset.data_prefix.get('img', 'data/tj4d/training/image_2')
    sample_id = raw_info.get('sample_idx', raw_info.get('image_id', ''))
    if isinstance(sample_id, int):
        sample_id = f"{sample_id:06d}"
    img_path = os.path.join(img_dir, f"{sample_id}.png")

print(f"👉 确认图片路径: {img_path}")
img_bytes = mmcv.imread(img_path, channel_order='rgb')

print("5. 正在渲染并保存...")
visualizer = Det3DLocalVisualizer()
for attr in ['dataset_meta', 'metainfo']:
    if hasattr(visualizer, attr):
        setattr(visualizer, attr, dataset.metainfo)

input_dict = dict(img=img_bytes)
visualizer.add_datasample(
    'result',
    input_dict,
    data_sample=results[0],
    draw_gt=False,
    out_file='final_viz_result.png',
    pred_score_thr=0.2
)

if os.path.exists('final_viz_result.png'):
    print("\n✅【大功告成】图片已生成：final_viz_result.png")
else:
    print("\n❌【保存失败】渲染过程出现未知错误。")
