import os
import copy
from mmengine.config import Config
from mmengine.runner import Runner

# 1. 加载配置文件
cfg = Config.fromfile('configs/centerpoint/centerpoint_voxel005_second_secfpn_8xb4-cyclic-20e_tj4d_mm.py')

# 🚨 破案关键：直接用健康的 val_evaluator 覆盖掉原配置里写崩的 test_evaluator！
cfg.test_evaluator = copy.deepcopy(cfg.val_evaluator)
cfg.test_dataloader = copy.deepcopy(cfg.val_dataloader)

# 2. 指定最优权重和工作目录
cfg.load_from = 'work_dirs/mssf_tj4d/epoch_40.pth'
cfg.work_dir = 'work_dirs/mssf_tj4d'

# 3. 创建输出目录并注入保存指令
out_dir = './detection_data_results'
os.makedirs(out_dir, exist_ok=True)
prefix = f'{out_dir}/mssf_predictions'

if isinstance(cfg.test_evaluator, dict):
    cfg.test_evaluator['pklfile_prefix'] = prefix
elif isinstance(cfg.test_evaluator, list):
    for i in range(len(cfg.test_evaluator)):
        if isinstance(cfg.test_evaluator[i], dict):
            cfg.test_evaluator[i]['pklfile_prefix'] = prefix

print("🚀 启动修正版推理进程...")
runner = Runner.from_cfg(cfg)
runner.test()
print(f"\n✅ 导出大功告成！pkl 数据已保存在 {out_dir} 文件夹中。")
