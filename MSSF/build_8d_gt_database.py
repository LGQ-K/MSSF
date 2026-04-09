import os
import pickle
import numpy as np
from tqdm import tqdm


def extract_points_in_box(points, box):
    """
    底层核心矩阵算子：判断 8维点云中有哪些点落在了 3D 边界框内部
    """
    center = box[0:3]
    l, w, h = box[3], box[4], box[5]
    yaw = box[6]

    pts_local = points[:, 0:3] - center

    cosa, sina = np.cos(-yaw), np.sin(-yaw)
    rot_mat = np.array([
        [cosa, -sina, 0],
        [sina, cosa, 0],
        [0, 0, 1]
    ])
    pts_local = np.dot(pts_local, rot_mat.T)

    mask = (np.abs(pts_local[:, 0]) <= l / 2.0) & \
           (np.abs(pts_local[:, 1]) <= w / 2.0) & \
           (np.abs(pts_local[:, 2]) <= h / 2.0)

    return mask


def main():
    data_root = './data/tj4d/'
    info_path = os.path.join(data_root, 'tj4d_infos_train.pkl')
    db_path = os.path.join(data_root, 'tj4d_gt_database')
    os.makedirs(db_path, exist_ok=True)

    print(f"正在读取原始数据索引: {info_path}")
    with open(info_path, 'rb') as f:
        infos = pickle.load(f)

    # 兼容新旧版本的字典解包
    if isinstance(infos, dict) and 'data_list' in infos:
        data_list = infos['data_list']
    elif isinstance(infos, dict) and 'infos' in infos:
        data_list = infos['infos']
    else:
        data_list = infos

    print(f"成功解包字典，实际需要处理的帧数为: {len(data_list)} 帧")

    db_infos = {'Pedestrian': [], 'Cyclist': [], 'Car': [], 'Truck': []}

    print("开始执行 8 维特征矩阵裁剪...")
    for info in tqdm(data_list):

        # 1. 嗅探并解析点云路径
        rel_path = None
        if 'point_cloud' in info:
            pc_info = info['point_cloud']
            rel_path = pc_info['velodyne_path'] if isinstance(pc_info, dict) else pc_info
        elif 'lidar_points' in info:
            pc_info = info['lidar_points']
            rel_path = pc_info.get('lidar_path', pc_info.get('data_path'))
        elif 'pts_path' in info:
            rel_path = info['pts_path']
        elif 'velodyne_path' in info:
            rel_path = info['velodyne_path']

        if rel_path is None:
            continue

        # 2. 强制对齐绝对物理路径
        filename = os.path.basename(rel_path)
        if not filename.endswith('.bin'):
            filename += '.bin'

        velodyne_path = os.path.join(data_root, 'training', 'velodyne_reduced', filename)
        if not os.path.exists(velodyne_path):
            velodyne_path = os.path.join(data_root, 'training', 'velodyne', filename)

        # 3. 读取 8 维数据流
        try:
            points = np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 8)
        except Exception:
            continue

        # 4. 兼容标注结构
        if 'annos' in info:
            annos = info['annos']
            names = annos['name']
            gt_boxes = annos['gt_boxes_3d']
        elif 'instances' in info:
            names = [inst['bbox_label_3d'] for inst in info['instances']]
            gt_boxes = [inst['bbox_3d'] for inst in info['instances']]
        else:
            continue

        # 5. 实例裁剪
        for i in range(len(names)):
            # ================= 核心修复：硬核映射 =================
            raw_name = str(names[i]).strip()  # 先强转字符串并去空格

            # 定义一个硬映射字典，完全对齐你 check_labels.py 的发现
            class_map = {
                '0': 'Pedestrian',
                '1': 'Cyclist',
                '2': 'Car',
                '3': 'Truck',
                '4': 'DonT'  # 类别 4 我们不需要，可以丢弃
            }
            # 翻译：把 '0' 变成 'Pedestrian'，如果原来就是英文，保持不变
            name = class_map.get(raw_name, raw_name)
            # ==========================================================

            if name not in db_infos:
                continue

            box = gt_boxes[i]
            mask = extract_points_in_box(points, box)
            obj_points = points[mask]

            # 既然行人很难捞，我们把小目标的点数过滤门槛也稍微放宽一点
            MIN_POINTS = {'Car': 5, 'Truck': 5, 'Pedestrian': 3, 'Cyclist': 3}
            if obj_points.shape[0] < MIN_POINTS.get(name, 5):
                continue

            # 几何归一化（保留雷达速度等特征）
            obj_points[:, 0:3] -= box[0:3]

            sample_name = f"{filename.split('.')[0]}_{name}_{i}.bin"
            filepath = os.path.join(db_path, sample_name)

            # 写入本地
            with open(filepath, 'w') as f:
                obj_points.tofile(f)

            # 记录索引
            db_infos[name].append({
                'name': name,
                'path': f'tj4d_gt_database/{sample_name}',
                'gt_idx': i,
                'box3d_lidar': box,
                'num_points_in_gt': obj_points.shape[0],
                'difficulty': 0
            })

    db_info_path = os.path.join(data_root, 'tj4d_dbinfos_train.pkl')
    with open(db_info_path, 'wb') as f:
        pickle.dump(db_infos, f)

    print(f"\n提取完成。8 维索引文件已保存至: {db_info_path}")


if __name__ == '__main__':
    main()