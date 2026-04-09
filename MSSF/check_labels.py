import pickle
import numpy as np

def main():
    info_path = './data/tj4d/tj4d_infos_train.pkl'
    with open(info_path, 'rb') as f:
        infos = pickle.load(f)

    data_list = infos.get('data_list', infos.get('infos', infos))
    dims_dict = {}

    for info in data_list:
        if 'annos' in info:
            names = info['annos']['name']
            boxes = info['annos']['gt_boxes_3d']
        elif 'instances' in info:
            names = [inst['bbox_label_3d'] for inst in info['instances']]
            boxes = [inst['bbox_3d'] for inst in info['instances']]
        else:
            continue

        for i, name in enumerate(names):
            name = str(name).strip()
            if name not in dims_dict:
                dims_dict[name] = []
            dims_dict[name].append(boxes[i][3:6]) # 提取 l, w, h

    print("\n📏 类别真实物理尺寸扫描报告：")
    for name, dims in dims_dict.items():
        avg_dims = np.mean(dims, axis=0)
        print(f"标签 '{name}': 平均 长 {avg_dims[0]:.2f}m, 宽 {avg_dims[1]:.2f}m, 高 {avg_dims[2]:.2f}m  (总数量: {len(dims)})")

if __name__ == '__main__':
    main()