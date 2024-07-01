import numpy as np
from mmdet.datasets import BaseDetDataset
from mmdet.registry import DATASETS
import pathlib
import json
import os

@DATASETS.register_module()
class GeneratedImageDataset(BaseDetDataset):
    METAINFO = {'classes': ('generated_caption', 'generated_image')}

    def load_data_list(self):
        text_folder, image_folder = self.ann_file.split('+')
        base_folder = pathlib.Path(text_folder)
        assert base_folder.exists()
        json_paths = list(base_folder.glob('*.json'))
        data_infos = []

        print(f'loading generated image dataset with expected size of {len(json_paths)}')

        for i, json_path in enumerate(json_paths):
            ann_file = json_path
            assert ann_file.exists(), ann_file
            with open(ann_file, 'r') as f:
                ann = json.load(f)

            img_path = os.path.join(image_folder, ann['img'])
            assert os.path.exists(img_path), img_path

            width, height = ann['imgwh']

            if len(ann['locations']) == 0:
                continue

            instances = []  # no image box

            whwh = np.array([[width, height, width, height]])
            bboxes = np.array(ann['locations'])
            bboxes = (bboxes * whwh).round().astype(np.int64)
            for bbox in bboxes:
                instance = {}
                instance['ignore_flag'] = 0
                instance['bbox'] = bbox.tolist()
                instance['bbox_label'] = 1
                instances.append(instance)  # region box

            data_infos.append(dict(
                img_path=img_path,
                img_id=i,
                image_ids=[i],
                width=width,
                height=height,
                instances=instances,
                captions=[phrase for phrase in ann['phrases']],
                tags=[[]]
            ))

        print('loaded generated data', len(data_infos))
        return data_infos
