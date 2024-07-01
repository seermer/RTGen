from mmdet.datasets import BaseDetDataset
from mmdet.registry import DATASETS
import pickle
import pathlib
import imagesize
import signal
import time
import cv2
import json


@DATASETS.register_module()
class GeneratedCaptionDataset(BaseDetDataset):
    METAINFO = {'classes': ('generated_caption', 'generated_image')}

    def load_data_list(self):
        ann_file = self.ann_file
        base_extract_path = pathlib.Path('extracted')
        assert base_extract_path.exists()
        with open(ann_file, 'rb') as f:
            generated_ann = pickle.load(f)
        size_file = pathlib.Path(ann_file).parent / 'train_sizes.json'
        if size_file.exists():
            with open(size_file, 'r') as f:
                size_info = json.load(f)
            print('loaded size info')
        else:
            size_info = {}

        data_infos = []
        base_img_path = pathlib.Path(ann_file).absolute().parent.parent / 'train'
        assert base_img_path.exists()
        print(f'loading generated dataset with expected size of {len(generated_ann)}')
        t0, t1, t2 = 0, 0, 0

        for i, item in enumerate(generated_ann):
            img_path = base_img_path / item['image']
            caption_path = base_extract_path / f'{img_path.stem}.pth'
            assert caption_path.exists(), caption_path
            assert img_path.exists(), img_path

            start = time.perf_counter()

            width, height = size_info.get(img_path.name, (None, None))

            img_path = str(img_path)

            if width is None:
                width, height = imagesize.get(img_path)

            t0 += time.perf_counter() - start
            start = time.perf_counter()

            instance = {}
            bbox = [0.0, 0.0, width, height]
            instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = 0
            instances = [instance]  # image box

            t1 += time.perf_counter() - start
            start = time.perf_counter()

            for bbox_id, region in enumerate(item['regions']):
                instance = {}
                bbox = region['box'].tolist()
                instance['ignore_flag'] = 0
                instance['bbox'] = bbox
                instance['bbox_label'] = 0
                instances.append(instance)  # region box

            data_infos.append(dict(
                img_path=img_path,
                img_id=i,
                image_ids=[i],
                width=width,
                height=height,
                instances=instances,
                captions=str(caption_path),
                tags=[[]]
            ))
            t2 += time.perf_counter() - start

            if len(data_infos) % (len(generated_ann) // 10) == 0:
                print(f'loaded {len(data_infos)}, '
                      f't0={t0 / (i + 1):.7f}, t1={t1 / (i + 1):.7f}, t2={t2 / (i + 1):.7f}')
        print('loaded generated data', len(data_infos))
        return data_infos
