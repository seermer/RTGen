# Data preparation
## Open-Vocabulary COCO
Prepare data following [MMDetection](https://mmdetection.readthedocs.io/en/latest/user_guides/useful_tools.html#dataset-download).\
Download open-vocabulary annotation files from [Google Drive](https://drive.google.com/drive/folders/16EOi35ffNFqsWVIKEak1NQ0YyaJl0neQ?usp=drive_link), and place into the following folder structure:

```text
RTGen/data
├── coco
    ├── annotations
        ├── instances_{train,val}2017.json
    ├── ov
        ├── instances_train2017_base.json
        ├── instances_val2017_base.json
        ├── instances_val2017_novel.json
    ├── train2017
    ├── val2017
    ├── test2017
```


## Open-Vocabulary LVIS
Prepare data following [MMDetection](https://mmdetection.readthedocs.io/en/latest/user_guides/useful_tools.html#dataset-download).\
Download open-vocabulary annotation file from [Google Drive](https://drive.google.com/file/d/1798tdJw3UD1LThFdAMu65Leqx2gS4okf/view?usp=drive_link), and place into the following folder structure:

```text
RTGen/data
├── lvis_v1
    ├── annotations
        ├── lvis_v1_val.json
        ├── lvis_v1_train.json
    ├── ov
        ├── lvis_v1_train_base.json
    ├── train2017
    ├── val2017
```


## RTGen
Download CC3M from our [Huggingface Datasets](https://huggingface.co/datasets/Dwrety/RTGen/tree/main). Download `cc3m_images.*` and `annotations` folder, unzip all zip files. Place into the following folder structure:

```text
RTGen/data
├── cc3m
    ├── annotations
        ├── cc3m_blip2_full_and_region_captions_v2.pkl
        ├── cleaned_inpainting_box_text_1M
        ├── extracted_v2
    ├── images
```

Download text-to-region generated samples from our [Huggingface Datasets](https://huggingface.co/datasets/Dwrety/RTGen/tree/main). Download `generation_samples.*`, unzip all zip files. Place into the following folder structure:

```text
RTGen/data
├── generation_samples
```


## Final Structure
If all data are downloaded, the final structure should be:

```text
RTGen/data
├── coco
    ├── annotations
        ├── instances_{train,val}2017.json
    ├── ov
        ├── instances_train2017_base.json
        ├── instances_val2017_base.json
        ├── instances_val2017_novel.json
    ├── train2017
    ├── val2017
    ├── test2017
├── lvis_v1
    ├── annotations
        ├── lvis_v1_val.json
        ├── lvis_v1_train.json
    ├── ov
        ├── lvis_v1_train_base.json
    ├── train2017
    ├── val2017
├── cc3m
    ├── annotations
        ├── cc3m_blip2_full_and_region_captions_v2.pkl
        ├── cleaned_inpainting_box_text_1M
        ├── extracted_v2
    ├── images
├── generation_samples
```
