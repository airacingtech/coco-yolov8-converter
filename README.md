# coco-yolov8-converter

 Convert COCO dataset to YOLOv8 format.

 Expected file structure:
```sh
coco/
├── converted/ # (will be generated)
│   └── 123/
│       ├── images/
│       └── labels/
├── unconverted/
│   └── 123/
│       ├── annotations/
│       └── images/
└── convert.py
```
Usage: `python convert.py unconverted`
optionally specify `--handle_images copy/move/none`