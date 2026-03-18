#!/bin/bash

yolo track \
  model=runs/detect/train2/weights/best.pt \
  source=/home/stasen/Desktop/plantas/videos/* \
  conf=0.4 \
  iou=0.5 \
  tracker=bytetrack.yaml \
  save_txt=True \
  save_conf=True
