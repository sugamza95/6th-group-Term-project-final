#!/bin/bash

cd tensorrt_demos
git clone https://github.com/NVIDIA-AI-IOT/trt_pose.git
cd trt_pose
sudo python3 setup.py install
cd ..
python3 trt_yolo_with_centernet.py --video /home/ee201511281/sample_mask.mp4 -m yolov4-tiny-416
