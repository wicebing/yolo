#!/bin/bash

echo "Download best_model.pth from Dropbox"
wget "https://www.dropbox.com/s/xk5cl1pquwcxx9n/yolo_best.pth?dl=1"
# Rename the downloaded zip file
mv ./yolo_best.pth?dl=1 ./yolo_best.pth

echo "Download baseline_model.pth from Dropbox"
wget "https://www.dropbox.com/s/ilqpotj6doramrz/yolo_baseline.pth?dl=1"
# Rename the downloaded zip file
mv ./yolo_baseline.pth?dl=1 ./yolo_baseline.pth
