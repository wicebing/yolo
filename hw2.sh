# TODO: create shell script for running your YoloV1-vgg16bn model
#!/bin/bash

mkdir -p model_zoo
mkdir -p $2

echo "Download Base_model.pth from Dropbox"
wget "https://www.dropbox.com/s/h6xnz60kvcygbb0/BW_yolo_baseline.pth?dl=1"
# Rename the downloaded zip file
echo "move Base_model to model_zoo"
mv ./BW_yolo_baseline.pth?dl=1 ./model_zoo/BW_yolo_baseline.pth

echo "====== Predict image by baseline model ======"
echo "plz output predict txt to ./Test_hbb/"

echo "Read image from: " $1 
echo "Output prediction to: "$2
echo "Start to Predict"

python3 predict_base.py $1 $2
echo "Predict Complete, have a nice day!!!"
