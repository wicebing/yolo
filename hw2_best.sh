# TODO: create shell script for running your YoloV1-vgg16bn model
#!/bin/bash

mkdir -p model_zoo
mkdir -p $2

echo "Download Best_model.pth from Dropbox"
wget "https://www.dropbox.com/s/0ygib566lnpsbeh/BW_yolo_best.pth?dl=1"
# Rename the downloaded zip file
echo "move Best_model to model_zoo"
mv ./BW_yolo_best.pth?dl=1 ./model_zoo/BW_yolo_best.pth

echo "====== Predict image by baseline model ======"
echo "plz output predict txt to ./Test_hbb/"

echo "Read image from: " $1 
echo "Output prediction to: "$2
echo "Start to Predict"

python3 predict_best.py $1 $2
echo "Predict Complete, have a nice day!!!"
