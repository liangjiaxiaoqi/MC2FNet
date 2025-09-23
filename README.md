# MC2FNet
Our new work about Multi-modal RGB-D/T Salient Object Detection.

# Usage
Step 1: Download MC2FNet from https://github.com/liangjiaxiaoqi/MC2FNet.  
Step 2: Download the backbone pretrained parameters.  
Step 3: Download the RGB-D/T dataset and set your data path in datasets.py.  
Step 4: Train and Evaluate  
Step 5: Use PySODEvalToolkit to test the corresponding metrics and plot curves such as PR from https://github.com/lartpang/PySODEvalToolkit.  

## Train & Evaluate
python main.py --config ./configs/rgbd-2dataset.py --model-name MC2FNet_ResNet --info rgbd-2dataset --pretrained ./pretrained/resnet101d.pth  
python main.py --config ./configs/rgbd-3dataset.py --model-name MC2FNet_ResNet --info rgbd-3dataset --pretrained ./pretrained/resnet101d.pth  
python main.py --config ./configs/rgbt.py --model-name MC2FNet_ResNet --info rgbt --pretrained ./pretrained/resnet101d.pth  

