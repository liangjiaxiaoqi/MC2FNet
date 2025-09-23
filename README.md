# MC2FNet
Our new work about Multi-modal RGB-D/T Salient Object Detection.
# Train & Evaluate
python main.py --config ./configs/rgbd-2dataset.py --model-name MC2FNet_ResNet --info rgbd-2dataset --pretrained ./pretrained/resnet101d.pth
python main.py --config ./configs/rgbd-3dataset.py --model-name MC2FNet_ResNet --info rgbd-3dataset --pretrained ./pretrained/resnet101d.pth
python main.py --config ./configs/rgbt.py --model-name MC2FNet_ResNet --info rgbt --pretrained ./pretrained/resnet101d.pth

