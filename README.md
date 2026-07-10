<!--
# MC2FNet
Our new work about Multi-modal RGB-D/T Salient Object Detection.

# Usage
Step 1: Download MC2FNet from https://github.com/liangjiaxiaoqi/MC2FNet.  
Step 2: Download the backbone pretrained parameters from https://pan.baidu.com/s/1rs7GbpSJP5FOdLgwiXTElA, and the extraction code is gnxq.  
Step 3: Download the RGB-D/T dataset and set your data path in datasets.py. The RGB-D/T dataset link is https://pan.baidu.com/s/1zV5C8ckiPcYNL18PLxHmYQ, and the extraction code is ain7.  
Step 4: Train and Evaluate  
<!--Step 5: Use PySODEvalToolkit to test the corresponding metrics and plot curves such as PR from https://github.com/lartpang/PySODEvalToolkit.-->  
<!--
## Train & Evaluate
python main.py --config ./configs/rgbd-2dataset.py --model-name MC2FNet_ResNet --info rgbd-2dataset --pretrained ./pretrained/resnet101d.pth  
python main.py --config ./configs/rgbd-3dataset.py --model-name MC2FNet_ResNet --info rgbd-3dataset --pretrained ./pretrained/resnet101d.pth  
python main.py --config ./configs/rgbt.py --model-name MC2FNet_ResNet --info rgbt --pretrained ./pretrained/resnet101d.pth  
-->


# MC2FNet: Multi-Scale Cross-Modal Competitive Fusion Network for RGB-D/T Salient Object Detection

🎉 **News:** Our paper has been officially accepted by **The Visual Computer**! 🎉

This repository contains the official PyTorch implementation of **MC2FNet**. In this work, we propose a Multi-Scale Cross-Modal Competitive Fusion Network designed for highly efficient multi-modal interaction in both RGB-D (Depth) and RGB-T (Thermal) Salient Object Detection (SOD) tasks.

---

## 🚀 Model Architecture

*Our MC2FNet effectively leverages multi-scale features and introduces a competitive fusion mechanism to deeply integrate cross-modal information, suppressing noise and highlighting salient regions across different modalities.*

<div align="center">
  <img src="docs/model.png" alt="MC2FNet Model Architecture" width="90%">
  <p><em>Figure 1: Overall Architecture of the proposed MC2FNet.</em></p>
</div>

---

## 📊 Experimental Results

Extensive experiments demonstrate that MC2FNet achieves state-of-the-art performance against other methods on multiple widely-used RGB-D and RGB-T Salient Object Detection benchmarks.

<div align="center">
  <img src="docs/results.png" alt="Experimental Results" width="90%">
  <p><em>Figure 2: Qualitative and Quantitative results of MC2FNet compared with State-of-the-Art methods.</em></p>
</div>

---

## 🛠️ Usage

### Step 1: Clone the Repository
Download MC2FNet from our GitHub repository:
```bash
git clone [https://github.com/liangjiaxiaoqi/MC2FNet.git](https://github.com/liangjiaxiaoqi/MC2FNet.git)
cd MC2FNet
