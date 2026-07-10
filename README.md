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


# Competitive fusion in multimodal networks for enhanced salient object detection

🎉🎉🎉 **News:** [Our paper](https://link.springer.com/article/10.1007/s00371-026-04602-y) has been officially accepted by ***The Visual Computer (CCF-C)***! 🎉🎉🎉

This repository contains the official PyTorch implementation of **MC2FNet**. In this work, we propose a Multi-Scale Cross-Modal Competitive Fusion Network designed for highly efficient multi-modal interaction in both RGB-D (Depth) and RGB-T (Thermal) Salient Object Detection (SOD) tasks.

---

## ✨ Motivation

In RGB-D/T salient object detection (SOD), the information representations of interference and target objects at the same depth or thermal are extremely similar, such as those of the red and yellow boxes in the figure, which may cause false detection. To address this issue, we propose MC2FNet.

<div align="center">
  <img src="docs/Examples of Failed Tests.jpg" alt="MC2FNet Model Architecture' Motivation" width=90%">
  <p><em>Figure 1: In cases where detection fails when the target and background are similar in depth and thermal, the proposed MC2FNet can effectively mitigate this issue.</em></p>
</div>

---

## 🚀 Model Architecture

*Our MC2FNet effectively leverages multi-scale features and introduces a competitive fusion mechanism to deeply integrate cross-modal information, suppressing noise and highlighting salient regions across different modalities.*

<div align="center">
  <img src="docs/model.jpg" alt="MC2FNet Model Architecture" width=90%">
  <p><em>Figure 2: Overall Architecture of the proposed MC2FNet.</em></p>
</div>

---

## 📊 Experimental Results

Extensive experiments demonstrate that MC2FNet achieves state-of-the-art performance against other methods on multiple widely-used RGB-D and RGB-T Salient Object Detection benchmarks.

<div align="center">
  <img src="docs/RGB-T results.jpg" alt="Experimental Results" width="90%">
  <p><em>Figure 3: <!--Qualitative and -->Quantitative results of MC2FNet compared with RGB-T SOD State-of-the-Art methods.</em></p>
</div>

<div align="center">
  <img src="docs/RGB-D results.jpg" alt="Experimental Results" width="90%">
  <p><em>Figure 4: <!--Qualitative and -->Quantitative results of MC2FNet compared with RGB-D SOD State-of-the-Art methods.</em></p>
</div>

<div align="center">
  <img src="docs/Visualization of MC2FNet.jpg" alt="Experimental Results" width="90%">
  <p><em>Figure 5: Visualization of RGB-D/T salient object detection by MC2FNet. CMFU<sub>i</sub> represents the decoder for each stage.</em></p>
</div>

---

## 🛠️ Usage

<!--
### Step 1: Clone the Repository
Download MC2FNet from our GitHub repository:
```bash
git clone [https://github.com/liangjiaxiaoqi/MC2FNet.git](https://github.com/liangjiaxiaoqi/MC2FNet.git)
cd MC2FNet
-->

### Preparations
Step 1: Cloned MC2FNet from https://github.com/liangjiaxiaoqi/MC2FNet.  
Step 2: Download the backbone pretrained parameters from https://pan.baidu.com/s/1rs7GbpSJP5FOdLgwiXTElA, and the extraction code is gnxq.  
Step 3: Download the RGB-D/T dataset and set your data path in datasets.py. The RGB-D/T dataset link is https://pan.baidu.com/s/1zV5C8ckiPcYNL18PLxHmYQ, and the extraction code is ain7.  
Step 4: Train and Evaluate  
<!--Step 5: Use PySODEvalToolkit to test the corresponding metrics and plot curves such as PR from https://github.com/lartpang/PySODEvalToolkit.-->  

#### Train & Evaluate
python main.py --config ./configs/rgbd-2dataset.py --model-name MC2FNet_ResNet --info rgbd-2dataset --pretrained ./pretrained/resnet101d.pth  
python main.py --config ./configs/rgbd-3dataset.py --model-name MC2FNet_ResNet --info rgbd-3dataset --pretrained ./pretrained/resnet101d.pth  
python main.py --config ./configs/rgbt.py --model-name MC2FNet_ResNet --info rgbt --pretrained ./pretrained/resnet101d.pth 

---

## 📚 Related Multi-Modal SOD Works

If you are interested in our research, please also check out related works in the field of multi-modal representation learning and salient object detection:

● [Multi-Modal Hierarchical Fusion with Cross-Agent for RGB-D Salient Object Detection (ICASSP 2026, CCF-B)](https://github.com/liangjiaxiaoqi/HMaT-D)  
● [HEFT: Hierarchical Enhanced Fusion Transformer for RGB-D Salient Object Detection (ICARM 2025, CAA-A)](https://ieeexplore.ieee.org/document/11293468)

---

## ✒️ Citation

If you find our work, model, or code useful for your research, please consider citing our paper in The Visual Computer:
```bibtex
@article{liang2026mc2fnet,
  title={MC2FNet: Multi-modal Cross-level Collaborative Fusion Network for RGB-D and RGB-T Salient Object Detection},
  author={Liang, Jiaxiao and Others},
  journal={The Visual Computer},
  year={2026},
  publisher={Springer},
  issn={0178-2789},
  doi={10.1007/s00371-026-02xxxx-x}
}
