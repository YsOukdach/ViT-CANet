# ViT-CANet

This repository represents the whole implementation of the proposed approach described in our paper titled 'ViTCA-Net: A Framework for Disease Detection in Video Capsule Endoscopy Images Using a Vision Transformer and Convolutional Neural Network with a Specific Attention Mechanism.

To train the model, please feel free to clone this repository using this command in your terminal:
```bash
git clone https://github.com/YsOukdach/ViT-CANet.git
```
Install install the dependencies by using this command:
```bash
pip install -r requerments 
```
Start training runing: 
```bash
python train.py 
```
The pretrained models, combined with ViT, are implemented in the pretrainedmodel folder. Feel free to test the concatenation experiments


In the testing folder, you can evaluate the performance by displaying a classification report, ROC curves, and heatmaps to assess the model's performance across various regions.

