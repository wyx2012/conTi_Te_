# ReMoE: Weight-Space Mixture-of-Experts for mRNA Translation Efficiency

This repository contains the official implementation of **ReMoE** (Robust Mixture-of-Experts), a framework designed for predicting mRNA Translation Efficiency (TE) across diverse tissues.

##  Abstract

Predicting mRNA Translation Efficiency (TE) across diverse tissues is critical yet challenging due to concept shift, where identical genomic sequences exhibit distinct functional profiles in different cellular environments. Existing static models fail to capture these continuous regulatory variations, often requiring expensive fine-tuning for unseen domains. 

##  Project Structure

The project is modularized into three main components:

- **`main.py`**: The main entry point for training and testing. It handles the training loop, validation with adaptation, and final testing.
- **`model.py`**: Contains the model architecture definitions, including the `HAC_Net` (implementation of ReMoE) and `DynamicConv1d`.
- **`DataLoad.py`**: Handles data loading, preprocessing (sequence truncation/padding), and PyTorch `Dataset` creation.

##  Requirements

To run this project, please ensure you have the following packages installed. 
You can install them using the commands below:

```bash
pip install pandas==2.3.3
pip install numpy==1.26.4
pip install torch==2.9.1
pip install scikit-learn==1.7.2
pip install scipy==1.16.3
pip install tqdm==4.67.1
```

##  Training & Testing
Run the main script to start the training process. The script includes training, validation (with few-shot adaptation), and final testing on unseen tissues.
```bash
python main.py
```


# README
