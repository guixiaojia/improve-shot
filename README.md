</center>
# Multi-view pedestrian detection via residual mask fusion and cosine similarity-based passive sampler for video surveillance systems
center>

## Project Overview

This project proposes a multi-view pedestrian detection method for video surveillance systems, which improves detection performance through residual masking and a passive sampler based on cosine similarity.

## Project Structure

<pre>
improve-shot/
├── multiview_detector/
│   ├── datasets/             
│   │   ├── MultiviewX.py     # MultiviewX dataset parser
│   │   ├── Wildtrack.py      # Wildtrack dataset parser
│   │   ├── __init__.py
│   │   └── frameDataset.py   # Frame Processing
│   │  
│   ├── evaluation/ 
│   │  
│   ├── loss/ 
│   │   ├── __init__.py
│   │   ├── gaussian_mse.py   # Gaussian mean square error loss
│   │   └── losses.py         # Focal loss
│   │
│   ├── models/               
│   │   ├── MultiViewDynamicMask.py # residual mask
│   │   ├── fusion3.py              # cosine similarity-based passive sampler
│   │   ├── resnet.py               # backbone
│   │   ├── shot.py                 # baseline
│   │   └── ops/
│   │
│   ├── utils/
│   │
│   └── trainer.py
│
├── LICENSE                   # MIT License
├── README.md
└── main.py
</pre>

## Environment Version

```
RTX 4090
ubuntu: 20.04
PyTorch: 1.12.0
torchvision: 0.13.0
Numpy: 1.21.2
tqdm: 4.65.2
kornia: 0.6.12
opencv-python: 4.9.0.80
matplotlib: 3.5.2
```

## Dataset Preparation

| Dataset | link |
|--------|----------|
| **Wildtrack** | [🔗 download link](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/) |
| **MultiviewX** | [🔗 download link](https://onedrive.live.com/?id=DFB1B9D32643ECDC%2182813&resid=DFB1B9D32643ECDC%2182813&e=Hm9Xdg&migratedtospo=true&redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBdHpzUXliVHViSGZoWVo5R2hoYWhicDIwT1g5a0E%5FZT1IbTlYZGc&cid=dfb1b9d32643ecdc&v=validatepermission) |



