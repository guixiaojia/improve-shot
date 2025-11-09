# Multi-view Pedestrian Detection with Residual Mask Fusion and Cosine Similarity-based Passive Sampler

## 项目简介

本项目提出了一种用于视频监控系统的多视角行人检测方法，通过残差掩码融合和基于余弦相似度的被动采样器来提高检测性能。

## 项目结构

<pre>
improve-shot/
├── LICENSE                   # MIT License
├── multiview_detector/
│   ├── models/               # Network architectures
│   │   ├── mvdetr.py         # Multi-view detection transformer model
│   │   ├── conv_world_feat.py # Convolutional world feature fusion
│   │   ├── trans_world_feat.py # Transformer-based world feature processing
│   │   ├── fusion3.py        # Feature fusion modules
│   │   ├── boostershot.py    # Reference map generation
│   │   ├── ops/              # Custom operations (CUDA extensions)
│   │   │   ├── setup.py      # Compilation script for CUDA ops
│   │   │   └── make.sh       # Build script for deformable convolutions
│   │   └── attn_module.py    # Attention utilities
│   │
│   ├── datasets/             # Dataset handling
│   │   ├── Wildtrack.py      # Wildtrack dataset parser
│   │   └── MultiviewX.py     # MultiviewX dataset parser
│   │
│   ├── evaluation/           # Evaluation tools
│   │   ├── README.md         # Evaluation setup guide
│   │   ├── gt-demo.txt       # Example ground truth data
│   │   ├── pyeval/           # Python evaluation API (no MATLAB required)
│   │   │   └── README.md     # Guide for Python evaluation
│   │   └── motchallenge-devkit/ # MOTChallenge evaluation toolkit
│   │       ├── README.md     # MOTChallenge devkit guide
│   │       ├── res/          # Tracking results storage
│   │       ├── utils/        # Evaluation utilities
│   │       │   ├── convertTXTToStruct.m # Convert results to MATLAB struct
│   │       │   └── external/ # External dependencies
│   │       │       ├── PrintTable.m # MATLAB table formatting for reports
│   │       │       ├── iniconfig/ # INI config parser (BSD licensed)
│   │       │       └── dollar/ # Detector utilities (based on Integral Channel Features)
│   │       └── seqmaps/      # Benchmark sequence lists
│   │
│   └── utils/                # Utility functions
│       ├── image_utils.py    # Image processing helpers
│       └── projection.py     # Camera projection utilities
</pre>
