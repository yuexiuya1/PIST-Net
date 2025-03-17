# Estimating Stellar Fundamental Parameters for Red Giants with Multimodal Method

Thank you for watching our code! 

The file structure of the project is shown below
```
main
├── model
├── data 
│   ├── data
│   │   ├── xxx.csv
│   │   ├── yyy.csv
│   │   └── ...
│   ├── pic16
│   │   ├── xxx.fits
│   │   ├── yyy.fits
│   │   └── ...
│   ├── fold_1_train_labels.csv
│   ├── fold_1_val_labels.csv
│   └── ...
├── cfg \\ configs for MSPC-Net
├── main.py \\ spectral data train file
├── picmain.py \\ image data train file
├── multimain.py \\ multimodal data train file
├── cutpic.py
├── preprocess.ipynb
├── requirements.txt 
└── ...
```

The training data is compressed in [data.tar.gz](https://drive.google.com/file/d/1frPUIh3F5jcylFoD-ifFsSm0o3F3YV_b/view?usp=sharing)

To save storage space, the picture data for training is saved only in the file of the largest picture. Before you try training, run cutpic.py to generate additional images.

All training code is included in this project. Data acquisition and data preprocessing code in preprocess.ipynb, unorganized, and some code may be missing. All training code is executed under WSL2 Ubuntu20.04, while data acquisition and preprocessing is executed under Windows 10.

