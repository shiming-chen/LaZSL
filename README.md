# LaZSL
This repository contains the code for the ICCV'25 paper titled with  "***Intrepretable Zero-Shot Learning with Locally-Aligned Vision-Language Model***".

## Requirements
First install the dependencies.

Either manually:
```
conda install pytorch torchvision -c pytorch
conda install matplotlib torchmetrics -c conda-forge
```

## Preparing Dataset
Please follow the instructions [DATASETS.md](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) to construct the datasets.

## Test

To reproduce accuracy results from the paper: edit the directories to match your local machine in `load_OP.py` and set `hparams['dataset']` accordingly. Then simply run `python main_OP.py`.
All hyperparameters can be modified in `load_OP.py`.

## Results
Results of our released models using various evaluation protocols on 6 datasets.


| Dataset | Acc(ViT-B/32) | Acc(ViT-B/16) | Acc(ViT-L/14) |
| :-----: | :-----: | :-----: | :-----: |
| Imagenet | 65.3 | 69.2| 75.7 |
| CUB | 56.5 | 60.3 | 66.1 |
| OxfordPets | 84.7 | 87.4 | 92.7 |
| Food101 | 85.9 | 89.7 | 93.5 |
| Place365 | 41.5 | 42.0 | 41.8 | 

## Citation
If you find LaZSL is useful in your research or applications, please consider giving us a star 🌟.





