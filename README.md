# LaZSL
## Usage

First install the dependencies.

Either manually:
```
conda install pytorch torchvision -c pytorch
conda install matplotlib torchmetrics -c conda-forge
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/modestyachts/ImageNetV2_pytorch
```



To reproduce accuracy results from the paper: edit the directories to match your local machine in `load_OP.py` and set `hparams['dataset']` accordingly. Then simply run `python main_OP.py`.
All hyperparameters can be modified in `load_OP.py`.

