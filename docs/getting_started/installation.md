# Installation

We use miniconda to create a conda environment for the ease of installation. Please type:
```
conda create -n libero python=3.8.13
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

Then, please install [robosuite](https://robosuite.ai/docs/installation.html), which is the underlying simulation environment we use.
```
pip install robosuite
```

Lastly, install the package.
```
git clone https://github.com/zhuyifengzju/libero-dev.git
pip install -e .
```

Now you can enjoy your lifelong learning research!
