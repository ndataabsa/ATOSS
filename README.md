### Setup

```sh
conda create -n mvp python=3.8
conda activate mvp
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
pip install jupyter notebook
python -m ipykernel install --user --name mvp --display-name mvp
```

## Requirements

We highly recommend you to install the specified version of the following packages to avoid unnecessary troubles:

typing-extensions==4.6.2
sentencepiece==0.1.99
numpy==1.24.3
pandas==1.5.3
pytorch_lightning==1.9.5
tqdm==4.65.0
transformers==4.14.1
openai==0.27.7
Requests==2.31.0


## git add 
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ndata_rsa

cd ~/ABSA
git add .
git commit -m "data upload"
git push -u origin main

