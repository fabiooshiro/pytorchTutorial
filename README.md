# Estudando PyTorch

https://www.youtube.com/watch?v=c36lUUr864M

## Virtual ENV
https://github.com/conda-forge/miniforge

Crie o seu virtual env:
`conda create --name pytorch`

Ative o seu virtual env:  
`conda activate pytorch`

Instale o que vc precisar, exemplos:

```
conda install pytorch torchvision -c pytorch
conda install matplotlib -c pytorch
conda install -c conda-forge scikit-learn 
```

para instalar o tensorboard no Apple Silicon M1
```
export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1

pip install grpcio
pip install tensorboard
```