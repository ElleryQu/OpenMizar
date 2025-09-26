# Mizar: Boosting Secure Three-Party Deep Learning with Co-Designed Sign-Bit Extraction and GPU Acceleration


## Introduction

Mizar is a secure computation protocol designed for three-party outsourced computing. By improving the sign-bit extraction protocol and introducing parallel computing acceleration, it delivers high-performance secure inference capabilities for neural networks. For detailed information on the relevant protocols and implementation specifics, please refer to the paper Mizar: Boosting Secure Three-Party Deep Learning with Co-Designed Sign-Bit Extraction and GPU Acceleration. 

This repository is built upon [Piranha](https://github.com/ucbrise/piranha), which provides a prototype implementation of Mizar, intended solely for experimental testing purposes and SHOULD NOT be used in production environments.

The paper is available on [Eprint/2025/1063](https://eprint.iacr.org/2025/1063).


## Build

```shell
# prepare deps.
apt install libgtest-dev libssl-dev
git submodule update --init --recursive

# please fill in the location of the nvcc compiler based on your CUDA toolkit installation status. 
# specifically, modify the `CUDA_VERSION` variables under `files/make/Makefile-*`.
NEW_CUDA_VERSION=$(nvcc --version | grep -oP '(?<=release )\d+\.\d+')
sed -i "s/^CUDA_VERSION=.*/CUDA_VERSION=$NEW_CUDA_VERSION/" files/make/Makefile-PFalcon
sed -i "s/^CUDA_VERSION=.*/CUDA_VERSION=$NEW_CUDA_VERSION/" files/make/Makefile-Mizar
sed -i "s/^CUDA_VERSION=.*/CUDA_VERSION=$NEW_CUDA_VERSION/" files/make/Makefile-Mizar

# build pfalcon, aegis and mizar.
./scripts/quick_make.sh
```

## Download Datasets
```shell
# download necessary datasets.
mkdir -p files/MNIST; mkdir -p files/CIFAR10
pip install torch torchvision
python download_{mnist, cifar}.py
```

## Run
```shell
# run the experiments.
# by default, only the online phase of the protocol is executed. 
# if you wish to run an end-to-end experiment, please add the "-s" parameter.
python3 ./scripts/quick_exp.py -m func  -p falcon aegis mizar -c 0 1 2                                                          # func test
python3 ./scripts/quick_exp.py -m snni  -p falcon aegis mizar -c 0 1 2 --models lenet vgg16 --num_iterations 10                 # snni test
python3 ./scripts/quick_exp.py -m train -p falcon aegis mizar -c 0 1 2 --models lenet vgg16 --num_iterations 10 --batch_size 32 # snnt test
```
For more details about artifacts, please move to the directory /claims.

## Citing Mizar

```text
@misc{cryptoeprint:2025/1063,
      author = {Ye Dong and Xudong Chen and Xiangfu Song and Yaxi Yang and Tianwei Zhang and Jin-Song Dong},
      title = {{MIZAR}: Boosting Secure Three-Party Deep Learning with Co-Designed Sign-Bit Extraction and {GPU} Acceleration},
      howpublished = {Cryptology {ePrint} Archive, Paper 2025/1063},
      year = {2025},
      url = {https://eprint.iacr.org/2025/1063}
}
```
