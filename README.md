# TorchAC CUDA/ROCm

For now this is a fork of [torchac_cuda Github](https://github.com/LMCache/torchac_cuda.git)

This library is needed by LMCache repositories.

## Setup
### CUDA
```bash
python3 setup.py develop
```

### ROCm
```bash
HCC_AMDGPU_TARGET=<HIP_ARCHITECTURE> python3 setup.py develop
# e.g. MI210 MI250
HCC_AMDGPU_TARGET=gfx90a python3 setup.py develop
```