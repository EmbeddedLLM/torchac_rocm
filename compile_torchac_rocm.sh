#!/bin/bash

export HCC_AMDGPU_TARGET=gfx90a  # Replace with your GPU architecture
export ROCM_PATH=/opt/rocm  # Adjust if your ROCm installation is elsewhere

python3 setup.py develop