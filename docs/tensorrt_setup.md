# TensorRT Setup

## Install steps

1) Ensure NVIDIA driver + CUDA toolkit are installed for your GPU.
2) Activate Pipenv:
   - pipenv shell
3) Install Python deps:
   - pipenv install --dev
4) Install CUDA-enabled PyTorch (per project convention):
   - pipenv run pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
5) Install PyCUDA (required by TensorRT engine loader):
   - pipenv run pip install pycuda

## Quick GPU check

- Validate TensorRT + CUDA before running the API:
  - pipenv run python scripts/check_tensorrt.py --config configs/tensorrt.json

If the check passes, start the API:
- pipenv run python -m src.main --config configs/tensorrt.json
