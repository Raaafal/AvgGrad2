Prerequisites
-------------
* Python version: 3.8.2 was used for this training configuration.


Installation Instructions
-------------------------

1. Install PyTorch (NVIDIA GPU / CUDA users only).
If you have an NVIDIA GPU, first install the versions of torch, torchvision, and torchtext that support CUDA 11.8:
```
    python -m pip install --no-cache-dir --force-reinstall torch==2.1.0 torchvision==0.16.0 torchtext==0.16.0 --index-url https://download.pytorch.org/whl/cu118
```
2. Install Remaining Dependencies. Choose ONE of the following commands based on your hardware:

* If you do NOT have an NVIDIA GPU:
```
    python -m pip install -r requirements_all.txt
```
* If you DO have an NVIDIA GPU (and already ran the CUDA installation above):
```
    python -m pip install -r requirements_without_torch_torchvision_torchtext.txt
```

Execution & Configuration
-------------------------
The primary entry point for the project is main.py. Once configured, execute the script using:
```
    python main.py
```
⚠️ Important Note on Reproducibility: To ensure exact experimental reproducibility for research purposes, this project uses manual source-code configuration. You must set your experiment parameters directly inside the Python files before running the script.


Step 1: Configure main.py.
Open main.py in your text editor and modify the following variables to set up your specific run:

* Model Selection (model_nr):
  - 6 = Model A (Supports: MNIST, Fashion MNIST)
  - 9 = Model B (Supports: MNIST, Fashion MNIST)
  - 12 = Model C (Supports: IMDb)
  - 13 = ResNet-152 (Supports: Imagenet-OOD)
  - 14 = ResNet-152-GELU (Supports: Imagenet-OOD)
  - 15 = ResNet-152-SiLU (Supports: Imagenet-OOD)
  - 18 = ResNet-152-Sig (Supports: Imagenet-OOD)

* Dataset Selection (dataset):
  - Set to 'imagenet_ood', 'mnist', 'fashion_mnist', or 'imdb'

* Gradient Method (method):
  - 20 = Standard gradient
  - 21 = AG-1 variant
  - 27 = AG-2 variant
  - 33 = AG-3-1 variant
  - 37 = AG-3 variant

* Optimizer Settings:
  - Set optimizer_type to 'SOAP', 'optim.RMSprop', or 'optim.Adam'
  - (Note: Modify the optim_args variable to adjust specific optimizer hyperparameters).

* Training Count:
  - Set trainings_same_model (e.g., 100) to define the total training count. 
  - (Use the trainings_different_model variable for hyperparameter searching).


Step 2: Configure algorithms.py (Linear Variants Only).
If you specifically want to run the variants marked 'linear', you must also edit the algorithms file:
1. Open algorithms.py.
2. Locate the variable average_gradient_of_linear_layers_enhancement.
3. Set it to True.
