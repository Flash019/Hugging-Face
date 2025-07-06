# install ----> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.get_device_name(0))  # Your GPU name
print(torch.__version__)


# Upgrade Pytorch -----> pip install torch==2.6.0+cu121 torchvision==0.19.0+cu121 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu121
