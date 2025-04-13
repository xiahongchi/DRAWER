conda create --name drawer_sdf -y python=3.8
conda activate drawer_sdf
# ensure gcc version 11.x and nvcc version 11.8
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

pip install -e .
pip install functorch --no-deps
pip install torchmetrics[image]
pip install torchtyping

pip install accelerate==0.27.2
pip install diffusers==0.30.2
pip install tokenizers==0.15.2
pip install transformers==4.37.2
pip install xformers==0.0.23 --index-url https://download.pytorch.org/whl/cu118 --no-deps
pip install omegaconf
pip install tabulate
pip install pandas

pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.2+cu118.html

pip install "typeguard==2.12.1"
pip install --upgrade tyro

pip install git+https://github.com/NVlabs/nvdiffrast.git

pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

pip install scikit-learn
pip install imageio[ffmpeg]

pip install hydra-core --upgrade --pre
pip install hydra-submitit-launcher --upgrade
pip install visdom

pip install kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.1_cu118.html

pip install transformations