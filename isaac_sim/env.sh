conda create -n isaacsim -y python=3.10
conda activate isaacsim

pip install isaacsim[all]==4.5.0 --extra-index-url https://pypi.nvidia.com
pip install isaacsim[extscache]==4.5.0 --extra-index-url https://pypi.nvidia.com

pip install trimesh
pip install transformations
pip install requests

git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh -i
cd ..
