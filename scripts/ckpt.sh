cd 3DOI
mkdir checkpoints
wget https://fouheylab.eecs.umich.edu/~syqian/3DOI/checkpoint_20230515.pth -P checkpoints --no-check-certificate
cd ..

cd grounded_sam
# download if not
# Swin-B (slow)
wget https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth
# Swin-T (fast)
wget https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth
# SAM
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

cd ..

