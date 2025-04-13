conda activate drawer_sdf

# install if not
pip install --no-build-isolation -e GroundingDINO
pip install diffusers[torch]

# download if not
# Swin-B (slow)
wget https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth
# Swin-T (fast)
wget https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth
# SAM
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# run
export CUDA_VISIBLE_DEVICES=0

data_dir="/home/hongchix/data/uw_kitchen_3/"
image_dir="images_2"

# Swin-B (slow)
python grounded_sam_detect_doors.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py \
  --grounded_checkpoint groundingdino_swinb_cogcoor.pth \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --input_dir ${data_dir}/${image_dir} \
  --output_dir ${data_dir}/grounded_sam \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --text_prompt "drawer. drawer door. cabinet door. drawer face. drawer front. fridge door. fridge front. refridgerator door. refridgerator front." \
  --device "cuda"

# Swin-T (fast)
python grounded_sam_detect_doors.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --input_dir ${data_dir}/${image_dir} \
  --output_dir ${data_dir}/grounded_sam \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --text_prompt "drawer. drawer door. cabinet door. drawer face. drawer front. fridge door. fridge front. refridgerator door. refridgerator front." \
  --device "cuda"