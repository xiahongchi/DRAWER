export CUDA_VISIBLE_DEVICES=0
python grounded_sam_demo_handle.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py \
  --grounded_checkpoint groundingdino_swinb_cogcoor.pth \
  --sam_checkpoint /home/hongchix/codes/SoM/sam_vit_h_4b8939.pth \
  --input_dir /home/hongchix/main/data/cs_kitchen_n/vis_groups_back_match \
  --output_dir /home/hongchix/main/data/cs_kitchen_n/vis_groups_back_match_gsam_handle \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --text_prompt "handle" \
  --device "cuda" 
