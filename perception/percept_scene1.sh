conda activate drawer_sdf

OPENAI_KEY="your key here"

root_dir="/home/hongchix/data/"
data_name="cs_kitchen_n"
data_dir=${root_dir}/${data_name}
image_dir="images_2"
sdf_dir="./outputs/cs_kitchen_n/cs_kitchen_n_sdf_recon/bakedsdf/2025-02-23_232628"

python percept_stage1.py \
    --data_dir ${data_dir} \
    --image_dir ${data_dir}/${image_dir} \
    --sdf_dir ../sdf/${sdf_dir} \
    --num_max_frames 1000

python percept_stage2.py \
    --data_dir ${data_dir}

python percept_stage3.py \
    --data_dir ${data_dir} \
    --image_dir ${data_dir}/${image_dir}

rm -rf ${data_dir}/perception/vis_groups_back_match_gsam_handle/*

cd ../grounded_sam

export CUDA_VISIBLE_DEVICES=0
python grounded_sam_detect_handles.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --input_dir ${data_dir}/perception/vis_groups_back_match \
  --output_dir ${data_dir}/perception/vis_groups_back_match_gsam_handle \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --text_prompt "handle" \
  --device "cuda" 

cd ../perception

python percept_stage4.py \
    --data_dir ${data_dir}

python percept_stage5.py \
    --api_key ${OPENAI_KEY} \
    --data_dir ${data_dir}

python percept_stage6.py \
    --data_dir ${data_dir} \
    --sdf_dir ../sdf/${sdf_dir} 

# It is suggested to check the results in ${data_dir}/perception/vis_groups_final_mesh/
# and then remove some failure cases or add some missing cases

cp ${data_dir}/perception/vis_groups_final_mesh/all.json ${data_dir}/grounded_sam/