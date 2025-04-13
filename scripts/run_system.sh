data_name="cs_kitchen"
data_dir=/home/hongchix/data/${data_name}
image_dir="images_2"
downscale_factor=2

# monocular depth and normal

cd marigold

conda activate drawer_sdf

python run.py \
    --checkpoint "GonzaloMG/marigold-e2e-ft-depth" \
    --modality depth \
    --input_rgb_dir ${data_dir}/${image_dir} \
    --output_dir ${data_dir}/marigold_ft

python run.py \
    --checkpoint "GonzaloMG/marigold-e2e-ft-normals" \
    --modality normals \
    --input_rgb_dir ${data_dir}/${image_dir} \
    --output_dir ${data_dir}/marigold_ft

python read_marigold.py --data_dir ${data_dir}/marigold_ft
ln -s ${data_dir}/marigold_ft/depth ${data_dir}/depth
ln -s ${data_dir}/marigold_ft/normal ${data_dir}/normal

# sdf reconstruction
cd ../sdf

python scripts/train.py bakedsdf --vis wandb \
    --output-dir outputs/${data_name} --experiment-name ${data_name}_sdf_recon \
    --trainer.steps-per-eval-image 2000 --trainer.steps-per-eval-all-images 250001 \
    --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 250001 \
    --optimizers.fields.scheduler.max-steps 250000 \
    --optimizers.field-background.scheduler.max-steps 250000 \
    --optimizers.proposal-networks.scheduler.max-steps 250000 \
    --pipeline.model.eikonal-anneal-max-num-iters 250000 \
    --pipeline.model.beta-anneal-max-num-iters 250000 \
    --pipeline.model.sdf-field.bias 1.5 --pipeline.model.sdf-field.inside-outside True \
    --pipeline.model.eikonal-loss-mult 0.01 --pipeline.model.num-neus-samples-per-ray 24 \
    --pipeline.datamanager.train-num-rays-per-batch 4096 \
    --machine.num-gpus 1 --pipeline.model.scene-contraction-norm inf \
    --pipeline.model.mono-normal-loss-mult 0.2 \
    --pipeline.model.mono-depth-loss-mult 1.0 \
    --pipeline.model.near-plane 1e-6 \
    --pipeline.model.far-plane 100 \
    panoptic-data \
    --data ${data_dir} \
    --panoptic_data False \
    --mono_normal_data True \
    --mono_depth_data True \
    --panoptic_segment False \
    --downscale_factor ${downscale_factor} \
    --num_max_image 2000 # only use if memory is not enough

sdf_dir=outputs/${data_name}/${data_name}_sdf_recon

# extract mesh from mesh
python scripts/extract_mesh.py --load-config ${sdf_dir}/config.yml \
   --output-path ${sdf_dir}/mesh.ply \
   --bounding-box-min -2.0 -2.0 -2.0 --bounding-box-max 2.0 2.0 2.0 \
   --resolution 2048 --marching_cube_threshold 0.0035 --create_visibility_mask True --simplify-mesh True

# extract texture for mesh
mkdir -p ${sdf_dir}/texture_mesh
python scripts/texture.py --load-config ${sdf_dir}/config.yml \
   --output-dir ${sdf_dir}/texture_mesh \
   --input_mesh_filename ${sdf_dir}/mesh-simplify.ply \
   --target_num_faces 300000

# save pose for mesh
python scripts/save_pose.py \
    --ckpt_dir ${sdf_dir} \
    --save_dir ${data_dir}

# grounded sam
cd ../grounded_sam
export CUDA_VISIBLE_DEVICES=0
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

# perception
cd ../perception

python percept_stage1.py \
    --data_dir ${data_dir} \
    --image_dir ${data_dir}/${image_dir} \
    --sdf_dir ../sdf/${sdf_dir} \
    --num_max_frames 1500 \
    --num_faces_simplified 80000

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

# fit doors
cd ../sdf


python scripts/sam_project.py \
    --data_dir ${data_dir} \
    --image_dir ${data_dir}/${image_dir} \
    --sdf_dir ../sdf/${sdf_dir}


cd ../sam

python propagate.py \
    --sam_ckpt ../grounded_sam/sam_vit_h_4b8939.pth \
    --data_dir ${data_dir} \
    --image_dir ${data_dir}/${image_dir} \
    --sdf_dir ../sdf/${sdf_dir}

cd ../3DOI


python art_infer.py \
    --config-name sam_inference \
    checkpoint_path=checkpoints/checkpoint_20230515.pth \
    output_dir=${data_dir}/art_infer \
    data_dir=${data_dir} \
    image_dir=${data_dir}/${image_dir}

cd ../sdf

python scripts/fit_doors.py \
    --data_dir ${data_dir} \
    --image_dir ${data_dir}/${image_dir} \
    --sdf_dir ../sdf/${sdf_dir}

conda deactivate
conda activate isaacsim

cd ../isaac_sim

python compose_usd.py \
    --sdf_dir ../sdf/${sdf_dir}

python run_simulation.py \
    --sdf_dir ../sdf/${sdf_dir}

cd ../splat

conda deactivate
conda activate drawer_splat
cd ../splat

mesh_path=../sdf/${sdf_dir}/texture_mesh/mesh-simplify.obj
save_extra_info_dir=./vis/${data_name}/gs_extra_info
run_note="dense_1"
area=2e-5 # increase this value if the splat is too dense

python nerfstudio/scripts/train.py splatfacto_on_mesh_uc \
    --vis wandb  \
    --output-dir outputs/${data_name} \
    --experiment-name ${data_name}_mesh_gauss_splat \
    --pipeline.model.mesh_area_to_subdivide ${area} \
    --pipeline.model.acm_lambda 1.0 \
    --pipeline.model.elevate_coef 2.0 \
    --pipeline.model.upper_scale 2.0 \
    --pipeline.model.continue_cull_post_densification True \
    --pipeline.model.gaussian_save_extra_info_path ${save_extra_info_dir}/${run_note}.pt \
    --pipeline.model.mesh_depth_lambda 1.0 \
    --pipeline.model.reset_alpha_every 30 \
    --pipeline.model.use_scale_regularization True \
    --pipeline.model.max_gauss_ratio 1.5 \
    --max-num-iterations 30000 \
    panoptic-data  \
    --data ${data_dir} \
    --mesh_gauss_path ${mesh_path} \
    --mesh_area_to_subdivide ${area} \
    --mesh_depth True \
    --downscale_factor ${downscale_factor} \
    --num_max_image 2000 

splat_dir=outputs/${data_name}/${data_name}_mesh_gauss_splat


python scripts/matfuse_texgen.py \
    --splat_dir ${splat_dir} \
    --output_dir vis/${data_name}/interior \
    --sdf_dir ../sdf/${sdf_dir} \
    --ckpt ckpts/matfuse-full.ckpt \
    --config scripts/matfuse_sd/src/configs/diffusion/matfuse-ldm-vq_f8.yaml

python scripts/paint_ao.py \
    --src_dir vis/${data_name}/interior

python scripts/splat_merge.py \
    --splat_dir ${splat_dir} \
    --sdf_dir ../sdf/${sdf_dir} \
    --interior_dir vis/${data_name}/interior \
    --save_note "default"

python scripts/splat_merge_val.py \
    --splat_dir ${splat_dir} \
    --sdf_dir ../sdf/${sdf_dir} \
    --interior_dir vis/${data_name}/interior \
    --save_note "default" \
    --save_dir vis/${data_name}/val

python scripts/splat_merge_objaverse.py \
    --splat_dir ${splat_dir} \
    --sdf_dir ../sdf/${sdf_dir} \
    --interior_dir vis/${data_name}/interior \
    --save_note "default" \
    --save_dir vis/${data_name}/objaverse