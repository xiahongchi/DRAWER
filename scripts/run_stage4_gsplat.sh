data_name="cs_kitchen"
data_dir=/home/hongchix/data/${data_name}
image_dir="images_2"
downscale_factor=2

sdf_dir=outputs/${data_name}/${data_name}_sdf_recon


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

python scripts/splat_export.py \
    --splat_dir ${splat_dir} \
    --save_note "default"