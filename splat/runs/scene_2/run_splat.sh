conda activate drawer_splat

root_dir="/home/hongchix/data/"
data_name="uw_kitchen_3"
data_dir=${root_dir}/${data_name}

sdf_dir="./outputs/uw_kitchen_3/uw_kitchen_3_sdf_recon/bakedsdf/2025-02-27_185326"


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
    --downscale_factor 2 \
    --num_max_image 2000 # only use if memory is not enough
