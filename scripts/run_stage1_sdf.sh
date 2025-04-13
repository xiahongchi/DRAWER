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