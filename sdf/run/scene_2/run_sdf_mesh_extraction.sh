conda activate drawer_sdf

root_dir="/home/hongchix/data/"
data_name="uw_kitchen_3"
data_dir=${root_dir}/${data_name}

output_dir="./outputs/uw_kitchen_3/uw_kitchen_3_sdf_recon/bakedsdf/2025-02-27_185326"

# extract mesh from mesh
python scripts/extract_mesh.py --load-config ${output_dir}/config.yml \
   --output-path ${output_dir}/mesh.ply \
   --bounding-box-min -2.0 -2.0 -2.0 --bounding-box-max 2.0 2.0 2.0 \
   --resolution 2048 --marching_cube_threshold 0.0035 --create_visibility_mask True --simplify-mesh True

# extract texture for mesh
mkdir -p ${output_dir}/texture_mesh
python scripts/texture.py --load-config ${output_dir}/config.yml \
   --output-dir ${output_dir}/texture_mesh \
   --input_mesh_filename ${output_dir}/mesh-simplify.ply \
   --target_num_faces 300000

# save pose for mesh
python scripts/save_pose.py \
    --ckpt_dir ${output_dir} \
    --save_dir ${data_dir}