conda activate drawer_splat

root_dir="/home/hongchix/data/"
data_name="cs_kitchen_n"
data_dir=${root_dir}/${data_name}
image_dir="images_2"
sdf_dir="./outputs/cs_kitchen_n/cs_kitchen_n_sdf_recon/bakedsdf/2025-02-23_232628"


python align_drawers.py \
    --sdf_dir ../sdf/${sdf_dir} \
    --interior_dir ../splat/vis/${data_name}/interior \
    --save_dir ../splat/vis/${data_name}/interior_local

texture_mesh_dir=../sdf/${sdf_dir}/texture_mesh/
drawer_result_dir=../sdf/${sdf_dir}/drawers/results/
drawer_tex_dir=../splat/vis/${data_name}/interior_local/

usd_dir=./outputs/${data_name}
mkdir -p ${usd_dir}

cp -r ${texture_mesh_dir} ${usd_dir}
cp -r ${drawer_result_dir} ${usd_dir}/drawers
cp -r ${drawer_tex_dir} ${usd_dir}/texture_doors

conda deactivate
conda activate isaacsim
python compose_usd_texture.py \
    --usd_dir ${usd_dir}
