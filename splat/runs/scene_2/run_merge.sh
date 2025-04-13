conda activate drawer_splat

root_dir="/home/hongchix/data/"
data_name="uw_kitchen_3"
data_dir=${root_dir}/${data_name}
sdf_dir="./outputs/uw_kitchen_3/uw_kitchen_3_sdf_recon/bakedsdf/2025-02-27_185326"

splat_dir=outputs/uw_kitchen_3/uw_kitchen_3_mesh_gauss_splat/splatfacto_on_mesh_uc/2025-02-28_232922

# download ckpt if not exist
mkdir -p ckpts
wget https://huggingface.co/gvecchio/MatFuse/resolve/main/matfuse-full.ckpt -P ckpts

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