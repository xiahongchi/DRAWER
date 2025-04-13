conda activate drawer_splat

root_dir="/home/hongchix/data/"
data_name="cs_kitchen_n"
data_dir=${root_dir}/${data_name}
sdf_dir="./outputs/cs_kitchen_n/cs_kitchen_n_sdf_recon/bakedsdf/2025-02-23_232628"

splat_dir=outputs/cs_kitchen_n/cs_kitchen_n_mesh_gauss_splat/splatfacto_on_mesh_uc/2025-02-26_160328

# download ckpt if not exist
mkdir -p ckpts
wget https://huggingface.co/gvecchio/MatFuse/resolve/main/matfuse-full.ckpt -P ckpts

python scripts/matfuse_texgen.py \
    --splat_dir ${splat_dir} \
    --output_dir vis/${data_name}/interior_v2 \
    --sdf_dir ../sdf/${sdf_dir} \
    --ckpt ckpts/matfuse-full.ckpt \
    --config scripts/matfuse_sd/src/configs/diffusion/matfuse-ldm-vq_f8.yaml

python scripts/paint_ao.py \
    --src_dir vis/${data_name}/interior_v2

python scripts/splat_merge.py \
    --splat_dir ${splat_dir} \
    --sdf_dir ../sdf/${sdf_dir} \
    --interior_dir vis/${data_name}/interior_v2 \
    --save_note "default"