conda activate drawer_sdf

root_dir="/home/hongchix/data/"
data_name="cs_kitchen_n"
data_dir=${root_dir}/${data_name}
image_dir="images_2"
sdf_dir="./outputs/cs_kitchen_n/cs_kitchen_n_sdf_recon/bakedsdf/2025-02-23_232628"

python propagate.py \
    --sam_ckpt ../grounded_sam/sam_vit_h_4b8939.pth \
    --data_dir ${data_dir} \
    --image_dir ${data_dir}/${image_dir} \
    --sdf_dir ../sdf/${sdf_dir}