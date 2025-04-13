conda activate isaacsim

# root_dir="/home/hongchix/data/"
# data_name="cs_kitchen_n"
# data_dir=${root_dir}/${data_name}
# image_dir="images_2"
# sdf_dir="./outputs/cs_kitchen_n/cs_kitchen_n_sdf_recon/bakedsdf/2025-02-23_232628"

root_dir="/home/hongchix/data/"
data_name="uw_kitchen_3"
data_dir=${root_dir}/${data_name}
image_dir="images_2"
sdf_dir="./outputs/uw_kitchen_3/uw_kitchen_3_sdf_recon/bakedsdf/2025-02-27_185326"


python compose_usd.py \
    --sdf_dir ../sdf/${sdf_dir}

python run_simulation.py \
    --sdf_dir ../sdf/${sdf_dir}