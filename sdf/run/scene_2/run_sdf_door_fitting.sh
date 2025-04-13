conda activate drawer_sdf

root_dir="/home/hongchix/data/"
data_name="uw_kitchen_3"
data_dir=${root_dir}/${data_name}
image_dir="images_2"
sdf_dir="./outputs/uw_kitchen_3/uw_kitchen_3_sdf_recon/bakedsdf/2025-02-27_185326"

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

# download ckpt if not exist
mkdir checkpoints
wget https://fouheylab.eecs.umich.edu/~syqian/3DOI/checkpoint_20230515.pth -P checkpoints --no-check-certificate

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