conda activate drawer_sdf

# download ckpt if not exist
mkdir checkpoints
wget https://fouheylab.eecs.umich.edu/~syqian/3DOI/checkpoint_20230515.pth -P checkpoints --no-check-certificate

# run art_infer.py
root_dir="/home/hongchix/data/"
data_name="cs_kitchen_n"
data_dir=${root_dir}/${data_name}
image_dir="images_2"

python art_infer.py \
    --config-name sam_inference \
    checkpoint_path=checkpoints/checkpoint_20230515.pth \
    output_dir=${data_dir}/art_infer \
    data_dir=${data_dir} \
    image_dir=${data_dir}/${image_dir}