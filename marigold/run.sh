conda activate drawer_sdf
data_dir="/home/hongchix/data/cs_kitchen_n"
image_dir="images_2"

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