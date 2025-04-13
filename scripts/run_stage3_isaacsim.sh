data_name="cs_kitchen"
data_dir=/home/hongchix/data/${data_name}
image_dir="images_2"
downscale_factor=2

sdf_dir=outputs/${data_name}/${data_name}_sdf_recon

conda deactivate
conda activate isaacsim

cd ../isaac_sim

python compose_usd.py \
    --sdf_dir ../sdf/${sdf_dir}

python run_simulation.py \
    --sdf_dir ../sdf/${sdf_dir}
