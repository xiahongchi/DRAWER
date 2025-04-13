conda activate drawer_sdf

root_dir="/home/hongchix/data/"
data_name="cs_kitchen_n"
data_dir=${root_dir}/${data_name}

python scripts/train.py bakedsdf --vis wandb \
    --output-dir outputs/${data_name} --experiment-name ${data_name}_sdf_recon \
    --trainer.steps-per-eval-image 2000 --trainer.steps-per-eval-all-images 250001 \
    --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 250001 \
    --optimizers.fields.scheduler.max-steps 250000 \
    --optimizers.field-background.scheduler.max-steps 250000 \
    --optimizers.proposal-networks.scheduler.max-steps 250000 \
    --pipeline.model.eikonal-anneal-max-num-iters 250000 \
    --pipeline.model.beta-anneal-max-num-iters 250000 \
    --pipeline.model.sdf-field.bias 1.5 --pipeline.model.sdf-field.inside-outside True \
    --pipeline.model.eikonal-loss-mult 0.01 --pipeline.model.num-neus-samples-per-ray 24 \
    --pipeline.datamanager.train-num-rays-per-batch 4096 \
    --machine.num-gpus 1 --pipeline.model.scene-contraction-norm inf \
    --pipeline.model.mono-normal-loss-mult 0.5 \
    --pipeline.model.mono-depth-loss-mult 2.0 \
    --pipeline.model.near-plane 1e-6 \
    --pipeline.model.far-plane 100 \
    panoptic-data \
    --data ${data_dir} \
    --panoptic_data False \
    --mono_normal_data True \
    --mono_depth_data True \
    --panoptic_segment False \
    --downscale_factor 2 \
    --num_max_image 1000 # only use if memory is not enough