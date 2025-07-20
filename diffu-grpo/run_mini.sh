#!/bin/bash
export WANDB_MODE=disabled
export LOGDIR=checkpoints
mkdir -p $LOGDIR

DATASET="cd4"   
RUN_NAME=${DATASET}_base_`date "+%Y%m%d-%H%M%S"`
MODEL_PATH=model_config #diffusionfamily/diffugpt-s
NUM_ITER=12 # number of policy gradient inner updates iterations

accelerate launch \
    --config_file accelerate.yaml \
    --main_process_port 12346 diffu_grpo_train_mini.py \
    --config slurm_scripts/train_mini.yaml \
    --model_path $MODEL_PATH \
    --num_iterations $NUM_ITER \
    --dataset $DATASET \
    --run_name $RUN_NAME \
    --output_dir checkpoints/$RUN_NAME 
