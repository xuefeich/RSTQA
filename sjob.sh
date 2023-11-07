#!/bin/bash
#SBATCH --job-name              ccir
#SBATCH --time                  48:00:00
#SBATCH --cpus-per-task         4
#SBATCH --gres                  gpu:04
#SBATCH --mem                   90G
#SBATCH --output                output.txt
#SBATCH --partition             v100_batch
source ~/.bashrc
source activate alan
#PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/tag_op python tag_op/prepare_dataset.py --mode train --num_arithmetic_operators 6
#export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:3072'
#PYTHONPATH=$PYTHONPATH:$(pwd) deepspeed --num_gpus=4 --no_local_rank tag_op/train_dp.py --data_dir tag_op/cache/ --save_dir ./checkpoint --batch_size 48 --eval_batch_size 8 --max_epoch 50 --warmup 0.06 --optimizer adam --learning_rate 5e-4 --weight_decay 5e-5 --seed 123 --gradient_accumulation_steps 12 --bert_learning_rate 1.5e-5 --bert_weight_decay 0.01 --log_per_updates 50 --eps 1e-6 --encoder roberta --num_ops 6 --gpu_num 4 --deepspeed_config dp_stage3.json


CUDA_VISIBLE_DEVICES=2,3,4,5 PYTHONPATH=$PYTHONPATH:$(pwd) nohup python tag_op/trainer.py --data_dir tag_op/cache/ --save_dir ./checkpoint --batch_size 32 --eval_batch_size 8 --max_epoch 100 --warmup 0.06 --optimizer adam --learning_rate 5e-4 --weight_decay 5e-5 --seed 123 --gradient_accumulation_steps 8 --bert_learning_rate 1.5e-5 --bert_weight_decay 0.01 --log_per_updates 50 --eps 1e-6 --encoder deberta --plm_path ./model --num_ops 6 &
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=$PYTHONPATH:$(pwd) python tag_op/predictor.py --data_dir tag_op/cache/ --test_data_dir tag_op/cache/ --save_dir tag_op/ --eval_batch_size 8 --model_path ./checkpoint --encoder deberta --set test --plm_path ./model

PYTHONPATH=$PYTHONPATH:$(pwd) python tag_op/trainer.py --data_dir tag_op/cache/ --save_dir ./checkpoint --batch_size 48 --eval_batch_size 8 --max_epoch 50 --warmup 0.06 --optimizer adam --learning_rate 5e-4 --weight_decay 5e-5 --seed 123 --gradient_accumulation_steps 8 --bert_learning_rate 1.5e-5 --bert_weight_decay 0.01 --log_per_updates 50 --eps 1e-6 --encoder roberta --num_ops 6 --roberta_model ./model/roberta.large
#PYTHONPATH=$PYTHONPATH:$(pwd) python tag_op/predictor.py --data_dir tag_op/cache/ --test_data_dir tag_op/cache/ --save_dir tag_op/ --eval_batch_size 8 --model_path ./checkpoint --encoder roberta --num_ops 6 --set dev --roberta_model ./model/roberta.large
