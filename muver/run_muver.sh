export PYTHONPATH='.'
CUDA_VISIBLE_DEVICES=0 python muver/multi_view/train.py \
    --pretrained_model /data/jiarui_ji/entity_link/MuVER/bert-base-chinese \
    --dataset_path /data/jiarui_ji/entity_link/entity_linking_project \
    --epoch 30 \
    --train_batch_size 128 \
    --learning_rate 1e-5 \
    --do_train --do_eval \
    --name distributed_multi_view \
    --data_parallel 
    