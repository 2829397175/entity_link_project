export PYTHONPATH='.'
CUDA_VISIBLE_DEVICES=0,1 python muver/multi_view/train.py \
    --pretrained_model /data/jiarui_ji/entity_link/MuVER/bert-base-chinese \
    --dataset_path /data/jiarui_ji/entity_link/entity_linking_project \
    --bi_ckpt_path runtime_log/distributed_multi_view/2024-04-25-20-18-04/epoch_8.bin \
    --max_cand_len 40 \
    --max_seq_len 128 \
    --do_test  \
    --test_mode test \
    --data_parallel \
    --eval_batch_size 128  \
    --accumulate_score \
    # --view_expansion  \
    # --merge_layers 4  \
    # --top_k 0.4