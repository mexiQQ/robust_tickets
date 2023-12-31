#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

log_file='draw-retrain_ticket.log'
result_file='retrain_ticket.csv'  # attack results
ckpt='./save_models/debug/'  # draw and retrain tickets saving path
lr=2e-5
epoch=10  # larger epoch is better

# # IMDB
# ticket_path='./your_search-ticket_path' # search ticket path
# for epoch in 19 17 15 13
# do
#   for sparsity in 0.2 0.3 0.4
#   do
#     masked_model_path=${model_path}${epoch} # retrain on tickets with different searching epochs
#     python draw_retrain_ticket.py \
#     --masked_model_path ${ticket_path} \
#     --model_name bert-base-uncased \
#     --lr $lr \
#     --max_seq_length 256 \
#     --dataset_name imdb \
#     --ckpt_dir ${ckpt} \
#     --result_file ${result_file} \
#     --sparsity $sparsity \
#     --num_examples 100 \
#     --bsz 32 \
#     --num_labels 2 \
#     --epochs $epoch >> ${log_file}
# done
# done

# # AGNEWS
# ticket_path='./your_search-ticket_path' # search ticket path
# for epoch in 19 17 15 13
# do
#   for sparsity in 0.2 0.3 0.4
#   do
#     masked_model_path=${model_path}${epoch} # retrain on tickets with different searching epochs
#     python draw_retrain_ticket.py \
#     --masked_model_path ${ticket_path} \
#     --model_name bert-base-uncased \
#     --lr $lr \
#     --max_seq_length 256 \
#     --dataset_name ag_news \
#     --ckpt_dir ${ckpt} \
#     --result_file ${result_file} \
#     --sparsity $sparsity \
#     --num_examples 200 \
#     --bsz 32 \
#     --num_labels 4 \
#     --epochs $epoch >> ${log_file}
# done
# done

# SST-2
ticket_path='/hdd1/jianwei/workspace/robust_ticket/save_models/search_ticket/search-robust-ticket_bert_glue-sst2_lr0.1_lambda0.5_adv-lr0.03_adv-step5_epochs20' # search ticket path
for epoch in 10
do
  for sparsity in 0.4
  do
    masked_model_path=${model_path}${epoch} # retrain on tickets with different searching epochs
    python draw_retrain_ticket_debug.py \
    --masked_model_path ${ticket_path} \
    --model_name bert-base-uncased \
    --lr $lr \
    --max_seq_length 128 \
    --dataset_name glue \
    --task_name sst2 \
    --ckpt_dir ${ckpt} \
    --result_file ${result_file} \
    --sparsity $sparsity \
    --num_examples 2 \
    --bsz 32 \
    --num_labels 2 \
    --epochs $epoch >> ${log_file}
  done
done
