python evaluation_robustness.py \
    --model_path /hdd1/jianwei/workspace/SparseOptimizer/prune+kd/sst2/research/adt/13-5 \
    --model_name bert-base-uncased \
    --dataset_name glue \
    --task_name sst2 \
    --num_examples 872 \
    --num_labels 2 \
    --result_file ./tmp_result.csv