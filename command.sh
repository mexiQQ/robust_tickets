textattack attack --model /hdd1/jianwei/workspace/robust_ticket/save_models/search_ticket/search-robust-ticket_bert_glue-sst2_lr0.1_lambda0.5_adv-lr0.03_adv-step5_epochs20/epoch19 --recipe textfooler --dataset-from-huggingface sst2 --dataset-split validation --num-examples -1 --random-seed 42 --parallel

# CUDA_VISIBLE_DEVICES=0 textattack eval --model /hdd1/jianwei/workspace/robust_ticket/save_models/fine-tune/finetune_bert-base-uncased_glue-sst2_lr2e-05_epochs25_seed42/epoch24 --dataset-from-huggingface sst2 --dataset-split validation --num-examples -1

# textattack attack --model /hdd1/jianwei/workspace/robust_ticket/save_models/fine-tune/finetune_bert-base-uncased_ag_news_lr2e-05_epochs20_seed42/epoch19 --recipe textfooler --dataset-from-huggingface ag_news --dataset-split test --num-examples -1 --random-seed 42 --parallel