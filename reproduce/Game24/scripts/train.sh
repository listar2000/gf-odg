python3 main.py \
	--seed 12 \
	--pretrained_model meta-llama/Meta-Llama-3-8B \
    --length 20 \
	--max-length 20 \
	--num-iters 2000 \
	--start 0 \
	--end 50 \
	--lr-nll-portion 1.0 \
    --topk 10 \
    --output-lgt-temp 1 \
	--verbose \
    --straight-through  \
	--buffer_size 50 \
	--batch-size 4 \
	--epoch_nums 1 \
	--train-data data/train.json\
	--do_train \
	--do_test \
	--test_sample_nums 20 \
	--cache_dir "/net/scratch/llama3" \
	--fp16 > "logs/train.txt" 