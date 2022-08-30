~/.conda/envs/paddle/bin/python -u -m paddle.distributed.launch --gpus "0,1" finetune.py > nohup.out 2>&1 &
