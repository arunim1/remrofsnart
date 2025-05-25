import time

out_dir = 'out-rev-openwebtext'
eval_interval = 5
eval_iters = 40
wandb_log = True # feel free to turn on
wandb_project = 'remrofsnart'
wandb_run_name = 'gpu-rev-ft-openwebtext-' + str(time.time())

dataset = 'openwebtext'
init_from = 'gpt2' # this is the largest GPT-2 model

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 4
gradient_accumulation_steps = 8
max_iters = 200


learning_rate = 3e-4
min_lr = 1e-5
decay_lr = True
warmup_iters = 20
lr_decay_iters = max_iters