import time

out_dir = 'out-openwebtext'
eval_interval = 5
eval_iters = 40
wandb_log = True # feel free to turn on
wandb_project = 'remrofsnart'
wandb_run_name = 'gpu-ft-openwebtext-' + str(time.time())

dataset = 'openwebtext'
init_from = 'gpt2' 

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 55246589 tokens, so 1 epoch ~= 169 iters
batch_size = 16
gradient_accumulation_steps = 4
max_iters = 400


learning_rate = 3e-4
min_lr = 1e-5
decay_lr = True
warmup_iters = 20
lr_decay_iters = max_iters