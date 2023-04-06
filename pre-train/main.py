import run_multitask_unified_pretraining
import sys
import os
import torch
from argparse import Namespace

model_name = sys.argv[1]
is_gpu = len(sys.argv) >= 3 and sys.argv[2].lower() != 'cpu'
should_continue = len(sys.argv) >= 4 and sys.argv[3].lower() == 'continue'
is_concat = False
print('is_gpu', is_gpu)
print('should_continue', should_continue)
print('is_concat', is_concat)

if is_concat:
  dataset_dir = '../../datasets/amrbart-concat'
else:
  dataset_dir = '../../datasets/amrbart-new'

output_dir = f'../outputs/{model_name}'
os.makedirs(output_dir, exist_ok=True)

# torch.set_num_threads(1)

batch_size = 4

args = Namespace(
  adam_epsilon=1e-08,
  block_size=512,
  cache_dir=None,
  config_name=None,
  device='gpu' if is_gpu else 'cpu',
  do_eval=True,
  do_train=True,
  eval_all_checkpoints=False,
  evaluate_during_training=True,
  fp16=False, #! causing nan losses
  fp16_opt_level='O1',
  freeze_decoder=False,
  freeze_embeds=False,
  freeze_encoder=False,
  gradient_accumulation_steps=batch_size,
  is_concat=is_concat,
  joint_train_interval=1,
  learning_rate=5e-07,
  line_by_line=False,
  local_rank=-1,
  logging_steps=2000,
  max_grad_norm=1.0,
  max_steps=0,
  mlm=True,
  mlm_amr=True,
  mlm_amr_plus_text=True,
  mlm_amr_plus_text_short=False,
  mlm_amr_short=False,
  mlm_joint_to_amr=True,
  mlm_joint_to_joint=False,
  mlm_joint_to_text=True,
  mlm_probability=0.15,
  mlm_text=True,
  mlm_text_plus_amr=True,
  mlm_text_plus_amr_short=False,
  mlm_text_short=False,
  model_name_or_path=f'../models/{model_name}',
  model_type='facebook/mbart-large-50',
  no_cuda=False,
  num_train_epochs=1,
  output_dir=output_dir,
  overwrite_cache=False,
  overwrite_output_dir=False,
  per_gpu_eval_batch_size=batch_size,
  per_gpu_train_batch_size=batch_size,
  save_steps=500,
  save_total_limit=2,
  seed=42,
  server_ip='',
  server_port='',
  should_continue=should_continue,
  smart_init=False,
  test_file=f'{dataset_dir}/test.jsonl',
  tokenizer_name=None,
  train_file=f'{dataset_dir}/pretrain.jsonl',
  val_file=f'{dataset_dir}/dev.jsonl',
  warmup_steps=2500,
  weight_decay=0.0
)

run_multitask_unified_pretraining.main(args)
