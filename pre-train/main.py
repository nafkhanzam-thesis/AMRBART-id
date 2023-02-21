import run_multitask_unified_pretraining
import sys
import os
from argparse import Namespace

model_name = sys.argv[1]

output_dir = f'../outputs/{model_name}'
os.makedirs(output_dir, exist_ok=True)

args = Namespace(
  adam_epsilon=1e-08,
  block_size=512,
  cache_dir=None,
  config_name=None,
  do_eval=True,
  do_train=True,
  eval_all_checkpoints=False,
  evaluate_during_training=True,
  fp16=False,
  fp16_opt_level='O1',
  freeze_decoder=False,
  freeze_embeds=False,
  freeze_encoder=False,
  gradient_accumulation_steps=1,
  joint_train_interval=1,
  learning_rate=5e-05,
  line_by_line=False,
  local_rank=-1,
  logging_steps=1000,
  max_grad_norm=1.0,
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
  model_type=f'../models/{model_name}',
  max_steps=300000,
  no_cache=False,
  no_cuda=False,
  num_train_epochs=1000.0,
  output_dir=output_dir,
  overwrite_cache=False,
  overwrite_output_dir=True,
  per_gpu_eval_batch_size=1,
  per_gpu_train_batch_size=1,
  save_steps=500,
  save_total_limit=2,
  seed=42,
  server_ip='',
  server_port='',
  should_continue=False,
  smart_init=False,
  test_file='../../datasets/amrbart/test.jsonl',
  tokenizer_name=None,
  train_file='../../datasets/amrbart/pretrain.jsonl',
  val_file='../../datasets/amrbart/dev.jsonl',
  warmup_steps=2500,
  weight_decay=0.0,
)

run_multitask_unified_pretraining.main(args)
