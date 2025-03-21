import os
import math
import time
import json
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from datetime import timedelta
from accelerate import Accelerator
from torch import distributed as dist
from flash_attn.losses.cross_entropy import CrossEntropyLoss
from flash_attn.utils.distributed import all_gather, all_reduce
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate.utils import (
    InitProcessGroupKwargs,
    set_seed,
    DummyOptim,
    DummyScheduler,
)

os.environ["WANDB_MODE"] = "offline"

from data import SFTData
from ring_attn_utils import Config, convert_to_local_input



def main(args):
    set_seed(args.seed)
    timeout = InitProcessGroupKwargs(timeout=timedelta(seconds=1_000_000))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        kwargs_handlers=[timeout],
        log_with="wandb"
    )
    accelerator.init_trackers(project_name=args.project_name)
    world_size = accelerator.num_processes
    sp_world_size = args.sequence_parallel_degree
    dp_world_size = world_size // sp_world_size

    accelerator.print(f"world_size: {world_size}")
    accelerator.print(f"sp_world_size: {sp_world_size}")
    accelerator.print(f"dp_world_size: {dp_world_size}")

    config = Config(ring_attn_size=sp_world_size, ring_head_stride=4)
    ring_group = config.setup_ring_attn()

    with open(args.data_config_path, 'r') as f:
        data_config = json.loads(f.read())
    idx2name = {}
    for file in data_config['ratio']:
        idx2name[file['category_id']] = file['category_name']

    sft_data = SFTData(args.data_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2",
    )

    num_steps_per_epoch = math.ceil(len(sft_data.data) / dp_world_size / args.gradient_accumulation_steps)
    max_train_steps = num_steps_per_epoch * args.num_epochs

    accelerator.print(f"数据总量: {len(sft_data.data)}")
    accelerator.print(f"训练步数: {max_train_steps}")

    optim = DummyOptim(model.parameters())
    scheduler = DummyScheduler(optim)

    local_ce = CrossEntropyLoss(reduction="none")

    model, optim, scheduler = accelerator.prepare(model, optim, scheduler)
    model.gradient_checkpointing_enable()
    accelerator.register_for_checkpointing(scheduler)
    model.train()

    global_step = 0
    for epoch in range(args.num_epochs):
        train_loader = sft_data.get_dataloader(dp_world_size, dist.get_rank() // sp_world_size, seed=args.seed,
                                               epoch=epoch, shuffle=True)
        train_steps = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
        accelerator.print(f"每个epoch数据总量: {len(train_loader)}")
        accelerator.print(f"每个epoch训练步数: {train_steps}")
        progress_bar = tqdm(
            range(train_steps), disable=not accelerator.is_local_main_process
        )
        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(accelerator.device)
            labels = batch['labels'].to(accelerator.device)
            attention_mask = batch['attention_mask'].to(accelerator.device)
            packed_seq_lens = batch['packed_seq_lens'].to(accelerator.device)
            category_ids = batch['category_ids'].to(accelerator.device)

            local_input_ids, local_labels, local_attention_mask, local_position_ids, local_category_ids = convert_to_local_input(
                input_ids, labels, attention_mask, packed_seq_lens, category_ids
            )

            with accelerator.accumulate(model):
                out = model(
                    input_ids=local_input_ids,
                    attention_mask=local_attention_mask,
                    position_ids=local_position_ids,
                )
                shift_logits = out.logits[..., :-1, :].contiguous()
                shift_labels = local_labels[..., 1:].contiguous()
                token_losses = local_ce(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                if args.use_turn_loss:
                    all_token_losses = all_gather(token_losses, ring_group)
                    all_shift_labels = all_gather(shift_labels, ring_group)
                    mask = all_shift_labels.view(-1) != -100
                    idx = torch.where(mask)[0]
                    diffs = torch.diff(idx)
                    split_points = torch.where(diffs > 1)[0] + 1
                    groups = torch.tensor_split(idx, split_points.tolist())
                    loss = torch.stack([all_token_losses[g].mean() for g in groups]).mean()
                    loss = torch.nan_to_num(loss, nan=0.0)
                else:
                    loss = token_losses[shift_labels.view(-1) != -100].mean()
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM, group=ring_group)
                    loss = loss / sp_world_size
                    loss = torch.nan_to_num(loss, nan=0.0)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    gathered_loss = accelerator.gather(loss.clone().detach())
                    mask = (gathered_loss != 0)
                    if mask.sum() == 0:
                        loss_ = torch.tensor(0.0, device=accelerator.device)
                    else:
                        loss_ = gathered_loss[mask].mean()
                    loss_log = {
                        "epoch": epoch,
                        "steps": step,
                        "lr": scheduler.get_last_lr()[0],
                        "loss": loss_.item()
                    }

                    progress_bar.set_postfix(loss_log)
                    progress_bar.update(1)
                    time.sleep(0.1)

                    token_losses_ = token_losses.clone().detach()
                    all_token_losses = accelerator.gather(token_losses_)
                    all_category_ids = accelerator.gather(local_category_ids)[..., 1:].contiguous()

                    idx2mean_loss = pd.DataFrame({
                        'category_id': all_category_ids.cpu().ravel(),
                        'token_loss': all_token_losses.cpu().numpy().ravel()
                    }).groupby('category_id')['token_loss'].mean().to_dict()

                    for idx, mean_loss in idx2mean_loss.items():
                        if idx == 0 or idx not in idx2name:
                            continue
                        loss_log[f"{idx2name[idx]}_loss"] = mean_loss

                    loss_log_str = '\n' + json.dumps(loss_log, ensure_ascii=False, indent=4)
                    accelerator.print(loss_log_str)
                    accelerator.log(loss_log, step=global_step)
                    global_step += 1

                optim.step()
                scheduler.step()
                optim.zero_grad()

        if args.save_checkpoint is not None:
            if not os.path.exists(args.save_checkpoint):
                os.makedirs(args.save_checkpoint, exist_ok=True)
            save_path = f"{args.save_checkpoint}/epoch{epoch}_end"
            accelerator.print(f"Saving model to {save_path}")
            accelerator.wait_for_everyone()
            state_dict = accelerator.get_state_dict(model)
            accelerator.unwrap_model(model).save_pretrained(
                save_path,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=state_dict,
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(save_path)
                os.remove(f"{save_path}/model.safetensors")

            accelerator.print(f"Saving Finished")

    accelerator.print(f"Training Finished")
    accelerator.end_training()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--project_name", type=str, default='')
    args.add_argument("--gradient_accumulation_steps", type=int, default=1)
    args.add_argument("--save_checkpoint", type=str, default="")
    args.add_argument("--seed", type=int, default=2025)
    args.add_argument("--model_path", type=str, default="")
    args.add_argument("--data_path", type=str, default="")
    args.add_argument("--data_config_path", type=str, default="")
    args.add_argument("--sequence_parallel_degree", type=int, default=8)
    args.add_argument("--num_epochs", type=int, default=4)
    args.add_argument("--use_turn_loss", action='store_true')
    main(args.parse_args())