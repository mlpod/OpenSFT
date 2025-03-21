import os
import torch
import transformers
from typing import Optional
import torch.distributed as dist
import torch.nn.functional as F
from transformers.modeling_flash_attention_utils import (
    _flash_supports_window_size,
    is_flash_attn_greater_or_equal,
)
from ring_flash_attn.llama3_flash_attn_varlen import (
    llama3_flash_attn_varlen_func,
    llama3_flash_attn_prepare_cu_seqlens
)
from flash_attn.losses.cross_entropy import CrossEntropyLoss
from flash_attn.ops.rms_norm import rms_norm

def flash_rms_norm(self, x):
    return rms_norm(x, self.weight, self.variance_epsilon)

RING_ATTN_GROUP = None


def set_ring_attn_group(group):
    global RING_ATTN_GROUP
    RING_ATTN_GROUP = group


def get_ring_attn_group():
    return RING_ATTN_GROUP


def reset_ring_attn_position_ids(start, end, packed_seq_lens):
    """
    Calculate position ids for packed_seq_ids[start:end].
    For example, if the packed_seq_lens is [3, 2, 4, 1], start=2, end=8,
    the position ids will be [2, 0, 1, 0, 1, 2].

    Args:
        start: the start position
        end: the end position
        packed_seq_lens: the sequence lengths of packed sequences
    """
    position_ids = torch.zeros((1, end - start), dtype=torch.long, device=torch.cuda.current_device())
    offset = 0
    for seqlen in packed_seq_lens:
        seq_start = max(offset, start)
        seq_end = min(offset + seqlen, end)
        if seq_start < seq_end:
            position_ids[0, seq_start - start: seq_end - start] = torch.arange(seq_start - offset, seq_end - offset)

        offset += seqlen
        if offset >= end:
            break
    return position_ids


def update_ring_attn_params(packed_seq_lens, total_seq_len):
    """
    Calculate the cu_seqlens for the current forward pass and pass the value to
    the substituted ring_flash_attn.

    Note that total_seq_len may be larger than the sum of packed_seq_lens because of padding.
    """
    assert RING_ATTN_GROUP is not None
    cu_seqlens = torch.cumsum(
        packed_seq_lens.clone().detach(), #torch.tensor(packed_seq_lens, device=torch.cuda.current_device(), dtype=torch.int32),
        dim=-1,
        dtype=torch.int32,
    )
    cu_seqlens = F.pad(F.pad(cu_seqlens, (1, 0), value=0), (0, 1), value=total_seq_len)
    update_ring_flash_attn_params(cu_seqlens, RING_ATTN_GROUP)



def convert_to_local_input(input_ids, labels, attention_mask, packed_seq_lens, category_ids):
    ring_attn_rank = dist.get_rank(group=RING_ATTN_GROUP)
    ring_attn_size = dist.get_world_size(group=RING_ATTN_GROUP)
    total_seq_len = input_ids.numel()
    local_seq_len = total_seq_len // ring_attn_size
    start, end = ring_attn_rank * local_seq_len, (ring_attn_rank + 1) * local_seq_len
    local_input_ids = input_ids[:, start:end]
    local_labels_ids = labels[:, start:end]
    local_attention_mask = attention_mask[:, start:end]
    local_category_ids = category_ids[:, start:end]
    local_position_ids = reset_ring_attn_position_ids(start, end, packed_seq_lens)
    update_ring_attn_params(packed_seq_lens, total_seq_len)
    return local_input_ids, local_labels_ids, local_attention_mask, local_position_ids, local_category_ids


DATA_PARAMS = {}


def update_ring_flash_attn_params(
        cu_seqlens: torch.Tensor, process_group: dist.ProcessGroup
):
    world_size = dist.get_world_size(group=process_group)
    rank = dist.get_rank(group=process_group)
    (
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        local_k_slice,
    ) = llama3_flash_attn_prepare_cu_seqlens(cu_seqlens, True, rank, world_size)
    DATA_PARAMS.update(
        {
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_k,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_k,
            "local_k_slice": local_k_slice,
        }
    )


def create_ring_flash_attention_forward(
        process_group: dist.ProcessGroup, heads_k_stride: int
):
    def _flash_attention_forward(
            query_states: torch.Tensor,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            attention_mask: torch.Tensor,
            query_length: int,
            is_causal: bool,
            dropout: float = 0.0,
            position_ids: Optional[torch.Tensor] = None,
            softmax_scale: Optional[float] = None,
            sliding_window: Optional[int] = None,
            use_top_left_mask: bool = False,
            softcap: Optional[float] = None,
            deterministic: bool = None,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
            use_top_left_mask (`bool`, defaults to `False`):
                flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference.
            softcap (`float`, *optional*):
                Softcap for the attention logits, used e.g. in gemma2.
            deterministic (`bool`, *optional*):
                Determines if the deterministic option introduced in flash_attn>=2.4.1 is enabled.
        """
        if not use_top_left_mask:
            causal = is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__.
            causal = is_causal and query_length != 1

        # Assuming 4D tensors, key_states.shape[1] is the key/value sequence length (source length).
        use_sliding_windows = (
                _flash_supports_window_size
                and sliding_window is not None
                and key_states.shape[1] > sliding_window
        )
        flash_kwargs = (
            {"window_size": (sliding_window, sliding_window)}
            if use_sliding_windows
            else {}
        )

        if is_flash_attn_greater_or_equal("2.4.1"):
            if deterministic is None:
                deterministic = (
                        os.environ.get("FLASH_ATTENTION_DETERMINISTIC", "0") == "1"
                )
        flash_kwargs["deterministic"] = deterministic
        assert (
                softcap is None
        ), "llama3_flash_attn_varlen_func does not support softcap yet."
        # flash_kwargs["softcap"] = softcap
        flash_kwargs["group"] = process_group

        # not sure why attention_mask can be not None...
        assert causal, "only causal attention is supported yet."
        batch_size = query_states.size(0)
        assert batch_size == 1, "varlen data should be processed in advance."

        attn_output = llama3_flash_attn_varlen_func(
            query_states.squeeze(dim=0),
            key_states.squeeze(dim=0),
            value_states.squeeze(dim=0),
            cu_seqlens_q=DATA_PARAMS["cu_seqlens_q"],
            cu_seqlens_k=DATA_PARAMS["cu_seqlens_k"],
            max_seqlen_q=DATA_PARAMS["max_seqlen_q"],
            max_seqlen_k=DATA_PARAMS["max_seqlen_k"],
            heads_k_stride=heads_k_stride,
            local_k_slice=DATA_PARAMS["local_k_slice"],
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
            **flash_kwargs,
        )

        attn_output = attn_output.unsqueeze(dim=0)

        return attn_output

    return _flash_attention_forward


class Config(object):
    def __init__(self, ring_attn_size, ring_head_stride):
        self.ring_attn_size = ring_attn_size
        self.ring_head_stride = ring_head_stride
        self.ring_attn_rank = None


    def setup_ring_attn(self):
        for i in range(dist.get_world_size() // self.ring_attn_size):
            ring_attn_ranks = list(
                range(
                    i * self.ring_attn_size,
                    (i + 1) * self.ring_attn_size,
                )
            )
            group = dist.new_group(ranks=ring_attn_ranks, backend="nccl")
            if dist.get_rank() in ring_attn_ranks:
                set_ring_attn_group(group)
                self.ring_attn_rank = dist.get_rank(group=group)

        transformers.models.qwen2.modeling_qwen2._flash_attention_forward = create_ring_flash_attention_forward(
            RING_ATTN_GROUP,
            heads_k_stride=self.ring_head_stride
        )
        transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm.forward = flash_rms_norm
        transformers.models.qwen2.modeling_qwen2.CrossEntropyLoss = CrossEntropyLoss

        transformers.models.llama.modeling_llama._flash_attention_forward = create_ring_flash_attention_forward(
            RING_ATTN_GROUP,
            heads_k_stride=self.ring_head_stride
        )
        transformers.models.llama.modeling_llama.LlamaRMSNorm.forward = flash_rms_norm
        transformers.models.llama.modeling_llama.CrossEntropyLoss = CrossEntropyLoss
        return RING_ATTN_GROUP

