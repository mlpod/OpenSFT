from tqdm import tqdm
import argparse
import random
import torch
import numpy as np
from functools import partial
from multiprocessing import Pool
from datasets import load_from_disk, Dataset


def process_single_group(group_indices, sequences, max_length, padding_value):
    """处理单个packed sequence组"""
    # 预分配numpy数组，使用padding_value初始化
    packed_input_ids = np.full(max_length, padding_value, dtype=np.int64)
    packed_labels = np.full(max_length, -100, dtype=np.int64)
    packed_category_ids = np.zeros(max_length, dtype=np.int64)
    attention_mask = np.zeros(max_length, dtype=np.int64)

    current_pos = 0
    packed_seq_lens = []

    # 随机打乱索引
    random.shuffle(group_indices)

    # 填充数据
    for idx in group_indices:
        seq_len = len(sequences[idx]['input_ids'])
        packed_seq_lens.append(seq_len)

        end_pos = current_pos + seq_len
        packed_input_ids[current_pos:end_pos] = sequences[idx]['input_ids']
        packed_labels[current_pos:end_pos] = sequences[idx]['token_labels']
        packed_category_ids[current_pos:end_pos] = sequences[idx]['category_ids']
        attention_mask[current_pos:end_pos] = 1

        current_pos = end_pos

    return {
        'input_ids': torch.from_numpy(packed_input_ids),
        'labels': torch.from_numpy(packed_labels),
        'attention_mask': torch.from_numpy(attention_mask),
        'packed_seq_lens': torch.tensor(packed_seq_lens),
        'category_ids': torch.from_numpy(packed_category_ids),
    }


def pack_sequences(sequences, max_length, num_workers=20, padding_value=0, max_attempt=2000):
    """
    使用贪心算法将多个序列打包成固定长度的序列，尽量减少packed_sequences的数量

    Args:
        sequences: 包含多个序列的列表，每个序列是一个字典，包含'input_ids'键
        max_length: 打包后序列的最大长度
        padding_value: 填充值

    Returns:
        packed_results: 包含打包后的序列和长度信息的字典列表
            - input_ids: 拼接后的序列
            - lengths: 原始序列长度列表
            - indices: 原始序列索引列表
    """

    def compute_length(example):
        return {'length': len(example['input_ids'])}

    data_with_length = sequences.map(
        compute_length,
        num_proc=num_workers,
        desc="计算序列长度"
    )
    idx2len = dict(enumerate(data_with_length['length']))

    unused_indices = set(idx2len.keys())
    packed_sequences = []

    pbar = tqdm(total=len(unused_indices), desc="打包进度")

    while unused_indices:
        current_seq = []
        current_len = 0
        attempt = max_attempt
        for idx in idx2len:
            if idx in unused_indices:
                if current_len + idx2len[idx] <= max_length:
                    current_seq.append(idx)
                    current_len += idx2len[idx]
                    unused_indices.remove(idx)
                    pbar.update(1)
                else:
                    if attempt > 0:
                        attempt -= 1
                        continue
                    else:
                        break
        packed_sequences.append(current_seq)
    pbar.close()

    with Pool(num_workers) as pool:
        process_fn = partial(
            process_single_group,
            sequences=sequences,
            max_length=max_length,
            padding_value=padding_value
        )

        # 使用imap显示进度条
        packed_results = list(tqdm(
            pool.imap(process_fn, packed_sequences),
            total=len(packed_sequences),
            desc="并行打包处理"
        ))

    return packed_results

class Packer(object):
    def __init__(self, input_path, output_path, max_length, padding_value, num_workers, max_attempt):
        self.dataset = load_from_disk(input_path)
        self.output_path = output_path
        self.max_length = max_length
        self.padding_value = padding_value
        self.num_workers = num_workers
        self.max_attempt = max_attempt
        self.packed_dataset = None

    def filter_fn(self, example):
        return len(example['input_ids']) <= self.max_length

    def pack(self):
        valid_items = self.dataset.filter(
            self.filter_fn,
            num_proc=self.num_workers
        )
        packed_data = pack_sequences(
            valid_items,
            max_length=self.max_length,
            padding_value=self.padding_value,
            max_attempt=self.max_attempt
        )
        self.packed_dataset = Dataset.from_list(packed_data)
        self.packed_dataset.set_format('torch')


    def save(self):
        self.packed_dataset.save_to_disk(self.output_path)




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, help='数据路径')
    parser.add_argument('--output-path', type=str, help='数据保存路径')
    parser.add_argument('--max-length', type=int, help=' 最大长度')
    parser.add_argument('--padding-value', type=int, help='padding值')
    parser.add_argument('--num-workers', type=int, help='并行处理的工作进程数')
    return parser.parse_args()

def main():
    args = parse_args()
    packer = Packer(
        input_path=args.input_path,
        output_path=args.output_path,
        max_length=args.max_length,
        padding_value=args.padding_value,
        num_workers=args.num_workers,
        max_attempt=2000) # 打包超出最大长度时，继续向后查找次数
    packer.pack()
    packer.save()

if __name__ == "__main__":
    main()


