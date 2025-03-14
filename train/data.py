import torch
import datasets
from datasets import disable_caching
from torch.utils.data import DataLoader, DistributedSampler

disable_caching()

class SFTData(object):
    def __init__(self, data_path):
        self.data = datasets.load_from_disk(data_path)

    def collate_fn(self, records):
        input_ids = records[0]["input_ids"].unsqueeze(0)
        labels = records[0]["labels"].unsqueeze(0)
        attention_mask = records[0]["attention_mask"].unsqueeze(0)
        category_ids = records[0]["category_ids"].unsqueeze(0)
        packed_seq_lens = records[0]['packed_seq_lens'].to(torch.int32)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            packed_seq_lens=packed_seq_lens,
            category_ids=category_ids
        )

    def get_dataloader(self, dp_world_size, dp_rank, seed, epoch, shuffle=True):
        sampler = DistributedSampler(
            self.data, num_replicas=dp_world_size, rank=dp_rank, seed=seed+epoch, shuffle=shuffle
        )
        train_dataloader = DataLoader(
            self.data,
            batch_size=1,
            sampler=sampler,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )
        return train_dataloader