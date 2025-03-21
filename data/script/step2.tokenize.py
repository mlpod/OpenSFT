import json
import torch
import argparse
from transformers import AutoTokenizer
from datasets import load_from_disk

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

class Tokenizer(object):
    def __init__(self, input_path, tokenizer_path, num_workers, output_path, ignore_index=-100):
        self.dataset = load_from_disk(input_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.num_workers = num_workers
        self.output_path = output_path
        self.ignore_index = ignore_index


    def process(self, record):
        input_ids = self.tokenizer.apply_chat_template(
            record["messages"],
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt")[0]

        token_labels = torch.full_like(input_ids, fill_value=self.ignore_index)
        category_ids = [0] * len(input_ids)
        for idx, message in enumerate(record["messages"]):
            if message["role"] == "assistant":
                prompt = self.tokenizer.apply_chat_template(record["messages"][:idx], tokenize=False,
                                                            add_generation_prompt=True)
                response = self.tokenizer.apply_chat_template(record["messages"][: idx + 1], tokenize=False)[
                           len(prompt):]
                start_idx = self.tokenizer(
                    prompt,
                    padding=False,
                    return_tensors="pt",
                    add_special_tokens=False,
                )["attention_mask"].int().sum().item()
                end_idx = start_idx + self.tokenizer(
                    response,
                    padding=False,
                    return_tensors="pt",
                    add_special_tokens=False,
                )["attention_mask"].int().sum().item()
                if record["labels"][idx] == 1:
                    token_labels[start_idx:end_idx] = input_ids[start_idx:end_idx]
                    if 'meta' in record:
                        category_ids[start_idx:end_idx] = [record['meta']['category_id']] * (end_idx - start_idx)
                    else:
                        category_ids[start_idx:end_idx] = [record['category_id_list'][idx]] * (end_idx - start_idx)
        return {
            "input_ids": input_ids,
            "token_labels": token_labels,
            "category_ids": category_ids,
        }

    def tokenize(self):
        self.dataset = self.dataset.map(
            self.process,
            num_proc=self.num_workers,
            remove_columns=self.dataset.column_names,
            desc="分词"
        )

    def save(self):
        self.dataset.save_to_disk(self.output_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, help='数据路径')
    parser.add_argument('--output-path', type=str, help='数据保存路径')
    parser.add_argument('--tokenizer-path', type=str, help='tokenizer路径')
    parser.add_argument('--num-workers', type=int, help='并行处理的工作进程数')
    return parser.parse_args()

def main():
    args = parse_args()
    tokenizer = Tokenizer(input_path=args.input_path,
                        tokenizer_path=args.tokenizer_path,
                        num_workers=args.num_workers,
                      output_path=args.output_path)
    tokenizer.tokenize()
    tokenizer.save()

if __name__ == "__main__":
    main()