import json
import random
import argparse
from tqdm import tqdm
from multiprocessing import Pool
from datasets import Dataset, load_from_disk

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def process_category(items, ratio, seed):
    random.seed(seed)
    int_part = int(ratio)
    float_part = ratio - int_part
    result = items * int_part
    extra_sample_size = int(len(items) * float_part)
    if extra_sample_size > 0:
        result += random.sample(items, extra_sample_size)
    return result

class Sampler(object):
    def __init__(self, config_path, preprocessed_data_path, num_workers, seed, output_path):
        self.config =load_config(config_path=config_path)
        self.num_workers = num_workers
        self.seed = seed
        self.output_path = output_path
        self.dataset = load_from_disk(preprocessed_data_path)

    def sample(self):
        category_ratios = {
            item['category_id']: item['sample_rate']
            for item in self.config['ratio']
        }
        categorized_data = {}
        for item in tqdm(self.dataset):
            category_id = item['meta']['category_id']
            if category_id not in categorized_data:
                categorized_data[category_id] = []
            categorized_data[category_id].append(item)
        with Pool(self.num_workers) as pool:
            tasks = [
                (items, category_ratios[category_id], self.seed + category_id)
                for category_id, items in categorized_data.items()
            ]
            results = pool.starmap(process_category, tasks)
        sampled_data = []
        for result in results:
            sampled_data.extend(result)
        random.seed(self.seed)
        random.shuffle(sampled_data)
        self.dataset = Dataset.from_list(sampled_data)

    def save(self):
        self.dataset.save_to_disk(self.output_path)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, help='数据配置文件路径')
    parser.add_argument('--preprocessed-data-path', type=str, help='数据保存路径')
    parser.add_argument('--output-path', type=str, help='数据保存路径')
    parser.add_argument('--seed', type=int, default=42, help='随机数种子')
    parser.add_argument('--num-workers', type=int, default=5, help='并行处理的工作进程数')
    return parser.parse_args()

def main():
    args = parse_args()
    sampler = Sampler(config_path=args.config_path,
                      preprocessed_data_path=args.preprocessed_data_path,
                      output_path=args.output_path,
                      seed=args.seed,
                      num_workers=args.num_workers)
    sampler.sample()
    sampler.save()

if __name__ == "__main__":
    main()