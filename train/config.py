import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--max_steps', type=int, default=2000)
parser.add_argument('--warmup_num_steps', type=int, default=200)
parser.add_argument("--learning_rate", type=float, default=2e-5)
args = parser.parse_args()

with open("config/zero3_offload.json") as f:
    config = json.loads(f.read())

config['scheduler']['params']['total_num_steps'] = args.max_steps
config['scheduler']['params']['warmup_num_steps'] = args.warmup_num_steps
config['scheduler']['params']['warmup_max_lr'] = args.learning_rate
config['scheduler']['params']['warmup_min_lr'] = args.learning_rate * 0.1

print(config)
with open("config/zero3_offload.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(config, indent=4))
