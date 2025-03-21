import random
from datasets import load_from_disk, Dataset


def merge_records(record1, record2):
    """
    合并两个记录。
    :param record1: 第一个记录 (dict)
    :param record2: 第二个记录 (dict)
    :return: 合并后的记录 (dict)
    """
    c1 = record1["messages"][0]["role"] == "system" and record2["messages"][0]["role"] == "system"
    c2 = record1["messages"][0]["content"] == record2["messages"][0]["content"]
    if c1 and c2:
        record2["messages"] = record2["messages"][1:]
        record2['labels'] = record2['labels'][1:]

    if 'meta' in record1:
        record1['category_id_list'] = [record1["meta"]["category_id"]] * len(record1["messages"])

    merged_record = {
        "messages": record1["messages"] + record2["messages"],  # 合并 messages
        "labels": record1["labels"] + record2["labels"],  # 合并 labels
        "category_id_list": record1['category_id_list'] +
                            [record2["meta"]["category_id"]] * len(record2["messages"]),  # 第二条记录的 category_id
    }
    assert len(merged_record['messages']) == len(merged_record['labels']) == len(merged_record['category_id_list'])
    return merged_record




def merge(input_file, output_file, merge_probability):
    """
    :param input_file: 输入 JSONL 文件路径
    :param output_file: 输出 JSONL 文件路径
    :param merge_probability: 合并的概率 (0～1)
    """
    # 读取 JSONL 文件内容
    dataset = load_from_disk(input_file)
    records = []
    for record in dataset:
        records.append(record)

    merged_records = []
    i = 0
    while i < len(records):
        record1 = records[i]
        i += 1
        # 按概率尝试合并连续记录
        while i < len(records) and random.random() < merge_probability:
            record2 = records[i]
            record1 = merge_records(record1, record2)
            i += 1
        # 将最终合并的记录加入结果
        merged_records.append(record1)

    turn_len_list = []
    for rec in merged_records:
        turn_len = len(rec['messages'])
        turn_len_list.append(turn_len)
    print("平均轮次:", sum(turn_len_list)/len(turn_len_list))

    # 保存结果到文件
    data = Dataset.from_list(merged_records)
    print("合并前:", len(dataset), "合并后:", len(data))
    data.save_to_disk(output_file)


if __name__ == '__main__':
    # 示例：运行脚本
    input_file = "dataset/distill_r1_110k.sampled.jsonl"  # 输入文件路径
    output_file = "dataset/distill_r1_110k.sampled.pseudo.multiturn.jsonl"  # 输出文件路径
    merge_probability = 0.5  # 合并概率 (50%)
    num_workers = 1
    merge(input_file, output_file, merge_probability)