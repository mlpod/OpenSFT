set -x


step0=0
step1=1
step2=0
step3=0

if [ $step0 -ne 0 ]; then
    python script/preprocess.example.py
fi

config_path=config/distill_r1_110k.json
preprocessed_data_path=dataset/distill_r1_110k.preprocessed.jsonl
sampled_data_path=dataset/distill_r1_110k.sampled.jsonl
tokenized_data_path=dataset/distill_r1_110k.tokenized.jsonl
packed_data_path=dataset/distill_r1_110k.packed.jsonl


if [ $step1 -ne 0 ]; then
    python script/step1.sample.py \
        --config-path=$config_path \
        --preprocessed-data-path=$preprocessed_data_path \
        --output-path=$sampled_data_path \
        --seed=2025 --num-workers=10
fi

tokenizer_path=
padding_value=

if [ $step2 -ne 0 ]; then
    python script/step2.tokenize.py \
        --input-path=$sampled_data_path \
        --output-path=$tokenized_data_path \
        --tokenizer-path=$tokenizer_path \
        --num-workers=10
fi


if [ $step3 -ne 0 ]; then
    python script/step3.pack.py \
        --input-path=$tokenized_data_path \
        --output-path=$packed_data_path \
        --max-length=131072 \
        --padding-value=$padding_value \
        --num-workers=10
fi