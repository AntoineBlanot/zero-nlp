MODEL_PATH="/home/chikara/ws/efficient-llm/exp/3way-nli-mixture/checkpoint-30000"

python classify.py \
    --model_path $MODEL_PATH \
    --task "intent" \
    --data_name "godel-generated" \
    --seq_length 512 \
    --out_file "godel-generated"

python classify.py \
    --model_path $MODEL_PATH \
    --task "intent" \
    --data_name "hri-forms" \
    --seq_length 512 \
    --out_file "hri-forms"

python classify.py \
    --model_path $MODEL_PATH \
    --task "intent" \
    --data_name "demos-chat" \
    --seq_length 512 \
    --out_file "demos-chat"

python classify.py \
    --model_path $MODEL_PATH \
    --task "intent" \
    --data_name "demos-hospital" \
    --seq_length 512 \
    --out_file "demos-hospital"

python classify.py \
    --model_path $MODEL_PATH \
    --task "boolqa" \
    --data_name "demos-boolqa" \
    --seq_length 512 \
    --out_file "demos-boolqa"

python classify.py \
    --model_path $MODEL_PATH \
    --task "boolqa" \
    --data_name "tung-boolqa" \
    --seq_length 512 \
    --out_file "tung-boolqa"

python classify.py \
    --model_path $MODEL_PATH \
    --task "sentiment" \
    --data_name "demos-sentiment" \
    --seq_length 512 \
    --out_file "demos-sentiment"

