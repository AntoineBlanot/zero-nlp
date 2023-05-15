MODEL_PATH="/home/chikara/ws/peft/flan-xxl-3way/best"

python classify.py \
    --model_path $MODEL_PATH \
    --task "intent" \
    --data_name "godel-generated" \
    --out_file "godel-generated"

python classify.py \
    --model_path $MODEL_PATH \
    --task "intent" \
    --data_name "hri-forms" \
    --out_file "hri-forms"

python classify.py \
    --model_path $MODEL_PATH \
    --task "intent" \
    --data_name "demos-chat" \
    --out_file "demos-chat"

python classify.py \
    --model_path $MODEL_PATH \
    --task "intent" \
    --data_name "demos-hospital" \
    --out_file "demos-hospital"

python classify.py \
    --model_path $MODEL_PATH \
    --task "boolqa" \
    --data_name "demos-boolqa" \
    --out_file "demos-boolqa"

python classify.py \
    --model_path $MODEL_PATH \
    --task "boolqa" \
    --data_name "tung-boolqa" \
    --out_file "tung-boolqa"

python classify.py \
    --model_path $MODEL_PATH \
    --task "sentiment" \
    --data_name "demos-sentiment" \
    --out_file "demos-sentiment"

