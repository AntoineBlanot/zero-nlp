MODEL_PATH="/home/chikara/ws/efficient-llm/exp/roberta-nli/checkpoint-70000"
BS=8
DO_MULTICLASS=False
THRESHOLD=...

python classify.py \
    --model_path $MODEL_PATH --true_class "entailment" --false_class "contradiction" \
    --task "intent" --bs $BS \
    --do_multiclass $DO_MULTICLASS --threshold $THRESHOLD \
    --data_name "godel-generated" \
    --seq_length 512 \
    --out_file "godel-generated"

python classify.py \
    --model_path $MODEL_PATH --true_class "entailment" --false_class "contradiction" \
    --task "intent" --bs $BS \
    --do_multiclass $DO_MULTICLASS --threshold $THRESHOLD \
    --data_name "hri-forms" \
    --seq_length 512 \
    --out_file "hri-forms"

python classify.py \
    --model_path $MODEL_PATH --true_class "entailment" --false_class "contradiction" \
    --task "intent" --bs $BS \
    --do_multiclass $DO_MULTICLASS --threshold $THRESHOLD \
    --data_name "demos-chat" \
    --seq_length 512 \
    --out_file "demos-chat"

python classify.py \
    --model_path $MODEL_PATH --true_class "entailment" --false_class "contradiction" \
    --task "intent" --bs $BS \
    --do_multiclass $DO_MULTICLASS --threshold $THRESHOLD \
    --data_name "demos-hospital" \
    --seq_length 512 \
    --out_file "demos-hospital"

python classify.py \
    --model_path $MODEL_PATH --true_class "entailment" --false_class "contradiction" \
    --task "boolqa" --bs $BS \
    --do_multiclass $DO_MULTICLASS --threshold $THRESHOLD \
    --data_name "demos-boolqa" \
    --seq_length 512 \
    --out_file "demos-boolqa"

python classify.py \
    --model_path $MODEL_PATH --true_class "entailment" --false_class "contradiction" \
    --task "boolqa" --bs $BS \
    --do_multiclass $DO_MULTICLASS --threshold $THRESHOLD \
    --data_name "tung-boolqa" \
    --seq_length 512 \
    --out_file "tung-boolqa"

python classify.py \
    --model_path $MODEL_PATH --true_class "entailment" --false_class "contradiction" \
    --task "sentiment" --bs $BS \
    --do_multiclass $DO_MULTICLASS --threshold $THRESHOLD \
    --data_name "demos-sentiment" \
    --seq_length 512 \
    --out_file "demos-sentiment"

## GLOBAL

python classify.py \
    --model_path $MODEL_PATH --true_class "entailment" --false_class "contradiction" \
    --task "global-boolqa" --bs $BS \
    --do_multiclass $DO_MULTICLASS --threshold $THRESHOLD \
    --data_name "global-boolqa" \
    --seq_length 512 \
    --out_file "global-boolqa"

python classify.py \
    --model_path $MODEL_PATH --true_class "entailment" --false_class "contradiction" \
    --task "global-sentiment" --bs $BS \
    --do_multiclass $DO_MULTICLASS --threshold $THRESHOLD \
    --data_name "global-sentiment" \
    --seq_length 512 \
    --out_file "global-sentiment"

python classify.py \
    --model_path $MODEL_PATH --true_class "entailment" --false_class "contradiction" \
    --task "global-paraphrase" --bs $BS \
    --do_multiclass $DO_MULTICLASS --threshold $THRESHOLD \
    --data_name "global-paraphrase" \
    --seq_length 512 \
    --out_file "global-paraphrase"

python classify.py \
    --model_path $MODEL_PATH --true_class "entailment" --false_class "contradiction" \
    --task "global-nli" --bs $BS \
    --do_multiclass $DO_MULTICLASS --threshold $THRESHOLD \
    --data_name "global-nli" \
    --seq_length 512 \
    --out_file "global-nli"
