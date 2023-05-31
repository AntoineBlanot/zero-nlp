MODEL_PATH="roberta-large-mnli"
BS=8
DO_MULTICLASS=False
THRESHOLD=...

python classify_fairseq.py \
    --model_path $MODEL_PATH --true_class "ENTAILMENT" --false_class "CONTRADICTION" \
    --task "intent" --bs $BS \
    --do_multiclass $DO_MULTICLASS --threshold $THRESHOLD \
    --data_name "godel-generated" \
    --seq_length 512 \
    --out_file "godel-generated"

python classify_fairseq.py \
    --model_path $MODEL_PATH --true_class "ENTAILMENT" --false_class "CONTRADICTION" \
    --task "intent" --bs $BS \
    --do_multiclass $DO_MULTICLASS --threshold $THRESHOLD \
    --data_name "hri-forms" \
    --seq_length 512 \
    --out_file "hri-forms"

python classify_fairseq.py \
    --model_path $MODEL_PATH --true_class "ENTAILMENT" --false_class "CONTRADICTION" \
    --task "intent" --bs $BS \
    --do_multiclass $DO_MULTICLASS --threshold $THRESHOLD \
    --data_name "demos-chat" \
    --seq_length 512 \
    --out_file "demos-chat"

python classify_fairseq.py \
    --model_path $MODEL_PATH --true_class "ENTAILMENT" --false_class "CONTRADICTION" \
    --task "intent" --bs $BS \
    --do_multiclass $DO_MULTICLASS --threshold $THRESHOLD \
    --data_name "demos-hospital" \
    --seq_length 512 \
    --out_file "demos-hospital"

python classify_fairseq.py \
    --model_path $MODEL_PATH --true_class "ENTAILMENT" --false_class "CONTRADICTION" \
    --task "boolqa" --bs $BS \
    --do_multiclass $DO_MULTICLASS --threshold $THRESHOLD \
    --data_name "demos-boolqa" \
    --seq_length 512 \
    --out_file "demos-boolqa"

python classify_fairseq.py \
    --model_path $MODEL_PATH --true_class "ENTAILMENT" --false_class "CONTRADICTION" \
    --task "boolqa" --bs $BS \
    --do_multiclass $DO_MULTICLASS --threshold $THRESHOLD \
    --data_name "tung-boolqa" \
    --seq_length 512 \
    --out_file "tung-boolqa"

python classify_fairseq.py \
    --model_path $MODEL_PATH --true_class "ENTAILMENT" --false_class "CONTRADICTION" \
    --task "sentiment" --bs $BS \
    --do_multiclass $DO_MULTICLASS --threshold $THRESHOLD \
    --data_name "demos-sentiment" \
    --seq_length 512 \
    --out_file "demos-sentiment"
