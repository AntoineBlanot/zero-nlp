from argparse import ArgumentParser
from pathlib import Path
import json
from transformers import BitsAndBytesConfig, AutoTokenizer, RobertaForSequenceClassification

from zero import ZeroDataset, ZeroClassifCollatorBERT, ZeroClassifier

DATA_PATH = {
    'godel-generated':[str(x) for x in Path('/home/chikara/data/zero-shot-intent/godel-generated/').glob('*/*.json')],
    'hri-forms': [str(x) for x in Path('/home/chikara/data/zero-shot-intent/hri-forms/').glob('*/*_v2.json')],
    'demos-chat':[str(x) for x in Path('/home/chikara/data/zero-shot-intent/demos/').glob('data_v3.json')],
    'demos-hospital': [str(x) for x in Path('/home/chikara/data/zero-shot-intent/demos/').glob('hospital_collected_clean.json')],
    'demos-boolqa':[str(x) for x in Path('/home/chikara/data/zero-shot-intent/demos/').glob('yes-no_collected_clean.json')],
    'tung-boolqa': [str(x) for x in Path('/home/chikara/data/zero-shot-intent/tung-yesno/').glob('*/data.json')],
    'demos-sentiment':[str(x) for x in Path('/home/chikara/data/zero-shot-intent/demos/').glob('sentiment_collected_clean.json')]
}

FALLBACK_ID = -1
FALLBACK_VALUE = 'FALLBACK'

parser = ArgumentParser()
parser.add_argument(
    '--model_path', type=str
)
parser.add_argument(
    '--task', type=str
)
parser.add_argument(
    '--bs', type=int
)
parser.add_argument(
    '--do_multiclass', default=False, type=lambda x: (str(x).lower() == 'true')
)
parser.add_argument(
    '--threshold', type=float
)
parser.add_argument(
    '--true_class', type=str
)
parser.add_argument(
    '--false_class', type=str
)
parser.add_argument(
    '--data_name', type=str
)
parser.add_argument(
    '--seq_length', type=int
)
parser.add_argument(
    '--out_file', type=str, default=None
)
args = parser.parse_args()
print(args)

# Load Model
device_map = {'': 0}
trainable_layers = ['classifier']
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_skip_modules=trainable_layers
)
model = RobertaForSequenceClassification.from_pretrained(args.model_path, quantization_config=quantization_config, device_map=device_map)
model.eval()

# Load Tokenizer and Data Collator
tokenizer = AutoTokenizer.from_pretrained(args.model_path, model_max_length=args.seq_length)
collator = ZeroClassifCollatorBERT(tokenizer)

# Load Data
data = ZeroDataset(task_name=args.task, fallback_id=FALLBACK_ID, fallback_value=FALLBACK_VALUE)
data.from_json(file=DATA_PATH[args.data_name])
print(data)

# Classification
true_id = model.config.label2id[args.true_class]
false_id = model.config.label2id[args.false_class]
all_hypothesis_classes = sorted(set(sum([xx['hypothesis_classes'] for x in data for xx in x], [])))
id2label = {i: l for i, l in enumerate(all_hypothesis_classes)}
id2label.update({FALLBACK_ID: FALLBACK_VALUE})

classifier = ZeroClassifier(model, tokenizer, do_mutliclass=args.do_multiclass, true_id=true_id, false_id=false_id, tqdm=True)
results = classifier.classify(data, id2label=id2label, batch_size=args.bs, collator=collator, threshold=args.threshold)

print('Classification scores: {}'.format(results['scores']))

if args.out_file:
    with open(args.out_file + '.json', 'w') as f:
        json.dump(results, f, indent='\t')
