from argparse import ArgumentParser
from pathlib import Path
import json
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoConfig

from model.modeling import T5ForClassification
from zero import ZeroDataset, ZeroClassifier

DATA_PATH = {
    'godel-generated':[str(x) for x in Path('/home/chikara/data/zero-shot-intent/godel-generated/').glob('*/*.json')],
    'hri-forms': [str(x) for x in Path('/home/chikara/data/zero-shot-intent/hri-forms/').glob('*/*_v2.json')],
    'demos-chat':[str(x) for x in Path('/home/chikara/data/zero-shot-intent/demos/').glob('data_v3.json')],
    'demos-hospital': [str(x) for x in Path('/home/chikara/data/zero-shot-intent/demos/').glob('hospital_collected_clean.json')],
    'demos-boolqa':[str(x) for x in Path('/home/chikara/data/zero-shot-intent/demos/').glob('yes-no_collected_clean.json')],
    'tung-boolqa': [str(x) for x in Path('/home/chikara/data/zero-shot-intent/tung-yesno/').glob('*/data.json')],
    'demos-sentiment':[str(x) for x in Path('/home/chikara/data/zero-shot-intent/demos/').glob('sentiment_collected_clean.json')]
}

parser = ArgumentParser()
parser.add_argument(
    '--task', type=str
)
parser.add_argument(
    '--model_path', type=str
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

# Load Model
peft_config = PeftConfig.from_pretrained(args.model_path)
base_config = AutoConfig.from_pretrained(args.model_path)
model = T5ForClassification.from_pretrained(pretrained_model_name_or_path=peft_config.base_model_name_or_path, **base_config.to_diff_dict(), load_in_8bit=True, device_map={'': 0})

model = PeftModel.from_pretrained(model, args.model_path, device_map={'': 0})
model.eval()

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_path, model_max_length=args.seq_length)

# Load Data
data = ZeroDataset(task_name=args.task)
data.from_json(file=DATA_PATH[args.data_name])

# Classification
classifier = ZeroClassifier(model, tokenizer, true_id=0, tqdm=True)
results = classifier.classify(data, batch_size=8, threshold=0.8)

print(results)

if args.out_file:
    with open(args.out_file + '.json', 'w') as f:
        json.dump(results, f, indent='\t')