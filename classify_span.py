import json
from transformers import BitsAndBytesConfig, RobertaTokenizerFast, RobertaConfig
from peft import PeftModel, PeftConfig

from model.modeling import RobertaForTokenClassification, RobertaForClassification
from span import TokenDataset, SpanCollatorBERT, SpanDetector
from zero import ZeroDataset, ZeroClassifCollatorBERT, ZeroClassifier
from task_bert import NERBERT

SPAN_MODEL_PATH = '/home/chikara/ws/efficient-llm/exp/best-token-classif'
CLASSIF_MODEL_PATH = '/home/chikara/ws/efficient-llm/exp/best-seq-classif'
DATA_PATH = '/home/chikara/ws/datasets/data/CoNLL2003/dev.json'

FALLBACK_ID = -1
FALLBACK_VALUE = 'FALLBACK'

#region Model
device_map = {'': 0}
trainable_layers = ['token_head']
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_skip_modules=trainable_layers
)
peft_config = PeftConfig.from_pretrained(SPAN_MODEL_PATH)
base_config = RobertaConfig.from_pretrained(SPAN_MODEL_PATH)

span_model = RobertaForTokenClassification.from_pretrained(pretrained_model_name_or_path=peft_config.base_model_name_or_path, **base_config.to_diff_dict(), quantization_config=quantization_config, device_map=device_map)
print('Base span model loaded')
span_model = PeftModel.from_pretrained(span_model, SPAN_MODEL_PATH, device_map=device_map)
print('Full checkpoint loaded')
span_model.eval()
#endregion


#region Tokenizer + Data
span_tokenizer = RobertaTokenizerFast.from_pretrained(SPAN_MODEL_PATH, model_max_length=512, add_prefix_space=True)
span_collator = SpanCollatorBERT(tokenizer=span_tokenizer, label2id=span_model.base_model.model.config.label2id, convert_to_span=True)
span_data = TokenDataset()
span_data.from_json(file=DATA_PATH)
print(span_data)
#endregion


#region Span detection
detector = SpanDetector(model=span_model, tokenizer=span_tokenizer, tqdm=True)
span_res = detector.detect_span(dataset=span_data, id2label=span_model.base_model.model.config.id2label, batch_size=8, collator=span_collator, do_score=True)
print('Span detection scores: {}'.format(span_res['scores']))
span_results = [x for x in span_res['results'] if len(x['texts']) >= 1]
#endregion


######################################################################


#region Model
device_map = {'': 0}
trainable_layers = ['classif_head']
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_skip_modules=trainable_layers
)
peft_config = PeftConfig.from_pretrained(CLASSIF_MODEL_PATH)
base_config = RobertaConfig.from_pretrained(CLASSIF_MODEL_PATH)

classif_model = RobertaForClassification.from_pretrained(pretrained_model_name_or_path=peft_config.base_model_name_or_path, **base_config.to_diff_dict(), quantization_config=quantization_config, device_map=device_map)
print('Base classification model loaded')
classif_model = PeftModel.from_pretrained(classif_model, CLASSIF_MODEL_PATH, device_map=device_map)
print('Full checkpoint loaded')
classif_model.eval()
#endregion


#region Tokenizer + Data
classif_tokenizer = RobertaTokenizerFast.from_pretrained(CLASSIF_MODEL_PATH, model_max_length=512)
classif_collator = ZeroClassifCollatorBERT(tokenizer=classif_tokenizer)
classif_data = ZeroDataset(task_name='ner', fallback_id=FALLBACK_ID, fallback_value=FALLBACK_VALUE)
classif_data.from_pandas(data=span_results)
print(classif_data)
#endregion


#region Classification
true_id = classif_model.base_model.model.config.label2id['entailment']
false_id = classif_model.base_model.model.config.label2id['contradiction']
all_hypothesis_classes = sorted(set(sum([xx['hypothesis_classes'] for x in classif_data for xx in x], [])))
id2label = {i: l for i, l in enumerate(all_hypothesis_classes)}
id2label.update({FALLBACK_ID: FALLBACK_VALUE})

classifier = ZeroClassifier(model=classif_model, tokenizer=classif_tokenizer, do_mutliclass=False, true_id=true_id, false_id=false_id, tqdm=True)
classif_res = classifier.classify(dataset=classif_data, id2label=id2label, batch_size=8, collator=classif_collator, threshold=1)
print('Classification scores: {}'.format(classif_res['scores']))
#endregion

with open('results.json', 'w') as f:
    json.dump(classif_res, f, indent='\t')

c = 0
for i in range(5):
    span = span_results[i]
    print(span['document'])
    for j in range(len(span['texts'])):
        entity = span['texts'][j]
        tag = [x.split('-')[-1] for x in span['ner_tags'][j]]
        tag = max(tag, key=tag.count)
        pred = classif_res['preds'][c]
        scores = classif_res['probs'][c]
        c+=1

        print('Entity: {}   Pred: {} ({})   Scores: {}'.format(entity, pred, tag, scores))
