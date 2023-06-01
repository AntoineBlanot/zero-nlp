from typing import Any, Callable, Dict, List
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import datasets
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from tqdm import tqdm

from task import (
    IntentRecognitionForDialog, BoolQA, SentimentAnalysisForDialog,
    GlobalBoolQA, SentimentAnalysis, NaturalLanguageInference, Paraphrase
)
from task_bert import (
    IntentRecognitionForDialogBERT, BoolQABERT, SentimentAnalysisForDialogBERT,
    GlobalBoolQABERT, SentimentAnalysisBERT, ParaphraseBERT, NaturalLanguageInferenceBERT,
    NERBERT
)

TASK_MAPPING = {
    'intent': IntentRecognitionForDialogBERT, 'boolqa': BoolQABERT, 'sentiment': SentimentAnalysisBERT,
    'global-boolqa': GlobalBoolQABERT, 'global-sentiment': SentimentAnalysisBERT, 'global-paraphrase': ParaphraseBERT, 'global-nli': NaturalLanguageInferenceBERT,
    'ner': NERBERT
}

def convert_columns(task: str, data: datasets.Dataset):
    if task == 'intent':
        renamed_data =  data.rename_columns(dict(
            topic='label_name',
            question='bot_question',
            answer='user_answer',
            possible_intents='candidate_labels'
        ))
    
    if task == 'boolqa':
        renamed_data =  data.rename_columns(dict(
            topic='label_name',
            possible_intents='candidate_labels'
        ))
    
    if task == 'sentiment':
        renamed_data =  data.rename_columns(dict(
            topic='label_name',
            # question='bot_question',
            answer='document',
            possible_intents='candidate_labels'
        ))

    if task == 'global-sentiment':
        renamed_data =  data.rename_columns(dict(
            text='document'
        ))

    if task == 'global-boolqa':
        renamed_data = data

    if task == 'global-paraphrase':
        renamed_data = data.rename_columns(dict(
            question1='document1',
            question2='document2'
        ))
    
    if task == 'global-nli':
        renamed_data = data.rename_columns(dict(
            sentence1='premise',
            sentence2='claim'
        ))

    if task == 'ner':
        renamed_data = data.rename_columns(dict(
            texts='detected_entities',
            ner_tags='label_list'
        ))
        renamed_data = renamed_data.add_column('candidate_labels', [['ORG', 'PER', 'LOC']]*len(renamed_data))

    return renamed_data

class ZeroDataset(Dataset):

    def __init__(self, task_name: str, fallback_id: int, fallback_value: str) -> None:
        super().__init__()
        self.task_name = task_name
        self.task = TASK_MAPPING[self.task_name](fallback_id=fallback_id, fallback_value=fallback_value)
    
    def from_json(self, file: str):
        self.data = datasets.load_dataset('json', data_files=file)['train']
        self.data = convert_columns(self.task_name, self.data)

    def from_dict(self, data_dict: str):
        self.data = datasets.Dataset.from_dict({k: [v] for k, v in data_dict.items()})

    def from_pandas(self, data: List[dict]):
        self.data = datasets.Dataset.from_pandas(pd.DataFrame(data=data))
        self.data = convert_columns(self.task_name, self.data)

    def column_names(self):
        return self.data.column_names
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self) -> str:
        return self.data.__repr__()

    def __getitem__(self, index):
        data_dict = self.data[index]
        return self.task.build_prompt(**data_dict)


class ZeroCollator():

    def __init__(self, tokenizer, with_labels: bool = False) -> None:
        self.tokenizer = tokenizer
        self.with_labels = with_labels

    def __call__(self, features):
        batched_input_text = [xx['input_text'] for x in features for xx in x]
        batched_target_text = [self.tokenizer.pad_token + xx['target_text'] for x in features for xx in x]
        batched_label = [xx['label'] for x in features for xx in x] if self.with_labels else None

        batched_hypothesis_classes = [xx['hypothesis_classes'] for x in features for xx in x]
        batched_group = [xx['group'] for x in features for xx in x]

        inputs = self.tokenizer(batched_input_text, padding=True, truncation=True, return_tensors='pt')
        targets = self.tokenizer(batched_target_text, padding=True, truncation=True, return_tensors='pt')
        labels = torch.as_tensor(batched_label, dtype=torch.long) if self.with_labels else None

        return dict(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            decoder_input_ids=targets.input_ids,
            decoder_attention_mask=targets.attention_mask,
            labels=labels,
            metadata=dict(
                hypothesis_classes=batched_hypothesis_classes,
                group=batched_group
            )
        )

class ZeroClassifCollatorBERT():
    """
    Data collator for `BERT` models for `zero-shot classification` task

    Required features in the dataset:
        - `input_text`: text to input to the model (str)
        - `hypothesis_classes`: list of hypothesis classes (list of str)
        - `group`: group number for batched predictions (int)
    
    Optional features in the dataset:
        - `label`: target class
    """

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features):
        batched_input_text = [xx['input_text'] for x in features for xx in x]
        batched_hypothesis_classes = [xx['hypothesis_classes'] for x in features for xx in x]
        batched_group = [xx['group'] for x in features for xx in x]

        inputs = self.tokenizer(batched_input_text, padding=True, truncation=True, return_tensors='pt')
        inputs['metadata'] = dict(
                hypothesis_classes=batched_hypothesis_classes,
                group=batched_group
        )

        has_label = any(['label' in xx.keys() for x in features for xx in x])   # check if label field is present
        if has_label:
            batched_label = [xx['label'] for x in features for xx in x]
            labels = torch.as_tensor(batched_label, dtype=torch.long)
            inputs['labels'] = labels
        
        return inputs

class ZeroClassifier():

    def __init__(self, model: torch.nn.Module, tokenizer, do_mutliclass: bool, true_id: int, false_id: int, tqdm: bool = False) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.do_mutliclass = do_mutliclass
        self.true_id = true_id
        self.false_id = false_id
        self.tqdm = tqdm

    def classify(self, dataset: Dataset, id2label: Dict[int, str], batch_size: int = 1, collator: Callable = None, threshold: float = 0.8):
        self.id2label = id2label
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)

        output_list, label_list, group_list = [], [], []
        for inputs in tqdm(dataloader, desc='Classifying', disable=not self.tqdm):
            metadata = inputs.pop('metadata')

            if 'labels' in inputs:
                label_list.append(inputs.pop('labels'))

            with torch.no_grad():
                outputs = self.model(**inputs)
                outputs = outputs['logits'][:, [self.true_id, self.false_id]].float()

            output_list.append(outputs)
            group_list.append(metadata['group'])

        group_list = sum(group_list, [])
        group_count = [group_list.count(g) for g in set(group_list)]
        outputs = torch.cat(output_list)

        preds, probs = self.predict(outputs, group_count, threshold=threshold)

        results_dict = dict(
            id2label=self.id2label,
            preds=preds,
            probs=[p.tolist() for p in probs]
        )

        if label_list != []:
            labels = [x[0].item() for x in torch.split(torch.cat(label_list), group_count)]
            scores = self.score(labels, preds)
            results_dict = {'scores': scores, **results_dict}
        
        return results_dict

    def predict(self, outputs, group_count, threshold: float):
        grouped_outputs = torch.split(outputs, group_count)

        if self.do_mutliclass:
            probs = [x.softmax(1)[:, 0] for x in grouped_outputs]
        else:
            probs = [x.softmax(0)[:, 0] for x in grouped_outputs]

        preds = [torch.argmax(x).item() if torch.max(x) >= 2 / (len(x)+threshold) else -1 for x in probs]
        preds = [self.id2label[x] for x in preds]

        return preds, probs
    
    def score(self, labels, preds):
        labels = [self.id2label[x] for x in labels]

        conf_matrix = confusion_matrix(y_true=labels, y_pred=preds, labels=list(self.id2label.values()))
        print(self.id2label)
        print(conf_matrix)

        acc = accuracy_score(y_true=labels, y_pred=preds)
        f1 = f1_score(y_true=labels, y_pred=preds, average='weighted')
    
        return dict(acc=acc, f1=f1)
