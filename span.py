from typing import Any, Callable, Dict, List

from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import datasets
from evaluate import load


class TokenDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
    
    def from_json(self, file: str):
        self.data = datasets.load_dataset('json', data_files=file)['train']

    def from_dict(self, data_dict: str):
        self.data = datasets.Dataset.from_dict({k: [v] for k, v in data_dict.items()})

    def column_names(self):
        return self.data.column_names
    
    def __repr__(self) -> str:
        return self.data.__repr__()
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class SpanCollatorBERT():
    """
    Data collator for `BERT` models for `span detection` task

    Required features in the dataset:
        - `tokens`: pre-tokenized (split into words) text (list of str)
    
    Optional features in the dataset:
        - `ner_tags`: target tags for each pre-tokenized token (list of int)
    """

    def __init__(self, tokenizer, label2id: Dict[str, int], convert_to_span: bool = True) -> None:
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.convert_to_span = self.__convert_span_fn if convert_to_span else lambda x: x
        self.preprocess_fn = self.tokenize_and_align_labels

    def __convert_span_fn(self, tag: str):
        if tag.startswith('B'):
            return 'B-TAG'
        elif tag.startswith('I'):
            return 'I-TAG'
        return tag

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batched_tokens = [x['tokens'] for x in features]
        examples_dict = dict(tokens=batched_tokens)

        has_labels = any(['ner_tags' in x.keys() for x in features])    # check if bin_tag field is present
        if has_labels:
            batched_str_tags = [x['ner_tags'] for x in features]
            batched_tags = [[self.label2id[self.convert_to_span(tag)] for tag in tag_list] for tag_list in batched_str_tags]
            examples_dict['bin_tags'] = batched_tags
            examples_dict['ner_tags'] = batched_str_tags

        inputs = self.preprocess_fn(examples=examples_dict)

        return inputs
    
    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(examples['tokens'], is_split_into_words=True, padding=True, truncation=True, pad_to_multiple_of=8, return_tensors='pt')

        if 'bin_tags' in examples.keys():
            labels = []
            for i, label in enumerate(examples['bin_tags']):
                word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:  # Set the special tokens to -100.
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                labels.append(label_ids)

            tokenized_inputs['labels'] = torch.as_tensor(labels, dtype=torch.long)

        if 'ner_tags' in examples.keys():
            labels = []
            for i, label in enumerate(examples['ner_tags']):
                word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:  # Set the special tokens to -100.
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(label[word_idx])
                    previous_word_idx = word_idx
                labels.append(label_ids)

            tokenized_inputs['metadata'] = dict()
            tokenized_inputs['metadata']['ner_labels'] = labels

        return tokenized_inputs

class SpanDetector():
    """
    Wrapper class for `Span Detection` task (i.e. binary token classification)
    """

    def __init__(self, model: torch.nn.Module, tokenizer, tqdm: bool = False) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.tqdm = tqdm

    def detect_span(self, dataset: Dataset, id2label: Dict[int, str], batch_size: int = 1, collator: Callable = None, do_score: bool = False) -> Dict[str, Any]:
        """
        Detect spans in a given dataset

        Return: list of results (each result is in a dictionary format). Dictionary keys are:
            - `tags`: detected binary I/B/O tags (List of str)
            - `ids`: corresponding input id in the vocabulary for each tag (List of int)
            - `tokens`: corresponding detokenized sub-words (List of str)
            - `texts`: words spans, grouped sub-words (List of str)
            - `document_ids`: inputs to the model (list of int)
            - `document`: input text to the model (str)
            - `ner_tags`: real NER I/B/O tags (List of str, optional)
        """
        self.id2label = id2label
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)

        input_list, pred_list, label_list, ner_tags = [], [], [], []

        for inputs in tqdm(dataloader, desc='Detecting spans', disable=not self.tqdm):
            metadata = inputs.pop('metadata')

            if do_score:
                label_list.append(inputs.pop('labels').tolist())

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs['logits']
                preds = logits.argmax(-1)
            
            input_list.append(inputs['input_ids'].tolist())
            pred_list.append(preds.tolist())
            ner_tags.append(metadata['ner_labels'])
        
        inputs = sum(input_list, [])
        preds = sum(pred_list, [])
        ner_tags = sum(ner_tags, [])

        res_list = [self.post_process(input, pred, ner_tag) for input, pred, ner_tag in zip(inputs, preds, ner_tags)]

        if do_score:
            labels = sum(label_list, [])
            scores = self.score(preds, labels)

        return dict(
            scores=scores,
            results=res_list
        )

    def post_process(self, inputs: List[int], preds: List[str], ner_tags: List[str] = None) -> Dict[str, List]:
        banned_ids = [self.tokenizer.convert_tokens_to_ids(x) for x in self.tokenizer.special_tokens_map.values()]

        res_dict = dict(tags=[], ids=[])
        current_tag, current_id = None, None

        if ner_tags is not None:
            res_dict['ner_tags'] = []
            current_ner_tag = None
        
        for i, pred in enumerate(preds):
            id = inputs[i]
            if id not in banned_ids:
                pred = self.id2label[pred]

                if pred.startswith('B'):
                    res_dict['tags'].append(current_tag) if current_tag is not None else None
                    res_dict['ids'].append(current_id) if current_id is not None else None
                    current_tag, current_id = [pred], [id]
                    if ner_tags is not None:
                        res_dict['ner_tags'].append(current_ner_tag) if current_ner_tag is not None else None
                        current_ner_tag = [ner_tags[i]]

                if pred.startswith('I'):
                    current_tag.append(pred) if current_tag is not None else None
                    current_id.append(id) if current_id is not None else None
                    if ner_tags is not None:
                        current_ner_tag.append(ner_tags[i]) if current_ner_tag is not None else None
        
        res_dict['tags'].append(current_tag) if current_tag is not None else None
        res_dict['ids'].append(current_id) if current_id is not None else None

        res_dict['tokens'] = [self.tokenizer.convert_ids_to_tokens(i) for i in res_dict['ids']]
        res_dict['texts'] = [''.join(toks).replace('Ġ', ' ').strip() for toks in res_dict['tokens']]

        res_dict['document_ids'] = inputs
        res_dict['document'] = ''.join(self.tokenizer.convert_ids_to_tokens(inputs, skip_special_tokens=True)).replace('Ġ', ' ').strip()
        if ner_tags is not None:
            res_dict['ner_tags'].append(current_ner_tag) if current_ner_tag is not None else None
    
        return res_dict
    
    def score(self, preds: List[List[int]], labels: List[List[int]]) -> List[str]:
        seqeval = load('seqeval')
        label_name_list = [self.id2label[i] for i in range(len(self.id2label))]

        true_predictions = [
            [label_name_list[p] for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(preds, labels)
        ]
        true_labels = [
            [label_name_list[l] for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(preds, labels)
        ]
        
        results = seqeval.compute(predictions=true_predictions, references=true_labels)

        return {'acc': results['overall_accuracy'], 'f1': results['overall_f1']}
