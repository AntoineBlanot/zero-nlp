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
        self.data = datasets.load_dataset('json', data_files=file)['train'].select(range(100))

    def from_dict(self, data_dict: str):
        self.data = datasets.Dataset.from_dict({k: [v] for k, v in data_dict.items()})

    def column_names(self):
        return self.data.column_names
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class TokenCollatorBERT():
    """
    Data collator for `BERT` models for `token classification` task

    Required features in the dataset:
        - `tokens`: pre-tokenized (split into words) text (list of str)
    
    Optional features in the dataset:
        - `bin_tags`: target tags for each pre-tokenized token (list of int)
    """

    def __init__(self, tokenizer, label2id: Dict[str, int]) -> None:
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.preprocess_fn = self.tokenize_and_align_labels

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batched_tokens = [x['tokens'] for x in features]
        examples_dict = dict(tokens=batched_tokens)

        has_labels = 'bin_tags' in features[0]      # check if bin_tag field is in the first feature
        if has_labels:
            batched_str_tags = [x['bin_tags'] for x in features]
            batched_tags = [[self.label2id[tag] for tag in tag_list] for tag_list in batched_str_tags]
            examples_dict['tags'] = batched_tags

        inputs = self.preprocess_fn(examples=examples_dict)

        return inputs
    
    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(examples['tokens'], is_split_into_words=True, padding=True, truncation=True, pad_to_multiple_of=8, return_tensors='pt')

        if 'tags' in examples.keys():
            labels = []
            for i, label in enumerate(examples['tags']):
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

        return tokenized_inputs

class SpanDetector():
    """
    Wrapper class for `Span Detection` task (i.e. binary token classification)
    """

    def __init__(self, model: torch.nn.Module, tokenizer, id2label: Dict[str, str], tqdm: bool = False) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.id2label = id2label
        self.tqdm = tqdm

    def detect_span(self, dataset: Dataset, batch_size: int = 1, collator: Callable = None, do_score: bool = False) -> List[Dict[str, List]]:
        """
        Detect spans in a given dataset

        Return: list of results (each result is in a dictionary format). Dictionary keys are:
            - `tags`: detected I/B/O tags (List of str)
            - `ids`: corresponding input id in the vocabulary for each tag (List of int)
            - `tokens`: corresponding detokenized sub-words (List of str)
            - `texts`: words spans, grouped sub-words (List of str)
            - `document_ids`: inputs to the model (list of int)
            - `document`: input text to the model (str)
        """
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)

        input_list, pred_list, label_list = [], [], []

        for inputs in tqdm(dataloader, desc='Detecting spans', disable=not self.tqdm):
            if do_score:
                label_list.append(inputs.pop('labels').tolist())

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs['logits']
                preds = logits.argmax(-1)
            
            input_list.append(inputs.input_ids.tolist())
            pred_list.append(preds.tolist())
        
        inputs = sum(input_list, [])
        preds = sum(pred_list, [])

        res_list = [self.post_process(input, pred) for input, pred in zip(inputs, preds)]

        if do_score:
            labels = sum(label_list, [])
            scores = self.score(preds, labels)
            print(scores)

        return res_list

    def post_process(self, inputs: List[int], preds: List[str]) -> Dict[str, List]:
        res_dict = dict(tags=[], ids=[])
        banned_ids = [self.tokenizer.convert_tokens_to_ids(x) for x in self.tokenizer.special_tokens_map.values()]
        
        current_tag, current_id = None, None

        for i, pred in enumerate(preds):
            id = inputs[i]
            if id not in banned_ids:
                pred = self.id2label[pred]
                if pred.startswith('B'):
                    res_dict['tags'].append(current_tag) if current_tag is not None else None
                    res_dict['ids'].append(current_id) if current_id is not None else None
                    current_tag, current_id = [pred], [id]
                if pred.startswith('I'):
                    current_tag.append(pred) if current_tag is not None else None
                    current_id.append(id) if current_id is not None else None
        
        res_dict['tags'].append(current_tag) if current_tag is not None else None
        res_dict['ids'].append(current_id) if current_id is not None else None

        res_dict['tokens'] = [self.tokenizer.convert_ids_to_tokens(i) for i in res_dict['ids']]
        res_dict['texts'] = [''.join(toks).replace('Ġ', ' ').strip() for toks in res_dict['tokens']]

        res_dict['document_ids'] = inputs
        res_dict['document'] = ''.join(self.tokenizer.convert_ids_to_tokens(inputs, skip_special_tokens=True)).replace('Ġ', ' ').strip()
    
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

        return {'accuracy': results['overall_accuracy'], 'f1': results['overall_f1']}
