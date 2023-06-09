import torch
from torch.utils.data import Dataset, DataLoader
import datasets
from evaluate import load
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from task import IntentRecognitionForDialog, BoolQA, SentimentAnalysisForDialog

TASK_MAPPING = {'intent': IntentRecognitionForDialog, 'boolqa': BoolQA, 'sentiment': SentimentAnalysisForDialog}


class ZeroDataset(Dataset):

    def __init__(self, task_name: str) -> None:
        super().__init__()
        self.task_name = task_name
        self.task = TASK_MAPPING[self.task_name]()
    
    def from_json(self, file: str):
        self.data = datasets.load_dataset('json', data_files=file)['train']

    def from_dict(self, data_dict: str):
        self.data = datasets.Dataset.from_dict({k: [v] for k, v in data_dict.items()})

    def column_names(self):
        return self.data.column_names
    
    def __len__(self):
        return len(self.data)

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


class ZeroClassifier():

    def __init__(self, model: torch.nn.Module, tokenizer, true_id: int, tqdm: bool = False) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.true_id = true_id
        self.tqdm = tqdm

    def classify(self, dataset: ZeroDataset, batch_size: int = 1, threshold: float = 0.8):
        do_score = 'label' in dataset.column_names()
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=ZeroCollator(self.tokenizer, with_labels=do_score))

        output_list, label_list, group_list = [], [], []
        for inputs in tqdm(dataloader, desc='Classifying', disable=not self.tqdm):
            labels = inputs.pop('labels') if do_score else None
            metadata = inputs.pop('metadata')
            outputs = self.model(**inputs)
            true_outputs = outputs['logits'][:, self.true_id].float()

            output_list.append(true_outputs)
            label_list.append(labels)
            group_list.append(metadata['group'])

        group_list = sum(group_list, [])
        group_count = [group_list.count(g) for g in set(group_list)]
        outputs = torch.cat(output_list)

        preds, probs = self.predict(outputs, group_count, threshold=threshold)

        if do_score:
            labels = [x[0].item() for x in torch.split(torch.cat(label_list), group_count)]
            scores = self.score(labels, preds)
            return dict(
                scores=scores if scores is not None else None,
                preds=preds,
                probs=[p.tolist() for p in probs]
            )
        
        return dict(
            preds=preds,
            probs=[p.tolist() for p in probs]
        )

    def predict(self, outputs, group_count, threshold: float):
        grouped_outputs = torch.split(outputs, group_count)

        probs = [x.softmax(0) for x in grouped_outputs]
        preds = [torch.argmax(x).item() if torch.max(x) > threshold else -1 for x in probs]

        return preds, probs
    
    def score(self, labels, preds):
        accuracy_metric = load("accuracy")
        recall_metric = load("recall")
        precision_metric = load("precision")
        f1_metric = load("f1")

        conf_matrix = confusion_matrix(y_true=labels, y_pred=preds)
        print(conf_matrix)

        acc = accuracy_metric.compute(predictions=preds, references=labels)
        rec = recall_metric.compute(predictions=preds, references=labels, average='weighted')
        prec = precision_metric.compute(predictions=preds, references=labels, average='weighted')
        f1 = f1_metric.compute(predictions=preds, references=labels, average='weighted')

        return {**acc, **rec, **prec, **f1}


