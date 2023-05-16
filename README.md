# Zero NLP

## Introduction
This repository gives you access to trained LLMs that can directly be used for any zero-shot NLP task.

## Installation
Please install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) (miniconda recommended) for the environment manager.<br>

Once conda is installed, you can create and activate the environment using the following commads.

```
conda env create -f zero-nlp.yml
conda activate zero-nlp
```

## 1. Zero-shot sentence classification

### Introduction
**Sentence classification** is the most basic NLP task. It consists of classifying one sentence (i.e a text) into a target class. It is necessary for most applications that requires NLU components.<br>
The most used models for this task are encoder-only architecture based on BERT and the most often used is [Roberta](https://huggingface.co/docs/transformers/model_doc/roberta). However, Roberta needs to be finetuned on every specific NLU task making it unpractical and impossible to use for domains where data is not largely available.<br>

We propose a unified model, based on Google's [T5](https://github.com/google-research/text-to-text-transfer-transformer) architecture that achieves extremly good results on any classification task without having been trained on it. This model can directly be used for:
- Topic classification
- Intent recognition
- Boolean question-answering
- Sentiment analysis
- and more...

### Usage
The base model is hosted on HuggingFace's hub [here](https://huggingface.co/AntoineBlanot/flan-t5-xxl-classif-3way). Please dowmload it from there.<br>
The additional trained weights are in the model folder.

#### Loading model and tokenizer
For this example, we will use a T5-based model (~5B parameters) and load it with the [peft](https://github.com/huggingface/peft) library.
```
model_path = 'model/t5-xxl-nli'

from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoConfig
from model.modeling import T5ForClassification

peft_config = PeftConfig.from_pretrained(model_path)
base_config = AutoConfig.from_pretrained(model_path)

model = T5ForClassification.from_pretrained(pretrained_model_name_or_path=peft_config.base_model_name_or_path, **base_config.to_diff_dict(), load_in_8bit=True, device_map={'': 0})
model = PeftModel.from_pretrained(model, model_path, device_map={'': 0})
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_path)
```

#### Creating data
We will instantiate a Zero-Shot Boolean Question-Answering dataset. For using it on other tasks, please look at the different tasks available. You can also create your own zero-shot task.
```
task_name = 'boolqa'
data = dict(question='Do you like being a child?', answer='I hate being a child', candidate_labels=['yes', 'no'])

from zero import ZeroDataset
dataset = ZeroDataset(task_name)
dataset.from_dict(data)
```

#### Prediction
The model `true_id` is 0. `true_id` corresponds to the output neuron we are interested in (other neurons will be ignored)
```
from zero import ZeroClassifier

classifier = ZeroClassifier(model, tokenizer, true_id=0)
res = classifier.classify(dataset, batch_size=1, threshold=0.8)

pred_text = data['candidate_labels'][res['preds'][0]]
print('Prediction: {} ({})'.format(pred_text, res['preds'][0]))
# Prediction: no (1)
print('Probabilities: {}'.format(res['probs'][0]))
# Probabilities: [0.043386060744524, 0.9566139578819275]
```
