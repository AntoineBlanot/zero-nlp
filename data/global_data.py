from datasets import load_dataset


def tweet_eval():
    data = load_dataset('tweet_eval', name='sentiment', split='validation')

    id2label = {'0': 'negative', '1': 'neutral', '2': 'positive'}
    candidates_labels = ['negative', 'neutral', 'positive']

    data = data.add_column(
        'label_name',
        [id2label[str(l)] for l in data['label']]
    )
    data = data.add_column(
        'candidate_labels',
        [candidates_labels] * len(data)
    )
    data.to_json('tweet_eval.json')


def tweet_sentiment_extraction():
    data = load_dataset('mteb/tweet_sentiment_extraction', split='test')

    id2label = {'0': 'negative', '1': 'neutral', '2': 'positive'}
    candidates_labels = ['negative', 'neutral', 'positive']

    data = data.add_column(
        'label_name',
        [id2label[str(l)] for l in data['label']]
    )
    data = data.add_column(
        'candidate_labels',
        [candidates_labels] * len(data)
    )
    data.to_json('tweet_sentiment_extraction.json')

def sst2():
    data = load_dataset('glue', name='sst2', split='validation')

    id2label = {'0': 'negative', '1': 'positive'}
    candidates_labels = ['negative', 'positive']

    data = data.add_column(
        'label_name',
        [id2label[str(l)] for l in data['label']]
    )
    data = data.add_column(
        'candidate_labels',
        [candidates_labels] * len(data)
    )
    data.to_json('sst2.json')

def boolq():
    label2id = {False: 0, True: 1}
    candidates_labels = ['false', 'true']

    data = load_dataset('boolq', split='validation')
    data = data.add_column(
        'label',
        [label2id[l] for l in data['answer']]
    )
    data = data.add_column(
        'candidate_labels',
        [candidates_labels] * len(data)
    )
    data.to_json('boolq.json')



boolq()
