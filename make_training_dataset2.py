from transformers import XLMTokenizer, RobertaModel
from sentimentpl.models import SentimentPLModel
import os
import pandas as pd
import pickle
import numpy as np
import tqdm
import torch

device = torch.device("cuda")

tokenizer = XLMTokenizer.from_pretrained(
    "models/politicalBERT")
model = RobertaModel.from_pretrained(
    "models/politicalBERT", return_dict=True)
model = model.to(device)


def text2vec(text):
    encoded = tokenizer.encode(text, return_tensors='pt')
    return model(encoded)['pooler_output'].detach().numpy()[0]


def texts2vec(text):
    encoded = tokenizer(text, return_tensors='pt', padding=True)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    output = model(**encoded)
    return output['pooler_output'].detach().cpu().numpy(), output['last_hidden_state'].detach().cpu().numpy(),


sentiment_model = SentimentPLModel(from_pretrained='latest').cuda()

tags = {
    '#LGBTtoLudzie': {
        'positive': ['liberal', 'center'],
        # 'negative': ['center', 'center']
    },
    # '#LGBTtoIdeologia': {
    #     'positive': ['center', 'center'],
    #     'negative': ['center', 'center']
    # },
    '#IdeologiaLGBT': {
        # 'positive': ['center', 'center'],
        'negative': ['conservative', 'center']
    },
    '#StopAgresjiLGBT': {
        'positive': ['center', 'center'],
        'negative': ['center', 'center']
    },
    '#homofobia': {
        'positive': ['liberal', 'center'],
        # 'negative': ['center', 'center'],
    },
    # '#BabiesLifesMatter': {
    #     'positive': ['center', 'center'],
    #     'negative': ['center', 'center'],
    # }
    # '#piekłodzieci': {
    #     'positive': ['center', 'center'],
    #     'negative': ['center', 'center'],
    # }
    '#AborcjaBezGranic': {
        'positive': ['liberal', 'center'],
        'negative': ['liberal', 'center']
    },
    '#reżimPiS': {
        'positive': ['center', 'right'],
        'negative': ['center', 'right']
    },
    '#ulgaabolicyjna': {
        'positive': ['center', 'right'],
        'negative': ['center', 'right']
    },
    '#500plus': {
        'positive': ['center', 'left'],
        'negative': ['center', 'right']
    },
    '#renty': {
        'positive': ['center', 'center'],
        'negative': ['center', 'center']
    },
    '#emerytury': {
        # tutaj trudno powiedzieć bo raczej brak nacechowania
        'positive': ['center', 'center'],
        'negative': ['center', 'center']  #
    },
    '#płacaminimalna': {
        # 'positive': ['center', 'center'],
        'negative': ['center', 'left']
    },
    '#wynagrodzenia': {
        'positive': ['center', 'left'],
        'negative': ['center', 'left']
    },
    '#firmy': {
        'positive': ['center', 'center'],  # głównie apolityczne
        # 'negative': ['center', 'center'], # tutaj różnie
    },
    '#pracownicy': {
        'positive': ['center', 'center'],  # głównie apolityczne
        # 'negative': ['center', 'center'], # tutaj różnie
    },
    '#socjalizm': {
        'positive': ['center', 'right'],
        'negative': ['center', 'right']
    },
    '#własność': {
        'positive': ['conservative', 'center'],
        'negative': ['conservative', 'center']
    },
}


def label_to_val(tag, sentiment):
    map1 = {
        'conservative': 1,
        'center': 0,
        'liberal': -1
    }
    map2 = {
        'left': -1,
        'center': 0,
        'right': 1
    }
    sentiment = 'positive' if sentiment > 0 else 'negative'
    label = tags[tag].get(sentiment, None)

    if label:
        return map2[label[1]], map1[label[0]]
    return None


embeddings = []
sequences = []
sentiments = []
labels = []
texts = []

BATCH_SIZE = 8

validation_data = pd.read_csv('twitter_data/data/validation_data.csv')
val_ids = set(validation_data['tweet_id'])

for tag in tags:
    label = tags[tag]

    data = pd.read_csv(os.path.join(
        'twitter_data', 'data', tag + '.csv'), sep='\t')
    tweets = list(data['tweet'])
    ids = list(data['id'])
    tweets = [tweet.replace(tag, '') for tweet, id in zip(tweets, ids) if id not in val_ids]
    
    for i in tqdm.tqdm(range(0, len(tweets), BATCH_SIZE)):
        # embedding, seq = texts2vec(tweets[i:i+BATCH_SIZE])
        sentiment = sentiment_model(
            tweets[i:i+BATCH_SIZE]).detach().cpu().numpy()
        torch.cuda.empty_cache()
        # sequences.extend(seq)
        # embeddings.append(embedding)
        sentiments.append(sentiment)
        labels.append([tag for _ in range(sentiment.shape[0])])
        texts.append(tweets[i:i+BATCH_SIZE])


sentiments = np.concatenate(sentiments)
labels = np.concatenate(labels)
texts = np.concatenate(texts)

# dataset = {
#     'X': np.concatenate(embeddings, axis=0),
#     'y_regression': [label_to_val(label, sentiment) for label, sentiment in zip(labels, sentiments)],
# }

# idx = []
# y = []

# for i, val in enumerate(dataset['y_regression']):
#     if val is not None:
#         idx.append(i)
#         y.append(val)

# idx = np.array(idx)

# dataset = {
#     'X': dataset['X'][idx],
#     'y_regression': np.array(y),
# }


# with open('dataset2 fitted bert.pickle', 'wb') as f:
#     f.write(pickle.dumps(dataset))


labels = [label_to_val(label, sentiment) for label, sentiment in zip(labels, sentiments)]
texts = [text for text, label in zip(texts, labels) if label is not None]
labels = np.array([label for label in labels if label is not None])
pd.DataFrame({"texts":texts, "economic":labels[:, 0], "worldview":labels[:, 1]}).to_csv(
    'twitter_data/data/dataset2.csv')