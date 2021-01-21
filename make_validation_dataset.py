from transformers import XLMTokenizer, RobertaModel
import os
import pandas as pd
import pickle
import numpy as np
import tqdm
import torch

device = torch.device("cuda")

tokenizer = XLMTokenizer.from_pretrained(
    "allegro/herbert-klej-cased-tokenizer-v1")
model = RobertaModel.from_pretrained(
    "allegro/herbert-klej-cased-v1", return_dict=True)
model = model.to(device)


def text2vec(text):
    encoded = tokenizer.encode(text, return_tensors='pt')
    return model(encoded)['pooler_output'].detach().numpy()[0]


def texts2vec(text):
    encoded = tokenizer(text, return_tensors='pt', padding=True)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    output = model(**encoded)
    return output['pooler_output'].detach().cpu().numpy(), output['last_hidden_state'].detach().cpu().numpy(), 

data = pd.read_csv('twitter_data/data/validation_data.csv')

embeddings = []
sequences = []
labels = []

BATCH_SIZE = 8

tweets = list(data['tweet'])

for i in tqdm.tqdm(range(0, len(tweets), BATCH_SIZE)):
    embedding, seq = texts2vec(tweets[i:i+BATCH_SIZE])
    # torch.cuda.empty_cache()
    embeddings.append(embedding)

labels = np.array([[row['economic'], row['worldview']] for i, row in data.iterrows()])

dataset = {
    'X': np.concatenate(embeddings, axis=0),
    'y': labels,
}

with open('validation_dataset.pickle', 'wb') as f:
    f.write(pickle.dumps(dataset))
