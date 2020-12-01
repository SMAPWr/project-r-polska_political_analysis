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


parties = [p.name for p in os.scandir('data') if os.path.isdir(p)]

party_to_label = {
    p: i for i, p in enumerate(parties)
}

parties_on_compass = {'pis': [-0.8, 0.7], 'po': [0.1, 0.0], 'lewica': [-0.5, -0.5],
                      'konfederacja': [0.5, 0.5], 'psl': [0.3, 0.1]}

parties_on_compass = {party_to_label[k]
    : v for k, v in parties_on_compass.items()}

users = {
    p: [user.path for user in os.scandir(os.path.join('data', p)) if os.path.isfile(user)] for p in parties
}


embeddings = []
sequences = []
labels = []

BATCH_SIZE = 8

for p in parties:
    label = party_to_label[p]
    for user in users[p]:
        data = pd.read_csv(user, sep='\t')
        tweets = list(data['tweet'])
        for i in tqdm.tqdm(range(0, len(tweets), BATCH_SIZE)):
            embedding, seq = texts2vec(tweets[i:i+BATCH_SIZE])
            torch.cuda.empty_cache()
            # sequences.extend(seq)
            embeddings.append(embedding)
            labels.append([label for _ in range(embedding.shape[0])])
        # for t in tqdm.tqdm(tweets):
        #     embedding = text2vec(t)
        #     embeddings.append(embedding)
        #     labels.append(label)

labels = np.concatenate(labels).astype(np.int32)

dataset = {
    'X': np.concatenate(embeddings, axis=0),
    # 'X_seq' : sequences,
    'y': labels,
    'y_regression': np.array([parties_on_compass[label] for label in labels]),
    'mapping': {v: k for k, v in party_to_label.items()}
}

with open('dataset.pickle', 'wb') as f:
    f.write(pickle.dumps(dataset))
