import flask
from models import RegressionMCDropoutModel, RobertaRegressionModel
import torch
from flask import Flask
from transformers import XLMTokenizer, RobertaModel
from flask_cors import CORS, cross_origin
import twint
import tempfile
import pandas as pd
import numpy as np
from flask import jsonify

device = torch.device("cuda")

# net = RegressionMCDropoutModel(
#     768, train_data_size=12755, lengthscale=1).to(device)
net = RobertaRegressionModel(device)
net.load_state_dict(torch.load('fitted_bert_epoch_15_mae_0_3427056074142456'))

# tokenizer = XLMTokenizer.from_pretrained(
#     "allegro/herbert-klej-cased-tokenizer-v1")
# model = RobertaModel.from_pretrained(
#     "allegro/herbert-klej-cased-v1", return_dict=True)

net = net.to(device)
net.train(False)

TWEETS_LIMIT = 500
BATCH_SIZE = 5

def text2vec(text):
    encoded = tokenizer.encode(text, return_tensors='pt').to(device)
    return model(encoded)['pooler_output'].detach()


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/', methods=['POST'])
@cross_origin()
def predict():
    data = flask.request.json
    # embedding = text2vec(data['text'])
    # prediction = net.predict(embedding, num_samples=200)[0]
    prediction = net(data['text'])
    [pred] = prediction.detach().cpu().numpy()

    return {
        "economic": float(pred[0]),
        "worldview": float(pred[1])
    }


# def texts2vec(text):
#     encoded = tokenizer(text, return_tensors='pt', padding=True)
#     encoded = {k: v.to(device) for k, v in encoded.items()}
#     output = model(**encoded)
#     return output['pooler_output'].detach()

@app.route('/hashtag', methods=['GET'])
@cross_origin()
def hashtag():
    hashtag = flask.request.args['hashtag']

    with tempfile.TemporaryDirectory() as tmpdirname:
        c = twint.Config()
        c.Search = f'(#{hashtag}) lang:pl'
        c.Store_csv = True
        c.Tabs = True
        c.Output = tmpdirname + '/tweets.csv'
        c.Limit = TWEETS_LIMIT
        c.Hide_output = True
        twint.run.Search(c)

        data = pd.read_csv(tmpdirname + '/tweets.csv', sep='\t')

        tweets = list(data['tweet'])
        predictions = []

        for i in range(0, len(tweets), BATCH_SIZE):
            predictions.append(net(tweets[i:i+BATCH_SIZE]).detach().cpu().numpy())

        predictions = np.concatenate(predictions)

        return jsonify({
            "tweets": [
                {
                    "text": tweet,
                    "economic": float(pred[0]),
                    "worldview": float(pred[1])
                }
                for (tweet, pred) in zip(tweets, predictions)
            ],
            "economic": float(predictions[:, 0].mean()),
            "worldview": float(predictions[:, 1].mean()),
        })


@app.route('/twitter_user', methods=['GET'])
@cross_origin()
def twitter_user():
    username = flask.request.args['user']

    with tempfile.TemporaryDirectory() as tmpdirname:
        print(f'saving data to {tmpdirname}')
        c = twint.Config()
        c.Search = f'lang:pl'
        c.Username = username
        c.Store_csv = True
        c.Tabs = True
        c.Output = tmpdirname + '/tweets.csv'
        c.Limit = TWEETS_LIMIT
        c.Hide_output = True
        twint.run.Search(c)

        data = pd.read_csv(tmpdirname + '/tweets.csv', sep='\t')

        tweets = list(data['tweet'])
        predictions = []

        for i in range(0, len(tweets), BATCH_SIZE):
            predictions.append(net(tweets[i:i+BATCH_SIZE]).detach().cpu().numpy())

        predictions = np.concatenate(predictions)

        return jsonify({
            "tweets": [
                {
                    "text": tweet,
                    "economic": float(pred[0]),
                    "worldview": float(pred[1])
                }
                for (tweet, pred) in zip(tweets, predictions)
            ],
            "economic": float(predictions[:, 0].mean()),
            "worldview": float(predictions[:, 1].mean()),
        })


# @app.route('/hashtag', methods=['GET'])
# @cross_origin()
# def hashtag():
#     hashtag = flask.request.args['hashtag']

#     with tempfile.TemporaryDirectory() as tmpdirname:
#         c = twint.Config()
#         c.Search = f'(#{hashtag}) lang:pl'
#         c.Store_csv = True
#         c.Tabs = True
#         c.Output = tmpdirname + '/tweets.csv'
#         c.Limit = TWEETS_LIMIT
#         c.Hide_output = True
#         twint.run.Search(c)

#         data = pd.read_csv(tmpdirname + '/tweets.csv', sep='\t')

#         embeddings = []

#         BATCH_SIZE = 8

#         tweets = list(data['tweet'])

#         for i in range(0, len(tweets), BATCH_SIZE):
#             embedding = texts2vec(tweets[i:i+BATCH_SIZE])
#             embeddings.append(embedding)

#         predictions = net.predict(torch.cat(embeddings, dim=0), num_samples=200)[0]
#         predictions = predictions.detach().cpu().numpy()

#         return jsonify({
#             "tweets": [
#                 {
#                     "text": tweet,
#                     "economic": float(pred[0]),
#                     "worldview": float(pred[1])
#                 }
#                 for (tweet, pred) in zip(tweets, predictions)
#             ],
#             "economic": float(predictions[:, 0].mean()),
#             "worldview": float(predictions[:, 1].mean()),
#         })


# @app.route('/twitter_user', methods=['GET'])
# @cross_origin()
# def twitter_user():
#     username = flask.request.args['user']

#     with tempfile.TemporaryDirectory() as tmpdirname:
#         print(f'saving data to {tmpdirname}')
#         c = twint.Config()
#         c.Search = f'lang:pl'
#         c.Username = username
#         c.Store_csv = True
#         c.Tabs = True
#         c.Output = tmpdirname + '/tweets.csv'
#         c.Limit = TWEETS_LIMIT
#         c.Hide_output = True
#         twint.run.Search(c)

#         data = pd.read_csv(tmpdirname + '/tweets.csv', sep='\t')

#         embeddings = []

#         BATCH_SIZE = 8

#         tweets = list(data['tweet'])

#         for i in range(0, len(tweets), BATCH_SIZE):
#             embedding = texts2vec(tweets[i:i+BATCH_SIZE])
#             embeddings.append(embedding)

#         predictions = net.predict(torch.cat(embeddings, dim=0), num_samples=200)[0]
#         predictions = predictions.detach().cpu().numpy()

#         return jsonify({
#             "tweets": [
#                 {
#                     "text": tweet,
#                     "economic": float(pred[0]),
#                     "worldview": float(pred[1])
#                 }
#                 for (tweet, pred) in zip(tweets, predictions)
#             ],
#             "economic": float(predictions[:, 0].mean()),
#             "worldview": float(predictions[:, 1].mean()),
#         })
