import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from transformers import XLMTokenizer, RobertaModel


class Classifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        hidden_size = 128
        output_size = 5
        input_size = input_size
        self.dropout = nn.Dropout(0.5)
        self.dense1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        self.dense2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.dropout(x)
        d1 = self.dense1(x)
        x = self.activation(d1)
        x = self.dense2(x)
        return x, d1


class RegressionModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(input_size, 256),
            nn.ReLU(),
        )
        self.lin1 = nn.Linear(256, 2)
        self.lin2 = nn.Linear(256, 2)

        self.activation = nn.Tanh()
        self.activation2 = nn.Softplus()

    def forward(self, x):
        x = self.net(x)
        mean = self.lin1(x)
        mean = self.activation(mean)

        sigma = self.lin2(x)
        sigma = self.activation2(sigma)
        return mean, sigma


class RobertaRegressionModel(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.net = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
            # nn.Linear(768, 2),
            nn.Tanh()
        )
        self.device = device
        self.tokenizer = XLMTokenizer.from_pretrained(
            # "models/politicalBERT")
            # "models/politicalHerBERT")
            "allegro/herbert-klej-cased-tokenizer-v1")
        self.model = RobertaModel.from_pretrained(
            # "models/politicalBERT",
            "models/politicalHerBERT",
            # "allegro/herbert-klej-cased-v1",
            return_dict=True)

    def forward(self, x):
        encoded = self.tokenizer(x, return_tensors='pt', padding=True)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        x = self.model(**encoded)['pooler_output']
        x = self.net(x)
        return x


class RegressionMCDropoutModel(nn.Module):
    '''
    Model described at https://arxiv.org/abs/1506.02142
    '''

    def __init__(self, input_size, train_data_size, dropout_probability=0.5, weight_decay: float = 1e-5, lengthscale: float = 1e-2):
        super().__init__()

        self.lin1 = nn.Linear(input_size, 256)
        self.lin2 = nn.Linear(256, 2)

        self.activation1 = nn.ReLU()
        self.activation2 = nn.Tanh()

        self.probability = dropout_probability
        self.train_data_size = train_data_size
        self.weight_decay: float = weight_decay
        self.lengthscale: float = lengthscale
        self.lossFn = torch.nn.MSELoss()

        self.tau = self.lengthscale**2*(1-self.probability) / \
            (2*self.train_data_size*self.weight_decay)

        self.layers = [
            self.lin1,
            self.lin2
        ]

    def forward(self, x, deterministic=False):
        x = F.dropout(x, p=self.probability, training=not deterministic)
        x = self.lin1(x)
        x = self.activation1(x)
        x = F.dropout(x, p=self.probability, training=not deterministic)
        x = self.lin2(x)
        x = self.activation2(x)
        return x

    def prediction_log_likelihood(self, predictions: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Implement calculation of model likelihood

        Predictions are in the form of N x K x T matrix, where N is a number
        of samples, K number of classes and T number of samplings while true - N x K.
        """
        tau = ((1-self.probability)*(self.lengthscale**2)) / \
            (2*self.train_data_size*self.weight_decay)
        K = true.shape[1]
        return (torch.logsumexp(-0.5*tau*((((predictions.permute((-1, 0, 1))-true)**2).sum(-1))), dim=0)-math.log(predictions.shape[2])-0.5*math.log(2*3.14)-0.5 * math.log(tau**-1)).mean()

    def getLoss(self, pred, true):
        a = self.lossFn(pred, true)
        b = self.probability*self.lengthscale**2/(2*self.tau*self.train_data_size)*(sum((l.weight**2).sum(
        ) for l in self.layers)) + self.lengthscale**2/(2*self.tau*self.train_data_size)*(sum((l.bias**2).sum() for l in self.layers))
        return a+b

    def predict(self, x, num_samples=10):
        samples = torch.stack([self.forward(x)
                               for _ in range(num_samples)], dim=0)
        # return samples.mean(dim=0), torch.sqrt(samples.var(dim=0)+1/self.tau)
        return samples.mean(dim=0), torch.sqrt(samples.var(dim=0))


class PoliticalTextsEncoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        hidden_size = 128
        output_size = 64
        input_size = input_size
        self.dropout = nn.Dropout(0.5)
        self.dense1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        self.dense2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.dropout(x)
        d1 = self.dense1(x)
        x = self.activation(d1)
        x = self.dense2(x)
        nn.functional.normalize(x, p=2, dim=1)
        return x


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(64, 64)
        self.activation = nn.ReLU()
        self.dense2 = nn.Linear(64, 2)
        self.dense3 = nn.Linear(2, 64)
        self.activation = nn.ReLU()
        self.dense4 = nn.Linear(64, 64)

    def forward(self, x):
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dense2(x)
        middle = x
        x = self.dense3(x)
        x = self.activation(x)
        x = self.dense4(x)
        return x, middle


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y, dtype=np.int64)
        assert self.X.shape[0] == self.y.shape[0]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i]
