import torch.nn as nn


class LSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_size,
        padding_idx,
        hidden_size,
        output_size,
        num_layers=1,
        emb_weights=None,
    ):
        super().__init__()
        if emb_weights != None:
            self.emb = nn.Embedding.from_pretrained(
                emb_weights, padding_idx=padding_idx
            )
        else:
            self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.lstm = nn.LSTM(emb_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0=None):
        x = self.emb(x)
        x, h = self.lstm(x, h0)
        x = x[:, -1, :]
        logits = self.fc(x)
        return logits


class RNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_size,
        padding_idx,
        hidden_size,
        output_size,
        num_layers=1,
        emb_weights=None,
    ):
        super().__init__()
        if emb_weights != None:
            self.emb = nn.Embedding.from_pretrained(
                emb_weights, padding_idx=padding_idx
            )
        else:
            self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.rnn = nn.RNN(emb_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0=None):
        x = self.emb(x)
        x, h = self.rnn(x, h0)
        x = x[:, -1, :]
        logits = self.fc(x)
        return logits
