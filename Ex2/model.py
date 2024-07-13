import torch.nn as nn
class RNNModel(nn.Module):

    def __init__(self, type, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp) # Token2Embeddings
        self.type = type
        if self.type == 'LSTM':
            self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)  # (seq_len, batch_size, emb_size)
        elif self.type == 'GRU':
            self.rnn = nn.GRU(ninp, nhid, nlayers, dropout=dropout)  # (seq_len, batch_size, emb_size)
        else:
            print('Unsupported RNN type')
            exit()
        self.decoder = nn.Linear(nhid, ntoken)
        self.init_weights()
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.05
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        # input size(bptt, bsz)
        emb = self.drop(self.encoder(input))
        # emb size(bptt, bsz, embsize)
        # hid size(layers, bsz, nhid)
        output, hidden = self.rnn(emb, hidden)
        # output size(bptt, bsz, nhid)
        output = self.drop(output)
        # decoder: nhid -> ntoken
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded, hidden

    def init_hidden(self, bsz):
        # LSTM h and c
        weight = next(self.parameters()).data
        if self.type == 'LSTM':
            return weight.new_zeros(self.nlayers, bsz, self.nhid), weight.new_zeros(self.nlayers, bsz, self.nhid)
        elif self.type == 'GRU':
            return weight.new_zeros(self.nlayers, bsz, self.nhid)
        else:
            print('Unsupported RNN type')
            exit()

    def hidden_to_device(self, hidden, device):
        if self.type == 'LSTM':
            hidden = [hidden[i].to(device) for i in range(len(hidden))]
        elif self.type == 'GRU':
            hidden = hidden.to(device)
        else:
            print('Unsupported RNN type')
            exit()
        return hidden

    def repackage_hidden(self, h):
        # detach
        if self.type == 'LSTM':
            return tuple(v.clone().detach() for v in h)
        elif self.type == 'GRU':
            return h.clone().detach()
        else:
            print('Unsupported RNN type')
            exit()


def param_selector(type):

    params = {}
    params['data'] = './data'
    params['checkpoint'] = ''
    params['type'] = 'GRU'
    params['emsize'] = 200
    params['nhid'] = 200
    params['nlayers'] = 2
    params['lr'] = 20
    params['clip'] = 0.25
    params['epochs'] = 5
    params['batch_size'] = 20
    params['bptt'] = 35
    params['dropout'] = 0.5
    params['save'] = './output/model_test.pt'
    params['opt'] = 'SGD'  # 'SGD, Adam, RMSprop, Momentum'

    if type == 'LSTM':
        params['type'] = 'LSTM'
        params['dropout'] = 0
        params['save'] = './output/model_lstm.pt'
    if type == 'LSTM+Drop':
        params['type'] = 'LSTM'
        params['dropout'] = 0.5
        params['save'] = './output/model_lstm_drop.pt'
    if type == 'GRU':
        params['type'] = 'GRU'
        params['dropout'] = 0
        params['save'] = './output/model_gru.pt'
    if type == 'GRU+Drop':
        params['type'] = 'GRU'
        params['dropout'] = 0.5
        params['save'] = './output/model_gru_drop.pt'

    return params