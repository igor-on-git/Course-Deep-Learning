from import_libs import *


class RNNModel(nn.Module):

    def __init__(self, type, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)  # Token2Embeddings
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
        initrange = 0.05 * 2
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
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
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
    params['name'] = 'LSTM'
    params['data'] = './data'
    params['train_model'] = 1
    params['continue_training'] = 0
    params['type'] = 'GRU'
    params['emsize'] = 200
    params['nhid'] = 200
    params['nlayers'] = 2
    params['clip'] = 0.25
    params['epochs'] = 10
    params['batch_size'] = 20
    params['seq_en'] = 20  #35
    params['dropout'] = 0.5
    params['save'] = './output/model_test'
    params['opt'] = 'SGD'  # 'SGD, Adam, RMSprop, Momentum'
    params['lr'] = 20

    if type == 'LSTM':
        params['name'] = 'LSTM'
        params['type'] = 'LSTM'
        params['dropout'] = 0
        params['epochs'] = 10
        params['lr'] = 10
        params['annealing_gamma'] = 2 / 3
        params['annealing_step'] = 1
        params['annealing_start'] = 4
        params['save'] = './output/model_lstm'
    if type == 'LSTM+Drop':
        params['name'] = 'LSTM+Drop'
        params['type'] = 'LSTM'
        params['dropout'] = 0.3
        params['epochs'] = 20
        params['lr'] = 20
        params['annealing_gamma'] = 15 / 20
        params['annealing_step'] = 1
        params['annealing_start'] = 9
        params['save'] = './output/model_lstm_drop'
    if type == 'GRU':
        params['name'] = 'GRU'
        params['type'] = 'GRU'
        params['dropout'] = 0
        params['epochs'] = 10
        params['lr'] = 5
        params['annealing_gamma'] = 2 / 3
        params['annealing_step'] = 1
        params['annealing_start'] = 4
        params['save'] = './output/model_gru'
    if type == 'GRU+Drop':
        params['name'] = 'GRU+Drop'
        params['type'] = 'GRU'
        params['dropout'] = 0.325
        params['epochs'] = 35
        params['lr'] = 13
        params['annealing_gamma'] = 7 / 8
        params['annealing_step'] = 1
        params['annealing_start'] = 8

        params['save'] = './output/model_gru_drop'

    return params


def load_model(file_path):
    with open(file_path, 'rb') as f:
        return torch.load(f)


def init_train_perf():
    train_perf = {}
    train_perf['train_loss'] = []
    train_perf['valid_loss'] = []
    train_perf['test_loss'] = []
    train_perf['train_ppl'] = []
    train_perf['valid_ppl'] = []
    train_perf['test_ppl'] = []
    train_perf['lr'] = 0

    return train_perf


def plot_all_results(name_list):
    fig, ax = plt.subplots(2, 2, sharex=False, sharey=False)
    fig.set_figheight(10)
    fig.set_figwidth(25)

    for model_ind, name in enumerate(name_list):
        params = param_selector(name)
        train_perf = np.load(params['save'] + '_train_perf.npy', allow_pickle=True).item()

        print(
            '| model: {:10}| train loss {:5.2f} | train ppl {:8.2f} | valid loss {:5.2f} | valid ppl {:8.2f} |  test loss {:5.2f} | test ppl {:8.2f} |'
            .format(name, train_perf['train_loss'][-1], train_perf['train_ppl'][-1], train_perf['valid_loss'][-1],
                    train_perf['valid_ppl'][-1], train_perf['test_loss'], train_perf['test_ppl']))

        ax[model_ind // 2][model_ind % 2].plot(range(len(train_perf['train_ppl'])), train_perf['train_ppl'],
                                               label='Train Perplexity')
        ax[model_ind // 2][model_ind % 2].plot(range(len(train_perf['valid_ppl'])), train_perf['valid_ppl'],
                                               label='Valid Perplexity')
        #ax[model_ind//2][model_ind % 2].title.set_text(params['name'] + ' - LR ' + str(train_perf['lr'],'.2f'))
        ax[model_ind // 2][model_ind % 2].title.set_text('{} - LR {:02.2f}'.format(params['name'], train_perf['lr']))
        ax[model_ind // 2][model_ind % 2].legend()
        ax[model_ind // 2][model_ind % 2].grid(which='both', axis='both')
        ax[model_ind // 2][model_ind % 2].set_ylim([0, 350])

    plt.savefig('./output/' + 'results.png')
    plt.show()


def plot_results(params, train_perf):
    plt.figure()
    train_perf = np.load(params['save'] + '_train_perf.npy', allow_pickle=True).item()

    print(
        '| train loss {:5.2f} | train ppl {:8.2f} | valid loss {:5.2f} | valid ppl {:8.2f} |  test loss {:5.2f} | test ppl {:8.2f} |'
        .format(train_perf['train_loss'][-1], train_perf['train_ppl'][-1], train_perf['valid_loss'][-1],
                train_perf['valid_ppl'][-1], train_perf['test_loss'], train_perf['test_ppl']))
    plt.plot(range(len(train_perf['train_ppl'])), train_perf['train_ppl'], label='Train Perplexity')
    plt.plot(range(len(train_perf['valid_ppl'])), train_perf['valid_ppl'], label='Valid Perplexity')
    plt.title('{} - LR {:02.2f}'.format(params['name'], train_perf['lr']))
    plt.legend()
    plt.xticks(minor=True)
    plt.yticks(minor=True)
    plt.grid(which='both', axis='both')
    plt.savefig(params['save'] + '_results.png')
    plt.show()
