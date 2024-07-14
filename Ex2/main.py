# code referenced from https://github.com/hjc18/language_modeling_lstm/blob/master/main.py

import time
import math
import torch
import torch.nn as nn
from corpus import *
from model import *
from torch import optim
import numpy as np

torch.manual_seed(1111)

name_list = ['LSTM', 'LSTM+Drop', 'GRU', 'GRU+Drop']
params = param_selector(name_list[3])
params['train_model'] = 1

# plot all results
#plot_all_results(name_list)

# Load data
corpus = Corpus(params['data'])

# select device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# divide data into batches
eval_batch_size = 10
train_data = batchify(corpus.train, params['batch_size']) # size(total_len//bsz, bsz)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

# Build the model
interval = 100 # interval to report
ntokens = len(corpus.dictionary) # 10000
rnn_model = []
if params['continue_training']:
    try:
        rnn_model = load_model(params['save']+'.pt')
        train_perf = np.load(params['save'] + '_train_perf.npy', allow_pickle=True).item()
        params['lr'] = train_perf['lr']
    except:
        print('saved model not found')

if rnn_model == []:
    rnn_model = RNNModel(params['type'], ntokens, params['emsize'], params['nhid'], params['nlayers'], params['dropout'])
    train_perf = init_train_perf()

print(rnn_model)

# loss criterion
criterion = nn.CrossEntropyLoss()

# evaluation code
def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    rnn_model.to(device)
    with torch.no_grad():
        rnn_model.eval()
        total_loss = 0
        ntokens = len(corpus.dictionary)
        hidden = rnn_model.init_hidden(eval_batch_size) #hidden size(nlayers, bsz, hdsize)
        hidden = rnn_model.hidden_to_device(hidden, device)
        for i in range(0, data_source.size(0) - 1, params['bptt']):# iterate over every timestep
            data, targets = get_batch(data_source, params['bptt'], i)
            data, targets = data.to(device), targets.to(device)
            output, hidden = rnn_model(data, hidden)
            # model input and output
            # inputdata size(bptt, bsz), and size(bptt, bsz, embsize) after embedding
            # output size(bptt*bsz, ntoken)
            total_loss += len(data) * criterion(output, targets).data
            hidden = rnn_model.repackage_hidden(hidden)

        rnn_model.to('cpu')
        return total_loss / len(data_source)

# training code
def train():
    # choose a optimizer
    rnn_model.to(device)
    rnn_model.train()
    total_loss = 0
    report_loss = 0
    start_time = time.time()
    hidden = rnn_model.init_hidden(params['batch_size'])
    hidden = rnn_model.hidden_to_device(hidden, device)
    # train_data size(batchcnt, bsz)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, params['bptt'])):
        data, targets = get_batch(train_data, params['bptt'], i)
        data, targets = data.to(device), targets.to(device)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = rnn_model.repackage_hidden(hidden)
        output, hidden = rnn_model(data, hidden)
        loss = criterion(output, targets)
        opt.zero_grad()
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(rnn_model.parameters(), params['clip'])
        opt.step()

        total_loss += loss.data
        report_loss += loss.data

        if batch % interval == 0 and batch > 0:
            cur_loss = report_loss / interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:4.2f}'.format(
                epoch, batch, len(train_data) // params['bptt'], lr_scheduler.get_last_lr()[0],
                elapsed * 1000 / interval, cur_loss, math.exp(cur_loss)))
            report_loss = 0
            start_time = time.time()

    total_loss /= len(train_data) / params['bptt']
    train_perf['train_loss'].extend([total_loss])
    train_perf['train_ppl'].extend([math.exp(total_loss)])
    train_perf['lr'] = lr_scheduler.get_last_lr()[0]
    #lr_scheduler.step()
    rnn_model.to('cpu')

# select learning rate and optimizer
lr = params['lr']
opt = torch.optim.SGD(rnn_model.parameters(), lr=lr)
if params['opt'] == 'Adam':
    lr = 0.002
    opt = torch.optim.Adam(rnn_model.parameters(), lr=lr, betas=(0.9, 0.99))
if params['opt'] == 'Momentum':
    opt = torch.optim.SGD(rnn_model.parameters(), lr=lr, momentum=0.8)
if params['opt'] == 'RMSprop':
    lr = 0.001
    opt = torch.optim.RMSprop(rnn_model.parameters(), lr=lr, alpha=0.9)

lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=params['annealing_step'], gamma=params['annealing_gamma'])

# start training
best_val_loss = None
if params['train_model']:
    try:
        for epoch in range(1, params['epochs']+1):
            epoch_start_time = time.time()
            train()
            val_loss = evaluate(val_data)
            train_perf['valid_loss'].extend([val_loss])
            train_perf['valid_ppl'].extend([math.exp(val_loss)])
            print('-' * 89)
            print('| end of epoch {:5d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss, math.exp(val_loss)))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(params['save']+'.pt', 'wb') as f:
                    torch.save(rnn_model, f)
                    np.save(params['save'] + '_train_perf.npy', train_perf)
                best_val_loss = val_loss
            elif val_loss/best_val_loss >= 1:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr_scheduler.step()

        # Load the best saved model.
        rnn_model = load_model(params['save'] + '.pt')

        # Run on test data.
        test_loss = evaluate(test_data)
        train_perf['test_loss'] = test_loss
        train_perf['test_ppl'] = math.exp(test_loss)
        np.save(params['save'] + '_train_perf.npy', train_perf)
        print('=' * 89)
        print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
            test_loss, math.exp(test_loss)))
        print('=' * 89)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

plot_results(params,train_perf)