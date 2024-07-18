from import_libs import *

from corpus import Dictionary, Corpus, batchify, get_batch

def train(rnn_model, params, device, train_data, criterion, opt, train_perf, lr_scheduler):
    # choose a optimizer
    rnn_model.to(device)
    rnn_model.train()
    total_loss = 0
    report_loss = 0
    start_time = time.time()
    hidden = rnn_model.init_hidden(params['batch_size'])
    hidden = rnn_model.hidden_to_device(hidden, device)
    # train_data size(batchcnt, bsz)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, params['seq_en'])):
        data, targets = get_batch(train_data, params['seq_en'], i)
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

    total_loss /= len(train_data) / params['seq_en']
    train_perf['train_loss'].extend([total_loss])
    train_perf['train_ppl'].extend([math.exp(total_loss)])
    train_perf['lr'] = lr_scheduler.get_last_lr()[0]
    # lr_scheduler.step()
    rnn_model.to('cpu')

    return total_loss

def evaluate(rnn_model, params, device, data_source, corpus, criterion, eval_batch_size):
    # Turn on evaluation mode which disables dropout.
    rnn_model.to(device)
    with torch.no_grad():
        rnn_model.eval()
        total_loss = 0
        ntokens = len(corpus.dictionary)
        hidden = rnn_model.init_hidden(eval_batch_size) #hidden size(nlayers, bsz, hdsize)
        hidden = rnn_model.hidden_to_device(hidden, device)
        for i in range(0, data_source.size(0) - 1, params['seq_en']):# iterate over every timestep
            data, targets = get_batch(data_source, params['seq_en'], i)
            data, targets = data.to(device), targets.to(device)
            output, hidden = rnn_model(data, hidden)
            # model input and output
            # inputdata size(bptt, bsz), and size(bptt, bsz, embsize) after embedding
            # output size(bptt*bsz, ntoken)
            total_loss += len(data) * criterion(output, targets).data
            hidden = rnn_model.repackage_hidden(hidden)

        rnn_model.to('cpu')
        return total_loss / len(data_source)