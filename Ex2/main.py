# code referenced from https://github.com/hjc18/language_modeling_lstm/blob/master/main.py
from import_libs import *
from corpus import Dictionary, Corpus, batchify, get_batch
from model import RNNModel, param_selector, load_model, init_train_perf, plot_all_results, plot_results
from train import train, evaluate

torch.manual_seed(1111)

#name_list = ['LSTM', 'LSTM+Drop', 'GRU', 'GRU+Drop']
name_list = ['LSTM']
# plot all results
# plot_all_results(name_list)

for name in name_list:
    params = param_selector(name)
    params['train_model'] = 1

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
    params['ntokens'] = corpus.dictionary.ntokens # 10000
    rnn_model = []
    if params['continue_training']:
        try:
            rnn_model = load_model(params['save']+'.pt')
            train_perf = np.load(params['save'] + '_train_perf.npy', allow_pickle=True).item()
            params['lr'] = train_perf['lr']
        except:
            print('saved model not found')

    if rnn_model == []:
        rnn_model = RNNModel(params['type'], params['ntokens'], params['emsize'], params['nhid'], params['nlayers'], params['dropout'])
        train_perf = init_train_perf()

    print(rnn_model)

    # loss criterion
    criterion = nn.CrossEntropyLoss()

    # select optimizer # notice that lr is dependent on type of optimizer
    if params['opt'] == 'SGD':
        opt = torch.optim.SGD(rnn_model.parameters(), lr=params['lr'])
    if params['opt'] == 'Adam':
        # lr = 0.002
        opt = torch.optim.Adam(rnn_model.parameters(), lr=params['lr'], betas=(0.9, 0.99))
    if params['opt'] == 'Momentum':
        opt = torch.optim.SGD(rnn_model.parameters(), lr=params['lr'], momentum=0.8)
    if params['opt'] == 'RMSprop':
        # lr = 0.001
        opt = torch.optim.RMSprop(rnn_model.parameters(), lr=params['lr'], alpha=0.9)

    # select lr scheduler
    #lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=params['annealing_step'], gamma=params['annealing_gamma'])
    lr_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=np.arange(params['annealing_start'],100,params['annealing_step']), gamma=params['annealing_gamma'])

    # training loop
    best_val_loss = None
    if params['train_model']:
        try:
            for epoch in range(1, params['epochs']+1):
                epoch_start_time = time.time()
                total_loss = train(rnn_model, params, device, train_data, criterion, opt, train_perf, lr_scheduler)
                val_loss = evaluate(rnn_model, params, device, val_data, corpus, criterion, eval_batch_size)
                train_perf['valid_loss'].extend([val_loss])
                train_perf['valid_ppl'].extend([math.exp(val_loss)])

                print('| end of epoch {:5d} | time: {:5.2f}s | lr {:2.5f} | valid loss {:5.2f} | valid ppl {:8.2f} | train loss {:5.2f} | train ppl {:8.2f}'
                      .format(epoch, (time.time() - epoch_start_time), lr_scheduler.get_last_lr()[0],
                                                   val_loss, math.exp(val_loss), total_loss, math.exp(total_loss)))

                # Save the model if the validation loss is the best we've seen so far.
                if not best_val_loss or val_loss < best_val_loss:
                    with open(params['save']+'.pt', 'wb') as f:
                        torch.save(rnn_model, f)
                        np.save(params['save'] + '_train_perf.npy', train_perf)
                    best_val_loss = val_loss
                #elif val_loss/best_val_loss >= 1:
                # Anneal the learning rate
                lr_scheduler.step()

            # end of training loop
            # Load the best saved model.
            rnn_model = load_model(params['save'] + '.pt')

            # Run on test data.
            test_loss = evaluate(rnn_model, params, device, test_data, corpus, criterion, eval_batch_size)
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