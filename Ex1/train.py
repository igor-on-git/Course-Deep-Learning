import numpy as np
import torch
import time

import metrics


def train_model(model_name, model, criterion, optimizer, train_loader, valid_loader, device, n_epochs, train_stop_criteria, train_stop_patience, save_state_dic, perf_prev=None):

    model.to(device)
    train_perf = perf_prev if perf_prev else init_train_perf()
    criteria_str, criteria_type, criteria_best, criteria_counter = init_criteria(train_stop_criteria, train_perf)

    for epoch in range(n_epochs):

        start_time = time.time()

        ###################
        # train the model #
        ###################
        model.train()  # prep model for training
        train_loss = 0
        train_len = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item() * data.size(0)
            train_len += data.shape[0]
        ######################
        # validate the model #
        ######################
        ref_vals = []
        est_vals = []
        valid_loss = 0
        model.eval()  # prep model for evaluation
        valid_len = 0
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data)
                # calculate the loss
                loss = criterion(output, target)
                # update running validation loss
                valid_loss += loss.item() * data.size(0)
                valid_len += data.shape[0]
                ps = torch.exp(output)
                top_prob, top_class = ps.topk(1, dim=1)
                ref_vals.extend(target.cpu().numpy().reshape(-1))
                est_vals.extend(top_class.cpu().numpy().reshape(-1))

        # calculate metrics over an epoch
        train_loss /= train_len
        valid_loss /= valid_len
        accuracy = metrics.accuracy(ref_vals, est_vals)
        #precision = metrics.precision(ref_vals, est_vals)
        #recall = metrics.recall(ref_vals, est_vals)
        #F1score = metrics.F1score(ref_vals, est_vals)

        train_perf['train_loss'].extend([train_loss])
        train_perf['valid_loss'].extend([valid_loss])
        train_perf['accuracy'].extend([accuracy])
        #train_perf['precision'].extend([precision])
        #train_perf['recall'].extend([recall])
        #train_perf['F1score'].extend([F1score])
        end_time = time.time()

        # print training/validation statistics
        print('Model: {}\tDevice: {}\tRun time: {:.2f} min'.format(model_name, device, (end_time - start_time)/60))
        #print('Epoch: {} \tTraining Loss: {:.3f} \tValidation Loss: {:.3f} \tPrecision: {:.3f}% \tRecall: {:.3f}% \tF1-score: {:.3f}'.format(
        #    epoch + 1, train_loss, valid_loss, precision*100, recall*100, F1score) )
        print(
            'Epoch: {} \tTraining Loss: {:.3f} \tValidation Loss: {:.3f} \tAccuracy: {:.3f}'.format(
                epoch + 1, train_loss, valid_loss, accuracy * 100))

        # stop criteria
        criteria_val = update_criteria_val(train_stop_criteria, train_perf)

        # save model if validation loss has decreased
        if (criteria_val <= criteria_best if criteria_type == 'loss' else criteria_val >= criteria_best):

            train_perf['criteria_best'] = criteria_best
            if save_state_dic:
                print(criteria_str + ' improved ({:.3f} --> {:.3f}).  Saving model ...'.format(criteria_best, criteria_val))
                if torch.cuda.is_available():
                    model.cpu()
                torch.save(model.state_dict(), 'models/' + model_name + '/state_dict.pth')
                np.save('models/' + model_name + '/train_perf.npy', train_perf)
                criteria_best = criteria_val
                criteria_counter = 0
                if torch.cuda.is_available():
                    model.to(device)

        else:
            criteria_counter += 1

        if epoch > train_stop_patience and criteria_counter >= train_stop_patience:
            print('Ending Training loop due to non imporving criteria\n')
            model.load_state_dict(torch.load('models/' + model_name + '/state_dict.pth')) # return best model
            break

    if torch.cuda.is_available():
        model.cpu()

    return model, train_perf


def init_train_perf():

    train_perf = {}
    train_perf['train_loss'] = []
    train_perf['valid_loss'] = []
    train_perf['accuracy'] = []
    train_perf['precision'] = []
    train_perf['recall'] = []
    train_perf['F1score'] = []
    train_perf['criteria_best'] = []

    return train_perf


def init_criteria(train_stop_criteria, train_perf):

    criteria_counter = 0
    if train_stop_criteria == 'valid loss':
        criteria_best = np.inf if train_perf['criteria_best'] == [] else train_perf['criteria_best']
        criteria_str = 'Validation loss'
        criteria_type = 'loss'
    if train_stop_criteria == 'F1-score':
        criteria_best = 0 if train_perf['criteria_best'] == [] else train_perf['criteria_best']
        criteria_str = 'F1 score'
        criteria_type = 'gain'

    return criteria_str, criteria_type, criteria_best, criteria_counter


def update_criteria_val(train_stop_criteria, train_perf):

    if train_stop_criteria == 'valid loss':
        criteria_val = train_perf['valid_loss'][-1]
    if train_stop_criteria == 'F1-score':
        criteria_val = train_perf['F1score'][-1]

    return criteria_val