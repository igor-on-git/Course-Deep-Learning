from train import *
from test import *
from model_selector import *
from torch.utils.data import Dataset, SubsetRandomSampler


if __name__ == '__main__':

    project_path = ''
    torch.manual_seed(0)

    # parameters
    model_name_list = ['LeNet5','LeNet5+Dropout','LeNet5+Weight_decay','LeNet5+batch_norm']#
    train_en = 0 # 0 - run test on saved net 1 - train network
    continue_training = 0 # 0 - train from scratch 1 - load saved net and continue training
    test_en = 1
    train_epochs = 30
    save_state_dic = 0

    # initiate data folders and transfers
    train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))])

    test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))])

    trainset = datasets.FashionMNIST(project_path + 'data', download=True, train=True, transform=train_transforms)
    testset = datasets.FashionMNIST(project_path + 'data', download=True, train=False, transform=test_transforms)
    testsampler = SubsetRandomSampler(np.arange(testset.data.shape[0] / 2, dtype=int))
    validsampler = SubsetRandomSampler(np.arange(testset.data.shape[0] / 2,testset.data.shape[0], dtype=int))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    test_perf_all = [dict() for x in range(len(model_name_list))]

    for model_ind, model_name in enumerate(model_name_list):

        model, optimizer, lr_scheduler, criterion, batch_size, train_stop_criteria, train_stop_patience = model_selector(model_name)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        validloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, sampler=validsampler)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, sampler=testsampler)

        if train_en:

            if continue_training:
                try:
                    model.load_state_dict(torch.load(project_path + 'models/' + model_name + '/state_dict.pth'))
                    train_perf = np.load(project_path + 'models/' + model_name + '/train_perf.npy', allow_pickle=True).item()
                except:
                    os.makedirs(project_path + 'models/' + model_name, exist_ok=True)
                    train_perf = None
            else:
                os.makedirs(project_path + 'models/' + model_name, exist_ok=True)
                train_perf = None

            model, train_perf = train_model(
                project_path, model_name, model, criterion, optimizer, lr_scheduler, trainloader, validloader, device, train_epochs, train_stop_criteria, train_stop_patience, save_state_dic, perf_prev=train_perf)

        else:

            model.load_state_dict(torch.load(project_path + 'models/' + model_name + '/state_dict.pth'))
            train_perf = np.load(project_path + 'models/' + model_name + '/train_perf.npy', allow_pickle=True).item()

        plot_train_results(model_name, train_perf, project_path)
        if test_en:
            test_perf_all[model_ind] = test(model_name, model, criterion, testloader, device)
            np.save(project_path + 'models/' + model_name + '/test_perf.npy', test_perf_all[model_ind])
        else:
            try:
                test_perf_all[model_ind] = np.load(project_path + 'models/' + model_name + '/test_perf.npy', allow_pickle=True).item()
            except:
                test_perf_all[model_ind] = None

    test_metric_all = [test_perf_all[i]['test_accuracy'] for i in range(len(model_name_list))]
    best_ind = np.argwhere(test_metric_all == np.max(test_metric_all))
    print('Best(Accuracy score) Net is: {}'.format(model_name_list[int(best_ind)]))

    plot_all_results(model_name_list, train_perf, project_path)
