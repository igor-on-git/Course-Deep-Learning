from imports import *
from data import get_data, get_labled_data, visualize_data
from models import Encoder, Decoder, VAE, init_xavier
from train_test import fit, generate_data, get_latent_data

if __name__ == "__main__":
    path = './'
    DATA_TYPE = 'FASHION-MNIST' #'MNIST'#
    TRAIN_VAE_EN = 0 # flag that enables training of VAE model
    TRAIN_SVM_EN = 0 # flag that enables training of SVM model
    BATCH_SIZE = 64  # number of data points in each batch
    N_EPOCHS = 20  # times to run the model on complete data
    LABELS_LIST = [100, 600, 1000, 3000]
    INPUT_DIM = 28 * 28  # size of each input
    HIDDEN_DIM = 600  # hidden dimension
    VAE_TYPE = 'linear' #'conv' #
    LATENT_DIM = 50  # latent vector dimension
    lr = 1e-3  # learning rate
    SEED = 0

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # train latent encoder decoder
    train_iterator, test_iterator = get_data(DATA_TYPE, BATCH_SIZE)

    batch, labels = next(iter(train_iterator))
    visualize_data(batch)

    # VAE
    encoder = Encoder(HIDDEN_DIM, LATENT_DIM, VAE_TYPE)
    decoder = Decoder(LATENT_DIM, HIDDEN_DIM, VAE_TYPE)
    model = VAE(encoder, decoder).cuda()
    model.apply(init_xavier)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model_file = path + 'data/vae_' + DATA_TYPE.lower() + '_' + VAE_TYPE + '_model.pt'
    if TRAIN_VAE_EN == 0 and os.path.isfile(model_file):
        model.load_state_dict(torch.load(model_file))
    else:
        fit(model, optimizer, train_iterator, test_iterator, device, N_EPOCHS)  # Training The model
        torch.save(model.state_dict(), model_file)

    batch = generate_data(model, device, BATCH_SIZE, LATENT_DIM)
    visualize_data(batch)

    train_acc = np.zeros(len(LABELS_LIST))
    test_acc = np.zeros(len(LABELS_LIST))
    # train svm on latent vector
    for i, labeled_size in enumerate(LABELS_LIST):
        train_labelled_iterator = get_labled_data(train_iterator, labeled_size)
        train_means, train_vars, train_labels = get_latent_data(model, train_labelled_iterator, device)

        train_latent_data = np.column_stack((train_means, train_vars))
        train_latent_df = pd.DataFrame(train_latent_data)
        train_labels = np.array(train_labels)

        # test latent data
        test_means, test_vars, test_labels = get_latent_data(model, test_iterator, device)

        test_latent_data = np.column_stack((test_means, test_vars))
        test_latent_df = pd.DataFrame(test_latent_data)
        test_labels = np.array(test_labels)

        #print('Shape of Training data : ', train_latent_data.shape)
        #print('Shape of Testing Data : ', test_latent_data.shape)

        svm_file = path + 'data/svm_' + DATA_TYPE.lower() + '_' + VAE_TYPE + '_' + str(labeled_size) + '_model.pt'
        if TRAIN_SVM_EN == 0 and os.path.isfile(svm_file):
            clf = joblib.load(svm_file)
        else:
            ### Training the SVM classifier on Latent Space"""
            clf = LinearSVC(random_state=0, tol=1e-5)
            clf.fit(train_latent_df, train_labels)
            joblib.dump(clf, svm_file)

        ypreds = clf.predict(train_latent_df)
        ytrue = train_labels
        train_acc[i] = accuracy_score(ytrue, ypreds)
        # prediction on test data
        ypreds = clf.predict(test_latent_df)
        ytrue = test_labels
        test_acc[i] = accuracy_score(ytrue, ypreds)

        print('Trainset size : {:3} \t SVM Train Accuracy : {:3.2f} \t SVM Test Accuracy : {:3.2f}'.format(labeled_size, train_acc[i]*100,test_acc[i]*100))
