import numpy as np

from imports import *
def get_data(type, batch_size):
    ### Loading Data ###
    transforms_list = transforms.Compose([transforms.ToTensor()])

    if type == 'MNIST':
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms_list)
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transforms_list)
    if type == 'FASHION-MNIST':
        train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transforms_list)
        test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transforms_list)

    train_iterator = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_iterator = DataLoader(test_dataset, batch_size=batch_size)

    return train_iterator, test_iterator

def get_labled_data(train_iterator, labeled_size):

    train_labelled, _ = torch.utils.data.random_split(train_iterator.dataset,[int(labeled_size), int(len(train_iterator.dataset) - labeled_size)])
    train_labelled_iterator = DataLoader(train_labelled, batch_size=train_iterator.batch_size, shuffle=True)

    return train_labelled_iterator

def visualize_data(batch, batch_size = 64):
    batch = torchvision.utils.make_grid(batch)
    batch = batch.numpy()
    batch = np.transpose(batch, (1, 2, 0))
    plt.figure(figsize=(int(np.sqrt(batch_size)), int(np.sqrt(batch_size))))
    plt.imshow(batch, cmap='Greys_r')
    plt.show()
