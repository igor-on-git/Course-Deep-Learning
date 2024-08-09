from imports import *

def train(model, optimizer, train_iterator, device):
    # set the train mode
    model.train()

    # loss of the epoch
    train_loss = 0

    for i, (x, _) in enumerate(train_iterator):
        x = x.to(device)

        # update the gradients to zero
        optimizer.zero_grad()

        # forward pass
        x_sample, z_mu, z_var = model(x)

        # reconstruction loss
        recon_loss = F.binary_cross_entropy(x_sample, x, size_average=False)

        # kl divergence loss
        kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1.0 - z_var)

        # total loss
        loss = recon_loss + kl_loss

        # backward pass
        loss.backward()
        train_loss += loss.item()

        # update the weights
        optimizer.step()

    return train_loss, model


def test(model, optimizer, test_iterator, device):
    # set the evaluation mode
    model.eval()

    # test loss for the data
    test_loss = 0

    with torch.no_grad():
        for i, (x, _) in enumerate(test_iterator):
            x = x.to(device)

            # forward pass
            x_sample, z_mu, z_var = model(x)

            # reconstruction loss
            recon_loss = F.binary_cross_entropy(x_sample, x, size_average=False)

            # kl divergence loss
            kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1.0 - z_var)

            # total loss
            loss = recon_loss + kl_loss
            test_loss += loss.item()

    return test_loss


def generate_data(model, device, batch_sizes, latent_dim):

    z = torch.randn(batch_sizes, latent_dim).to(device)
    reconstructed_img = model.dec(z)
    img = reconstructed_img.view(-1, 1, 28, 28).data

    return img.cpu()

def get_latent_data(model, data_iterator, device):
    means = np.zeros((1, model.enc.mu.out_features))
    vars = np.zeros((1, model.enc.mu.out_features))

    data_labels = []
    with torch.no_grad():
        for batch, labels in data_iterator:
            labels = list(labels.numpy())
            batch = batch.to(device)

            mean, var = model.enc(batch)

            mean = mean.to('cpu').numpy()
            var = var.to('cpu').numpy()

            means = np.vstack((means, mean))
            vars = np.vstack((vars, var))

            data_labels.extend(labels)

    means = means[1:]
    vars = vars[1:]

    return means, vars, data_labels

def fit(model, optimizer, train_iterator, test_iterator, device, N_EPOCHS):
    best_test_loss = float('inf')
    print('Before Training : ')
    #img = generate_data()

    for e in range(N_EPOCHS):

        train_loss, model = train(model, optimizer, train_iterator, device)
        test_loss = test(model, optimizer, test_iterator, device)

        train_loss /= len(train_iterator.dataset)
        test_loss /= len(test_iterator.dataset)

        print(f'Epoch {e}, Train Loss: {train_loss:.2f}, Test Loss: {test_loss:.2f}')
        #generate_data()

        if best_test_loss > test_loss:
            best_test_loss = test_loss
            patience_counter = 1
        else:
            patience_counter += 1

        if patience_counter > 3:
            break

    return model