# Ex1 - Training Lenet5 on FashionMNIST dataset
Purpose of exercise is to use regularization(L2 weight decay and dropout) and batch-norm methods to improve training.
Following results plot contains a pair of plots(accuracy and loss) for each network.
## Baseline network
Most left plots are for original Lenet5 network without any regularization and normalization applied. It shows overfitting while training.
## Baseline + Batch-Normalization
Most right plots show original network with added batch-normalization to hidden layers with all other hyperparameters without a change. Here we can see that the training is much faster than the case of baseline network. The overfitting is evident.
## Dropout
Second from the left plots show that overfitting issue is solved using dropout 20% on hidden layers without performance penalty
## L2 weight decay
Second plots from the right. This method also solved the overfitting issue but some performance penalty is present due to regularization applied to network coefficients.
![plot](https://github.com/igor-on-git/Course-Deep-Learning/blob/main/Ex1/results.png?raw=true)
## Conclusions
Dropout regularization seems more effective than L2 for this network and dataset. Batch-Normalization reduces training time.
