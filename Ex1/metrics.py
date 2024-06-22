import numpy as np


def accuracy(reference, estimation):

    return np.mean(np.array(reference) == np.array(estimation))


def confusion_matrix(reference, estimation):

    TP = np.sum( (np.array(estimation) == 1) & (np.array(reference) == 1) )
    TN = np.sum( (np.array(estimation) == 0) & (np.array(reference) == 0) )
    FP = np.sum( (np.array(estimation) == 1) & (np.array(reference) == 0) )
    FN = np.sum( (np.array(estimation) == 0) & (np.array(reference) == 1) )

    return {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}


def precision(reference, estimation):

    cm = confusion_matrix(reference, estimation)

    return cm['TP'] / (cm['TP'] + cm['FP'])


def recall(reference, estimation):

    cm = confusion_matrix(reference, estimation)

    return cm['TP'] / (cm['TP'] + cm['FN'])


def F1score(reference, estimation):

    return 2 / ( 1/precision(reference, estimation) + 1/recall(reference, estimation) )


def Fscore(beta_sq, reference, estimation):

    prec = precision(reference, estimation)
    rec = recall(reference, estimation)
    return (1+beta_sq) * prec * rec / (beta_sq * prec + rec)