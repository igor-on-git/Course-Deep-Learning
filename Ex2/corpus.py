from import_libs import *


class Dictionary(object):
    def __init__(self):
        self.word2idx = {} # word: index
        self.idx2word = [] # position(index): word

    def __len__(self):
        return len(self.idx2word)

    def translate_word2idx(self, inp):
        return torch.tensor([self.word2idx[c] for c in inp])

    def translate_idx2word(self, inp):
        return [self.idx2word[c] for c in inp.numpy()]


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()

        # three tensors of word index
        with open("./data/ptb.train.txt") as f:
            file = f.read()
            trn = file[1:].split(' ')
        with open("./data/ptb.valid.txt") as f:
            file = f.read()
            vld = file[1:].split(' ')
        with open("./data/ptb.test.txt") as f:
            file = f.read()
            tst = file[1:].split(' ')

        words = sorted(set(trn))
        self.dictionary.word2idx = {c: i for i, c in enumerate(words)}
        self.dictionary.idx2word = {i: c for i, c in enumerate(words)}
        self.dictionary.ntokens = len(self.dictionary.word2idx)

        self.train = self.dictionary.translate_word2idx(trn)
        self.valid = self.dictionary.translate_word2idx(vld)
        self.test  = self.dictionary.translate_word2idx(tst)


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data


def get_batch(source, bptt, i):

    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len].clone().detach()
    target = source[i+1:i+1+seq_len].clone().detach().view(-1)
    return data, target