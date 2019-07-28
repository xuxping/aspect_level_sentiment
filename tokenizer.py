# -*- coding:utf-8 -*-
import codecs
import os
import pickle

import numpy as np
import six


class BaseTokenizer:

    def __init__(self, maxlen=100, lower=True):
        self.maxlen = maxlen
        self.lower = lower

    def text_to_sequence(self, text, padding='post', truncating='post'):
        raise NotImplementedError

    @staticmethod
    def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
        x = (np.ones(maxlen) * value).astype(dtype)
        if truncating == 'pre':
            trunc = sequence[-maxlen:]
        else:
            trunc = sequence[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == 'post':
            x[:len(trunc)] = trunc
        else:
            x[-len(trunc):] = trunc
        return x


class Tokenizer(BaseTokenizer):

    def __init__(self, maxlen=100, lower=True):
        super(Tokenizer, self).__init__(maxlen=maxlen, lower=lower)
        self.maxlen = maxlen
        self.token2id = {}
        self.id2token = {}
        self.token_cnt = {}

        self.pad_token = '<blank>'
        self.unk_token = '<unk>'
        self.split_token = '<splitter>'

        self.lower = lower
        self.embed_dim = None
        self.embeddings = None

        self.initial_tokens = [self.pad_token, self.unk_token, self.split_token]
        for token in self.initial_tokens:
            self.add(token)

    def fit(self, text):
        if self.lower:
            text = text.lower()
        tokens = text.strip().split()
        for token in tokens:
            self.add(token)

    def add(self, token, cnt=1):
        """
        adds the token to vocab
        Args:
            token: a string
            cnt: a num indicating the count of the token to add, default is 1
        """
        token = token.lower() if self.lower else token
        if token in self.token2id:
            idx = self.token2id[token]
        else:
            idx = len(self.id2token)
            self.id2token[idx] = token
            self.token2id[token] = idx

        if cnt > 0:
            if token in self.token_cnt:
                self.token_cnt[token] += cnt
            else:
                self.token_cnt[token] = cnt
        return idx

    def size(self):
        return len(self.token2id)

    def text_to_sequence(self, text, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        tokens = text.strip().split()
        unk = len(self.token2id) + 1

        sequence = [self.token2id[token] if token in self.token2id else unk for token in tokens]
        pad = self.get_id(self.pad_token)

        return self.pad_and_truncate(sequence, self.maxlen, padding=padding, truncating=truncating, value=pad)

    def get_id(self, token):
        """
        gets the id of a token, returns the id of unk token if token is not in vocab
        Args:
            key: a string indicating the word
        Returns:
            an integer
        """
        token = token.lower() if self.lower else token
        try:
            return self.token2id[token]
        except KeyError:
            return self.token2id[self.unk_token]

    def get_token(self, idx):
        """
        gets the token corresponding to idx, returns unk token if idx is not in vocab
        Args:
            idx: an integer
        returns:
            a token string
        """
        try:
            return self.id2token[idx]
        except KeyError:
            return self.unk_token

    def load_pretrained_embeddings(self, embedding_path):
        """
        loads the pretrained embeddings from embedding_path,
        tokens not in pretrained embeddings will be filtered
        Args:
            embedding_path: the path of the pretrained embedding file
        """
        trained_embeddings = {}
        with codecs.open(embedding_path, 'r', 'utf-8') as fin:
            for line_no, line in enumerate(fin):
                contents = line.strip().split(' ')
                if len(contents) <= 2:
                    continue
                token = contents[0]

                if token not in self.token2id:
                    continue
                try:
                    trained_embeddings[token] = np.asarray(contents[1:], dtype='float32')
                except:
                    continue

                if self.embed_dim is None:
                    self.embed_dim = len(contents) - 1

        # load embeddings
        self.embeddings = np.zeros((self.size(), self.embed_dim))
        for token in self.token2id.keys():
            if token in trained_embeddings:
                self.embeddings[self.get_id(token)] = trained_embeddings[token]

    def randomly_init_embeddings(self, embed_dim=300):
        """
        randomly initializes the embeddings for each token
        Args:
            embed_dim: the size of the embedding for each token
        """
        self.embed_dim = embed_dim
        self.embeddings = np.random.rand(self.size(), embed_dim)
        for token in [self.pad_token, self.unk_token, self.split_token]:
            self.embeddings[self.get_id(token)] = np.zeros([self.embed_dim])

    @staticmethod
    def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
        x = (np.ones(maxlen) * value).astype(dtype)
        if truncating == 'pre':
            trunc = sequence[-maxlen:]
        else:
            trunc = sequence[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == 'post':
            x[:len(trunc)] = trunc
        else:
            x[-len(trunc):] = trunc
        return x

    def save(self, vocab_path):
        with codecs.open(vocab_path, 'wb') as fout:
            pickle.dump(self, fout)

    @classmethod
    def load(cls, vocab_path):
        with codecs.open(vocab_path, 'rb') as fin:
            if six.PY2:
                cls = pickle.load(fin)
            else:
                cls = pickle.load(fin, encoding='bytes')

        return cls


class TokenizerGlove(Tokenizer):
    pass


class TokenizerWord2Vec(Tokenizer):
    pass


class TokenizerELMo(BaseTokenizer):
    pass


def fit_tokenizer(args, tokenizer):
    for fname in [args.train_file, args.test_file]:
        with codecs.open(fname, 'r', encoding='utf-8') as fin:
            all_lines = fin.readlines()

        for i in range(0, len(all_lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in all_lines[i].partition("$T$")]
            aspect = all_lines[i + 1].lower().strip()
            text_raw = text_left + " " + aspect + " " + text_right
            tokenizer.fit(text_raw)


def build_tokenizer(args, tokenizer):
    print('build tokenizer from: {}, {}'.format(args.train_file, args.test_file))

    fit_tokenizer(args, tokenizer)

    print('vocab size:{}'.format(tokenizer.size()))
    # default pretrain_path ./pretrain/glove.840B.300d.txt
    if args.pretrain_type != 'baseline':
        print('Load pretrain embeddings from {}'.format(args.pretrain_path))
        tokenizer.load_pretrained_embeddings(args.pretrain_path)

    print('save pretrain embeddings to {}'.format(args.vocab_path))
    tokenizer.save(args.vocab_path)

    return tokenizer


def get_tokenizer(args):
    tokenizer_map = {
        'baseline': Tokenizer,
        'word2vec': TokenizerWord2Vec,
        'glove': TokenizerGlove
    }
    if args.pretrain_type not in tokenizer_map:
        raise ValueError('invalid pretrain_type:{}'.format(args.pretrain_type))

    TokenizerClass = tokenizer_map[args.pretrain_type]

    if args.vocab_path and os.path.exists(args.vocab_path):
        print('Load tokenizer from {}'.format(args.vocab_path))
        tokenizer = TokenizerClass.load(args.vocab_path)
    else:
        tokenizer = TokenizerClass(args.maxlen, lower=True)
        tokenizer = build_tokenizer(args, tokenizer)

    return tokenizer


if __name__ == '__main__':
    from args import args_parser

    args = args_parser()
    tokenizer = build_tokenizer(args)
    tokenizer.save(args.vocab_path)
