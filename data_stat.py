# -*- coding:utf-8 -*-
from collections import Counter

from args import parse_args
from dataset import SEResDataset
from tokenizer import TokenizerGlove


def stat(data):
    y = [item['polarity'] for item in data]
    counts = Counter(y)
    print('data size {}'.format(len(y)))

    for key, val in counts.items():
        print(key, val)


if __name__ == '__main__':
    args = parse_args()
    args.inputs_cols = ['text_bert_indices', 'bert_segments_ids']
    tokenizer = TokenizerGlove(maxlen=args.maxlen)
    train_data = SEResDataset(args.train_file, tokenizer=tokenizer)
    test_data = SEResDataset(args.test_file, tokenizer=tokenizer)

    stat(train_data.data)
    stat(test_data.data)
