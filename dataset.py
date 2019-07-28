# -*- coding:utf-8 -*-
import codecs
import os

import numpy as np
from torch.utils.data import Dataset

from tokenizer import Tokenizer


class SEResDataset(Dataset):
    """Restaurants Dataset"""

    def __init__(self, data_path=None, tokenizer=None):

        if not data_path or not os.path.exists(data_path):
            raise ValueError('set data path')

        self.tokenizer = tokenizer
        self.data = self._load_dataset(data_path)

    def _load_dataset(self, data_path):
        with codecs.open(data_path, 'r', encoding='utf-8') as fin:
            all_lines = fin.readlines()
        all_data = []
        for i in range(0, len(all_lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in all_lines[i].partition("$T$")]
            aspect = all_lines[i + 1].lower().strip()
            polarity = all_lines[i + 2].strip()
            polarity = int(polarity) + 1

            # for word2vec、glove
            text_raw_indices = self.tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
            aspect_indices = self.tokenizer.text_to_sequence(aspect)

            data = {
                'text_raw_indices': text_raw_indices,
                'aspect_indices': aspect_indices,
                'polarity': polarity,
            }

            all_data.append(data)

        return all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


if __name__ == '__main__':
    from args import parse_args
    from tokenizer import TokenizerBert

    args = parse_args()
    args.inputs_cols = ['text_bert_indices', 'bert_segments_ids']
    from torch.utils.data import DataLoader

    tokenizer = TokenizerBert(maxlen=80, pretrained_bert_name='bert-base-uncased')
    seres_data = SEResDataset('./data/semeval14/Restaurants_Train.xml.seg', tokenizer=tokenizer)
    dataloader = DataLoader(seres_data, batch_size=32, shuffle=True)
    for batch_id, batch_data in enumerate(dataloader, 1):
        print(batch_data)
        inputs = [batch_data[col] for col in args.inputs_cols]
        targets = batch_data['polarity']
        print(inputs)
        print(targets)
        break
