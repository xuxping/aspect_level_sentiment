# -*- coding:utf-8 -*-

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from codecs import open

from tokenizer import *


def _save_embeddings(save_path, tokenizer):
    print('Save embeddings to {}'.format(save_path))
    with open(save_path, 'w', encoding='utf-8') as fout:
        for token in tokenizer.token2id:
            embedding = tokenizer.embeddings[tokenizer.get_id(token)]
            embedding = [str(e) for e in embedding.tolist()]
            fout.write('{} {}\n'.format(token, ' '.join(embedding)))


if __name__ == '__main__':

    """
    For Word2Vec:
    ```
        python create_embedding.py --ptype=word2vec \
        --pretrain_path ./pretrain/GoogleNews-vectors-negative300.txt \
        --save_path ./pretrain/sn.word2vec.300d.txt
    ```
    
    For Glove:
    ```
        python create_embedding.py --ptype=glove \
        --pretrain_path ./pretrain/glove.840B.300d.txt \
        --save_path ./pretrain/sn.glove.300d.txt
    ```
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--ptype', type=str, choices=['word2vec', 'glove'], default=None)
    parser.add_argument('--pretrain_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None)

    parser.add_argument('--train_file', type=str, default='./data/semeval14/Restaurants_Train.xml.seg',
                        help='path of train file')
    parser.add_argument('--test_file', type=str, default='./data/semeval14/Restaurants_Test_Gold.xml.seg',
                        help='path of test file')

    args = parser.parse_args()
    print('build tokenizer from: {}, {}'.format(args.train_file, args.test_file))

    if args.ptype == 'word2vec':
        tokenizer = TokenizerWord2Vec()
    elif args.ptype == 'glove':
        tokenizer = TokenizerGlove()
    else:
        raise ValueError('invalid ptype')

    fit_tokenizer(args, tokenizer)
    print('Load pretrain embeddings from {}'.format(args.pretrain_path))
    tokenizer.load_pretrained_embeddings(args.pretrain_path)

    _save_embeddings(args.save_path, tokenizer)
