# -*- coding:utf-8 -*-


import distutils.util
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # run model
    parser.add_argument('--train', action='store_true', help='train model')
    parser.add_argument('--test', action='store_true', help='test model')

    parser.add_argument('--train_file', type=str, default='./data/semeval14/Restaurants_Train.xml.seg',
                        help='path of train file')
    parser.add_argument('--test_file', type=str, default='./data/semeval14/Restaurants_Test_Gold.xml.seg',
                        help='path of test file')

    parser.add_argument('--embed_dim', type=int, default=300, help='embedding size')
    parser.add_argument('--pretrain_path', type=str, default='./pretrain/glove.840B.300d.txt',
                        help='pretrained embedding path')
    parser.add_argument('--pretrain_type', type=str, default='glove',
                        choices=['baseline', 'word2vec', 'glove', 'bert', 'elmo'],
                        help='pretrain type')

    parser.add_argument('--freeze', type=distutils.util.strtobool, default=True,
                        help='freeze embedding')

    parser.add_argument('--vocab_path', type=str, default='./data/vocab/vocab.data', help='vocab path')

    parser.add_argument('--maxlen', type=int, default=128, help='max sentence length')
    parser.add_argument('--max_epoch', type=int, default=10, help='train epoch')
    parser.add_argument('--model_name', type=str, default='atae_lstm', choices=['atae_lstm'],
                        help='which model use to train')

    parser.add_argument('--use_gpu', action='store_true', default=True, help='use gpu')

    parser.add_argument('--seed', type=int, default=123, help='seed')

    parser.add_argument('--batch_size', type=int, default=32, help='train batch size')

    parser.add_argument('--weight_decay', type=float, default=0.001, help='train batch size')
    parser.add_argument('--optim', type=str, default='adam', choices=['adam', 'sgd', 'adagrad'], help='train optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

    parser.add_argument('--hidden_dim', type=int, default=300, help='hidden size')
    parser.add_argument('--drop_rate', type=float, default=0.2, help='hidden size')

    parser.add_argument('--test_step', type=int, default=50, help='test interval when train')

    parser.add_argument('--save_dir', type=str, default='./stat_dict', help='model save path')
    parser.add_argument('--load_dir', type=str, default=None, help='where to load model')
    parser.add_argument('--log_dir', type=str, default='./runs', help='where to tensorboardx log ')

    args = parser.parse_args()

    return args
