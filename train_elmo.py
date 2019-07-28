# -*- coding:utf-8 -*-
# -*- coding:utf-8 -*-

import codecs
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optims
from allennlp.modules.elmo import batch_to_ids
from sklearn.metrics import f1_score, classification_report
from tensorboardX import SummaryWriter

from args import parse_args

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from models.atae_lstm import ELMoModel


class SEResDataset():
    """Restaurants Dataset"""

    def __init__(self, data_path=None):

        if not data_path or not os.path.exists(data_path):
            raise ValueError('set data path')

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

            text = text_left + " " + aspect + " " + text_right
            text_raw_indices = text.split()
            aspect_indices = text.split()
            # text_raw_indices = batch_to_ids(text.split())
            # aspect_indices = batch_to_ids(aspect.split())
            data = (text_raw_indices, aspect_indices, polarity)
            # data = {
            #     'text_raw_indices': text_raw_indices,
            #     'aspect_indices': aspect_indices,
            #     'polarity': polarity,
            # }

            all_data.append(data)

        return all_data

    def _one_mini_batch(self, data, indices):
        """
        Get one mini batch
        Args:
            data: all data
            indices: the indices of the samples to be selected
        Returns:
            one batch of data
        """

        # batch_data = {
        #     'text_raw_indices': torch.cat([data[i]['text_raw_indices'] for i in indices]),
        #     'aspect_indices': torch.cat([data[i]['aspect_indices'] for i in indices]),
        #     'polarity': torch.cat([data[i]['polarity'] for i in indices]),
        # }
        # print(len(data))
        # print(torch.cat([data[i][0] for i in indices]))
        # print(torch.cat([data[i][0] for i in indices], dim=-1).shape)
        batch_data = {
            'text_raw_indices': batch_to_ids([data[i][0] for i in indices]),
            'aspect_indices': batch_to_ids([data[i][1] for i in indices]),
            'polarity': torch.tensor([data[i][2] for i in indices])
        }
        # batch_data = [(data[i] for i in indices]
        return batch_data

    def gen_mini_batch(self, batch_size=128, shuffle=True):
        """Generate data batches for a specific dataset (train/dev/test)\
        Args:
            set_name: train/dev/test to indicate the set
            batch_size: defautl 128
            shuffle: shuffle
        """

        data_size = len(self.data)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)

        for batch_start in np.arange(0, data_size, batch_size):
            batch_indices = indices[batch_start:batch_start + batch_size]
            yield self._one_mini_batch(self.data, batch_indices)


class TorchProgram:

    def __init__(self, args):

        if args.use_gpu and torch.cuda.is_available():
            cudnn.benchmark = True
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            use_gpu = True
        else:
            print("Currently using CPU, however, GPU is highly recommended")
            use_gpu = False

        self.args = args
        self.device = torch.device('cuda' if use_gpu else 'cpu')

        # data loader
        if args.train:
            self.train_dataset = SEResDataset(data_path=args.train_file)

        self.test_dataset = SEResDataset(data_path=args.test_file)

        self.model = ELMoModel(hidden_dim=args.hidden_dim, device=self.device)
        args.inputs_cols = ['text_raw_indices', 'aspect_indices']

        if use_gpu:
            self.model = nn.DataParallel(self.model).cuda()

        self.log_writer = SummaryWriter(comment='elmo')
        # self.log_writer.add_graph(self.model, torch.LongTensor(2, 2, 5).random_(0, 10))

        if args.load_dir and os.path.exists(args.load_dir):
            checkpoint = torch.load(args.load_dir)
            self.model.load_state_dict(checkpoint['model'])

    def train(self):

        criterion = nn.CrossEntropyLoss()
        optim = self.get_optimizer()
        optimizer = optim(params=self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        total_correct, total_num = 0, 0
        max_test_acc, this_test_f1 = 0, 0
        best_mode_path = None
        global_step = 1
        for epoch in range(0, args.max_epoch):
            trainloader = self.train_dataset.gen_mini_batch(batch_size=args.batch_size, shuffle=True)
            total_loss = 0

            for batch_id, batch_data in enumerate(trainloader, 1):
                self.model.train()
                start_time = time.time()
                optimizer.zero_grad()

                # to device
                inputs = [batch_data[col].to(self.device) for col in self.args.inputs_cols]
                outputs = self.model(inputs)
                targets = batch_data['polarity'].to(self.device)
                # targets = batch_data['polarity']

                loss = criterion(outputs, targets)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

                end_time = time.time()

                total_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                total_num += len(targets)
                train_acc = total_correct / total_num

                self.log_writer.add_scalar('data/loss', total_loss / batch_id, global_step)
                self.log_writer.add_scalar('data/train_acc', train_acc, global_step)

                if args.test_step > 0 and global_step % args.test_step == 0:
                    test_acc, test_f1 = self.test(None)

                    self.log_writer.add_scalar('data/test_acc', test_acc, global_step)
                    self.log_writer.add_scalar('data/test_f1', test_f1, global_step)

                    print(
                        'Epoch:{}, Batch_id:{}, Time:{:.2f}, Loss:{:.4f}, Train_acc:{:.4f},  Test_acc:{:.4f}, Test_f1:{:.4f}'.format(
                            epoch, batch_id, (end_time - start_time), total_loss / batch_id, train_acc, test_acc, test_f1
                        ))
                    if max_test_acc < test_acc:
                        max_test_acc = test_acc
                        this_test_f1 = test_f1
                        if args.save_dir:
                            os.makedirs(args.save_dir, exist_ok=True)

                        best_mode_path = '{}/elmo_{}_{}_{:.4f}'.format(args.save_dir, args.pretrain_type,
                                                                       epoch, test_acc)
                        print('save model to {}'.format(best_mode_path))

                        torch.save({
                            'model': self.model.state_dict(),
                            'epoch': epoch
                        }, best_mode_path)
                else:
                    print('Epoch:{}, Batch:{:.2f}, Time:{:.2f}, Loss:{:.4f}, Train_acc:{:.4f}'.format(
                        epoch, batch_id, (end_time - start_time), total_loss / batch_id, train_acc
                    ))

                global_step += 1

        self.log_writer.close()

        print('max test acc:{:.4f}'.format(max_test_acc))
        print('test f1:{:.4f}'.format(this_test_f1))
        print('best model path:{}'.format(best_mode_path))

    def test(self, testloader):
        total_correct, total_num = 0, 0
        y_true, y_pred = None, None
        testloader = self.test_dataset.gen_mini_batch(batch_size=args.batch_size, shuffle=False)
        self.model.eval()
        with torch.no_grad():
            for batch_id, batch_data in enumerate(testloader):
                # to device
                inputs = [batch_data[col].to(self.device) for col in args.inputs_cols]
                outputs = self.model(inputs)
                targets = batch_data['polarity'].to(self.device)

                if y_true is None:
                    y_true = targets
                    y_pred = outputs
                else:
                    y_true = torch.cat((y_true, targets), dim=0)
                    y_pred = torch.cat((y_pred, outputs), dim=0)

                total_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                total_num += len(targets)

        y_pred = torch.argmax(y_pred, -1)
        y_true, y_pred = y_true.cpu(), y_pred.cpu()

        test_acc = total_correct / total_num
        test_f1 = f1_score(y_true, y_pred, average='macro')
        if self.args.test:
            print("[Test]: acc:{:.4f}".format(test_acc))
            print("\n" + classification_report(y_true, y_pred))

        return test_acc, test_f1

    def get_optimizer(self):
        optimizers = {
            'adagrad': optims.Adagrad,  # default lr=0.01
            'adam': optims.Adam,  # default lr=0.001
            'sgd': optims.SGD,
        }
        return optimizers[self.args.optim]


def main(args):
    if args.train:
        program = TorchProgram(args)
        program.train()

    if args.test:
        program = TorchProgram(args)
        program.test(program.testloader)


if __name__ == '__main__':
    args = parse_args()
    main(args)
