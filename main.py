# -*- coding:utf-8 -*-

import os
import time
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optims
from sklearn.metrics import f1_score, classification_report
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from args import parse_args
from dataset import SEResDataset
from models.atae_lstm import MYATAE_LSTM

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tokenizer import get_tokenizer


class TorchProgram:

    def __init__(self, args):

        if args.use_gpu and torch.cuda.is_available():
            cudnn.benchmark = True
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            use_gpu = True
        else:
            print("Currently using CPU, however, GPU is highly recommended")
            use_gpu = False

        self.args = args
        self.device = torch.device('cuda' if use_gpu else 'cpu')
        self.model = None

        tokenizer = get_tokenizer(args)

        if args.model_name == 'atae_lstm':

            self.model = MYATAE_LSTM(tokenizer.embeddings, embed_dim=tokenizer.embed_dim,
                                     hidden_dim=args.hidden_dim, device=self.device)
            args.inputs_cols = ['text_raw_indices', 'aspect_indices']
        else:
            raise ValueError('invalid model')

        # data loader
        if args.train:
            train_dataset = SEResDataset(data_path=args.train_file, tokenizer=tokenizer)
            self.trainloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

        test_dataset = SEResDataset(data_path=args.test_file, tokenizer=tokenizer)
        self.testloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

        if use_gpu:
            self.model = nn.DataParallel(self.model).cuda()

        self.log_writer = SummaryWriter(comment=args.pretrain_type)
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

            total_loss = 0
            for batch_id, batch_data in enumerate(self.trainloader, 1):
                self.model.train()
                start_time = time.time()
                optimizer.zero_grad()

                # to device
                inputs = [batch_data[col].to(self.device) for col in self.args.inputs_cols]
                outputs = self.model(inputs)
                targets = batch_data['polarity'].to(self.device)

                loss = criterion(outputs, targets)
                total_loss += loss
                loss.backward()
                optimizer.step()

                end_time = time.time()

                total_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                total_num += len(targets)
                train_acc = total_correct / total_num

                # return
                self.log_writer.add_scalar('data/loss', total_loss / batch_id, global_step)
                self.log_writer.add_scalar('data/train_acc', train_acc, global_step)

                if args.test_step > 0 and global_step % args.test_step == 0:
                    test_acc, test_f1 = self.test(self.testloader)

                    self.log_writer.add_scalar('data/test_acc', test_acc, global_step)
                    self.log_writer.add_scalar('data/test_f1', test_f1, global_step)

                    print(
                        'Epoch:{}, Batch_id:{}, Time:{:.2f}, Loss:{:.4f}, Train_acc:{:.4f},  Test_acc:{:.4f}, Test_f1:{:.4f}'.format(
                            epoch, batch_id, (end_time - start_time), total_loss / batch_id, train_acc, test_acc,
                            test_f1
                        ))
                    if max_test_acc < test_acc:
                        max_test_acc = test_acc
                        this_test_f1 = test_f1
                        if args.save_dir:
                            os.makedirs(args.save_dir, exist_ok=True)

                        best_mode_path = '{}/{}_{}_{}_{:.4f}'.format(args.save_dir, args.model_name, args.pretrain_type,
                                                                     epoch, test_acc)
                        print('save model to {}'.format(best_mode_path))

                        torch.save({
                            'model': self.model.state_dict(),
                            'epoch': epoch
                        }, best_mode_path)
                else:
                    print('Epoch:{}, Batch:{}, Time:{:.2f}, Loss:{:.4f}, Train_acc:{:.4f}'.format(
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


def write_config(args):
    configs = vars(args)
    run_time = datetime.now().strftime('%Y%m%d-%H%M')
    with open('config/{}-{}-{}.conf'.format(args.model_name, args.pretrain_type, run_time), 'w',
              encoding='utf-8') as fout:
        for config, value in configs.items():
            fout.write('{}={}\n'.format(config, value))


def main(args):
    if args.train:
        program = TorchProgram(args)
        program.train()
        write_config(args)

    if args.test:
        program = TorchProgram(args)
        program.test(program.testloader)


if __name__ == '__main__':
    args = parse_args()
    main(args)
