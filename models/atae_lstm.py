# -*- coding:utf-8 -*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.modules.elmo import Elmo


class MYATAE_LSTM(nn.Module):
    """Reproduction of ATAE_LSTM"""
    def __init__(self, embeddings, embed_dim, hidden_dim, device, polarities_dim=3):
        super(MYATAE_LSTM, self).__init__()
        self.device = device
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embeddings, dtype=torch.float))

        self.lstm = nn.LSTM(embed_dim * 2, hidden_dim, num_layers=1, batch_first=True)

        self.Wh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.Wv = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        self.w = nn.Parameter(torch.Tensor(embed_dim + hidden_dim, 1))

        self.reset_atten_para(embed_dim, hidden_dim)
        self.dense = nn.Linear(hidden_dim, polarities_dim)

    def reset_atten_para(self, embed_dim, hidden_dim):
        stdv = 1. / math.sqrt(hidden_dim)
        self.Wh.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(embed_dim)
        self.Wv.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(embed_dim + hidden_dim)
        self.w.data.uniform_(-stdv, stdv)

    def attention(self, H, aspect):
        # H:(batch, seq_len, hidden_dim)
        # aspect: batch_size, seq_len, embed_dim

        # H:(batch, hidden_dim, seq_len)
        H = H.permute(0, 2, 1)
        m1 = torch.matmul(self.Wh, H)

        aspect = aspect.permute(0, 2, 1)
        m2 = torch.matmul(self.Wv, aspect)

        # batch_size, embed_dim + hidden_dim, seq_len
        M = torch.cat((m1, m2), dim=1)

        # batch_size, 1, seq_len
        x = torch.matmul(self.w.transpose(0, 1), M)

        # batch_size, seq_len
        x = x.squeeze(1)
        # print(x)
        a = F.softmax(x, dim=-1)
        # print(a)

        return a

    def forward(self, inputs):
        text_raw_indices, aspect_indices = inputs[0], inputs[1]
        seq_len = text_raw_indices.size(1)

        # batch_size, seq_len, embed_size
        x = self.embed(text_raw_indices)
        aspect = self.embed(aspect_indices)

        # batch_size, embed_size
        aspect = torch.sum(aspect, dim=1)

        # batch_size, seq_len, embed_size
        aspect = aspect.unsqueeze(dim=1).expand(-1, seq_len, -1)

        # batch_size, seq_len, embed_size * 2
        x = torch.cat((x, aspect), dim=-1)

        # out:(batch, seq_len, hidden_dim)
        out, (ht, ct) = self.lstm(x)

        # batch_size, seq_len
        score = self.attention(out, aspect)
        # batch_size, seq_len, 1
        score = score.unsqueeze(-1)

        # [batch, hidden_dim, seq_len] * [batch_size, seq_len, 1] = [batch, hidden_dim, 1]
        output = torch.squeeze(torch.bmm(out.permute(0, 2, 1), score), dim=-1)

        out = self.dense(output)

        return out



class ELMoModel(nn.Module):
    """
    https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md
    """

    def __init__(self, hidden_dim, device, polarities_dim=3,
                 embed_dim=1024,
                 options_file='./pretrain/elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json',
                 weight_file='./pretrain/elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'
                 ):
        super(ELMoModel, self).__init__()
        self.elmo = Elmo(options_file, weight_file,
                         num_output_representations=3,
                         requires_grad=False)
        self.device = device
        self.lstm = nn.LSTM(embed_dim * 2, hidden_dim, num_layers=1, batch_first=True, bidirectional=False)

        self.Wh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.Wv = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        self.w = nn.Parameter(torch.Tensor(embed_dim + hidden_dim, 1))

        self.reset_atten_para(embed_dim, hidden_dim)

        self.dense = nn.Linear(hidden_dim, polarities_dim)

    def reset_atten_para(self, embed_dim, hidden_dim):
        stdv = 1. / math.sqrt(hidden_dim)
        self.Wh.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(embed_dim)
        self.Wv.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(embed_dim + hidden_dim)
        self.w.data.uniform_(-stdv, stdv)

    def embed(self, inputs):
        embeddings = self.elmo(inputs)
        # batch_size, seq_len, embed_siz=1024
        # char_emb = embeddings['elmo_representations'][0]
        bi1_emb = embeddings['elmo_representations'][1]
        bi2_emb = embeddings['elmo_representations'][2]

        # batch_size, seq_len, embed_size=1024
        word_emb = torch.add(bi1_emb, bi2_emb)
        # word_emb = torch.add(word_emb, char_emb)
        return word_emb

    def attention(self, H, aspect):
        # H:(batch, seq_len, hidden_dim)
        # aspect: batch_size, seq_len, embed_dim

        # H:(batch, hidden_dim, seq_len)
        H = H.permute(0, 2, 1)
        m1 = torch.matmul(self.Wh, H)

        aspect = aspect.permute(0, 2, 1)
        m2 = torch.matmul(self.Wv, aspect)

        # batch_size, embed_dim + hidden_dim, seq_len
        M = torch.cat((m1, m2), dim=1)

        # batch_size, 1, seq_len
        x = torch.matmul(self.w.transpose(0, 1), M)

        # batch_size, seq_len
        x = x.squeeze(1)
        # print(x)
        a = F.softmax(x, dim=-1)
        # print(a)

        return a

    def forward(self, inputs):
        text_raw_indices, aspect_indices = inputs[0], inputs[1]
        seq_len = text_raw_indices.size(1)

        # batch_size, seq_len, embed_size
        x = self.embed(text_raw_indices)
        aspect = self.embed(aspect_indices)

        # batch_size, embed_size
        aspect = torch.sum(aspect, dim=1)

        # batch_size, seq_len, embed_size
        aspect = aspect.unsqueeze(dim=1).expand(-1, seq_len, -1)

        # batch_size, seq_len, embed_size * 2
        x = torch.cat((x, aspect), dim=-1)

        # out:(batch, seq_len, hidden_dim)
        out, (ht, ct) = self.lstm(x)

        # batch_size, seq_len
        score = self.attention(out, aspect)
        # batch_size, seq_len, 1
        score = score.unsqueeze(-1)

        # [batch, hidden_dim, seq_len] * [batch_size, seq_len, 1] = [batch, hidden_dim, 1]
        output = torch.squeeze(torch.bmm(out.permute(0, 2, 1), score), dim=-1)

        out = self.dense(output)

        return out
