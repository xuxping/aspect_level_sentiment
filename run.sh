#!/usr/bin/env bash

# create word2vec
python create_embedding.py --pretrain_type word2vec \
--pretrain_path ./pretrain/GoogleNews-vectors-negative300.txt \
--vocab_path ./data/vocab/word2vec.vocab.data


# train word2vec
python main.py --train --pretrain_type word2vec \
--pretrain_path ./pretrain/sn.word2vec.300d.txt \
--vocab_path ./data/vocab/word2vec.vocab.data \
--model_name atae_lstm \
--test_step 10 \
--lr 0.001 \
--save_dir w2c_stat_dict \
--max_epoch=20 \
--freeze=0

# train glove
python main.py --train --pretrain_type glove \
--pretrain_path ./pretrain/sn.glove.300d.txt \
--vocab_path ./data/vocab/glove.vocab.data \
--model_name atae_lstm \
--test_step 10 \
--save_dir glove_stat_dict \
--max_epoch=20 \
--lr 0.001


#train elmo
python train_elmo.py --train --lr 0.0005 --max_epoch=20 --test_step=100 --save_dir elmo_stat_dict

# train bert

python train_bert.py --do_train --output_dir bert_models/
python train_bert.py --do_test --output_dir bert_models/



# test
python main.py --test --pretrain_type baseline \
--vocab_path ./data/vocab/baseline.vocab.data \
--load_dir ./stat_dict/atae_lstm_baseline_0_0.6500

# test word2vec
 python main.py --test --pretrain_type word2vec \
 --load_dir stat_dict/atae_lstm_word2vec_5_0.7812 \
 --vocab_path data/vocab/word2vec.vocab.data

# test glove
 python main.py --test --pretrain_type glove \
 --load_dir stat_dict/atae_lstm_glove_2_0.7786 \
 --vocab_path data/vocab/glove.vocab.data
