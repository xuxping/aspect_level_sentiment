# -*- coding:utf-8 -*-
import gensim

if __name__ == '__main__':
    # Load Google's pre-trained Word2Vec model.
    model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
    # Convert word2vec bin file to text
    model.save_word2vec_format('GoogleNews-vectors-negative300.txt', binary=False)
