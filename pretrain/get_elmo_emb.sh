#!/usr/bin/env bash

wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json
wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5

# test elmo
echo "The cryptocurrency space is now figuring out to have the highest search on Google globally ." > sentences.txt
echo "Bitcoin alone has a sixty percent share of global search ." >> sentences.txt

allennlp elmo \
--options-file elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json \
--weight-file elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5 \
test_elmo_sentences.txt elmo_layers.hdf5 --all