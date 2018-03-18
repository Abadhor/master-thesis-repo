import sys
import os
import pickle
import io
import json
sys.path.append('../dataset-scripts')
from Preprocessing import Tokenizer, MWUAnnotator, Vectorizer
from MWTDatasetCreator import MWTDatasetCreator
import tensorflow as tf
import numpy as np
from gensim.models.word2vec import Word2Vec
import argparse
import options
import time
import random
random.seed(5)

"""
parser = argparse.ArgumentParser()
parser.add_argument(
      "--data_dir",
      type=str,
      default="",
      required=True,
      help="The location of the GATE inline MWT annotated text files.")
parser.add_argument(
      "--mwt_dir",
      type=str,
      default="",
      required=True,
      help="The name of the output file of the pickled set.")
args = parser.parse_args()
"""

# TODO:
# parse file locations

text_dir = "D:/data/datasets/patent_mwt/plaintext/"
mwt_file = "D:/data/datasets/patent_mwt/mwts/mwts.set"
tmp = "D:/Uni/MasterThesis/master-thesis-repo/tensorflow/tmp/data.tfrecord"
LM_DICT = "vocab-2016-09-10.txt"
LM_DATA = "embeddings_char_cnn.npy"

WORD2VEC = "D:/data/other/clefipnostem/300-1/skipgram.model"
NEW_WORDS = ['.', ',', ';', '$', '(', ')', '§']

# model parameters
params = {}
params['sent_length'] = 60
params['word_length'] = 32
params['gazetteer_count'] = 0
params['hidden_size'] = 350
params['gazetteers_dense_size'] = 50
params['bow_feature_size'] = 100
params['boc_feature_size'] = 21
params['char_cnn_filter_width'] = 3
params['char_cnn_out_features'] = 32
# 32->16, 16->8, 8->4, 4->2, 2->1
params['charCNN_layer_depths'] = [32, 64, 128, 256, 512]
params['char_dense_out_features'] = 50
"""
params['vocab_size'] = len(meta['invDict'])
params['alphabet_size'] = len(meta['alphabet']) + 1
params['num_classes'] = len(meta['labelNames'])
params['word_embeddings'] = meta['word_vectors']
params['LM_embeddings'] = meta['char_CNN_vectors']
params['LM_hidden_size'] = 512
params['no_vector_count'] = meta['no_vector_count']
"""
batch_size = 64
LSTM_layer_count = 1
batch_testing = True


# training parameters
num_epochs = 600
early_stopping_epoch_limit = 60
params['starter_learning_rate'] = 0.01
params['l2-coefficient'] = 0.01
params['grad_clip_norm'] = 5.0
params['decay_steps'] = 200
params['decay_rate'] = 0.96

# selection of files used for training, validation and testing
filenames = [text_dir + fname for fname in os.listdir(text_dir)]
sample = random.sample(filenames, 4)
test_files = sample[:2]
dev_files = sample[2:]
train_files = [fname for fname in filenames if fname not in sample]

# load Multi Word Terms set from files
with io.open(mwt_file, 'rb') as fp:
  mwt_set = pickle.load(fp)

model = Word2Vec.load(WORD2VEC)
dictionary = model.wv.vocab
dictionary = {k:v.count for k, v in dictionary.items()}
for w in NEW_WORDS:
  dictionary[w] = 1

tokenizer = Tokenizer()

# MWT preprocessing
mwt_tokens_list = []
for mwt in mwt_set:
  tokens = tokenizer.substituteTokenize(mwt)
  mwt_tokens_list.append(tokens)

mwt_tokens_list = [x for x in mwt_tokens_list if len(x) > 1]

# MWTDatasetCreator handles text preprocessing, tokenization
# and feature extraction / vectorization
# Sentences are represented as samples that have the following features
# 1. sequence of word IDs for each token in the sentence
# 2. length of the sentence
# 3. sequence of label IDs for each token in the sentence
# 4. sequence of char sequences for each token in the sentence,
# whereas each char sequence consists of char IDs for each char in a word
# 5. sequence of word lengths for each token in the sentence
# 6. ID of the whole sentence
# 7. ID of the sentence part
# 8. number of parts the whole sentence has been split into
# Samples are stored in a temporary file as TFRecords
annotator = MWUAnnotator(mwt_tokens_list)
vectorizer = Vectorizer(annotator, dictionary, {l:1 for l in tokenizer.alphabet}, gazetteers=None)

creator = MWTDatasetCreator(tmp, tokenizer, vectorizer, 60, 32)
dataset = creator.createDataset(train_files)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

times = []
with tf.Session() as sess:
  for i in range(10):
    start_time = time.time()
    sess.run(iterator.initializer)
    while True:
      try:
        t = sess.run(next_element)
      except tf.errors.OutOfRangeError:
        print("End of dataset 2")
        break
    times.append(time.time() - start_time)
print(times)

# clean-up
os.remove(tmp)
