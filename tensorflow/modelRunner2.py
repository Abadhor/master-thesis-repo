import sys
import os
import pickle
import io
import json
sys.path.append('../dataset-scripts')
from Preprocessing import Tokenizer, MWUAnnotator, Vectorizer
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

text_dir = "D:/data/datasets/patent_mwt/plaintext/"
mwt_file = "D:/data/datasets/patent_mwt/mwts/mwts.set"
tmp = "D:/Uni/MasterThesis/master-thesis-repo/tensorflow/tmp/data.tfrecord"
LM_DICT = "vocab-2016-09-10.txt"
LM_DATA = "embeddings_char_cnn.npy"

WORD2VEC = "D:/data/other/clefipnostem/300-1/skipgram.model"
NEW_WORDS = ['.', ',', ';', '$', '(', ')', '§']
WORD_MAX_LEN = 16
SENT_MAX_LEN = 50

filenames = [text_dir + fname for fname in os.listdir(text_dir)]
sample = random.sample(filenames, 4)
test_files = sample[:2]
dev_files = sample[2:]
train_files = [fname for fname in filenames if fname not in sample]

with io.open(mwt_file, 'rb') as fp:
  mwt_set = pickle.load(fp)

model = Word2Vec.load(WORD2VEC)
dictionary = model.wv.vocab
dictionary = {k:v.count for k, v in dictionary.items()}
for w in NEW_WORDS:
  dictionary[w] = 1

tokenizer = Tokenizer()

mwt_tokens_list = []
for mwt in mwt_set:
  tokens = tokenizer.substituteTokenize(mwt)
  mwt_tokens_list.append(tokens)

mwt_tokens_list = [x for x in mwt_tokens_list if len(x) > 1]

annotator = MWUAnnotator(mwt_tokens_list)
vectorizer = Vectorizer(annotator, dictionary, {l:1 for l in tokenizer.alphabet}, gazetteers=None)
#text = "A Test."
#tokens = tokenizer.substituteTokenize(text)
#rt = tuple(vectorizer.vectorize(tokens, 3, 4))

class MWTDatasetCreator:
  
  def __init__(self, out_file, tokenizer, vectorizer, maxSentLenght, maxWordLength):
    self.tokenizer = tokenizer
    self.vectorizer = vectorizer
    self.maxSentLenght = maxSentLenght
    self.maxWordLength = maxWordLength
    self.out_file = out_file
  
  def _get_sentences(self, files):
    dataset = tf.data.TextLineDataset(files)
    get_next = dataset.make_one_shot_iterator().get_next()
    sents = []
    with tf.Session() as sess:
      while True:
        try:
          line = sess.run(get_next).decode('utf-8')
          sents.extend(self.tokenizer.splitSentences(line))
        except tf.errors.OutOfRangeError:
          print("End of dataset")
          break
    return sents
  
  def _split_sentences(self, tokens, sentence_id):
    length = len(tokens)
    sentences = []
    sentence_count = (length // self.maxSentLenght)
    remainder = (length % self.maxSentLenght)
    i = 0
    while i < sentence_count:
      start_token = i * self.maxSentLenght
      end_token = start_token + self.maxSentLenght
      sentences.append({
        'tokens':tokens[start_token:end_token],
        'sentence_id': sentence_id,
        'sentence_part_id': i,
        'sentence_part_count': (sentence_count if remainder == 0 else sentence_count+1)
      })
      i += 1
    if remainder > 0:
      start_token = i * self.maxSentLenght
      end_token = start_token + remainder
      sentences.append({
        'tokens':tokens[start_token:end_token],
        'sentence_id': sentence_id,
        'sentence_part_id': i,
        'sentence_part_count': sentence_count+1
      })
    return sentences
  
  def createDataset(self, files):
    tokenized_sentences = []
    for idx, sent in enumerate(self._get_sentences(files)):
      tokens = self.tokenizer.substituteTokenize(sent)
      tokenized_sentences.extend(self._split_sentences(tokens,idx))
    
    writer = tf.python_io.TFRecordWriter(self.out_file)
    for sent in tokenized_sentences:
      token_vector, length, label_vector, char_vector, char_lengths = self.vectorizer.vectorize(sent['tokens'], self.maxSentLenght, self.maxWordLength)
      feature = {
        'tokens': tf.train.Feature(int64_list=tf.train.Int64List(value=token_vector.flatten())),
        'length': tf.train.Feature(int64_list=tf.train.Int64List(value=length.flatten())),
        'labels': tf.train.Feature(int64_list=tf.train.Int64List(value=label_vector.flatten())),
        'chars': tf.train.Feature(int64_list=tf.train.Int64List(value=char_vector.flatten())),
        'char_lengths': tf.train.Feature(int64_list=tf.train.Int64List(value=char_lengths.flatten())),
        'sentence_id': tf.train.Feature(int64_list=tf.train.Int64List(value=np.array((sent['sentence_id'],)).flatten())),
        'sentence_part_id': tf.train.Feature(int64_list=tf.train.Int64List(value=np.array((sent['sentence_part_id'],)).flatten())),
        'sentence_part_count': tf.train.Feature(int64_list=tf.train.Int64List(value=np.array((sent['sentence_part_count'],)).flatten())),
      }
      example = tf.train.Example(features=tf.train.Features(feature=feature))
      serialized = example.SerializeToString()
      writer.write(serialized)
    
    writer.close()
    
    return None
  
  def parse_proto(example_proto):
    features = {
      'X': tf.FixedLenFeature((345,), tf.float32),
      'y': tf.FixedLenFeature((5,), tf.float32),
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    return parsed_features['X'], parsed_features['y']
  
  

def _decode_encode_wrapper(x, fun):
    return np.array(
      [ele.encode('utf-8') for ele in fun(x.decode('utf-8'))],
      dtype='object')


def createDataset(files, tokenizer, annotator):
  dataset = tf.data.TextLineDataset(files)
  # split sentences flat map
  dataset = dataset.flat_map(
    lambda x: tf.data.Dataset.from_tensor_slices(tf.py_func(
      lambda y: _decode_encode_wrapper(y, tokenizer.splitSentences),
      [x], tf.string)))
  # tokenize map
  dataset = dataset.map(lambda x: tf.py_func(
    lambda y: _decode_encode_wrapper(y, tokenizer.substituteTokenize), [x], tf.string))
  get_next = dataset.make_one_shot_iterator().get_next()
  with tf.Session() as sess, io.open('./test.txt', 'w', encoding='utf-8') as fp:
    while True:
      try:
        line = sess.run(get_next)
        #line = line.decode('utf-8')
        #tokens = tokenizer.substituteTokenize(line)
        #labels = annotator.getLabels(tokens)
        #fp.write(str([[t[0],t[1]] for t in zip(tokens, labels)]) + '\n')
        fp.write(str([ele.decode('utf-8') for ele in line]) + '\n')
      except tf.errors.OutOfRangeError:
        print("End of dataset")
        break

#createDataset(train_files, tokenizer, annotator)
c = MWTDatasetCreator(tmp, tokenizer, vectorizer, 2, 4)
d2 = c.createDataset(train_files)
exit()
iterator = d2.make_initializable_iterator()
next_element = iterator.get_next()
times = []
with tf.Session() as sess, io.open('./test2.txt', 'w', encoding='utf-8') as fp:
  for i in range(1):
    start_time = time.time()
    sess.run(iterator.initializer)
    while True:
      try:
        t = sess.run(next_element)
        #fp.write('tokens:'+str(t[0])+';'+ 'length:'+str(t[1])+';'+'labels:'+str(t[2])+';'+'chars:'+str(t[3])+';'+'charLens:'+str(t[4])+';' + '\n')
      except tf.errors.OutOfRangeError:
        print("End of dataset 2")
        break
    times.append(time.time() - start_time)
print(times)
