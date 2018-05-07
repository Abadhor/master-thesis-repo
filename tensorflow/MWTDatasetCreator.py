import tensorflow as tf
import numpy as np
import os
import io

class MWTDatasetCreator:
  
  def __init__(self, tokenizer, vectorizer, maxSentLenght, maxWordLength, cache_outfile=False):
    self.tokenizer = tokenizer
    self.vectorizer = vectorizer
    self.maxSentLenght = maxSentLenght
    self.maxWordLength = maxWordLength
    self.cache_outfile = cache_outfile
  
  def _get_sentences(self, files, prefix):
    dataset = tf.data.TextLineDataset(files)
    get_next = dataset.make_one_shot_iterator().get_next()
    sents = []
    with tf.Session() as sess:
      while True:
        try:
          line = sess.run(get_next).decode('utf-8')
          sents.extend(self.tokenizer.splitSentences(line))
        except tf.errors.OutOfRangeError:
          print(prefix, "Finished reading text files.")
          break
    return sents
  
  def _split_sentences(self, z_tokens, sentence_id):
    length = len(z_tokens)
    sentences = []
    sentence_count = (length // self.maxSentLenght)
    remainder = (length % self.maxSentLenght)
    i = 0
    while i < sentence_count:
      start_token = i * self.maxSentLenght
      end_token = start_token + self.maxSentLenght
      sentences.append({
        'z_tokens':z_tokens[start_token:end_token],
        'sentence_id': sentence_id,
        'sentence_part_id': i,
        'sentence_part_count': (sentence_count if remainder == 0 else sentence_count+1)
      })
      i += 1
    if remainder > 0:
      start_token = i * self.maxSentLenght
      end_token = start_token + remainder
      sentences.append({
        'z_tokens':z_tokens[start_token:end_token],
        'sentence_id': sentence_id,
        'sentence_part_id': i,
        'sentence_part_count': sentence_count+1
      })
    return sentences
  
  def _parse_proto(self, example_proto):
    features = {
      'tokens': tf.FixedLenFeature((self.maxSentLenght,), tf.int64),
      'length': tf.FixedLenFeature((), tf.int64),
      'labels': tf.FixedLenFeature((self.maxSentLenght,), tf.int64),
      'chars': tf.FixedLenFeature((self.maxSentLenght,self.maxWordLength), tf.int64),
      'char_lengths': tf.FixedLenFeature((self.maxSentLenght,), tf.int64),
      'pos_tags': tf.FixedLenFeature((self.maxSentLenght,), tf.int64),
      'sentence_id': tf.FixedLenFeature((), tf.int64),
      'sentence_part_id': tf.FixedLenFeature((), tf.int64),
      'sentence_part_count': tf.FixedLenFeature((), tf.int64),
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    #return parsed_features['tokens'], parsed_features['length'], parsed_features['labels'], parsed_features['chars'], parsed_features['char_lengths'], parsed_features['sentence_id'], parsed_features['sentence_part_id'], parsed_features['sentence_part_count']
    labels = parsed_features['labels']
    del parsed_features['labels']
    return parsed_features, labels
  
  
  def createDataset(self, files, out_file, prefix=""):
    if not (self.cache_outfile and os.path.isfile(out_file) and os.path.isfile(out_file+'.len')):
      tokenized_sentences = []
      full_lengths = []
      sum_full_lengths = 0
      for idx, sent in enumerate(self._get_sentences(files, prefix)):
        zipped = self.tokenizer.substituteTokenize_with_POS(sent)
        full_lengths.append(len(zipped))
        sum_full_lengths += len(zipped)
        tokenized_sentences.extend(self._split_sentences(zipped,idx))
      full_lengths = np.array(full_lengths)
      print(prefix,"Finished tokenization")
      print(prefix,"Token Count:", sum_full_lengths)
      print(prefix,"Sentence Count:", len(full_lengths))
      print(prefix,"Sentence Count (Split):", len(tokenized_sentences))
      print(prefix,"Sentence Length Mean:", np.mean(full_lengths, axis=0))
      print(prefix,"Sentence Length Stdev:", np.std(full_lengths, axis=0))
      num_samples = len(tokenized_sentences)
      with io.open(out_file+'.len', 'w', encoding='utf-8') as fp:
        fp.write(str(num_samples))
    # write to outfile
      writer = tf.python_io.TFRecordWriter(out_file)
      for sent in tokenized_sentences:
        t = tuple(zip(*sent['z_tokens'])) #unzip
        features = {
          'tokens':t[0],
          'pos_tags':t[1]
        }
        vectors = self.vectorizer.vectorize(features, self.maxSentLenght, self.maxWordLength)
        feature = {
          'tokens': tf.train.Feature(int64_list=tf.train.Int64List(value=vectors['token_vector'].flatten())),
          'length': tf.train.Feature(int64_list=tf.train.Int64List(value=vectors['length'].flatten())),
          'labels': tf.train.Feature(int64_list=tf.train.Int64List(value=vectors['label_vector'].flatten())),
          'chars': tf.train.Feature(int64_list=tf.train.Int64List(value=vectors['char_vector'].flatten())),
          'char_lengths': tf.train.Feature(int64_list=tf.train.Int64List(value=vectors['char_lengths'].flatten())),
          'pos_tags': tf.train.Feature(int64_list=tf.train.Int64List(value=vectors['pos_tags'].flatten())),
          'sentence_id': tf.train.Feature(int64_list=tf.train.Int64List(value=np.array((sent['sentence_id'],)).flatten())),
          'sentence_part_id': tf.train.Feature(int64_list=tf.train.Int64List(value=np.array((sent['sentence_part_id'],)).flatten())),
          'sentence_part_count': tf.train.Feature(int64_list=tf.train.Int64List(value=np.array((sent['sentence_part_count'],)).flatten())),
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        serialized = example.SerializeToString()
        writer.write(serialized)
      writer.close()
      print(prefix,"End of preprocessing")
    else:
      with io.open(out_file+'.len', 'r', encoding='utf-8') as fp:
        num_samples = int(fp.readline())
    # read from outfile
    dataset = tf.data.TFRecordDataset([out_file])
    dataset = dataset.map(self._parse_proto)
    return dataset, num_samples
  
  def deleteOutfiles(self, out_file):
    os.remove(out_file)
    os.remove(out_file+'.len')
 