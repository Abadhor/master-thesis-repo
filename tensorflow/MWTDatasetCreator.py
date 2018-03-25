import tensorflow as tf
import numpy as np

class MWTDatasetCreator:
  
  def __init__(self, tokenizer, vectorizer, maxSentLenght, maxWordLength):
    self.tokenizer = tokenizer
    self.vectorizer = vectorizer
    self.maxSentLenght = maxSentLenght
    self.maxWordLength = maxWordLength
  
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
          print("Finished reading text files.")
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
  
  def _parse_proto(self, example_proto):
    features = {
      'tokens': tf.FixedLenFeature((self.maxSentLenght,), tf.int64),
      'length': tf.FixedLenFeature((), tf.int64),
      'labels': tf.FixedLenFeature((self.maxSentLenght,), tf.int64),
      'chars': tf.FixedLenFeature((self.maxSentLenght,self.maxWordLength), tf.int64),
      'char_lengths': tf.FixedLenFeature((self.maxSentLenght,), tf.int64),
      'sentence_id': tf.FixedLenFeature((), tf.int64),
      'sentence_part_id': tf.FixedLenFeature((), tf.int64),
      'sentence_part_count': tf.FixedLenFeature((), tf.int64),
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    #return parsed_features['tokens'], parsed_features['length'], parsed_features['labels'], parsed_features['chars'], parsed_features['char_lengths'], parsed_features['sentence_id'], parsed_features['sentence_part_id'], parsed_features['sentence_part_count']
    labels = parsed_features['labels']
    del parsed_features['labels']
    return parsed_features, labels
  
  
  def createDataset(self, files, out_file):
    tokenized_sentences = []
    full_lengths = []
    for idx, sent in enumerate(self._get_sentences(files)):
      tokens = self.tokenizer.substituteTokenize(sent)
      full_lengths.append(len(tokens))
      tokenized_sentences.extend(self._split_sentences(tokens,idx))
    full_lengths = np.array(full_lengths)
    print("Finished tokenization")
    print("Sentence Count:", len(full_lengths))
    print("Sentence Count (Split):", len(tokenized_sentences))
    print("Sentence Length Mean:", np.mean(full_lengths, axis=0))
    print("Sentence Length Stdev:", np.std(full_lengths, axis=0))
    
    writer = tf.python_io.TFRecordWriter(out_file)
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
    print("End of preprocessing")
    dataset = tf.data.TFRecordDataset([out_file])
    dataset = dataset.map(self._parse_proto)
    return dataset, len(tokenized_sentences)
 