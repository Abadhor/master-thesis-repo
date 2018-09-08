import io
import pickle
import os
from Preprocessing import Tokenizer, MWUAnnotator, Vectorizer
from MWUHashTree import MWUHashTree
import tensorflow as tf
import numpy as np

text_dir = "D:/data/datasets/patent_mwt/plaintext/"

filenames = [text_dir + fname for fname in os.listdir(text_dir)]

mwt_file = "D:/data/datasets/patent_mwt/mwts/mwts.dict"
with io.open(mwt_file, 'rb') as fp:
  mwt_dict = pickle.load(fp)

print("Unique MWTS:", len(mwt_dict))
print("count 1 MWTS:", len([k for k,v in mwt_dict.items() if v == 1]))
print("count >= 5 MWTS:", len([k for k,v in mwt_dict.items() if v >= 5]))
v = np.array(list(mwt_dict.values()), dtype='float32')
print("Mean count:", np.mean(v))
print("Stddev count:", np.std(v))
x = [len(' '.join(m.split('-')).split()) for m in list(mwt_dict.keys())]
v = np.array(x, dtype='float32')
print("Mean len:", np.mean(v))
print("Stddev len:", np.std(v))

tokenizer = Tokenizer(NLTK=False)

mwt_tokens_list = []
for mwt in mwt_dict.keys():
  tokens = tokenizer.substituteTokenize(mwt)
  mwt_tokens_list.append(tokens)

mwt_tokens_list = [x for x in mwt_tokens_list if len(x) > 1]
mwt_hash = MWUHashTree()
for mwt in mwt_tokens_list:
  mwt_hash[mwt] = 0

mwt_dict2 = {" ".join(k):0 for k in mwt_hash.keys()}

annotator = MWUAnnotator(mwt_tokens_list)


def get_sentences(files, prefix, tokenizer):
  dataset = tf.data.TextLineDataset(files)
  get_next = dataset.make_one_shot_iterator().get_next()
  sents = []
  with tf.Session() as sess:
    while True:
      try:
        line = sess.run(get_next).decode('utf-8')
        sents.extend(tokenizer.splitSentences(line))
      except tf.errors.OutOfRangeError:
        print(prefix, "Finished reading text files.")
        break
  return sents

def getCounts(files, prefix, tokenizer):
  for idx, sent in enumerate(get_sentences(files, prefix, tokenizer)):
    zipped = tokenizer.substituteTokenize_with_POS(sent)
    tokens, pos = tuple(zip(*zipped))
    for idx, token in enumerate(tokens):
      found = mwt_hash.getAll(tokens[idx:])
      for mwt in found:
        mwt_dict2[" ".join(mwt)] += 1

getCounts(filenames, "All", tokenizer)
print("Unique MWTS:", len(mwt_dict2))
print("count 1 MWTS:", len([k for k,v in mwt_dict2.items() if v == 1]))
print("count >= 5 MWTS:", len([k for k,v in mwt_dict2.items() if v >= 5]))
print("count >= 10 MWTS:", len([k for k,v in mwt_dict2.items() if v >= 10]))
print("count >= 100 MWTS:", len([k for k,v in mwt_dict2.items() if v >= 100]))
v = np.array(list(mwt_dict2.values()), dtype='float32')
print("Mean count:", np.mean(v))
print("Stddev count:", np.std(v))