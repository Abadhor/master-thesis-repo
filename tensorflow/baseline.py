from Preprocessing import Tokenizer, MWUAnnotator, Vectorizer
from TrainHelpers import TrainHelpers
import tensorflow as tf
import re
import os
import io
import pickle
import random
random.seed(5)


text_dir = "D:/data/datasets/patent_mwt/plaintext/"
mwt_file = "D:/data/datasets/patent_mwt/mwts/mwts.set"


# selection of files used for training, validation and testing

filenames = [text_dir + fname for fname in os.listdir(text_dir)]
sample = random.sample(filenames, 4)
test_files = sample[:2]
val_files = sample[2:]
train_files = [fname for fname in filenames if fname not in sample]
print("------------------------------Train Files------------------------------")
print(train_files)
print("----------------------------Validation Files---------------------------")
print(val_files)
print("------------------------------Test Files-------------------------------")
print(test_files)

# load Multi Word Terms set from files

with io.open(mwt_file, 'rb') as fp:
  mwt_set = pickle.load(fp)

# tokenizer and params

tokenizer = Tokenizer()

# MWT preprocessing

mwt_tokens_list = []
for mwt in mwt_set:
  tokens = tokenizer.substituteTokenize(mwt)
  mwt_tokens_list.append(tokens)

mwt_tokens_list = [x for x in mwt_tokens_list if len(x) > 1]


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

def getMWTs(sentence):
  # get MWTs in a sentence
  MWTs = []
  B_idx = None
  for idx, t in enumerate(sentence):
    if t == 'B':
      B_idx = idx
    elif t == 'L':
      if B_idx != None:
        MWTs.append((B_idx,idx))
      B_idx = None
  return MWTs

def getMWTs_POS(pos, tokenizer):
  # get MWTs in a list of POS tags
  # valid sequences:
  # 1. (Adj|Noun)+ Noun
  # 2. ((<ADJ|NOUN>+<ADP>)+<ADJ|NOUN>*)<NOUN>
  char = tokenizer.POS2Char(pos) # convert pos to chars a,n,p,o
  p = re.compile('(a|n)+n')
  #p2 = re.compile('((a|n)+p)+(a|n)*n')
  MWTs = set()
  for i in re.finditer(p, char):
    span = i.span()
    MWTs.add((span[0], span[1]-1))
  #for i in re.finditer(p2, char):
  #  span = i.span()
  #  MWTs.add((span[0], span[1]-1))
  return list(MWTs)
  
  
def baseline(files, prefix, tokenizer):
  sents = []
  GT_count = 0
  TP_count = 0
  FP_count = 0
  for idx, sent in enumerate(get_sentences(files, prefix, tokenizer)):
    zipped = tokenizer.substituteTokenize_with_POS(sent)
    tokens, pos = tuple(zip(*zipped)) #unzip
    labels = annotator.getLabels(tokens)
    GT_list = getMWTs(labels)
    GT_count += len(GT_list)
    GT_set = set()
    for mwt in GT_list:
      GT_set.add(mwt)
    predictions = getMWTs_POS(pos, tokenizer)
    for mwt in predictions:
      if mwt in GT_set:
        TP_count += 1
      else:
        FP_count += 1
  FN_count = GT_count - TP_count
  precision, recall, F1 = TrainHelpers.getScores(TP_count, FP_count, FN_count)
  print(prefix+"set precision: {:.3%}".format(precision))
  print(prefix+"set recall: {:.3%}".format(recall))
  print(prefix+"set F1: {:.3%}".format(F1))

baseline(train_files, "Train", tokenizer)
baseline(val_files, "Validation", tokenizer)
baseline(test_files, "Test", tokenizer)

