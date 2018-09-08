from Preprocessing import Tokenizer, MWUAnnotator, Vectorizer
from TrainHelpers import TrainHelpers
from MWUHashTree import MWUHashTree
import tensorflow as tf
import re
import os
import io
import pickle
import random
random.seed(5) # Split A
#random.seed(15) # Split B

text_dir = "D:/data/datasets/patent_mwt/plaintext/"
mwt_file = "D:/data/datasets/patent_mwt/mwts/mwts.dict"

NLTK = False
prepositions = False


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

# load Multi Word Terms dict from files

with io.open(mwt_file, 'rb') as fp:
  mwt_dict = pickle.load(fp)

# tokenizer and params

tokenizer = Tokenizer(NLTK=NLTK)

# MWT preprocessing

mwt_tokens_list = []
for mwt in mwt_dict.keys():
  tokens = tokenizer.substituteTokenize(mwt)
  mwt_tokens_list.append(tokens)

mwt_tokens_list = [x for x in mwt_tokens_list if len(x) > 1]
mwt_hash = MWUHashTree()
for mwt in mwt_tokens_list:
  mwt_hash[mwt] = 0

mwt_counts_dict = {" ".join(k):0 for k in mwt_hash.keys()}


annotator = MWUAnnotator(mwt_tokens_list)

def getCounts(files, prefix, tokenizer):
  for idx, sent in enumerate(get_sentences(files, prefix, tokenizer)):
    zipped = tokenizer.substituteTokenize_with_POS(sent)
    tokens, pos = tuple(zip(*zipped))
    for idx, token in enumerate(tokens):
      found = mwt_hash.getAll(tokens[idx:])
      for mwt in found:
        mwt_counts_dict[" ".join(mwt)] += 1

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
  if prepositions:
    #p = re.compile('((a|n)+p)+(a|n)*n')
    p = re.compile('((a|n)+|((a|n)*(np)?)(a|n)*)n')
  else:
    p = re.compile('(a|n)+n')
  MWTs = set()
  for i in re.finditer(p, char):
    span = i.span()
    MWTs.add((span[0], span[1]-1))
  #for i in re.finditer(p2, char):
  #  span = i.span()
  #  MWTs.add((span[0], span[1]-1))
  return list(MWTs)
  
  
def baseline(files, prefix, tokenizer, mwt_dict, chunking=False):
  sents = []
  GT_count = 0
  TP_count = 0
  FP_count = 0
  for idx, sent in enumerate(get_sentences(files, prefix, tokenizer)):
    if NLTK:
      zipped = tokenizer.substituteTokenize_with_POS_NLTK(sent)
    else:
      zipped = tokenizer.substituteTokenize_with_POS(sent)
    tokens, pos = tuple(zip(*zipped)) #unzip
    labels = annotator.getLabels(tokens)
    GT_list = getMWTs(labels)
    GT_count += len(GT_list)
    GT_set = set()
    for mwt in GT_list:
      GT_set.add(mwt)
      #add mwt to rare dict
      mwt_tokens = tokens[mwt[0]:mwt[1]+1]
      key = " ".join(mwt_tokens)
      if key not in mwt_dict:
        print(prefix+':','"',key, '" not found!')
      else:
        ls = mwt_dict[key]
        ls[1] += 1
    if chunking:
      pass
    else:
      predictions = getMWTs_POS(pos, tokenizer)
    for mwt in predictions:
      if mwt in GT_set:
        TP_count += 1
        #add mwt to rare dict
        mwt_tokens = tokens[mwt[0]:mwt[1]+1]
        key = " ".join(mwt_tokens)
        if key in mwt_dict:
          ls = mwt_dict[key]
          ls[2] += 1
      else:
        FP_count += 1
  FN_count = GT_count - TP_count
  precision, recall, F1 = TrainHelpers.getScores(TP_count, FP_count, FN_count)
  print(prefix+"set GT count:",GT_count)
  print(prefix+"set precision: {:.3%}".format(precision))
  print(prefix+"set recall: {:.3%}".format(recall))
  print(prefix+"set F1: {:.3%}".format(F1))
  process_rare(mwt_dict)

def process_rare(mwt_dict):
  #filter mwts not in file
  mwt_dict = {k:v for k,v in mwt_dict.items() if v[1] != 0}
  # sort that rarest is first
  mwt_list = sorted(list(mwt_dict.items()), key=lambda t: t[1][0])
  rare_list = [mwt for mwt in mwt_list if mwt[1][0] < 5]
  rare_set_count = sum([mwt[1][1] for mwt in rare_list])
  rare_pred_count = sum([mwt[1][2] for mwt in rare_list])
  print("Rare MWT (total count < 5) recall: {:.3%}\n".format(rare_pred_count/rare_set_count))

getCounts(filenames, "All", tokenizer)
"""
baseline(train_files, "Train", tokenizer,
{mwt:[total_count,0,0] for mwt, total_count in mwt_counts_dict.items()})
baseline(val_files, "Validation", tokenizer,
{mwt:[total_count,0,0] for mwt, total_count in mwt_counts_dict.items()})
baseline(test_files, "Test", tokenizer,
{mwt:[total_count,0,0] for mwt, total_count in mwt_counts_dict.items()})
baseline(filenames, "All", tokenizer,
{mwt:[total_count,0,0] for mwt, total_count in mwt_counts_dict.items()})
"""
print(filenames[0])
baseline(filenames[0:1], "00", tokenizer,
{mwt:[total_count,0,0] for mwt, total_count in mwt_counts_dict.items()})
print(filenames[1])
baseline(filenames[1:2], "01", tokenizer,
{mwt:[total_count,0,0] for mwt, total_count in mwt_counts_dict.items()})
print(filenames[7])
baseline(filenames[7:8], "08", tokenizer,
{mwt:[total_count,0,0] for mwt, total_count in mwt_counts_dict.items()})
print(filenames[16])
baseline(filenames[16:17], "17", tokenizer,
{mwt:[total_count,0,0] for mwt, total_count in mwt_counts_dict.items()})

