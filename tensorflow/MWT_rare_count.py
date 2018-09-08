import io
import pickle
import os
import json
from Preprocessing import Tokenizer, MWUAnnotator, Vectorizer
from MWUHashTree import MWUHashTree
import tensorflow as tf
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
      "--text_dir",
      type=str,
      required=True,
      help="The location of the text dataset.")
parser.add_argument(
      "--mwt_file",
      type=str,
      required=True,
      help="The location of the ground truth MWTs.")
parser.add_argument(
      "--out_dir",
      type=str,
      required=True,
      help="The location of the processed output files.")
parser.add_argument(
      "--data_files",
      type=str,
      required=True,
      nargs='+',
      help="List of files containing demo output.")
args = parser.parse_args()

#text_dir = "D:/data/datasets/patent_mwt/plaintext/"
text_dir = args.text_dir

filenames = [text_dir + fname for fname in os.listdir(text_dir)]

#mwt_file = "D:/data/datasets/patent_mwt/mwts/mwts.dict"
mwt_file = args.mwt_file
with io.open(mwt_file, 'rb') as fp:
  mwt_dict = pickle.load(fp)


tokenizer = Tokenizer(NLTK=False)

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
        mwt_counts_dict[" ".join(mwt)] += 1

def process_datafile(df, mwt_dict):
  print("File: ", df)
  fname = os.path.basename(df)
  out_path = args.out_dir + fname + '.rare'
  out_file = io.open(out_path, 'w')
  f = io.open(df, 'r')
  lines = f.read().splitlines()
  i = 0
  for line in lines:
    if i % 3 == 1: #ground truth line
      process_GT(json.loads(line), mwt_dict)
    elif i % 3 == 2: #prediction line
      process_pred(json.loads(line), mwt_dict)
    i += 1
  #filter mwts not in file
  mwt_dict = {k:v for k,v in mwt_dict.items() if v[1] != 0}
  # sort that rarest is first
  mwt_list = sorted(list(mwt_dict.items()), key=lambda t: t[1][0])
  rare_list = [mwt for mwt in mwt_list if mwt[1][0] < 5]
  rare_file_count = sum([mwt[1][1] for mwt in rare_list])
  rare_pred_count = sum([mwt[1][2] for mwt in rare_list])
  out_file.write("Rare MWT (total count < 5) recall: {:.3%}\n".format(rare_pred_count/rare_file_count))
  out_file.write(json.dumps(mwt_list))
  f.close()
  out_file.close()

def process_GT(list, mwt_dict):
  for mwt in list:
    key = " ".join(mwt)
    key = key.lower()
    if key not in mwt_dict:
      print('"',key, '" not found!')
      return
    ls = mwt_dict[key]
    ls[1] += 1

def process_pred(list, mwt_dict):
  for mwt in list:
    key = " ".join(mwt)
    if key in mwt_dict:
      ls = mwt_dict[key]
      ls[2] += 1

getCounts(filenames, "All", tokenizer)
for df in args.data_files:
  #format: (total count, file count, predicted count)
  process_datafile(df, {mwt:[total_count,0,0] for mwt, total_count in mwt_counts_dict.items()})


