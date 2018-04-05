import os
import io
import json
from Preprocessing import Tokenizer, Vectorizer
from TrainHelpers import TrainHelpers as hlp
from EntityModel import EntityModel
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


BEST_MODEL = "./models/best_model.session"
DICTIONARY = "./models/dictionary"
PARAMS = "./models/params"


# selection of files used for training, validation and testing
"""
filenames = [text_dir + fname for fname in os.listdir(text_dir)]
sample = random.sample(filenames, 4)
test_files = sample[:2]
val_files = sample[2:]
train_files = [fname for fname in filenames if fname not in sample]
"""
# load dictionary from file
with io.open(DICTIONARY, 'r', encoding='utf-8') as fp:
  dictionary = json.load(fp)

# tokenizer and vectorizer
tokenizer = Tokenizer()

vectorizer = Vectorizer(dictionary, {l:1 for l in tokenizer.alphabet})

# load model params from file
with io.open(PARAMS, 'r', encoding='utf-8') as fp:
  params = json.load(fp)

# train + evaluate
print("Loading model...")
with EntityModel(params, word_features='emb', char_features='boc', LM=None, gazetteers=False) as clf:
  clf.restoreModel(BEST_MODEL)
  print("Enter a sentence or type exit() to quit!")
  print("Sentence max length:", params['sent_length'], "Tokens")
  while True:
    text = input(">>> ")
    if text == "exit()":
      print("Exiting...")
      break
    sentences = tokenizer.splitSentences(text)
    print("Recognized", len(sentences), "sentences")
    for idx, sentence in enumerate(sentences):
      print("Sentence", idx+1)
      tokens = tokenizer.substituteTokenize(sentence)
      token_vector, length, _, char_vector, char_lengths = vectorizer.vectorize(
        tokens[:params['sent_length']], 
        params['sent_length'], 
        params['word_length'])
      data = {
        'sentences': token_vector,
        'sent_lengths': length,
        'sentence_chars': char_vector,
        'word_lengths': char_lengths
      }
      # normalize performance based on batch length
      fetched = clf.run(data, mode='predict')
      predictions = hlp.getMWTs(fetched['crf_decode'][0])
      print(tokens[:params['sent_length']])
      for p in predictions:
        start, end = p
        end += 1
        print(tokens[start:end])
    





