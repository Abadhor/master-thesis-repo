import os
import io
import json
import tensorflow as tf
from Preprocessing import Tokenizer, Vectorizer, MWUAnnotator
from TrainHelpers import TrainHelpers as hlp
from EntityModel import EntityModel
import argparse
import options
import pickle


parser = argparse.ArgumentParser()
parser.add_argument(
      "--model_dir",
      type=str,
      required=True,
      help="The location of the MWT classifier model.")
parser.add_argument(
      "--out_dir",
      type=str,
      required=False,
      help="The location of the processed output files. Defaults to stdout.")
parser.add_argument(
      "--mwt_file",
      type=str,
      required=False,
      help="The location of the ground truth MWTs.")
parser.add_argument(
      "--data_files",
      type=str,
      required=False,
      nargs='+',
      help="List of files containing text data for MWT extraction.")
args = parser.parse_args()

#BEST_MODEL = "./models/best_model.session"
#DICTIONARY = "./models/dictionary"
#PARAMS = "./models/params"

FOLDER = args.model_dir #"./logs/005_standard/"
BEST_MODEL = FOLDER+"best_model.session"
DICTIONARY = FOLDER+"dictionary"
PARAMS = FOLDER+"params"

def _split_sentences(z_tokens, sentence_id):
  length = len(z_tokens)
  sentences = []
  sentence_count = (length // params['sent_length'])
  remainder = (length % params['sent_length'])
  i = 0
  while i < sentence_count:
    start_token = i * params['sent_length']
    end_token = start_token + params['sent_length']
    sentences.append({
      'z_tokens':z_tokens[start_token:end_token],
      'sentence_id': sentence_id,
      'sentence_part_id': i,
      'sentence_part_count': (sentence_count if remainder == 0 else sentence_count+1)
    })
    i += 1
  if remainder > 0:
    start_token = i * params['sent_length']
    end_token = start_token + remainder
    sentences.append({
      'z_tokens':z_tokens[start_token:end_token],
      'sentence_id': sentence_id,
      'sentence_part_id': i,
      'sentence_part_count': sentence_count+1
    })
  return sentences

def process_text(text, out_file=None):
  sentences = tokenizer.splitSentences(text)
  if out_file == None:
    print("Recognized", len(sentences), "sentences")
  for idx, sentence in enumerate(sentences):
    if out_file == None:
      print("Sentence", idx+1)
    zipped = tokenizer.substituteTokenize_with_POS(sentence)
    tokenized_sentences = _split_sentences(zipped, idx)
    for sent in tokenized_sentences:
      #print("Max size sentence:",sent)
      t = tuple(zip(*sent['z_tokens'])) #unzip
      features = {
        'tokens':t[0],
        'pos_tags':t[1]
      }
      vectors = vectorizer.vectorize(
        features,
        params['sent_length'],
        params['word_length'])
      data = {
        'sentences': vectors['token_vector'],
        'sent_lengths': vectors['length'],
        'sentence_chars': vectors['char_vector'],
        'word_lengths': vectors['char_lengths'],
        'pos_tags': vectors['pos_tags']
      }
      # normalize performance based on batch length
      fetched = clf.run(data, mode='predict')
      predictions = hlp.getMWTs(fetched['crf_decode'][0])
      if args.mwt_file != None:
        GT_list = hlp.getMWTs(vectors['label_vector'][0])
      if out_file != None:
        out_file.write(str(features['tokens'][:params['sent_length']]) + '\n')
      else:
        print(features['tokens'][:params['sent_length']])
      if args.mwt_file != None: 
        GT_list = [features['tokens'][start:end+1] for start, end in GT_list]
        if out_file != None:
          out_file.write(str(GT_list).upper()+'\n')
        else:
          print(features['tokens'][start:end])
      p_list = [features['tokens'][start:end+1] for start, end in predictions]
      if out_file != None:
        out_file.write(str(p_list)+'\n')
      else:
        print(features['tokens'][start:end])


# load dictionary from file
with io.open(DICTIONARY, 'r', encoding='utf-8') as fp:
  dictionary = json.load(fp)

# tokenizer and vectorizer
tokenizer = Tokenizer()

# load Multi Word Terms set from files
if args.mwt_file != None:
  with io.open(args.mwt_file, 'rb') as fp:
    mwt_set = pickle.load(fp)
  # MWT preprocessing
  mwt_tokens_list = []
  for mwt in mwt_set:
    tokens = tokenizer.substituteTokenize(mwt)
    mwt_tokens_list.append(tokens)
  mwt_tokens_list = [x for x in mwt_tokens_list if len(x) > 1]
  annotator = MWUAnnotator(mwt_tokens_list)
  vectorizer = Vectorizer(dictionary, {l:1 for l in tokenizer.alphabet}, annotator=annotator)
else:
  vectorizer = Vectorizer(dictionary, {l:1 for l in tokenizer.alphabet})

# load model params from file
with io.open(PARAMS, 'r', encoding='utf-8') as fp:
  params = json.load(fp)

print("Loading model...")
#with EntityModel(params, word_features='emb', char_features='cnn', LM=None, gazetteers=False) as clf:

with EntityModel(params,
                 word_features='emb',
                 char_features=params['char_feature_type'],
                 LM=None,
                 gazetteers=False,
                 pos_features=params['pos_features'],
                 hidden_dense_out=params['hidden_dense_out']
                 ) as clf:
  clf.restoreModel(BEST_MODEL)
  if args.data_files == None:
    print("Enter a sentence or type exit() to quit!")
    print("Sentence max length:", params['sent_length'], "Tokens")
    while True:
      text = input(">>> ")
      if text == "exit()":
        print("Exiting...")
        break
      process_text(text)
  else: # process whole files
    for f in args.data_files:
      print("File: ", f)
      out_file = None
      if args.out_dir != None:
        fname = os.path.basename(f)
        out_path = args.out_dir + fname
        out_file = io.open(out_path, 'w')
      # read data and process
      dataset = tf.data.TextLineDataset([f])
      get_next = dataset.make_one_shot_iterator().get_next()
      while True:
        try:
          text = clf.getSession().run(get_next).decode('utf-8')
          process_text(text, out_file=out_file)
        except tf.errors.OutOfRangeError:
          print("Finished processing file", f)
          break
      # close
      if out_file != None:
        out_file.close()
    
    





