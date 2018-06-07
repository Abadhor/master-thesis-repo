import os
import pickle
import io
import json
import sys
from Preprocessing import Tokenizer, MWUAnnotator, Vectorizer
from MWTDatasetCreator import MWTDatasetCreator
from TrainHelpers import TrainHelpers as hlp
from EntityModel import EntityModel
from TFLogger import TFLogger
from gensim.models.word2vec import Word2Vec
import argparse
import options
import time
import random
random.seed(5) # Split A
#random.seed(15) # Split B

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
LM_DICT = "vocab-2016-09-10.txt"
LM_DATA = "embeddings_char_cnn.npy"

TMP_MODEL = "./tmp/model.ckpt"
LOG_DIR = "./log/"
BEST_MODEL = "./models/best_model.session"
DICTIONARY = "./models/dictionary"
PARAMS = "./models/params"

cache_tmp = False
tmp_train = "./tmp/tmp_train.tfrecord"
tmp_test = "./tmp/tmp_test.tfrecord"
tmp_val = "./tmp/tmp_val.tfrecord"

WORD2VEC = "D:/data/other/clefipnostem/300-1/skipgram.model"
#WORD2VEC = "D:/data/other/wikipedia/300-2/skipgram.model"
NEW_WORDS = ['.', ',', ';', '$', '(', ')', '§']


# model parameters
params = {}
# call params
params['small_training_set'] = False

params['char_feature_type'] = 'cnn'
params['pos_features'] = None
params['hidden_dense_out'] = True


# depths
params['sent_length'] = 75
params['word_length'] = 32
params['hidden_size'] = 350
params['boc_feature_size'] = 21
# char CNN
# 32->16, 16->8, 8->4, 4->2, 2->1
params['charCNN_layer_depths'] = [32, 64, 128, 256, 256]
params['char_dense_out_features'] = 100
params['char_cnn_filter_width'] = 3
params['final_dense_hidden_depth'] = 300
# gazetteers
"""
params['gazetteer_count'] = 0
params['gazetteers_dense_size'] = 50
"""
params['num_classes'] = len(MWUAnnotator.LABEL_NAMES) + 1 # + padding label
params['pos_depth'] = len(Tokenizer.get_POS_tags()) + 1 # + padding label
"""
params['LM_embeddings'] = meta['char_CNN_vectors']
params['LM_hidden_size'] = 512
"""

# training parameters
num_epochs = 200
batch_size = 32
early_stopping_epoch_limit = 20
performance_metric = 'F1'
#performance_metric = 'recall'
params['starter_learning_rate'] = 0.01
params['l2-coefficient'] = 0.01
params['grad_clip_norm'] = 5.0
params['decay_steps'] = 200
params['decay_rate'] = 0.96


# selection of files used for training, validation and testing
filenames = [text_dir + fname for fname in os.listdir(text_dir)]
sample = random.sample(filenames, 4)
test_files = sample[:2]
val_files = sample[2:]
train_files = [fname for fname in filenames if fname not in sample]
if params['small_training_set']:
  train_files = train_files[:len(train_files)//2]
print("------------------------------Train Files------------------------------")
print(train_files)
print("----------------------------Validation Files---------------------------")
print(val_files)
print("------------------------------Test Files-------------------------------")
print(test_files)

# load Multi Word Terms set from files
with io.open(mwt_file, 'rb') as fp:
  mwt_set = pickle.load(fp)

model = Word2Vec.load(WORD2VEC)
dictionary = model.wv.vocab
dictionary = {k:v.count for k, v in dictionary.items()}
for w in NEW_WORDS:
  dictionary[w] = 1

# save dictionary to file
with io.open(DICTIONARY, 'w', encoding='utf-8') as fp:
  json.dump(dictionary, fp, ensure_ascii=False)


# tokenizer and params
tokenizer = Tokenizer()
# + not-in-dictionary token, + padding token
params['vocab_size'] = len(dictionary) + 2
# + padding char
params['alphabet_size'] = len(tokenizer.alphabet) + 1

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

vectorizer = Vectorizer(dictionary, {l:1 for l in tokenizer.alphabet}, annotator=annotator, gazetteers=None)

# Load word embeddings based on dictionary sequence
word_embeddings = hlp.getWordVectors(model, vectorizer.inverseDictionary)
params['word_embeddings_shape'] = word_embeddings.shape
params['no_vector_count'] = len(vectorizer.inverseDictionary) - model.wv.syn0.shape[0]

# create datasets and dataset iterators
creator = MWTDatasetCreator(tokenizer, vectorizer, params['sent_length'], params['word_length'], cache_tmp)
# train
dataset_train, num_samples_train = creator.createDataset(train_files, tmp_train, "Train:")
dataset_train = dataset_train.cache(tmp_train+".cache").shuffle(num_samples_train).batch(batch_size)
iterator_train = dataset_train.make_initializable_iterator()
next_train = iterator_train.get_next()
# test
dataset_test, num_samples_test = creator.createDataset(test_files, tmp_test, "Test:")
dataset_test = dataset_test.batch(batch_size)
iterator_test = dataset_test.make_initializable_iterator()
next_test = iterator_test.get_next()
# development/validation
dataset_val, num_samples_val = creator.createDataset(val_files, tmp_val, "Val:")
dataset_val = dataset_val.batch(batch_size)
iterator_val = dataset_val.make_initializable_iterator()
next_val = iterator_val.get_next()

# save model params to file
with io.open(PARAMS, 'w', encoding='utf-8') as fp:
  json.dump(params, fp)

# train + evaluate
with EntityModel(params,
                 word_features='emb',
                 char_features=params['char_feature_type'],
                 LM=None,
                 gazetteers=False,
                 pos_features=params['pos_features'],
                 hidden_dense_out=params['hidden_dense_out']
                 ) as clf:
  clf.load_word_embeddings(word_embeddings)
  
  no_imp_ep_count = 0
  best_accuracy = 0
  best_performance = 0
  best_sent_accuracy = 0
  best_epoch = 0
  epoch_times = []
  
  log_fields = ['loss', 'accuracy', 'recall', 'precision', 'F1']
  logger = TFLogger(log_fields, clf.getSession())
  train_writer = logger.create_writer(LOG_DIR+'/train')
  eval_writer = logger.create_writer(LOG_DIR+'/eval')
  
  # For each epoch, train on the whole training set once
  # and validate on the validation set once
  for ep in range(0,num_epochs):
    print("Epoch", ep, "training...")
    epoch_start_time = time.time()
    clf.getSession().run(iterator_train.initializer)
    clf.getSession().run(iterator_val.initializer)
    performance_training = hlp.runMWTDataset(next_train, num_samples_train, clf, 'train')
    print("Train loss at Epoch", ep, ":", performance_training['loss'])
    print("Train accuracy: {:.3%}".format(performance_training['accuracy']))
    print("Train recall: {:.3%}".format(performance_training['recall']))
    print("Train F1: {:.3%}".format(performance_training['F1']))
    print("                                                ", end='\r')
    performance_validation = hlp.runMWTDataset(next_val, num_samples_val, clf, 'eval')
    print("Validation loss at Epoch", ep, ":", performance_validation['loss'])
    #print("Validation label counts: ", performance_validation['label_counts'])
    print("Validation accuracy: {:.3%}".format(performance_validation['accuracy']))
    print("Validation sentence accuracy: {:.3%}".format(performance_validation['accuracy_sentence']))
    print("Validation precision: {:.3%}".format(performance_validation['precision']))
    print("Validation recall: {:.3%}".format(performance_validation['recall']))
    print("Validation F1: {:.3%}".format(performance_validation['F1']))
    
    # logging
    logger.log(train_writer, performance_training, ep)
    logger.log(eval_writer, performance_validation, ep)
    
    # early stopping
    if performance_validation[performance_metric] > best_performance:
      no_imp_ep_count = 0
      best_accuracy = performance_validation['accuracy']
      best_sent_accuracy = performance_validation['accuracy_sentence']
      best_performance = performance_validation[performance_metric]
      best_epoch = ep
      save_path = clf.saveModel(TMP_MODEL)
      print("Model saved in file: %s" % save_path)
    else:
      no_imp_ep_count += 1
      if no_imp_ep_count == early_stopping_epoch_limit:
        break
    epoch_time = time.time() - epoch_start_time
    epoch_times.append(epoch_time)
    print("Epoch duration:", epoch_time)
    sys.stdout.flush()
  print("                                                ", end='\r')
  print("Best Epoch:", best_epoch)
  print("Best accuracy: {:.3%}".format(best_accuracy))
  print("Best sentence accuracy: {:.3%}".format(best_sent_accuracy))
  print("Best "+performance_metric+": {:.3%}".format(best_performance))
  
  # validation on test set
  print("Validating on test set...")
  clf.restoreModel(TMP_MODEL)
  print("Model restored.")
  clf.getSession().run(iterator_test.initializer)
  performance = hlp.runMWTDataset(next_test, num_samples_test, clf, 'eval')
  print("Testset loss at Epoch", best_epoch, ":", performance['loss'])
  print("Testset label counts: ", performance['label_counts'])
  print("Testset accuracy: {:.3%}".format(performance['accuracy']))
  print("Testset sentence accuracy: {:.3%}".format(performance['accuracy_sentence']))
  print("Testset precision: {:.3%}".format(performance['precision']))
  print("Testset recall: {:.3%}".format(performance['recall']))
  print("Testset F1: {:.3%}".format(performance['F1']))
  
  save_path = clf.saveModel(BEST_MODEL)
  print("Best Model saved in file: %s" % save_path)

# clean-up
if not cache_tmp:
  creator.deleteOutfiles(tmp_train)
  creator.deleteOutfiles(tmp_test)
  creator.deleteOutfiles(tmp_val)




