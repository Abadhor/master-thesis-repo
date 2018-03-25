import os
import pickle
import io
from Preprocessing import Tokenizer, MWUAnnotator, Vectorizer
from MWTDatasetCreator import MWTDatasetCreator
from TrainHelpers import TrainHelpers as hlp
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

# TODO:
# parse file locations

text_dir = "D:/data/datasets/patent_mwt/plaintext/"
mwt_file = "D:/data/datasets/patent_mwt/mwts/mwts.set"
LM_DICT = "vocab-2016-09-10.txt"
LM_DATA = "embeddings_char_cnn.npy"

TMP_MODEL = "./tmp/model.ckpt"
BEST_MODEL = "./models/best_model.session"

tmp_train = "./tmp/tmp_train.tfrecord"
tmp_test = "./tmp/tmp_test.tfrecord"
tmp_val = "./tmp/tmp_val.tfrecord"

WORD2VEC = "D:/data/other/clefipnostem/300-1/skipgram.model"
NEW_WORDS = ['.', ',', ';', '$', '(', ')', '§']

# model parameters
params = {}
params['sent_length'] = 60
params['word_length'] = 32
params['hidden_size'] = 350
params['boc_feature_size'] = 21
# char CNN
# 32->16, 16->8, 8->4, 4->2, 2->1
params['charCNN_layer_depths'] = [32, 64, 128, 256, 512]
params['char_dense_out_features'] = 50
params['char_cnn_filter_width'] = 3
params['char_cnn_out_features'] = 32
# gazetteers
"""
params['gazetteer_count'] = 0
params['gazetteers_dense_size'] = 50
"""
params['num_classes'] = len(MWUAnnotator.LABEL_NAMES) + 1 # + padding label
"""
params['LM_embeddings'] = meta['char_CNN_vectors']
params['LM_hidden_size'] = 512
"""

# training parameters
num_epochs = 600
batch_size = 64
early_stopping_epoch_limit = 60
performance_metric = 'F1'
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

# load Multi Word Terms set from files
with io.open(mwt_file, 'rb') as fp:
  mwt_set = pickle.load(fp)

model = Word2Vec.load(WORD2VEC)
dictionary = model.wv.vocab
dictionary = {k:v.count for k, v in dictionary.items()}
for w in NEW_WORDS:
  dictionary[w] = 1

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

vectorizer = Vectorizer(annotator, dictionary, {l:1 for l in tokenizer.alphabet}, gazetteers=None)

# Load word embeddings based on dictionary sequence
params['word_embeddings'] = hlp.getWordVectors(model, vectorizer.inverseDictionary)
params['no_vector_count'] = len(vectorizer.inverseDictionary) - model.wv.syn0.shape[0]

# create datasets and dataset iterators
creator = MWTDatasetCreator(tokenizer, vectorizer, params['sent_length'], params['word_length'])
# train
dataset_train, num_samples_train = creator.createDataset(train_files, tmp_train)
dataset_train = dataset_train.shuffle(num_samples_train).batch(batch_size)
iterator_train = dataset_train.make_initializable_iterator()
next_train = iterator_train.get_next()
# test
dataset_test, num_samples_test = creator.createDataset(test_files, tmp_test)
dataset_test = dataset_test.batch(batch_size)
iterator_test = dataset_test.make_initializable_iterator()
next_test = iterator_test.get_next()
# development/validation
dataset_val, num_samples_val = creator.createDataset(val_files, tmp_val)
dataset_val = dataset_val.batch(batch_size)
iterator_val = dataset_val.make_initializable_iterator()
next_val = iterator_val.get_next()

with EntityModel(params, word_features='emb', char_features='boc', LM=None, gazetteers=False) as clf:
  
  no_imp_ep_count = 0
  best_accuracy = 0
  best_performance = 0
  best_sent_accuracy = 0
  best_epoch = 0
  epoch_times = []
  
  # For each epoch, train on the whole training set once
  # and validate on the validation set once
  for ep in range(0,num_epochs):
    epoch_start_time = time.time()
    clf.getSession().run(iterator_train.initializer)
    clf.getSession().run(iterator_dev.initializer)
    performance = hlp.runMWTDataset(next_train, num_samples_train, clf, train=True)
    print("Train loss at Epoch", ep, ":", performance['loss'])
    #print("Train label counts: ", performance['label_counts'])
    print("Train accuracy: {:.3%}".format(performance['accuracy']))
    #print("Train sentence accuracy: {:.3%}".format(performance['accuracy_sentence']))
    #print("Train precision: {:.3%}".format(performance['precision']))
    print("Train recall: {:.3%}".format(performance['recall']))
    print("Train F1: {:.3%}".format(performance['F1']))
    print("                                                ", end='\r')
    performance = hlp.runMWTDataset(next_val, num_samples_val, clf)
    print("Validation loss at Epoch", ep, ":", performance['loss'])
    #print("Validation label counts: ", performance['label_counts'])
    print("Validation accuracy: {:.3%}".format(performance['accuracy']))
    print("Validation sentence accuracy: {:.3%}".format(performance['accuracy_sentence']))
    print("Validation precision: {:.3%}".format(performance['precision']))
    print("Validation recall: {:.3%}".format(performance['recall']))
    print("Validation F1: {:.3%}".format(performance['F1']))
    if performance[performance_metric] > best_performance:
      no_imp_ep_count = 0
      best_accuracy = performance['accuracy']
      best_sent_accuracy = performance['accuracy_sentence']
      best_performance = performance[performance_metric]
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
  performance = hlp.runMWTDataset(next_test, num_samples_test, clf)
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
os.remove(tmp_train)
os.remove(tmp_test)
os.remove(tmp_val)




