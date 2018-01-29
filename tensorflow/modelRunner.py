import tensorflow as tf
import math
import pickle
import numpy as np
import io
import random
from EntityModel import EntityModel


random.seed(10)

# data path
PATH = "./data/"

TEST = "test_sem2017.pickle"
DEV = "dev_sem2017.pickle"
TRAIN = "train_sem2017.pickle"
META = "meta_sem2017.pickle"

TMP_MODEL = "./tmp/model.ckpt"
BEST_MODEL = "./models/best_model.session"

with io.open(PATH+META, "rb") as fp:
  meta = pickle.load(fp)

params = {}
# model parameters
params['sent_length'] = meta['sent_length']
params['word_length'] = meta['word_length']
params['gazetteer_count'] = meta['gazetteer_count']
params['hidden_size'] = 350
params['gazetteers_dense_size'] = 50
params['bow_feature_size'] = 100
params['boc_feature_size'] = 21
params['char_cnn_filter_width'] = 3
params['char_cnn_out_features'] = 32
# 16->8, 8->4, 4->2, 2->1
params['charCNN_layer_depths'] = [32, 64, 128, 256]
params['char_dense_out_features'] = 50
params['vocab_size'] = len(meta['invDict'])
params['alphabet_size'] = len(meta['alphabet']) + 1
params['num_classes'] = len(meta['labelNames'])
params['word_embeddings'] = meta['word_vectors']
params['LM_embeddings'] = meta['char_CNN_vectors']
params['LM_hidden_size'] = 512
params['no_vector_count'] = meta['no_vector_count']
batch_size = 64
LSTM_layer_count = 1
batch_testing = True


# training parameters
num_epochs = 600
early_stopping_epoch_limit = 60
params['starter_learning_rate'] = 0.01
params['l2-coefficient'] = 0.01
params['decay_steps'] = 200
params['decay_rate'] = 0.96

# load data
with io.open(PATH+TRAIN, "rb") as fp:
  train = pickle.load(fp)

train_data = train['data']
train_labels = train['labels']
train_lengths = train['lengths']

with io.open(PATH+TEST, "rb") as fp:
  test = pickle.load(fp)

test_data = test['data']
test_labels = test['labels']
test_lengths = test['lengths']

with io.open(PATH+DEV, "rb") as fp:
  dev = pickle.load(fp)

dev_data = dev['data']
dev_labels = dev['labels']
dev_lengths = dev['lengths']
"""
# define the number of batches
num_batches = len(train_data) // batch_size
if len(train_data) % batch_size == 0:
  batch_size_last = batch_size
else:
  batch_size_last = len(train_data) - (num_batches * batch_size)
  num_batches += 1
"""
# create batches, randomizing the order of the samples
def createBatches(dataset, batch_size, randomize=True):
  # define the number of batches
  num_batches = len(dataset['data']) // batch_size
  if len(dataset['data']) % batch_size == 0:
    batch_size_last = batch_size
  else:
    batch_size_last = len(dataset['data']) - (num_batches * batch_size)
    num_batches += 1
  
  # create batches
  batches = {}
  for k in dataset.keys():
    batches[k] = []
  b_idx = [x for x in range(0,len(dataset['data']))]
  if randomize:
    random.shuffle(b_idx)
  offset = 0
  for b in range(0, num_batches-1):
    cur_b_idx = b_idx[offset:(offset+batch_size)]
    for k in dataset.keys():
      batches[k].append([dataset[k][x] for x in cur_b_idx])
    offset += batch_size
  
  cur_b_idx = b_idx[offset:(offset+batch_size_last)]
  for k in dataset.keys():
      batches[k].append([dataset[k][x] for x in cur_b_idx])
  return batches

def getMWTs(sentence):
  # get MWTs in s
  MWTs = []
  B_idx = None
  for idx, t in enumerate(sentence):
    if t == meta['labelDict']['B']:
      B_idx = idx
    elif t == meta['labelDict']['L']:
      if B_idx != None:
        MWTs.append((B_idx,idx))
      B_idx = None
  return MWTs

def countLabels(sentence, sent_length, counts_dict):
  for t in range(sent_length):
    if sentence[t] == meta['labelDict']['B']:
      counts_dict['B'] += 1
    if sentence[t] == meta['labelDict']['I']:
      counts_dict['I'] += 1
    if sentence[t] == meta['labelDict']['L']:
      counts_dict['L'] += 1
    if sentence[t] == meta['labelDict']['O']:
      counts_dict['O'] += 1

def getScores(TP,FP,FN):
  if TP == 0:
    return 0.0, 0.0, 0.0
  precision = TP/(TP+FP)
  recall = TP/(TP+FN)
  F1 = 2*TP /(2*TP+FP+FN)
  return precision, recall, F1

def runBatches(batches, train=False):
  # batch validation
  num_samples = sum([len(b) for b in batches['data']])
  losses = 0
  accuracies = 0
  s_accuracies = 0
  # ground truth, true and false positive counts
  GT_count = 0
  TP_count = 0
  FP_count = 0
  count_dict = {x:0 for x in meta['labelNames']}
  for b in range(len(batches['data'])):
    print("Batch:",b+1, '/', len(batches['data']), end='\r')
    data = {
      'sentences': batches['data'][b],
      'labels': batches['labels'][b],
      'sent_lengths': batches['lengths'][b],
      'sentence_chars': batches['chars'][b],
      'word_lengths': batches['word_lengths'][b],
      'gazetteers': batches['gazetteers'][b],
      'dictionary': meta['invDict'],
      'label_names': meta['labelNames']
    }
    fetched = clf.run(data, train=train)
    losses += fetched['loss_value'] * len(batches['data'][b])
    accuracies += fetched['accuracy_value'] * len(batches['data'][b])
    s_accuracies += fetched['sentence_accuracy_value'] * len(batches['data'][b])
    # find ground truth for each sentence and evaluate predictions
    for idx,s in enumerate(batches['labels'][b]):
      GT_list = getMWTs(s)
      GT_count += len(GT_list)
      GT_set = set()
      for mwt in GT_list:
        GT_set.add(mwt)
      pred_s = fetched['crf_decode'][idx]
      predictions = getMWTs(pred_s)
      countLabels(pred_s, batches['lengths'][b][idx], count_dict)
      for mwt in predictions:
        #print(mwt, end=" ")
        if mwt in GT_set:
          TP_count += 1
        else:
          FP_count += 1
  FN_count = GT_count - TP_count
  precision, recall, F1 = getScores(TP_count, FP_count, FN_count)
  losses_avg = losses/num_samples
  accuracy_avg = accuracies/num_samples
  sentence_accuracy_avg = s_accuracies/num_samples
  performance = {
    'loss': losses_avg,
    'accuracy':accuracy_avg,
    'accuracy_sentence': sentence_accuracy_avg,
    'precision': precision,
    'recall': recall,
    'F1': F1,
    'label_counts':count_dict
  }
  return performance
  

with EntityModel(params, word_features='emb') as clf:
  # we'll train with batches of size 128.  This means that we run 
  # our model on 128 examples and then do gradient descent based on the loss
  # over those 128 examples.
  step = 0
  
  no_imp_ep_count = 0
  best_accuracy = 0
  best_F1 = 0
  best_sent_accuracy = 0
  best_epoch = 0
  
  for ep in range(0,num_epochs):
    batches_train = createBatches(train, batch_size)
    performance = runBatches(batches_train, train=True)
    print("Train loss at Epoch", ep, ":", performance['loss'])
    #print("Train label counts: ", performance['label_counts'])
    print("Train accuracy: {:.3%}".format(performance['accuracy']))
    #print("Train sentence accuracy: {:.3%}".format(performance['accuracy_sentence']))
    #print("Train precision: {:.3%}".format(performance['precision']))
    #print("Train recall: {:.3%}".format(performance['recall']))
    print("Train F1: {:.3%}".format(performance['F1']))
    print("                                                ", end='\r')
    batches_dev = createBatches(dev, batch_size, randomize=False)
    performance = runBatches(batches_dev)
    print("Devset loss at Epoch", ep, ":", performance['loss'])
    #print("Devset label counts: ", performance['label_counts'])
    print("Devset accuracy: {:.3%}".format(performance['accuracy']))
    #print("Devset sentence accuracy: {:.3%}".format(performance['accuracy_sentence']))
    #print("Devset precision: {:.3%}".format(performance['precision']))
    #print("Devset recall: {:.3%}".format(performance['recall']))
    print("Devset F1: {:.3%}".format(performance['F1']))
    if performance['F1'] > best_F1:
      no_imp_ep_count = 0
      best_accuracy = performance['accuracy']
      best_sent_accuracy = performance['accuracy_sentence']
      best_F1 = performance['F1']
      best_epoch = ep
      save_path = clf.saveModel(TMP_MODEL)
      print("Model saved in file: %s" % save_path)
    else:
      no_imp_ep_count += 1
      if no_imp_ep_count == early_stopping_epoch_limit:
        break
  print("                                                ", end='\r')
  print("Best Epoch:", best_epoch)
  print("Best accuracy: {:.3%}".format(best_accuracy))
  print("Best sentence accuracy: {:.3%}".format(best_sent_accuracy))
  print("Best F1: {:.3%}".format(best_F1))
  
  # validation on test set
  print("Validating on test set...")
  clf.restoreModel(TMP_MODEL)
  print("Model restored.")
  batches_test = createBatches(test, batch_size, randomize=False)
  performance = runBatches(batches_test)
  print("Testset loss at Epoch", best_epoch, ":", performance['loss'])
  print("Testset label counts: ", performance['label_counts'])
  print("Testset accuracy: {:.3%}".format(performance['accuracy']))
  print("Testset sentence accuracy: {:.3%}".format(performance['accuracy_sentence']))
  print("Testset precision: {:.3%}".format(performance['precision']))
  print("Testset recall: {:.3%}".format(performance['recall']))
  print("Testset F1: {:.3%}".format(performance['F1']))
  
  save_path = clf.saveModel(BEST_MODEL)
  print("Best Model saved in file: %s" % save_path)



