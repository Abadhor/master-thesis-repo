import tensorflow as tf
import math
import pickle
import numpy as np
import io
import random
from EntityModel_BoW import EntityModel_BoW


"""
TODO:
evaluate with function
single sentences
check % of sentences fully correct
count number of precise MWT matches
compare with expected number of MWTs
check invalid labeling eg "BOL"
count number of wrongly predicted MWTs
-> MWTs with boundary mismatches
do evaluation on index level
"""

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
#sent_length = 20
params['hidden_size'] = 350
params['dense_hidden_size'] = 300
params['vocab_size'] = len(meta['invDict'])
params['num_classes'] = len(meta['labelNames'])
batch_size = 32
LSTM_layer_count = 1
batch_testing = True


# training parameters
num_epochs = 100
early_stopping_epoch_limit = 10
params['starter_learning_rate'] = 0.1
params['decay_steps'] = 500
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

# define the number of batches
num_batches = len(train_data) // batch_size
if len(train_data) % batch_size == 0:
  batch_size_last = batch_size
else:
  batch_size_last = len(train_data) - (num_batches * batch_size)
  num_batches += 1

# create batches, randomizing the order of the samples
def createBatches(data, labels, lengths, num_batches, batch_size_last, randomize=True):
  batches_sent = []
  batches_labels = []
  batches_lengths = []
  b_idx = [x for x in range(0,len(data))]
  if randomize:
    random.shuffle(b_idx)
  offset = 0
  for b in range(0, num_batches-1):
    cur_b_idx = b_idx[offset:(offset+batch_size)]
    batches_sent.append([data[x] for x in cur_b_idx])
    batches_labels.append([labels[x] for x in cur_b_idx])
    batches_lengths.append([lengths[x] for x in cur_b_idx])
    offset += batch_size
  
  cur_b_idx = b_idx[offset:(offset+batch_size_last)]
  batches_sent.append([data[x] for x in cur_b_idx])
  batches_labels.append([labels[x] for x in cur_b_idx])
  batches_lengths.append([lengths[x] for x in cur_b_idx])
  return batches_sent, batches_labels, batches_lengths

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

def getScores(TP,FP,FN):
  if TP == 0:
    return 0.0, 0.0, 0.0
  precision = TP/(TP+FP)
  recall = TP/(TP+FN)
  F1 = 2*TP /(2*TP+FP+FN)
  return precision, recall, F1

with EntityModel_BoW(params) as clf:
  # we'll train with batches of size 128.  This means that we run 
  # our model on 128 examples and then do gradient descent based on the loss
  # over those 128 examples.
  step = 0
  
  no_imp_step_count = 0
  best_accuracy = 0
  best_sent_accuracy = 0
  best_epoch = 0
  
  for ep in range(0,num_epochs):
    batches_sent, batches_labels, batches_lengths = createBatches(
    train_data, train_labels, train_lengths,
    num_batches, batch_size_last)
    for b in range(0,num_batches):
      print("Batch:",b+1, '/', num_batches, end='\r')
      data = {
        'sentences': batches_sent[b],
        'labels': batches_labels[b],
        'sent_lengths': batches_lengths[b],
        'dictionary': meta['invDict'],
        'label_names': meta['labelNames']
      }
      # TODO: replace this list of variables with dictionary
      loss_value_train, accuracy_value_train, sentence_accuracy_value_train = clf.train(data)
      if (step % 50000 == 0):
        print("Minibatch train loss at step", step, ":", loss_value_train)
        print("Minibatch accuracy: {:.3%}".format(accuracy_value_train))
        print("Minibatch sentence accuracy: {:.3%}".format(sentence_accuracy_value_train))
      step += 1
    # validation on dev set
    # one sentence at a time
    losses = []
    accuracies = []
    s_accuracies = []
    # ground truth, true and false positive counts
    GT_count = 0
    TP_count = 0
    FP_count = 0
    for s in range(len(dev_data)):
      print("Sentence:",s+1, '/', len(dev_data), end='\r')
      data_dev = {
        'sentences': dev_data[s:s+1,:],
        'labels': dev_labels[s:s+1,:],
        'sent_lengths': dev_lengths[s:s+1],
        'dictionary': meta['invDict'],
        'label_names': meta['labelNames']
      }
      loss_value_dev, accuracy_value_dev, sentence_accuracy_value_dev, pred_dev = clf.predict(data_dev)
      losses.append(loss_value_dev)
      accuracies.append(accuracy_value_dev)
      s_accuracies.append(sentence_accuracy_value_dev)
      GT_list = getMWTs(dev_labels[s,:dev_lengths[s]])
      GT_count += len(GT_list)
      GT_set = set()
      for mwt in GT_list:
        GT_set.add(mwt)
      predictions = getMWTs(pred_dev[0,:dev_lengths[s]])
      for mwt in predictions:
        if mwt in GT_set:
          TP_count += 1
        else:
          FP_count += 1
    FN_count = GT_count - TP_count
    precision, recall, F1 = getScores(TP_count, FP_count, FN_count)
    accuracy_value_dev = sum(accuracies)/len(accuracies)
    sentence_accuracy_value_dev = sum(s_accuracies)/len(s_accuracies)
    print("Devset loss at Epoch", ep, ":", sum(losses)/len(losses))
    print("Devset accuracy: {:.3%}".format(accuracy_value_dev))
    print("Devset sentence accuracy: {:.3%}".format(sentence_accuracy_value_dev))
    print("Devset precision: {:.3%}".format(precision))
    print("Devset recall: {:.3%}".format(recall))
    print("Devset F1: {:.3%}".format(F1))
    if accuracy_value_dev > best_accuracy:
      no_imp_step_count = 0
      best_accuracy = accuracy_value_dev
      best_sent_accuracy = sentence_accuracy_value_dev
      best_epoch = ep
      save_path = clf.saveModel(TMP_MODEL)
      print("Model saved in file: %s" % save_path)
    else:
      no_imp_step_count += 1
      if no_imp_step_count == early_stopping_epoch_limit:
        break
  
  print("Best Epoch:", best_epoch)
  print("Best accuracy: {:.3%}".format(best_accuracy))
  print("Best sentence accuracy: {:.3%}".format(best_sent_accuracy))
  
  # validation on test set
  print("Validating on test set...")
  clf.restoreModel(TMP_MODEL)
  print("Model restored.")
  losses = []
  accuracies = []
  s_accuracies = []
  # ground truth, true and false positive counts
  GT_count = 0
  TP_count = 0
  FP_count = 0
  for s in range(len(test_data)):
    print("Sentence:",s+1, '/', len(test_data), end='\r')
    data_testing = {
      'sentences': test_data[s:s+1,:],
      'labels': test_labels[s:s+1,:],
      'sent_lengths': test_lengths[s:s+1],
      'dictionary': meta['invDict'],
      'label_names': meta['labelNames']
    }
    loss_value_test, accuracy_value_test, sentence_accuracy_value_test, pred_test = clf.predict(data_testing)
    losses.append(loss_value_test)
    accuracies.append(accuracy_value_test)
    s_accuracies.append(sentence_accuracy_value_test)
    GT_list = getMWTs(test_labels[s,:test_lengths[s]])
    GT_count += len(GT_list)
    GT_set = set()
    for mwt in GT_list:
      GT_set.add(mwt)
    predictions = getMWTs(pred_test[0,:test_lengths[s]])
    for mwt in predictions:
      if mwt in GT_set:
        TP_count += 1
      else:
        FP_count += 1
  FN_count = GT_count - TP_count
  precision, recall, F1 = getScores(TP_count, FP_count, FN_count)
  accuracy_value_test = sum(accuracies)/len(accuracies)
  sentence_accuracy_value_test = sum(s_accuracies)/len(s_accuracies)
  print("Testset loss at Epoch", ep, ":", sum(losses)/len(losses))
  print("Testset accuracy: {:.3%}".format(accuracy_value_test))
  print("Testset sentence accuracy: {:.3%}".format(sentence_accuracy_value_test))
  print("Testset precision: {:.3%}".format(precision))
  print("Testset recall: {:.3%}".format(recall))
  print("Testset F1: {:.3%}".format(F1))
  
  save_path = clf.saveModel(BEST_MODEL)
  print("Best Model saved in file: %s" % save_path)



