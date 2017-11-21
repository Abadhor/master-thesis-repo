import tensorflow as tf
import math
import pickle
import numpy as np
import io
import random

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

# model parameters
sent_length = meta['sent_length']
#sent_length = 20
hidden_size = 350
dense_hidden_size = 300
vocab_size = len(meta['invDict'])
num_classes = len(meta['labelNames'])
batch_size = 32
LSTM_layer_count = 1
batch_testing = True


# training parameters
num_epochs = 100
early_stopping_epoch_limit = 10
starter_learning_rate = 0.1
decay_steps = 500
decay_rate = 0.96

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

# create a session variable that we can run later.
session = tf.Session()

# Create a weight matrix + bias layer
# with duplicated matrix for each batch entry
# input shape [batch_size, sent_length, input_depth] -->
# output shape [batch_size, sent_length, output_depth]
def linearLayerTiled(input_, input_depth, output_depth, name):
  cur_batch_size = tf.shape(input_)[0]
  with tf.variable_scope(name):
    W = tf.get_variable("linear_weights", (1, output_depth, input_depth), tf.float32, tf.random_normal_initializer())
    W = tf.tile(W, multiples=[cur_batch_size,1,1])
    b = tf.get_variable("bias", (output_depth), initializer=tf.constant_initializer(0.0))
  return tf.transpose(tf.matmul(W, tf.transpose(input_, perm=[0,2,1])),perm=[0,2,1]) + b


# create LSTM Cells using BasicLSTMCell.  Note that this creates a *layer* of
# LSTM cells, not just a single one.
def createBasicLSTMLayers(num_layers, hidden_size):
  LSTM_layers = []
  for i in range(num_layers):
    LSTM_layers.append(tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True))
  return LSTM_layers

# implement exponential learning rate decay
# optimizer gets called with this learning_rate and updates
# global_step during its call to minimize()
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, decay_steps, decay_rate)

# the placeholder for sentences has first dimension batch_size for each
# sentence in a batch,
# second dimension sent_length for each word in the sentence.
# Note that we use 'None' instead of batch_size for the first dimsension.
# This allows us 
# to deal with variable batch sizes
sentences = tf.placeholder(tf.int32, [None, sent_length], name="sentences")

# one hot encoding
sent_one_hot = tf.one_hot(sentences, vocab_size, axis=-1, name="sent_one_hot")

# the placeholder for labels has first dimension batch_size for each
# sentence in a batch and
# second dimension sent_length for each word in the sentence
# Note that we use 'None' instead of batch_size for the first dimsension.
# This allows us 
# to deal with variable batch sizes
labels = tf.placeholder(tf.int32, [None, sent_length], name="labels")

# one hot encoding
lbl_one_hot = tf.one_hot(labels, num_classes, name="lbl_one_hot")


# the placeholder for the lengths of each sentence
sent_lengths = tf.placeholder(tf.int32, [None], name="sent_lengths")

# mask that indicates the length of the sentence
# each padded position after the last word is equal to 0
# all others are equal to 1
seq_len_mask = tf.sequence_mask(sent_lengths, maxlen=sent_length, dtype=tf.float32)

# placeholder for the dictionary
dictionary = tf.placeholder(tf.string, [vocab_size], name="invDict")

# placeholder for the label_names
label_names = tf.placeholder(tf.string, [num_classes], name="label_names")


# dense input matrix

lstm_inputs = linearLayerTiled(sent_one_hot, vocab_size, dense_hidden_size, "lstm_input")



# Create a forward and a backward LSTM layer for the bidirectional_dynamic_rnn
l1_fw = tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
l1_bw = tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)

# Returns a tuple (outputs, output_states) where:
# outputs: A tuple (output_fw, output_bw) containing
# the forward and the backward rnn output Tensor.
# If time_major == False (default), output_fw will be a Tensor shaped: 
# [batch_size, max_time, cell_fw.output_size]
# and output_bw will be a Tensor shaped: 
# [batch_size, max_time, cell_bw.output_size].
l1_outputs, l1_final_state = tf.nn.bidirectional_dynamic_rnn(
    l1_fw,
    l1_bw,
    lstm_inputs,
    sequence_length=sent_lengths,
    dtype=tf.float32,
    scope="l1_bi-lstm"
)

l1_out_fw, l1_out_bw = l1_outputs
l1_concat_outputs = tf.concat([l1_out_fw, l1_out_bw], 2)

# Shape of the output is [batch_size, sent_length, 2*hidden_size]

# Create a forward and a backward LSTM layer for the bidirectional_dynamic_rnn
l2_fw = tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
l2_bw = tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)

# Returns a tuple (outputs, output_states) where:
# outputs: A tuple (output_fw, output_bw) containing
# the forward and the backward rnn output Tensor.
# If time_major == False (default), output_fw will be a Tensor shaped: 
# [batch_size, max_time, cell_fw.output_size]
# and output_bw will be a Tensor shaped: 
# [batch_size, max_time, cell_bw.output_size].
l2_outputs, l1_final_state = tf.nn.bidirectional_dynamic_rnn(
    l2_fw,
    l2_bw,
    l1_concat_outputs,
    sequence_length=sent_lengths,
    dtype=tf.float32,
    scope="l2_bi-lstm"
)

l2_out_fw, l2_out_bw = l2_outputs
l2_concat_outputs = tf.concat([l2_out_fw, l2_out_bw], 2)

# Shape of the output is [batch_size, sent_length, 2*hidden_size]


# pass the final state into this linear function to multiply it 
# by the weights and add bias to get our output.
# Shape of classes is [batch_size, sent_length, num_classes]
classes = linearLayerTiled(l2_concat_outputs, 2*hidden_size, num_classes, "output")

# define our loss function.
loss = tf.nn.softmax_cross_entropy_with_logits(logits=classes, labels=lbl_one_hot)

# our loss with softmax_cross_entropy_with_logits gives us a loss for each word 
# in each sentence.  We take the sum of all losses per sentence.
loss = tf.reduce_sum(loss, axis=1)

# our loss with softmax_cross_entropy_with_logits gives us a loss for each 
# example in the batch.  We take the mean of all these losses.
loss = tf.reduce_mean(loss)

# calculate accuracy of word class predictions
# first gather the indices of the highest results
# since the classes and words are one-hot encoded,
# this gives us the id of the class
sent_decoded = tf.argmax(sent_one_hot,2)
pred_classes = tf.argmax(classes,2)
true_classes = tf.argmax(lbl_one_hot,2)
correct_prediction = tf.cast(tf.equal(pred_classes, true_classes), tf.float32)

# ensure that results outside the sequence length are 0
correct_prediction = tf.multiply(correct_prediction, seq_len_mask)

# calculate accuracy only for the part in the sequence
# that is part of the sentence
sent_sum = tf.reduce_sum(correct_prediction,1)
mask_sum = tf.reduce_sum(seq_len_mask, 1)
accuracy = tf.reduce_mean(sent_sum / mask_sum)

# calculate fraction of sentences in which each word has been predicted correctly
correct_sent = tf.cast(tf.equal(sent_sum, mask_sum), tf.float32)
sentence_accuracy = tf.reduce_mean(correct_sent)


# decode sentences into readable words and labels
# lookup the string of the ID of the words and classes
# in the word dictionary and label name dictionary
sent_decoded = tf.gather(dictionary, sent_decoded)
pred_classes_names = tf.gather(label_names, pred_classes)
true_classes = tf.gather(label_names, true_classes)


'''Define the operation that specifies the AdamOptimizer and tells
   it to minimize the loss.''';
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

# initialize any variables
init_op = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# run init_op
init_op.run(session=session)

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

def printSent(sent, mask, pred, true):
  sent = np.vectorize(bytes.decode)(sent)
  pred = np.vectorize(bytes.decode)(pred)
  true = np.vectorize(bytes.decode)(true)
  print("Sent: "," ".join(sent))
  print("Mask: "," ".join([str(int(x)) for x in mask]))
  print("Pred: "," ".join(pred))
  print("True: "," ".join(true))

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
"""
data = {
  sentences:train_data[0:2,:],
  labels:train_labels[0:2,:],
  sent_lengths:train_lengths[0:2],
  dictionary:meta['invDict'],
  label_names:meta['labelNames']
}
x = session.run([accuracy], feed_dict=data)
print(x)
exit()
"""

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
      sentences: batches_sent[b],
      labels: batches_labels[b],
      sent_lengths: batches_lengths[b],
      dictionary:meta['invDict'],
      label_names:meta['labelNames']
    }
    # TODO: replace this list of variables with dictionary
    _, loss_value_train, accuracy_value_train, sentence_accuracy_value_train, a1, a2, a3, a4 = session.run([train_step, loss, accuracy, sentence_accuracy, sent_decoded, seq_len_mask, pred_classes,true_classes], feed_dict=data)
    if (step % 50000 == 0):
      print("Minibatch train loss at step", step, ":", loss_value_train)
      print("Minibatch accuracy: {:.3%}".format(accuracy_value_train))
      print("Minibatch sentence accuracy: {:.3%}".format(sentence_accuracy_value_train))
      #printSent(a1[0],a2[0],a3[0],a4[0])
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
      sentences: dev_data[s:s+1,:],
      labels: dev_labels[s:s+1,:],
      sent_lengths: dev_lengths[s:s+1],
      dictionary:meta['invDict'],
      label_names:meta['labelNames']
    }
    loss_value_dev, accuracy_value_dev, sentence_accuracy_value_dev, pred_dev = session.run([loss, accuracy, sentence_accuracy,pred_classes], feed_dict=data_dev)
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
    save_path = saver.save(session, TMP_MODEL)
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
saver.restore(session, TMP_MODEL)
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
    sentences: test_data[s:s+1,:],
    labels: test_labels[s:s+1,:],
    sent_lengths: test_lengths[s:s+1],
    dictionary:meta['invDict'],
    label_names:meta['labelNames']
  }
  loss_value_test, accuracy_value_test, sentence_accuracy_value_test, pred_test = session.run([loss, accuracy, sentence_accuracy, pred_classes], feed_dict=data_dev)
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

save_path = saver.save(session, BEST_MODEL)
print("Best Model saved in file: %s" % save_path)

"""
for step in range(num_steps):
    # get data for a batch
    offset = (step * batch_size) % (len(train_data) - batch_size)
    batch_tweets = one_hot(train_tweets[offset : (offset + batch_size)], vocab_size)
    batch_labels = train_labels[offset : (offset + batch_size)]
    
    # put this data into a dictionary that we feed in when we run 
    # the graph.  this data fills in the placeholders we made in the graph.
    data = {sentences: batch_tweets, labels: batch_labels}
    
    # run the 'optimizer', 'loss', and 'pred_err' operations in the graph
    _, loss_value_train, error_value_train = session.run(
      [optimizer, loss, pred_err], feed_dict=data)
    
    # print stuff every 50 steps to see how we are doing
    if (step % 50 == 0):
        print("Minibatch train loss at step", step, ":", loss_value_train)
        print("Minibatch train error: %.3f%%" % error_value_train)
        
        # get test evaluation
        test_loss = []
        test_error = []
        for batch_num in range(int(len(test_data)/batch_size)):
            test_offset = (batch_num * batch_size) % (len(test_data) - batch_size)
            test_batch_tweets = one_hot_test_tweets[test_offset : (test_offset + batch_size)]
            test_batch_labels = test_labels[test_offset : (test_offset + batch_size)]
            data_testing = {sentences: test_batch_tweets, labels: test_batch_labels}
            loss_value_test, error_value_test = session.run([loss, pred_err], feed_dict=data_testing)
            test_loss.append(loss_value_test)
            test_error.append(error_value_test)
        
        print("Test loss: %.3f" % np.mean(test_loss))
        print("Test error: %.3f%%" % np.mean(test_error))
"""