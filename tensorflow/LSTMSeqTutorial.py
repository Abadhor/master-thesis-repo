import tensorflow as tf
import math
import pickle
import numpy as np
import io
import random

random.seed(10)

# data path
PATH = "./data/"
#"""
TEST = "test_seq_data.pickle"
DEV = "dev_seq_data.pickle"
TRAIN = "train_seq_data.pickle"
"""
TEST = "test_seq_data_adv.pickle"
DEV = "dev_seq_data_adv.pickle"
TRAIN = "train_seq_data_adv.pickle"
#"""
TMP_MODEL = "./tmp/model.ckpt"
BEST_MODEL = "./models/best_model.session"

# model parameters
sent_length = 10
#sent_length = 20
hidden_size = 5
vocab_size = 9
num_classes = 3
batch_size = 128
LSTM_layer_count = 1

# training parameters
num_epochs = 100
early_stopping_epoch_limit = 10
starter_learning_rate = 0.01
decay_steps = 100
decay_rate = 0.96

# load data
with io.open(PATH+TRAIN, "rb") as fp:
  train = pickle.load(fp)

train_data = train['data']
train_labels = train['labels']

with io.open(PATH+TEST, "rb") as fp:
  test = pickle.load(fp)

test_data = test['data']
test_labels = test['labels']

with io.open(PATH+DEV, "rb") as fp:
  dev = pickle.load(fp)

dev_data = dev['data']
dev_labels = dev['labels']

# define the number of batches
num_batches = len(train_data) // batch_size
if len(train_data) % batch_size == 0:
  batch_size_last = batch_size
else:
  batch_size_last = len(train_data) - (num_batches * batch_size)
  num_batches += 1

# create a session variable that we can run later.
session = tf.Session()

# implement exponential learning rate decay
# optimizer gets called with this learning_rate and updates
# global_step during its call to minimize()
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, decay_steps, decay_rate)

# the placeholder for sentences has first dimension batch_size for each
# sentence in a batch,
# second dimension sent_length for each word in the sentence,
# and third dimension vocab_size
# since each word itself is represented by a one-hot vector of size vocab_size.
# Note that we use 'None' instead of batch_size for the first dimsension.
# This allows us 
# to deal with variable batch sizes
sentences = tf.placeholder(tf.float32, [None, sent_length, vocab_size])

# the placeholder for labels has first dimension batch_size for each
# sentence in a batch and
# second dimension sent_length for each word in the sentence and 
# third dimension num_classes since we want
# to predict a label for each word
# Note that we use 'None' instead of batch_size for the first dimsension.
# This allows us 
# to deal with variable batch sizes
labels = tf.placeholder(tf.float32, [None, sent_length, num_classes])

# mask that indicates the length of the sentence
# each padded position after the last word is equal to 0
# all others are equal to 1
seq_len_mask = tf.reduce_max(labels, 2)

# placeholder for the dictionary
dictionary = tf.placeholder(tf.string, [vocab_size])

# placeholder for the label_names
label_names = tf.placeholder(tf.string, [num_classes])

# create LSTM Cells using BasicLSTMCell.  Note that this creates a *layer* of
# LSTM cells, not just a single one.
LSTM_layers = []
for i in range(0, LSTM_layer_count):
  LSTM_layers.append(tf.contrib.rnn.BasicLSTMCell(hidden_size))

# create n+1 LSTM layers by wrapping n instances of 
# lstm_cell from above in tf.contrib.rnn_cell.MultiRNNCell. Note that
# you can use multiple cells by doing [cell1, cell2]. Also note
# that you should use state_is_tuple=True as an argument.  This will allow
# us to access the part of the cell state that we need later on.
multi_lstm_cells = tf.contrib.rnn.MultiRNNCell(LSTM_layers, state_is_tuple=True)

'''TODO: define the operation to create the RNN graph across time.  
   tf.nn.dynamic_rnn dynamically constructs the graph when it is executed,
   and returns the final cell state.''';
outputs, final_state = tf.nn.dynamic_rnn(multi_lstm_cells, sentences, dtype=tf.float32)

# Shape of the output is [batch_size, sent_length, hidden_size]

# Create a weight matrix + bias layer
# Applies the matrix to every output of the LSTM layer
# input shape [batch_size, sent_length, input_depth] -->
# output shape [batch_size, sent_length, output_depth]
def linearLayer(input_, input_depth, output_depth, name):
  cur_batch_size = tf.shape(input_)[0]
  with tf.variable_scope(name):
    W = tf.get_variable("linear_weights", (1, output_depth, input_depth), tf.float32, tf.random_normal_initializer())
    W = tf.tile(W, multiples=[cur_batch_size,1,1])
    b = tf.get_variable("bias", (output_depth), initializer=tf.constant_initializer(0.0))
  return tf.transpose(tf.matmul(W, tf.transpose(input_, perm=[0,2,1])),perm=[0,2,1]) + b

# pass the final state into this linear function to multiply it 
# by the weights and add bias to get our output.
# Shape of classes is [batch_size, sent_length, num_classes]
classes = linearLayer(outputs, hidden_size, num_classes, "output")

# define our loss function.
loss = tf.nn.softmax_cross_entropy_with_logits(logits=classes, labels=labels)

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
sent_decoded = tf.argmax(sentences,2)
pred_classes = tf.argmax(classes,2)
true_classes = tf.argmax(labels,2)
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
pred_classes = tf.gather(label_names, pred_classes)
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
def createBatches(data, labels, num_batches, batch_size_last):
  batches_sent = []
  batches_labels = []
  b_idx = [x for x in range(0,len(data))]
  random.shuffle(b_idx)
  offset = 0
  for b in range(0, num_batches-1):
    cur_b_idx = b_idx[offset:(offset+batch_size)]
    batches_sent.append([data[x] for x in cur_b_idx])
    batches_labels.append([labels[x] for x in cur_b_idx])
    offset += batch_size
  
  cur_b_idx = b_idx[offset:(offset+batch_size_last)]
  batches_sent.append([data[x] for x in cur_b_idx])
  batches_labels.append([labels[x] for x in cur_b_idx])
  return batches_sent, batches_labels

def printSent(sent, mask, pred, true):
  sent = np.vectorize(bytes.decode)(sent)
  pred = np.vectorize(bytes.decode)(pred)
  true = np.vectorize(bytes.decode)(true)
  print("Sent: "," ".join(sent))
  print("Mask: "," ".join([str(int(x)) for x in mask]))
  print("Pred: "," ".join(pred))
  print("True: "," ".join(true))

#data = {sentences:train['data'][0:2,:,:], labels:train['labels'][0:2,:,:], dictionary:train['dictionary'], label_names:train['label_names'] }
#x, a1,a2,a3 = session.run([accuracy, sent_decoded,pred_classes,true_classes], feed_dict=data)
#print(x)
#printSent(a1[0],a2[0],a3[0])
#exit()

# we'll train with batches of size 128.  This means that we run 
# our model on 128 examples and then do gradient descent based on the loss
# over those 128 examples.
step = 0

no_imp_step_count = 0
best_accuracy = 0
best_sent_accuracy = 0
best_epoch = 0

for ep in range(0,num_epochs):
  batches_sent, batches_labels = createBatches(train_data, train_labels, num_batches, batch_size_last)
  for b in range(0,num_batches):
    data = {
      sentences: batches_sent[b],
      labels: batches_labels[b],
      dictionary:train['dictionary'],
      label_names:train['label_names']
    }
    # TODO: replace this list of variables with dictionary
    _, loss_value_train, accuracy_value_train, sentence_accuracy_value_train, a1, a2, a3, a4 = session.run([train_step, loss, accuracy, sentence_accuracy, sent_decoded, seq_len_mask, pred_classes,true_classes], feed_dict=data)
    if (step % 50000 == 0):
      print("Minibatch train loss at step", step, ":", loss_value_train)
      print("Minibatch accuracy: {:.3%}".format(accuracy_value_train))
      print("Devset sentence accuracy: {:.3%}".format(sentence_accuracy_value_train))
      printSent(a1[0],a2[0],a3[0],a4[0])
    step += 1
  # validation on dev set
  data_dev = {
    sentences: dev_data,
    labels: dev_labels,
    dictionary:dev['dictionary'],
    label_names:dev['label_names']
  }
  loss_value_dev, accuracy_value_dev, sentence_accuracy_value_dev, a1, a2, a3, a4 = session.run([loss, accuracy, sentence_accuracy, sent_decoded, seq_len_mask,pred_classes,true_classes], feed_dict=data_dev)
  print("Devset loss at Epoch", ep, ":", loss_value_dev)
  print("Devset accuracy: {:.3%}".format(accuracy_value_dev))
  print("Devset sentence accuracy: {:.3%}".format(sentence_accuracy_value_dev))
  printSent(a1[0],a2[0],a3[0],a4[0])
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
data_testing = {
  sentences: test_data,
  labels: test_labels,
  dictionary:test['dictionary'],
  label_names:test['label_names']
}
loss_value_test, accuracy_value_test, sentence_accuracy_value_test, a1, a2, a3, a4 = session.run([loss, accuracy, sentence_accuracy, sent_decoded, seq_len_mask,pred_classes,true_classes], feed_dict=data_testing)
print("Test set loss at Epoch", best_epoch, ":", loss_value_test)
print("Test set accuracy: {:.3%}".format(accuracy_value_test))
print("Test set sentence accuracy: {:.3%}".format(sentence_accuracy_value_test))
printSent(a1[0],a2[0],a3[0],a4[0])

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