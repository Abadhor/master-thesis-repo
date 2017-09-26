import tensorflow as tf
import math
import pickle
import numpy as np
import io

# data path
PATH = "./data/"
TEST = "test_seq_data.pickle"
TRAIN = "train_seq_data.pickle"

# set variables
sent_length = 10
hidden_size = 5
vocab_size = 9
num_classes = 3
batch_size = 128
LSTM_layer_count = 1


# create a session variable that we can run later.
session = tf.Session()

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
correct_prediction = tf.equal(tf.argmax(classes,2), tf.argmax(labels,2))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# calculate fraction of sentences in which each word has been predicted correctly
sent_sum = tf.reduce_sum(tf.cast(correct_prediction, tf.float32),1)
correct_sent = tf.equal(sent_sum, tf.cast(tf.shape(correct_prediction[1]), tf.float32))
sentence_accuracy = tf.reduce_mean(tf.cast(correct_sent, tf.float32))
#sh = tf.shape(loss)

'''Define the operation that specifies the AdamOptimizer and tells
   it to minimize the loss.''';
train_step = tf.train.AdamOptimizer().minimize(loss)

# initialize any variables
tf.global_variables_initializer().run(session=session)


# load data
with io.open(PATH+TRAIN, "rb") as fp:
  train = pickle.load(fp)

train_data = train['data']
train_labels = train['labels']

with io.open(PATH+TEST, "rb") as fp:
  test = pickle.load(fp)

# create batches
num_batches = len(train_data) // batch_size
if len(train_data) % batch_size == 0:
  batch_size_last = batch_size
else:
  batch_size_last = len(train_data) - (num_batches * batch_size)
  num_batches += 1
  

batches_sent = []
batches_labels = []
offset = 0
for b in range(0, num_batches-1):
  batches_sent.append(train_data[offset:(offset+batch_size)])
  batches_labels.append(train_labels[offset:(offset+batch_size)])
  offset += batch_size

batches_sent.append(train_data[offset:(offset+batch_size_last)])
batches_labels.append(train_labels[offset:(offset+batch_size_last)])

#data = {sentences:train['data'], labels:train['labels'] }
#print(session.run(sh, feed_dict=data))

# we'll train with batches of size 128.  This means that we run 
# our model on 128 examples and then do gradient descent based on the loss
# over those 128 examples.
num_steps = 1000
num_epochs = 100
step = 0

for ep in range(0,num_epochs):
  for b in range(0,num_batches):
    data = {sentences: batches_sent[b], labels: batches_labels[b]}
    _, loss_value_train, accuracy_value_train = session.run([train_step, loss, accuracy], feed_dict=data)
    if (step % 50 == 0):
      print("Minibatch train loss at step", step, ":", loss_value_train)
      print("Minibatch accuracy: {:.3%}".format(accuracy_value_train))
      #print(str(accuracy_value_train))
    step += 1

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