"""
Taken from https://github.com/nicholaslocascio/bcs-lstm/blob/master/Lab.ipynb
"""



import tensorflow as tf
import math
import pickle as p
import numpy as np
import json

def one_hot(raw_data, vocab_size):
    data = np.zeros((len(raw_data), 20, vocab_size))
    for tweet_index in range(len(raw_data)):
        tweet = raw_data[tweet_index]
        for word_index in range(20):
            word_id = tweet[word_index]
            data[tweet_index, word_index, word_id] = 1
    return data

# set variables
tweet_size = 20
hidden_size = 100
vocab_size = 7597
batch_size = 64

# this just makes sure that all our following operations will be placed in the right graph.
tf.reset_default_graph()

# create a session variable that we can run later.
session = tf.Session()

# the placeholder for tweets has first dimension batch_size for each tweet in a batch,
# second dimension tweet_size for each word in the tweet, and third dimension vocab_size
# since each word itself is represented by a one-hot vector of size vocab_size.
# Note that we use 'None' instead of batch_size for the first dimsension.  This allows us 
# to deal with variable batch sizes
tweets = tf.placeholder(tf.float32, [None, tweet_size, vocab_size])

'''TODO: create a placeholder for the labels (our predictions).  
   This should be a 1D vector with size = None, 
   since we are predicting one value for each tweet in the batch,
   but we want to be able to deal with variable batch sizes.''';
labels = tf.placeholder(tf.float32, [None])


'''TODO: create 2 LSTM Cells using BasicLSTMCell.  Note that this creates a *layer* of LSTM
   cells, not just a single one.''';
lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(hidden_size)
lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(hidden_size)

'''TODO: create three LSTM layers by wrapping two instances of 
   lstm_cell from above in tf.contrib.rnn_cell.MultiRNNCell. Note that
   you can use multiple cells by doing [cell1, cell2]. Also note
   that you should use state_is_tuple=True as an argument.  This will allow
   us to access the part of the cell state that we need later on.''';
multi_lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)

'''TODO: define the operation to create the RNN graph across time.  
   tf.nn.dynamic_rnn dynamically constructs the graph when it is executed,
   and returns the final cell state.''';
_, final_state = tf.nn.dynamic_rnn(multi_lstm_cells, tweets, dtype=tf.float32)


## We define this function that creates a weight matrix + bias parameter
## and uses them to do a matrix multiplication.
def linear(input_, output_size, name, init_bias=0.0):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(name):
        W = tf.get_variable("weights", [shape[-1], output_size], tf.float32, tf.random_normal_initializer(stddev=1.0 / math.sqrt(shape[-1])))
    if init_bias is None:
        return tf.matmul(input_, W)
    with tf.variable_scope(name):
        b = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(init_bias))
    return tf.matmul(input_, W) + b

'''TODO: pass the final state into this linear function to multiply it 
   by the weights and add bias to get our output.
   
   {Quick note that we need to feed in final_state[-1][-1] into linear since 
   final_state is actually a tuple consisting of the cell state 
   (used internally for the cell to keep track of things) 
   as well as the hidden state (the output of the cell), and one of these 
   tuples for each layer. We want the hidden state for the last layer, so we use 
   final_state[-1][-1]}''';

sentiment = linear(final_state[-1][-1], 1, "linear")
sentiment = tf.squeeze(sentiment, [1])


'''TODO: define our loss function.  
   We will use tf.nn.sigmoid_cross_entropy_with_logits, which will compare our 
   sigmoid-ed prediction (sentiment from above) to the ground truth (labels).''';

loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=sentiment, labels=labels)

# our loss with sigmoid_cross_entropy_with_logits gives us a loss for each 
# example in the batch.  We take the mean of all these losses.
loss = tf.reduce_mean(loss)

# to get actual results like 'positive' or 'negative' , 
# we round the prediction probability to 0 or 1.
prediction = tf.to_float(tf.greater_equal(sentiment, 0.5))

# calculate the error based on which predictions were actually correct.
pred_err = tf.to_float(tf.not_equal(prediction, labels))
pred_err = tf.reduce_sum(pred_err)


'''Define the operation that specifies the AdamOptimizer and tells
   it to minimize the loss.''';
optimizer = tf.train.AdamOptimizer().minimize(loss)

# initialize any variables
tf.global_variables_initializer().run(session=session)

# load our data and separate it into tweets and labels
train_data = json.load(open('data/trainTweets_preprocessed.json', 'r'))
train_data = list(map(lambda row:(np.array(row[0],dtype=np.int32),str(row[1])),train_data))
train_tweets = np.array([t[0] for t in train_data])
train_labels = np.array([int(t[1]) for t in train_data])

test_data = json.load(open('data/testTweets_preprocessed.json', 'r'))
test_data = list(map(lambda row:(np.array(row[0],dtype=np.int32),str(row[1])),test_data))
# we are just taking the first 1000 things from the test set for faster evaluation
test_data = test_data[0:1000] 
test_tweets = np.array([t[0] for t in test_data])
one_hot_test_tweets = one_hot(test_tweets, vocab_size)
test_labels = np.array([int(t[1]) for t in test_data])

# we'll train with batches of size 128.  This means that we run 
# our model on 128 examples and then do gradient descent based on the loss
# over those 128 examples.
num_steps = 1000

for step in range(num_steps):
    # get data for a batch
    offset = (step * batch_size) % (len(train_data) - batch_size)
    batch_tweets = one_hot(train_tweets[offset : (offset + batch_size)], vocab_size)
    batch_labels = train_labels[offset : (offset + batch_size)]
    
    # put this data into a dictionary that we feed in when we run 
    # the graph.  this data fills in the placeholders we made in the graph.
    data = {tweets: batch_tweets, labels: batch_labels}
    
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
            data_testing = {tweets: test_batch_tweets, labels: test_batch_labels}
            loss_value_test, error_value_test = session.run([loss, pred_err], feed_dict=data_testing)
            test_loss.append(loss_value_test)
            test_error.append(error_value_test)
        
        print("Test loss: %.3f" % np.mean(test_loss))
        print("Test error: %.3f%%" % np.mean(test_error))
