import tensorflow as tf
import math
import pickle
import numpy as np
import io
import random


class EntityModel_BoW:
  
  def __init__(self, params, useBOW=False):
    self.session = tf.Session()
    # implement exponential learning rate decay
    # optimizer gets called with this learning_rate and updates
    # global_step during its call to minimize()
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(params['starter_learning_rate'], global_step, params['decay_steps'], params['decay_rate'])
    # the placeholder for sentences has first dimension batch_size for each
    # sentence in a batch,
    # second dimension sent_length for each word in the sentence.
    # Each word is represented by its index in the dictionary
    # Note that we use 'None' instead of batch_size for the first dimsension.
    # This allows us 
    # to deal with variable batch sizes
    self.sentences = tf.placeholder(tf.int32, [None, params['sent_length']], name="sentences")
    # one hot encoding
    #sent_one_hot = tf.one_hot(self.sentences, params['vocab_size'], axis=-1, name="sent_one_hot")
    
    # the placeholder for labels has first dimension batch_size for each
    # sentence in a batch and
    # second dimension sent_length for each word in the sentence
    # Note that we use 'None' instead of batch_size for the first dimsension.
    # This allows us 
    # to deal with variable batch sizes
    self.labels = tf.placeholder(tf.int32, [None, params['sent_length']], name="labels")
    
    # one hot encoding
    lbl_one_hot = tf.one_hot(self.labels, params['num_classes'], name="lbl_one_hot")
    
    # the placeholder for the lengths of each sentence
    self.sent_lengths = tf.placeholder(tf.int32, [None], name="sent_lengths")
    
    # mask that indicates the length of the sentence
    # each padded position after the last word is equal to 0
    # all others are equal to 1
    seq_len_mask = tf.sequence_mask(self.sent_lengths, maxlen=params['sent_length'], dtype=tf.float32)
    
    # placeholder for the dictionary
    self.dictionary = tf.placeholder(tf.string, [params['vocab_size']], name="invDict")
    
    # placeholder for the label_names
    self.label_names = tf.placeholder(tf.string, [params['num_classes']], name="label_names")
    
    # word embeddings
    wv_shape = params['word_embeddings'].shape
    word_embeddings = tf.get_variable("word_embeddings",shape=wv_shape, trainable=False)
    embedding_placeholder = tf.placeholder(tf.float32, shape=wv_shape)
    init_embeddings = tf.assign(word_embeddings, embedding_placeholder)
    embedded_word_ids = tf.nn.embedding_lookup(word_embeddings, self.sentences)
    
    # dense input matrix
    
    #lstm_inputs = self.linearLayerTiled(sent_one_hot, params['vocab_size'], params['dense_hidden_size'], "lstm_input")
    
    
    layer1 = self.createBiDirectionalLSTMLayer(embedded_word_ids, params['hidden_size'], self.sent_lengths, 'LSTM_l1')
    
    layer2 = self.createBiDirectionalLSTMLayer(layer1, params['hidden_size'], self.sent_lengths, 'LSTM_l2')
    
    # pass the final state into this linear function to multiply it 
    # by the weights and add bias to get our output.
    # Shape of class_scores is [batch_size, sent_length, num_classes]
    class_scores = self.linearLayerTiled(layer2, 2*params['hidden_size'], params['num_classes'], "output")
    
    # define our loss function.
    #loss = tf.nn.softmax_cross_entropy_with_logits(logits=class_scores, labels=lbl_one_hot)
    
    # Compute the log-likelihood of the gold sequences and keep the transition
    # params for inference at test time.
    log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
        class_scores, self.labels, self.sent_lengths)
    
    # our loss with softmax_cross_entropy_with_logits gives us a loss for each word 
    # in each sentence.  We take the sum of all losses per sentence.
    #loss = tf.reduce_sum(loss, axis=1)
    
    # our loss with softmax_cross_entropy_with_logits gives us a loss for each 
    # example in the batch.  We take the mean of all these losses.
    #self.loss = tf.reduce_mean(loss)
    
    self.loss = tf.reduce_mean(-log_likelihood)
    
    # calculate accuracy of word class predictions
    # first gather the indices of the highest results
    # since the class_scores and words are one-hot encoded,
    # this gives us the id of the class
    #sent_decoded = tf.argmax(sent_one_hot,2)
    self.pred_classes = tf.argmax(class_scores,2, output_type=tf.int32)
    true_classes = self.labels
    correct_prediction = tf.cast(tf.equal(self.pred_classes, true_classes), tf.float32)
    
    # ensure that results outside the sequence length are 0
    correct_prediction = tf.multiply(correct_prediction, seq_len_mask)
    
    # calculate accuracy only for the part in the sequence
    # that is part of the sentence
    sent_sum = tf.reduce_sum(correct_prediction,1)
    mask_sum = tf.reduce_sum(seq_len_mask, 1)
    self.accuracy = tf.reduce_mean(sent_sum / mask_sum)
    
    # calculate fraction of sentences in which each word has been predicted correctly
    correct_sent = tf.cast(tf.equal(sent_sum, mask_sum), tf.float32)
    self.sentence_accuracy = tf.reduce_mean(correct_sent)
    
    # decode sentences into readable words and labels
    # lookup the string of the ID of the words and classes
    # in the word dictionary and label name dictionary
    #sent_decoded = tf.gather(self.dictionary, sent_decoded)
    pred_classes_names = tf.gather(self.label_names, self.pred_classes)
    true_classes = tf.gather(self.label_names, true_classes)
    
    '''Define the operation that specifies the AdamOptimizer and tells
      it to minimize the loss.''';
    self.train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss, global_step=global_step)

    # initialize any variables
    init_op = tf.global_variables_initializer()

    # Add ops to save and restore all the variables.
    self.saver = tf.train.Saver()

    # run init_op
    init_op.run(session=self.session)
    self.session.run(init_embeddings, feed_dict={embedding_placeholder: params['word_embeddings']})
  
  def closeSession(self):
    self.session.close()
  
  def __enter__(self):
    return self
  
  def __exit__(self, exception_type, exception_value, traceback):
    if exception_type is not None:
      print(exception_type, exception_value, traceback)
    self.closeSession()
  
  def train(self, data):
    data = {
        self.sentences: data['sentences'],
        self.labels: data['labels'],
        self.sent_lengths: data['sent_lengths'],
        self.dictionary: data['dictionary'],
        self.label_names: data['label_names']
    }
    _, loss_value, accuracy_value, sentence_accuracy_value = self.session.run([self.train_step, self.loss, self.accuracy, self.sentence_accuracy], feed_dict=data)
    return loss_value, accuracy_value, sentence_accuracy_value
  
  def run(self, data, train=False):
    data = {
        self.sentences: data['sentences'],
        self.labels: data['labels'],
        self.sent_lengths: data['sent_lengths'],
        self.dictionary: data['dictionary'],
        self.label_names: data['label_names']
    }
    fetches = {
      'loss_value':self.loss,
      'accuracy_value':self.accuracy,
      'sentence_accuracy_value':self.sentence_accuracy,
      'prediction':self.pred_classes
    }
    if train:
      fetches['train_step'] = self.train_step
    return self.session.run(fetches, feed_dict=data)
  
  def saveModel(self, path):
    return self.saver.save(self.session, path)
  
  def restoreModel(self, path):
    self.saver.restore(self.session, path)
    
  
  # Create a weight matrix + bias layer
  # with duplicated matrix for each batch entry
  # input shape [batch_size, sent_length, input_depth] -->
  # output shape [batch_size, sent_length, output_depth]
  def linearLayerTiled(self, input_, input_depth, output_depth, name):
    cur_batch_size = tf.shape(input_)[0]
    with tf.variable_scope(name):
      W = tf.get_variable("linear_weights", (1, output_depth, input_depth), tf.float32, tf.random_normal_initializer())
      W = tf.tile(W, multiples=[cur_batch_size,1,1])
      b = tf.get_variable("bias", (output_depth), initializer=tf.constant_initializer(0.0))
    return tf.transpose(tf.matmul(W, tf.transpose(input_, perm=[0,2,1])),perm=[0,2,1]) + b
  
  def createBiDirectionalLSTMLayer(self, lstm_inputs, hidden_size, sent_lengths, name):
    with tf.variable_scope(name): 
      # Create a forward and a backward LSTM layer for the bidirectional_dynamic_rnn
      fw = tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
      bw = tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
      
      # Returns a tuple (outputs, output_states) where:
      # outputs: A tuple (output_fw, output_bw) containing
      # the forward and the backward rnn output Tensor.
      # If time_major == False (default), output_fw will be a Tensor shaped: 
      # [batch_size, max_time, cell_fw.output_size]
      # and output_bw will be a Tensor shaped: 
      # [batch_size, max_time, cell_bw.output_size].
      outputs, final_state = tf.nn.bidirectional_dynamic_rnn(
          fw,
          bw,
          lstm_inputs,
          sequence_length=sent_lengths,
          dtype=tf.float32,
          scope="bi-lstm"
      )
      
      out_fw, out_bw = outputs
      concat_outputs = tf.concat([out_fw, out_bw], 2)
      return concat_outputs
