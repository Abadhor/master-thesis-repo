import tensorflow as tf
import math
import pickle
import numpy as np
import io
import random


class EntityModel:
  
  def __init__(self, params, word_features='emb', char_features='boc', LM='emb', gazetteers=True):
    self.gazetteers = gazetteers
    self.session = tf.Session()
    # implement exponential learning rate decay
    # optimizer gets called with this learning_rate and updates
    # global_step during its call to minimize()
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(params['starter_learning_rate'], global_step, params['decay_steps'], params['decay_rate'])
    
    
    # dropout keep probability
    self.dropout_05 = tf.placeholder(tf.float32, (), name="dropout_05")
    self.dropout_08 = tf.placeholder(tf.float32, (), name="dropout_08")
    
    # the placeholder for sentences has first dimension batch_size for each
    # sentence in a batch,
    # second dimension sent_length for each word in the sentence.
    # Each word is represented by its index in the dictionary
    # Note that we use 'None' instead of batch_size for the first dimsension.
    # This allows us 
    # to deal with variable batch sizes
    self.sentences = tf.placeholder(tf.int32, [None, params['sent_length']], name="sentences")

    
    # the placeholder for labels has first dimension batch_size for each
    # sentence in a batch and
    # second dimension sent_length for each word in the sentence
    # Note that we use 'None' instead of batch_size for the first dimsension.
    # This allows us 
    # to deal with variable batch sizes
    self.labels = tf.placeholder(tf.int32, [None, params['sent_length']], name="labels")
    
    
    # the placeholder for the lengths of each sentence
    self.sent_lengths = tf.placeholder(tf.int32, [None], name="sent_lengths")
    
    # mask that indicates the length of the sentence
    # each padded position after the last word is equal to 0
    # all others are equal to 1
    seq_len_mask = tf.sequence_mask(self.sent_lengths, maxlen=params['sent_length'], dtype=tf.float32)
    
    # the placeholder for sentences_chars has first dimension batch_size for each
    # sentence in a batch,
    # second dimension sent_length for each word in the sentence,
    # third dimension word_length for each char in a word
    # Each word is represented by its index in the dictionary
    # Note that we use 'None' instead of batch_size for the first dimsension.
    # This allows us 
    # to deal with variable batch sizes
    self.sentences_chars = tf.placeholder(tf.int32, [None, params['sent_length'], params['word_length']], name="sentences_chars")
    
    # the placeholder for the lengths of each word
    self.word_lengths = tf.placeholder(tf.int32, [None, params['sent_length']], name="word_lengths")
    
    self.word_len_mask = tf.sequence_mask(self.word_lengths, maxlen=params['word_length'], dtype=tf.float32)
    
    if gazetteers:
      # placeholder for the binary gazetteers feature of each word
      self.gazetteers_binary = tf.placeholder(tf.float32, [None, params['sent_length'], params['gazetteer_count']], name="gazetteers_binary")
    
    # word embeddings
    if word_features=='emb':
      self.wordEmbeddings(params)
    elif word_features=='bow':
      self.BOW(params)
    else:
      raise ValueError('word_features='+word_features)
    
    # Language Model
    if LM == 'emb':
      LM_features = self.LM_wordEmbeddings(params)
    elif LM == None:
      print('LM == None')
    else:
      raise ValueError('LM='+LM)
    
    # char embeddings
    if char_features=='boc':
      char_embeddings = self.BOC(self.sentences_chars, params)
      # reshape mask for elementwise mult with broadcasting
      new_shape = [
        tf.shape(self.word_len_mask)[0],
        tf.shape(self.word_len_mask)[1],
        tf.shape(self.word_len_mask)[2],
        1
      ]
      w_m_re = tf.reshape(self.word_len_mask, new_shape)
      masked_char_embeddings = tf.multiply(char_embeddings, w_m_re, name="mask_char_embeddings")
    else:
      raise ValueError('char_features='+char_features)
    
    # char CNN
    #self.token_char_features = self.simpleCharCNN(masked_char_embeddings, params)
    self.token_char_features = self.layeredCharCNN(masked_char_embeddings, params)
    
    layer1_inputs = tf.concat([self.token_features, self.token_char_features],
    axis=2, name="layer1_inputs")
    
    layer1 = self.createBiDirectionalLSTMLayer(layer1_inputs, params['hidden_size'], self.sent_lengths, 'LSTM_l1')
    
    # add LM to layer 1 output
    if LM == 'emb':
      layer2_inputs = tf.concat([layer1, LM_features],
      axis=2, name="layer2_inputs")
    else:
      layer2_inputs = layer1
    
    layer2 = self.createBiDirectionalLSTMLayer(layer2_inputs, params['hidden_size'], self.sent_lengths, 'LSTM_l2')
    
    if gazetteers:
      # gazetteers layer
      gazetteers_dense_rs = tf.reshape(self.gazetteers_binary, [tf.shape(self.gazetteers_binary)[0]*params['sent_length'], params['gazetteer_count']],name="gazetteers_dense_rs")
      gazetteers_dense_activation = tf.tanh(self.linearLayer(gazetteers_dense_rs, params['gazetteer_count'], params['gazetteers_dense_size'], "gazetteers_linear"))
      gazetteers_dense = tf.reshape(gazetteers_dense_activation, [tf.shape(self.gazetteers_binary)[0], params['sent_length'], params['gazetteers_dense_size']], name="gazetteers_dense_rs2")
      
      # add gazetteers to layer 2 output
      final_dense_inputs = tf.concat([layer2, gazetteers_dense], axis=2, name="final_dense_inputs")
    else:
      final_dense_inputs = layer2
    
    # pass the final state into this linear function to multiply it 
    # by the weights and add bias to get our output.
    # Shape of class_scores is [batch_size, sent_length, num_classes]
    class_scores_rs_pre = tf.reshape(final_dense_inputs, [tf.shape(final_dense_inputs)[0]*params['sent_length'], 2*params['hidden_size']+params['gazetteers_dense_size']], name="class_scores_rs_pre")
    class_scores_linear = self.linearLayer(class_scores_rs_pre, 2*params['hidden_size']+params['gazetteers_dense_size'], params['num_classes'], "output")
    class_scores = tf.reshape(class_scores_linear, [tf.shape(final_dense_inputs)[0],params['sent_length'], params['num_classes']], name="class_scores_rs2")
        
    # Compute the log-likelihood of the gold sequences and keep the transition
    # params for inference at test time.
    log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
        class_scores, self.labels, self.sent_lengths)
    
    self.decode_tags, best_score = tf.contrib.crf.crf_decode(class_scores, transition_params, self.sent_lengths)
    
    # L2 regularization
    # punish if weights are too big
    weights_list = tf.trainable_variables()
    variables_names =[v.name for v in weights_list]
    regularizer = tf.contrib.layers.l2_regularizer(params['l2-coefficient'])
    
    # our crf loss gives us a loss for each 
    # example in the batch.  We take the mean of all these losses.
    self.loss = tf.reduce_mean(-log_likelihood) + tf.contrib.layers.apply_regularization(regularizer, weights_list=weights_list)
    
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
    
    '''Define the operation that specifies the AdamOptimizer and tells
      it to minimize the loss.'''    
    self.train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss, global_step=global_step)
    
    # AdamOptimizer with gradiant clipping
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    gvs = optimizer.compute_gradients(self.loss)
    capped_gvs = [(tf.clip_by_norm(grad,params['grad_clip_norm']), var) for grad, var in gvs]
    self.train_step = optimizer.apply_gradients(capped_gvs, global_step=global_step)

    # initialize any variables
    init_op = tf.global_variables_initializer()

    # Add ops to save and restore all the variables.
    self.saver = tf.train.Saver()

    # run init_op
    init_op.run(session=self.session)
    if word_features=='emb':
      self.session.run(self.init_embeddings, feed_dict={self.embedding_placeholder: params['word_embeddings']})
    
    if LM=='emb':
      self.session.run(self.init_LM_embeddings, feed_dict={self.LM_embedding_placeholder: params['LM_embeddings']})
  
  def closeSession(self):
    self.session.close()
  
  def __enter__(self):
    return self
  
  def __exit__(self, exception_type, exception_value, traceback):
    if exception_type is not None:
      print(exception_type, exception_value, traceback)
    self.closeSession()
  
  def run(self, data, train=False):
    feed_dict = {
        self.sentences: data['sentences'],
        self.labels: data['labels'],
        self.sent_lengths: data['sent_lengths'],
        self.sentences_chars: data['sentence_chars'],
        self.word_lengths: data['word_lengths'],
        self.dropout_05: 1.0,
        self.dropout_08: 1.0
    }
    fetches = {
      'loss_value':self.loss,
      'accuracy_value':self.accuracy,
      'sentence_accuracy_value':self.sentence_accuracy,
      'prediction':self.pred_classes,
      'crf_decode':self.decode_tags
    }
    if self.gazetteers:
      feed_dict[self.gazetteers_binary] = data['gazetteers']
    if train:
      fetches['train_step'] = self.train_step
      feed_dict[self.dropout_05] = 0.5
      feed_dict[self.dropout_08] = 0.8
    return self.session.run(fetches, feed_dict=feed_dict)
  
  def getSession(self):
    return self.session
  
  def saveModel(self, path):
    return self.saver.save(self.session, path)
  
  def restoreModel(self, path):
    self.saver.restore(self.session, path)
    
  # Create a weight matrix + bias layer
  # input shape [batch_size, input_depth] -->
  # output shape [batch_size, output_depth]
  def linearLayer(self, input_, input_depth, output_depth, name):
    with tf.variable_scope(name):
      W = tf.get_variable("linear_weights", (input_depth,output_depth), tf.float32, tf.random_normal_initializer())
      b = tf.get_variable("bias", (output_depth), initializer=tf.constant_initializer(0.0))
    return tf.matmul(input_, W) + b
  
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
  
  
  def createBiDirectionalLSTMLayer(self, lstm_inputs, hidden_size, sent_lengths, scope, concat=True):
    with tf.variable_scope(scope): 
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
      if concat:
        concat_outputs = tf.concat([out_fw, out_bw], 2)
        return concat_outputs
      else:
        return out_fw, out_bw
  
  def createLSTMLayer(self, lstm_inputs, hidden_size, sent_lengths, scope):
    with tf.variable_scope(scope): 
      # Create a forward LSTM layer for the dynamic_rnn
      fw = tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
      
      # Returns a tuple (outputs, output_states) where:
      # outputs:
      # If time_major == False (default), outputs will be a Tensor shaped: 
      # [batch_size, max_time, cell.output_size]
      outputs, final_state = tf.nn.dynamic_rnn(
          fw,
          lstm_inputs,
          sequence_length=sent_lengths,
          dtype=tf.float32,
          scope="fw-lstm"
      )
      return outputs
  
  def wordEmbeddings(self, params):
    # init word embeddings
    # zero for words with index greater wv_shape[0]
    wv_shape = params['word_embeddings'].shape
    word_embeddings = tf.get_variable("word_embeddings",shape=wv_shape, trainable=False)
    self.embedding_placeholder = tf.placeholder(tf.float32, shape=wv_shape)
    self.init_embeddings = tf.assign(word_embeddings, self.embedding_placeholder)
    self.token_features = tf.nn.embedding_lookup(word_embeddings, self.sentences)
    # trainable embeddings for unknown words
    # non-zero only for words with index greater wv_shape[0]
    extra_word_embeddings = tf.get_variable("extra_word_embeddings",
    shape=[params['no_vector_count'], wv_shape[1]])
    word_embeddings_vocab_const = tf.constant(wv_shape[0], dtype=tf.int32)
    extra_idx = self.sentences - word_embeddings_vocab_const
    extra_features = tf.nn.embedding_lookup(extra_word_embeddings, extra_idx)
    self.token_features = self.token_features + extra_features
  
  def LM_wordEmbeddings(self, params):
    # init word embeddings
    LM_wv_shape = params['LM_embeddings'].shape
    word_embeddings = tf.get_variable("LM_word_embeddings",shape=LM_wv_shape, trainable=False)
    self.LM_embedding_placeholder = tf.placeholder(tf.float32, shape=LM_wv_shape)
    self.init_LM_embeddings = tf.assign(word_embeddings, self.LM_embedding_placeholder)
    self.LM_token_features = tf.nn.embedding_lookup(word_embeddings, self.sentences)
    # LSTM layers for LM
    fwd_l1, bw = self.createBiDirectionalLSTMLayer(self.LM_token_features, params['LM_hidden_size'], self.sent_lengths, 'LM_BI_LSTM', concat=False)
    fwd_l2 = self.createLSTMLayer(fwd_l1, params['LM_hidden_size'], self.sent_lengths, 'LM_FWD_LSTM')
    return tf.concat([fwd_l2, bw], 2)
  
  def BOW(self, params):
    # Bag of Words
    # learn word features during training
    bow_features = tf.get_variable("bow_features",shape=[params['vocab_size'],params['bow_feature_size']], dtype=tf.float32)
    self.token_features = tf.nn.embedding_lookup(bow_features, self.sentences)
  
  def BOC(self, sentences_chars, params):
    # Bag of Characters
    # learn character features during training
    boc_features = tf.get_variable("boc_features",shape=[params['alphabet_size'],params['boc_feature_size']], dtype=tf.float32)
    return tf.nn.embedding_lookup(boc_features, sentences_chars)
  
  def simpleCharCNN(self, char_embeddings, params):
    new_batch_size = tf.shape(char_embeddings)[0]*params['sent_length']
    # reshape to [batch*sent_length, word_length, depth]
    # use tf.shape(char_embeddings)[0] for batch since batch size is dynamic
    ba_se_merge = tf.reshape(char_embeddings, 
    [
      new_batch_size,
      params['word_length'],
      params['boc_feature_size']
    ])
    # kernel for CNN
    filter = tf.get_variable("char_cnn_filter",shape=
    [
      params['char_cnn_filter_width'],
      params['boc_feature_size'],
      params['char_cnn_out_features']
    ], dtype=tf.float32)
    char_cnn = tf.nn.conv1d(
      ba_se_merge,
      filter,
      1,
      padding='SAME',
      data_format="NHWC",
      name="char_cnn"
    )
    # dense layer after Char-CNN
    dense_input = tf.reshape(char_cnn,
    [
      tf.shape(char_embeddings)[0]*params['sent_length'],
      params['word_length']*params['char_cnn_out_features']
    ])
    with tf.variable_scope("char_CNN"):
      W = tf.get_variable("linear_weights", (params['word_length']*params['char_cnn_out_features'], params['char_dense_out_features']), tf.float32, tf.random_normal_initializer())
      b = tf.get_variable("bias", (params['char_dense_out_features']), initializer=tf.constant_initializer(0.0))
    dense_op = tf.nn.relu(tf.matmul(dense_input, W) + b)
    return tf.reshape(dense_op, 
    [
      tf.shape(char_embeddings)[0],
      params['sent_length'],
      params['char_dense_out_features']
    ])
  
  def layeredCharCNN(self, char_embeddings, params):
    """
      Creates a CharCNN with multiple layers.
      the list params['charCNN_layer_depths'] contains the depth
      of each layer and its length is the number of layers.
    """
    new_batch_size = tf.shape(char_embeddings)[0]*params['sent_length']
    # reshape to [batch*sent_length, word_length, depth]
    # use tf.shape(char_embeddings)[0] for batch since batch size is dynamic
    embeddings_rs = tf.reshape(char_embeddings, 
    [
      new_batch_size,
      params['word_length'],
      params['boc_feature_size']
    ])
    cur_layer = embeddings_rs
    in_depth = params['boc_feature_size']
    for idx, depth in enumerate(params['charCNN_layer_depths']):
      out_depth = depth
      cur_layer = self.charCNNLayer(cur_layer, in_depth, out_depth, params, "charCNN_layer_"+str(idx))
      in_depth = out_depth
    final_layer = cur_layer
    final_layer_rs = tf.reshape(final_layer, [new_batch_size, params['charCNN_layer_depths'][-1]])
    # after final CNN layer use dense layer
    # final layer: [batch*sent_length, lastCharCNN_layer_depth]
    linear = self.linearLayer(final_layer_rs, params['charCNN_layer_depths'][-1], params['char_dense_out_features'], "charCNN_dense")
    dense_activation = tf.nn.relu(linear)
    return tf.reshape(dense_activation, 
    [
      tf.shape(char_embeddings)[0],
      params['sent_length'],
      params['char_dense_out_features']
    ]) 
  
  def charCNNLayer(self, input, in_depth, out_depth, params, name):
    """
      Each layer halves the length of the input
    """
    with tf.variable_scope(name):
      # kernel for CNN
      filter1 = tf.get_variable("char_cnn_filter1",shape=
      [
        params['char_cnn_filter_width'],
        in_depth,
        out_depth
      ], dtype=tf.float32)
      filter2 = tf.get_variable("char_cnn_filter2",shape=
      [
        params['char_cnn_filter_width'],
        out_depth,
        out_depth
      ], dtype=tf.float32)
      # CNN
      char_cnn1 = tf.nn.conv1d(
        input,
        filter1,
        1,
        padding='SAME',
        data_format="NHWC",
        name="cnn1"
      )
      char_cnn2 = tf.nn.conv1d(
        tf.nn.relu(char_cnn1),
        filter2,
        1,
        padding='SAME',
        data_format="NHWC",
        name="cnn2"
      )
      # Pool
      pool = tf.layers.max_pooling1d(
        tf.nn.relu(char_cnn2),
        2,
        2,
        padding='same',
        data_format='channels_last',
        name="max_pool"
      )
    return pool
      
    
