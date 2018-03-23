


class TrainHelpers:
  
  @staticmethod
  def getWordVectors(gensim_model, inverse_dictionary):
    # create matrix with word embeddings, indexed based on the collection dictionary
    #no_vector_count = len(inverse_dictionary) - gensim_model.wv.syn0.shape[0]
    word_vectors = np.zeros(gensim_model.wv.syn0.shape)
    for i in range(gensim_model.wv.syn0.shape[0]):
      print("Vector:", i+1, "/", gensim_model.wv.syn0.shape[0], end='\r')
      word = inverse_dictionary[i]
      if word in gensim_model.wv:
        v = gensim_model.wv[word].reshape((1,gensim_model.wv.syn0.shape[1]))
        word_vectors[i,:] = v
      else:
        v = np.zeros((1,gensim_model.wv.syn0.shape[1]))
        word_vectors[i,:] = v
    print()
    return word_vectors
  
  @staticmethod
  def loadLMData(lm_data_path, lm_dict_path, inverse_dictionary):
    # load pre-trained language model word embeddings
    with io.open(lm_dict_path, "r", encoding='utf-8') as fp:
      lm_dict = fp.readlines()
      lm_dict = {x.strip():idx for idx,x in enumerate(lm_dict)}
    
    matches = []
    non_matches = 0
    for w in inverse_dictionary:
      if w in lm_dict:
        matches.append((w,lm_dict[w]))
      else:
        matches.append((w,lm_dict['<UNK>']))
        non_matches += 1
    
    # outputs the number of words not found in the LM dictionary
    print("Non-Matches:",non_matches)
    print("Non-Matches: {:.3%}".format(non_matches/len(inverse_dictionary)))
    
    emb_char_cnn = np.load(lm_data_path)
    
    emb_out = np.zeros((len(matches),emb_char_cnn.shape[1]))
    
    for i in range(len(matches)):
      id = matches[i][1]
      emb_out[i,:] = emb_char_cnn[id]
    return emb_out
  
  @staticmethod
  def getMWTs(label_dict, sentence):
    # get MWTs in a vectorized sentence
    MWTs = []
    B_idx = None
    for idx, t in enumerate(sentence):
      if t == label_dict['B']:
        B_idx = idx
      elif t == label_dict['L']:
        if B_idx != None:
          MWTs.append((B_idx,idx))
        B_idx = None
    return MWTs
  
  @staticmethod
  def countLabels(label_dict, sentence, sent_length, counts_dict):
    for t in range(sent_length):
      if sentence[t] == label_dict['B']:
        counts_dict['B'] += 1
      if sentence[t] == label_dict['I']:
        counts_dict['I'] += 1
      if sentence[t] == label_dict['L']:
        counts_dict['L'] += 1
      if sentence[t] == label_dict['O']:
        counts_dict['O'] += 1
  
  @staticmethod
  def getScores(TP,FP,FN):
    if TP == 0:
      return 0.0, 0.0, 0.0
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    F1 = 2*TP /(2*TP+FP+FN)
    return precision, recall, F1
  
  @staticmethod
  def createBatches(dataset, batch_size, randomize=True):
    # create batches, randomizing the order of the samples
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
  
  @staticmethod
  def runNext(next_element, classifier, train=False):
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
    while True:
        try:
          features, labels = classifier.getSession().run(next_element)
          data = {
            'sentences': features['tokens'],
            'labels': labels,
            'sent_lengths': features['length'],
            'sentence_chars': features['chars'],
            'word_lengths': features['char_lengths']
          }
          fetched = classifier.run(data, train=train)
          losses += fetched['loss_value'] * len(features['tokens'])
          accuracies += fetched['accuracy_value'] * len(features['tokens'])
          s_accuracies += fetched['sentence_accuracy_value'] * len(features['tokens'])
          # find ground truth for each sentence and evaluate predictions
          for idx,s in enumerate(labels):
            GT_list = TrainHelpers.getMWTs(s)
            GT_count += len(GT_list)
            GT_set = set()
            for mwt in GT_list:
              GT_set.add(mwt)
            pred_s = fetched['crf_decode'][idx]
            predictions = TrainHelpers.getMWTs(pred_s)
            TrainHelpers.countLabels(pred_s, features['length'][idx], count_dict)
            for mwt in predictions:
              if mwt in GT_set:
                TP_count += 1
              else:
                FP_count += 1
        except tf.errors.OutOfRangeError:
          break
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
  
  
