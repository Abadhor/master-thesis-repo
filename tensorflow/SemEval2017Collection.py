import sys
sys.path.append('../dataset-scripts')
from SemEval2017Dataset import SemEval2017Dataset
from MWUHashTree import MWUHashTree
import numpy as np
import spacy

UNKNOWN_TOKEN = '????'
LB_BEGINNING = 'B'
LB_INSIDE = 'I'
LB_LAST = 'L'
LB_OUTSIDE = 'O'
LB_UNIGRAM = 'U'
LABEL_NAMES = [LB_BEGINNING, LB_INSIDE, LB_LAST, LB_OUTSIDE, LB_UNIGRAM]
LABEL_DICT = {LABEL_NAMES[i]:i for i in range(len(LABEL_NAMES))}

class SemEval2017Collection():
  
  def __init__(self, trainPath, devPath, testPath, verbose=False):
    self.verbose = verbose
    # if dictionary remains none, use train dictionary
    self.dictionary = None
    self.max_length = None
    self.nlp = spacy.load('en_core_web_md')
    self.train = SemEval2017Dataset(spacyNLP = self.nlp)
    self.dev = SemEval2017Dataset(spacyNLP = self.nlp)
    self.test = SemEval2017Dataset(spacyNLP = self.nlp)
    # init datasets
    self.initDS(self.train, trainPath)
    self.initDS(self.dev, devPath)
    self.initDS(self.test, testPath)
    self.train_labels = self.createLabels(self.train)
    self.dev_labels = self.createLabels(self.dev)
    self.test_labels = self.createLabels(self.test)
  
  def initDS(self, ds, path):
    ds.extractKeywords(path, "ann")
    ds.filterUnigramKeywords()
    ds.extractSentences(path, "txt", xml=False, verbose=self.verbose)
    ds.tokenize()
    
  
  def setDictionary(self, dictionary=None):
    hasInput = (dictionary != None)
    if not hasInput:
      dictionary = self.train.getDictionary()
    dic_ls = list(dictionary.keys())
    ordered_dic = {dic_ls[i]:i for i in range(len(dic_ls))}
    special = self.train.getSpecialTokens()
    for s in special:
      if s not in ordered_dic:
        ordered_dic[s] = len(ordered_dic)
    self.dictionary = ordered_dic
    print("Train")
    self.train.setDictionary(ordered_dic)
    print("Dev")
    self.dev.setDictionary(ordered_dic)
    print("Test")
    self.test.setDictionary(ordered_dic)
  
  def getDictionary(self):
    return self.dictionary
  
  def getLabelDict(self):
    return LABEL_DICT
  
  def getInverseDictionary(self):
    ret = [0] * len(self.dictionary)
    items = self.dictionary.items()
    for k, v in items:
      ret[v] = k
    return ret
  
  def getInverseLabelDict(self):
    return LABEL_NAMES
  
  def getSentenceLength(self):
    if not self.max_length:
      maxes = [
        max([len(s) for s in self.train.corpus]), 
        max([len(s) for s in self.dev.corpus]),
        max([len(s) for s in self.test.corpus])
      ]
      self.max_length = max(maxes)
    return self.max_length
  
  def createLabels(self, ds):
    """ Create labels for all sentences in the corpus """
    labels = []
    corpus = ds.getCorpus()
    print(max([len(s) for s in corpus]))
    #keys = list(ds.text_files.keys())
    #corpus = ds.text_files['S0022311513010313']
    keywords = ds.keywords
    ls_added = MWUHashTree()
    if self.verbose:
      print("Labeling...")
    for idx, sentence in enumerate(corpus):
      if self.verbose:
        print("Sentence:",idx+1, '/', len(corpus), end='\r')
      sent_labels = self.createSentenceLabels(sentence, keywords, ls_added)
      labels.append(sent_labels)
    if self.verbose:
      #for i in range(len(corpus)):
      #  print("Sentence", i)
      #  print(corpus[i])
      #  print(labels[i])
      print("The following keywords where not found:")
      nf = 0
      for item in keywords.items():
        #if item[1] != 'S0022311513010313':
        #  continue
        if item[0] not in ls_added:
          nf += 1
          #print(item[0], keywords[item[0]])
      print(nf, '/' , len(keywords))
    return labels
  
  def createSentenceLabels(self, sentence, keywords, ls_added):
    """ Create labels for a single sentence in the corpus """
    sentence_labels = [LB_OUTSIDE for i in range(len(sentence))]
    kw_tokens = 0
    for s_token_id in range(len(sentence)):
      if kw_tokens > 0:
        if kw_tokens == 1:
          sentence_labels[s_token_id] = LB_LAST
        else:
          sentence_labels[s_token_id] = LB_INSIDE
        kw_tokens -= 1
      else:
        found_kws = keywords.getAll(sentence[s_token_id:])
        if len(found_kws) > 0:
          last = found_kws[len(found_kws)-1]
          kw_tokens += len(last)
          sentence_labels[s_token_id] = LB_BEGINNING
          ls_added[last] = 1
          kw_tokens -= 1
    return sentence_labels
  
  def getKeywords(self, sentence, labels):
    kwds = []
    for i in range(len(sentence)):
      if labels[i] == LB_BEGINNING:
        kwd = []
        cur = 0
        while labels[i+cur] != LB_LAST:
          kwd.append(sentence[i+cur])
          if cur > len(sentence):
            raise NameError("Out of Bounds")
          cur += 1
    return kwds
  
  def encode(self, corpus, labels):
    # 1. Create array with dictionary indices for each word in each sentence
    # 2. Create array with label indices for each word in each sentence
    # 3. Create array with sentence lengths for each sentence
    x = np.zeros((len(corpus),self.getSentenceLength()), dtype='int32')
    y = np.zeros((len(corpus),self.getSentenceLength()), dtype='int32')
    lengths = np.zeros((len(corpus)), dtype='int32')
    for s_idx, s in enumerate(corpus):
      for t_idx, t in enumerate(s):
        token = t
        if token not in self.dictionary:
          token = UNKNOWN_TOKEN
        x[s_idx,t_idx] = self.dictionary[token]
        y[s_idx,t_idx] = LABEL_DICT[labels[s_idx][t_idx]]
      lengths[s_idx] = len(s)
    return x, y, lengths
        
    
    
