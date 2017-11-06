import sys
sys.path.append('../dataset-scripts')
from SemEval2017Dataset import SemEval2017Dataset
from MWUHashTree import MWUHashTree
import numpy as np

UNKNOWN_TOKEN = '$unknown$'
LB_OUTSIDE = 'O'
LB_INSIDE = 'I'
LB_BEGINNING = 'B'
LB_LAST = 'L'
LB_UNIGRAM = 'U'

class SemEval2017Collection():
  """
  TODO:
  -dont check all keywords, check words in sentence to see if keywords match
  -convert corpus to one hot features and labels
  """
  
  def __init__(self, trainPath, devPath, testPath, verbose=False):
    self.verbose = verbose
    # if dictionary remains none, use train dictionary
    self.dictionary = None
    self.train = SemEval2017Dataset()
    self.dev = SemEval2017Dataset()
    self.test = SemEval2017Dataset()
    # init datasets
    #self.initDS(self.train, trainPath)
    self.initDS(self.dev, devPath)
    #self.initDS(self.test, testPath)
  
  
  def initDS(self, ds, path):
    ds.extractKeywords(path, "ann")
    ds.filterUnigramKeywords()
    ds.extractSentences(path, "txt", xml=False, verbose=self.verbose)
    ds.tokenize()
    self.createLabels(ds)
  
  def setDictionary(self, dictionary):
    self.dictionary = dictionary
  
  def createLabels(self, ds):
    """ Create labels for all sentences in the corpus """
    labels = []
    corpus = ds.getCorpus()
    #keys = list(ds.text_files.keys())
    #corpus = ds.text_files['S0003491615001839']
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
      #for i in range(4,6):
      #  print("Sentence", i)
      #  print(corpus[i])
      #  print(labels[i])
      print("The following keywords where not found:")
      for item in keywords.items():
        #if item[1] != 'S0003491615001839':
        #  continue
        if item[0] not in ls_added:
          print(item[0], keywords[item[0]])
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
  
  """
  def labelTokens(self, sentence, sent_labels, keyword_tokens):
    """ """Check and label the tokens in a sentence""" """
    one_found = 0
    # iterate through each token in the sentence
    for s_token_id in range(1+len(sentence)-len(keyword_tokens)):
      match = False
      # iterate through each token in the keyword and check
      for kw_token_id in range(len(keyword_tokens)):
        if sentence[s_token_id+kw_token_id] != keyword_tokens[kw_token_id]:
          # break if there is a mismatch
          break
        if kw_token_id == len(keyword_tokens)-1:
          # if there was no mismatch until the last token
          match = True
          one_found = True
      if match:
        for kw_token_id in range(len(keyword_tokens)):
          if kw_token_id == 0:
            sent_labels[s_token_id+kw_token_id] = LB_BEGINNING
          elif kw_token_id == len(keyword_tokens)-1:
            sent_labels[s_token_id+kw_token_id] = LB_LAST
          else:
            sent_labels[s_token_id+kw_token_id] = LB_INSIDE
    return one_found
    """
  
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
  
  def oneHot(self, token, ordered_dic):
    # move outside
    #dic_ls = list(dictionary.keys())
    #ordered_dic = {dic_ls[i]:i for i in range(len(dic_ls))}
    #ordered_dic[UNKNOWN_TOKEN] = len(ordered_dic)
    
    if token not in ordered_dic:
      token = UNKNOWN_TOKEN
    x = np.zeros(len(ordered_dic))
    x[ordered_dic[token]]
    return x
    
  
