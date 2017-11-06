

class KeywordDataset:
  
  def __init__(self):
    self.keywords = None
    self.keywords_debug = None
    self.text_files = None
    self.corpus = None
    self.dictionary = None
  
  def getDataset(self):
    if self.corpus == None:
      return self.keywords, self.text_files
    else:
      #return self.keywords, self.corpus
      return self.keywords, self.text_files
  
  def getCorpus(self):
    if not self.corpus:
      corpus = []
      for sentences in self.text_files.values():
        corpus.extend(sentences)
      self.corpus = corpus
    return self.corpus
  
  def tokenize(self):
    raise NotImplementedError("Please Implement")
  
  def getDictionary(self):
    if not self.dictionary:
      self.tokenize()
    return self.dictionary