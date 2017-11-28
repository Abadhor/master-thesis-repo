
from TextDataset import TextDataset

class KeywordDataset(TextDataset):
  
  def __init__(self):
    super().__init__()
    self.keywords = None
    self.keywords_debug = None
  
  def getDataset(self):
    if self.corpus == None:
      return self.keywords, self.text_files
    else:
      #return self.keywords, self.corpus
      return self.keywords, self.text_files