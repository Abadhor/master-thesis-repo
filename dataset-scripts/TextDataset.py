
import re

class TextDataset:
  
  def __init__(self):
    self.text_files = None
    self.corpus = None
    self.dictionary = None
    self.RE_OTHERS = re.compile(r'[^a-zA-Z0-9\. ,\/#!$%\^&\*;:{}=\-_`~()\+\[\]]+')
    self.RE_OTHERS_SUB = '§§§§'
    self.RE_NUMBERS = re.compile(r'^[-]?[\d]+[\W\d]*$')
    self.RE_NUMBERS_SUB = '$$$$'
    self.RE_DASH = re.compile(r'\s-\s')
    self.RE_DASH_SUB = ' -- '
    self.BRACKETS_B = set()
    self.BRACKETS_B.add('(')
    self.BRACKETS_B.add('[')
    self.BRACKETS_B.add('{')
    self.BRACKETS_B_SUB = '[[[['
    self.BRACKETS_L = set()
    self.BRACKETS_L.add(')')
    self.BRACKETS_L.add(']')
    self.BRACKETS_L.add('}')
    self.BRACKETS_L_SUB = ']]]]'
    self.UNKNOWN = '????'
  
  def getSpecialTokens(self):
    return ['.', ',', ';', self.RE_NUMBERS_SUB, self.RE_OTHERS_SUB, self.RE_DASH_SUB,
            self.BRACKETS_B_SUB, self.BRACKETS_L_SUB, self.UNKNOWN]
  
  def getCorpus(self):
    if not self.corpus:
      corpus = []
      for sentences in self.text_files.values():
        corpus.extend(sentences)
      self.corpus = corpus
    return self.corpus
  
  def removeApostrophe(self, text):
    text = text.replace('"', '')
    text = text.replace('“', '')
    text = text.replace('”', '')
    return text
  
  def preprocess(self, text, toLower=False, doubleDash=False):
    if toLower:
      text = text.lower()
    text = text.replace(chr(8211), '-')
    text = self.removeApostrophe(text)
    if doubleDash:
      text = self.RE_DASH.sub(self.RE_DASH_SUB, text)
    return text
  
  def tokenize(self):
    raise NotImplementedError("Please Implement")
  
  def replaceWithSpecial(self, tokens, numbers=True, brackets=True, others=True):
    # replace numbers with symbol
    if numbers:
      tokens = [self.RE_NUMBERS.sub(self.RE_NUMBERS_SUB, token) for token in tokens]
    if others:
      tokens = [self.RE_OTHERS.sub(self.RE_OTHERS_SUB, token) for token in tokens]
    # replace brackets
    if brackets:
      tokens = [ x if x not in self.BRACKETS_B else self.BRACKETS_B_SUB for x in tokens]
      tokens = [ x if x not in self.BRACKETS_L else self.BRACKETS_L_SUB for x in tokens]
    return tokens
  
  def postprocess(self, tokens, removeHyphens=False):
    # remove empty
    tmp = []
    for i in range(len(tokens)):
      stripped = tokens[i].strip()
      if len(stripped) != 0:
        tmp.append(tokens[i])
    tokens = tmp
    # combine: 'semi', '-', 'optimized' -> 'semi-optimized'
    if not removeHyphens:
      tmp = []
      minus = False
      prev = -1
      for i in range(len(tokens)):
        if not minus:
          if tokens[i] != '-':
            tmp.append(tokens[i])
            prev += 1
          else:
            if prev == -1:
              tmp.append(tokens[i])
              prev += 1
            else:
              tmp[prev] = tmp[prev] + tokens[i]
              minus = True
        else:
          tmp[prev] = tmp[prev] + tokens[i]
          minus = False
      tokens = tmp
    else:
      tmp = []
      for i in range(len(tokens)):
        if tokens[i] != '-':
          tmp.append(tokens[i])
      tokens = tmp
    tokens = self.replaceWithSpecial(tokens)
    return tokens
  
  def replaceUnknownTokens(self):
    corpus = self.getCorpus()
    dictionary = self.getDictionary()
    unknown_count = 0
    token_count = 0
    for idx_s, sent in enumerate(corpus):
      for idx_t, t in enumerate(sent):
        token_count += 1
        if t not in dictionary:
          #print("'",t,"'", end=' ')
          sent[idx_t] = self.UNKNOWN
          unknown_count += 1
    print("Unknown Tokens:")
    print("{} / {} ({:.2%})".format(unknown_count,token_count,unknown_count/token_count))
  
  def getDictionary(self):
    if not self.dictionary:
      self.tokenize()
    return self.dictionary
  
  def setDictionary(self, dictionary):
    if not self.dictionary:
      self.tokenize()
    self.dictionary = dictionary
    self.replaceUnknownTokens()
    
    
    