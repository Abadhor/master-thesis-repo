import spacy
import string
import re
from nltk.tokenize import RegexpTokenizer

class Preprocessing:
  
  def __init__(self, spacynlp=None):
    if spacynlp == None:
      self.nlp = spacy.load('en_core_web_md')
    else:
      self.nlp = spacynlp
    self.alphabet = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ §'
    
    # char replacements
    self.RE_UNKNOWN = re.compile('[^'+re.escape(self.alphabet)+']')
    self.RE_UNKNOWN_SUB = ' § '
    self.RE_NUMBERS = re.compile(r'[-]?[\d]+[^a-zA-Z_ ]*')
    self.RE_NUMBERS_SUB = ' $ '
    self.RE_BRACKETS_B = re.compile('['+re.escape('({[')+']')
    self.RE_BRACKETS_B_SUB = ' ( '
    self.RE_BRACKETS_L = re.compile('['+re.escape(']})')+']')
    self.RE_BRACKETS_L_SUB = ' ) '
    self.RE_NEWLINE = re.compile('[\n]')
    self.RE_NEWLINE_SUB = ' '
    self.RE_HYPHEN = re.compile('['+re.escape('-')+']')
    self.RE_HYPHEN_SUB = ' '
  
  def cleanseTokenize(self, inputtext):
    text=inputtext.lower()
    cleanr =re.compile('<.*?>')
    text=re.sub(cleanr,'', text)
    text=re.sub('\d+', ' ', text)
    tokenizer = RegexpTokenizer(r'\w+')
    processed_tokens = tokenizer.tokenize(text)
    return processed_tokens

  def substituteUnknown(self, inputtext):
    return re.sub(self.RE_UNKNOWN, self.RE_UNKNOWN_SUB, inputtext)

  def substituteNumbers(self, inputtext):
    return re.sub(self.RE_NUMBERS, self.RE_NUMBERS_SUB, inputtext)

  def substituteBrackets(self, inputtext):
    text = re.sub(self.RE_BRACKETS_B, self.RE_BRACKETS_B_SUB, inputtext)
    return re.sub(self.RE_BRACKETS_L, self.RE_BRACKETS_L_SUB, text)
  
  def substituteNewline(self, inputtext):
    return re.sub(self.RE_NEWLINE, self.RE_NEWLINE_SUB, inputtext)
  
  def substituteHyphen(self, inputtext):
    text = inputtext.replace(chr(8211), '-')
    return re.sub(self.RE_HYPHEN, self.RE_HYPHEN_SUB, text)
  
  def substituteTokenize(self, inputtext):
    text = self.substituteNewline(inputtext)
    text = self.substituteUnknown(text)
    text = self.substituteBrackets(text)
    text = self.substituteNumbers(text)
    doc = self.nlp(text)
    tokens = [x.lemma_ if x.tag_ in ["NN","NNS"] else x.text for x in doc]
    tokens = [self.substituteHyphen(x) for x in tokens]
    tokens = [x.strip() for x in tokens if len(x.strip()) > 0]
    return tokens