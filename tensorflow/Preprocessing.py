import spacy
import string
import re
import numpy as np
from MWUHashTree import MWUHashTree

class Tokenizer:
  """
    Various functions for general text preprocessing and tokenization.
  """
  
  def __init__(self, spacynlp=None):
    if spacynlp == None:
      self.nlp = spacy.load('en_core_web_md')
      print("Spacy model 'en_core_web_md' loaded")
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
    # TODO: save hyphen locations in extra feature
  
  def splitSentences(self, inputtext):
    doc = self.nlp(inputtext)
    return [sent.text for sent in doc.sents]
  
  def substituteTokenize(self, inputtext, cleanseHyphens=True):
    text = self.substituteNewline(inputtext)
    text = self.substituteUnknown(text)
    text = self.substituteBrackets(text)
    text = self.substituteNumbers(text)
    text = text.lower()
    doc = self.nlp(text)
    # lemma to handle ending s
    # ending s is either noun plural or 3rd person singular verb
    # only take nouns
    tokens = [x.lemma_ if x.tag_ in ["NN","NNS"] else x.text for x in doc]
    if cleanseHyphens:
      tokens = [self.substituteHyphen(x) for x in tokens]
    tokens = [x.strip() for x in tokens if len(x.strip()) > 0]
    return tokens

class MWUAnnotator:
  """
    Creates labels by annotating all MWUs in a text in BILOU format.
  """
  LB_BEGINNING = 'B'
  LB_INSIDE = 'I'
  LB_LAST = 'L'
  LB_OUTSIDE = 'O'
  LB_UNIGRAM = 'U'
  LABEL_NAMES = [LB_BEGINNING, 
                 LB_INSIDE, 
                 LB_LAST, 
                 LB_OUTSIDE, 
                 LB_UNIGRAM]
  label_dictionary = {label: idx for idx, label in enumerate(LABEL_NAMES)}
  
  # Pass a list of tokenized MWUs 
  def __init__(self, mwu_tokens_list):
    self.MWUs = MWUHashTree()
    for idx, MWU in enumerate(mwu_tokens_list):
      self.MWUs[MWU] = idx
  
  def getLabels(self, tokens):
    labels = [self.LB_OUTSIDE for i in range(len(tokens))]
    MWU_token_count = 0
    for token_id in range(len(tokens)):
      if MWU_token_count > 0:
        if MWU_token_count == 1:
          labels[token_id] = self.LB_LAST
        else:
          labels[token_id] = self.LB_INSIDE
        MWU_token_count -= 1
      else:
        found_MWUs = self.MWUs.getAll(tokens[token_id:])
        if len(found_MWUs) > 0:
          longest = found_MWUs[len(found_MWUs)-1]
          MWU_token_count += len(longest)
          labels[token_id] = self.LB_BEGINNING
          MWU_token_count -= 1
    return labels
  
  def getLabelID(self, label):
    return self.label_dictionary[label]

class Vectorizer:
  """
    Translates words and other features to vectors based on dictionary and alphabet.
  """
  
  def __init__(self, annotator, dictionary, alphabet, gazetteers=None):
    self.annotator = annotator
    # Unknown word index = len(dictionary)
    # Padding word index = len(dictionary)+1
    self.inverseDictionary = [word for word in dictionary.keys()]
    self.dictionary = {word: idx for idx, word in enumerate(self.inverseDictionary)}
    # Padding char index = len(alphabet)
    self.inverseAlphabet = [ch for ch in alphabet.keys()]
    self.alphabet = {ch: idx for idx, ch in enumerate(self.inverseAlphabet)}
    if gazetteers != None:
      self.gazetteers = MWUHashTree()
      for idx,gaz in enumerate(gazetteers):
        self.gazetteers[gaz] = idx
    else:
      self.gazetteers = None
  
  def tokenVectorize(self, tokens, maxSentLenght):
    # Returns the sentence as a vector
    # Not-in-dictionary token ID = len(self.dictionary)
    # Padding token ID = len(self.dictionary) + 1
    token_vector = np.full((1,maxSentLenght), len(self.dictionary)+1, dtype='int32')
    for t_idx, token in enumerate(tokens):
      if t_idx >= maxSentLenght:
        break
      if token not in self.dictionary:
        token_vector[0,t_idx] = len(self.dictionary)
      else:
        token_vector[0,t_idx] = self.dictionary[token]
    return token_vector
  
  def labelVectorize(self, tokens, maxSentLenght):
    labels = self.annotator.getLabels(tokens)
    label_vector = np.full((1,maxSentLenght), len(self.annotator.LABEL_NAMES), dtype='int32')
    for l_idx, label in enumerate(labels):
      if l_idx >= maxSentLenght:
        break
      label_vector[0,l_idx] = self.annotator.getLabelID(label)
    return label_vector
  
  def characterVectorize(self, tokens, maxSentLenght, maxWordLength):
    # Returns the sentence as a list of char vectors
    # and the char vector lengths
    # Padding char ID = len(self.dictionary)
    char_vector = np.full((1,maxSentLenght,maxWordLength), len(self.alphabet), dtype='int32')
    lengths = np.zeros((1, maxSentLenght), dtype='int32')
    for t_idx, token in enumerate(tokens):
      if t_idx >= maxSentLenght:
        break
      for c_idx, c in enumerate(token):
        if c_idx >= maxWordLength:
          break
        char_vector[0, t_idx, c_idx] = self.alphabet[c]
      lengths[0, t_idx] = min(len(token), maxWordLength)
    return char_vector, lengths
  
  def gazetteerVectorize(self, tokens, maxSentLenght):
    gaz_vector = np.zeros((1,maxSentLenght,len(self.gazetteers)), dtype='int32')
    for token_id in range(len(tokens)):
      if token_id >= maxSentLenght:
        break
      found_gaz = self.gazetteers.getAll(tokens[token_id:])
      for gaz in found_gaz:
        gaz_id = self.gazetteers[gaz]
        for i in range(len(gaz)):
          gaz_data[0, token_id+i, gaz_id] = 1
    return gaz_vector
  
  def vectorize(self, tokens, maxSentLenght, maxWordLength):
    token_vector = self.tokenVectorize(tokens, maxSentLenght)
    label_vector = self.labelVectorize(tokens, maxSentLenght)
    char_vector, char_lengths = self.characterVectorize(tokens, maxSentLenght, maxWordLength)
    if self.gazetteers != None:
      gazetteer_vector = gazetteerVectorize(tokens, maxSentLenght)
      return token_vector, np.array((len(tokens),)), label_vector, char_vector, char_lengths, gazetteer_vector
    return token_vector, np.array((len(tokens),)), label_vector, char_vector, char_lengths
    
    
  
    