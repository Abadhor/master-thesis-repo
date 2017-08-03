import io
import os
import nltk
import math
from operator import itemgetter
from nltk.tokenize import TweetTokenizer
from collections import Counter

DATA_FOLDER = "D:\\Uni\\MasterThesis\\Data\\SemEval2017_AutomaticKeyphraseExtraction\\scienceie2017_train\\train2_modified\\"
OUT_FOLDER_KWDS = "D:\\Uni\\MasterThesis\\Data\\SemEval2017_AutomaticKeyphraseExtraction\\scienceie2017_train\\train2_keywords\\"

def writeCountedList(counted, outFile):
  for c in sorted(list(counted.keys())):
    outFile.write(c + '\t' + str(counted[c]) + '\n')

class Word:
  
  def __init__(self):
    self.f = 1
    self.r = {}
    self.l = {}
  
  def calculateEntropies(self):
    r_sum = sum(self.r.values())
    l_sum = sum(self.l.values())
    self.entropy_r = 0
    self.entropy_l = 0
    for key in self.r.keys():
      p = self.r[key] / r_sum
      self.entropy_r -= p * math.log(p,2)
    for key in self.l.keys():
      p = self.l[key] / l_sum
      self.entropy_l -= p * math.log(p,2)


class Perplexity:
  
  def __init__(self):
    self.dictionary = {}
  
  def loadFile(self, filePath):
    """update frequencies in dictonary"""
    #TweetTokenizer to handle non-alphabetic characters better
    tknzr = TweetTokenizer()
    infile = io.open(filePath, 'r', encoding='utf8')
    for line in infile:
      line_tokens = tknzr.tokenize(line)
      if len(line_tokens) == 0:
        continue
      for idx,token in enumerate(line_tokens):
        if token not in self.dictionary:
          self.dictionary[token] = Word()
        else:
          self.dictionary[token].f += 1
        if idx > 0:
          l = line_tokens[idx-1]
          if l not in self.dictionary[token].l:
            self.dictionary[token].l[l] = 1
          else:
            self.dictionary[token].l[l] += 1
        if idx < len(line_tokens)-1:
          r = line_tokens[idx+1]
          if r not in self.dictionary[token].r:
            self.dictionary[token].r[r] = 1
          else:
            self.dictionary[token].r[r] += 1
    infile.close()
  
  def calculateEntropies(self):
    keys = self.dictionary.keys()
    for idx, word in enumerate(keys):
      print(idx, '/', len(keys), end='\r')
      self.dictionary[word].calculateEntropies()
  
  def calculateLogPerplexity(self, tokens):
    """calculate the log_2 perplexity of a multi word unit"""
    sum = 0
    for token in tokens:
      if token not in self.dictionary:
        continue
      w = self.dictionary[token]
      sum += w.entropy_r + w.entropy_l
    log_pp = sum / (2*len(tokens))
    return log_pp
  
  def loadFolder(self, folderName, extension='sents'):
    fileList = os.listdir(folderName)
    for fname in fileList:
      if fname[len(fname)-len(extension):] == extension:
        self.loadFile(folderName + fname)
    self.calculateEntropies()

