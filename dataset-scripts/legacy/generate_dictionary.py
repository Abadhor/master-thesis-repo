import io
import os
import nltk
import math
from operator import itemgetter
#from nltk.tokenize import TweetTokenizer
from collections import Counter

DATA_FOLDER = "D:\\Uni\\MasterThesis\\Data\\SemEval2017_AutomaticKeyphraseExtraction\\scienceie2017_train\\train2_modified\\"
OUT_FOLDER_KWDS = "D:\\Uni\\MasterThesis\\Data\\SemEval2017_AutomaticKeyphraseExtraction\\scienceie2017_train\\train2_keywords\\"

#TweetTokenizer to handle non-alphabetic characters better
#tknzr = TweetTokenizer()

def writeCountedList(counted, outFile):
  for c in sorted(list(counted.keys())):
    outFile.write(c + '\t' + str(counted[c]) + '\n')

def addToDictionary(filePath, dictionary):
  infile = io.open(filePath, 'r', encoding='utf8')
  for line in infile:
    #line_tokens = tknzr.tokenize(line)
    line_tokens = nltk.word_tokenize(line)
    if len(line_tokens) == 0:
      continue
    for token in line_tokens:
      if token not in dictionary:
        dictionary[token] = 1
      else:
        dictionary[token] += 1
  infile.close()

dictionary = {}

fileList = os.listdir(DATA_FOLDER)
for fname in fileList:
  if fname[len(fname)-5:] == 'sents':
    addToDictionary(DATA_FOLDER + fname, dictionary)

with io.open(OUT_FOLDER_KWDS + 'dictionary', 'w', encoding='utf8') as outfile:
  writeCountedList(dictionary, outfile)