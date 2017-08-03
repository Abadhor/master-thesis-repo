import io
import csv
import os
import nltk
import nltk.tokenize.punkt as punkt
import re
import xml.etree.ElementTree as ET
from nltk.tokenize import TweetTokenizer

DATA_FOLDER = "D:\\Uni\\MasterThesis\\Data\\SemEval2017_AutomaticKeyphraseExtraction\\scienceie2017_train\\train2\\"
OUT_FOLDER = "D:\\Uni\\MasterThesis\\Data\\SemEval2017_AutomaticKeyphraseExtraction\\scienceie2017_train\\train2_modified\\"
OUT_FOLDER_KWDS = "D:\\Uni\\MasterThesis\\Data\\SemEval2017_AutomaticKeyphraseExtraction\\scienceie2017_train\\train2_keywords\\"

tknzr = TweetTokenizer()

def readKeywords(filePath, keywords):
  infile = io.open(filePath, 'r', encoding='utf8')
  reader = csv.reader(infile, delimiter='\t')
  for row in reader:
    if (row[0] == '*') or (row[0][0] == 'R'):
      continue
    kw = " ".join(tknzr.tokenize(row[2]))
    if kw not in keywords:
      keywords[kw] = []
    keywords[kw].append({'ID':row[0], 'Info':row[1]})
  infile.close()
  

def writeKeywordList(keywords, outFile):
  for kwd in sorted(list(keywords.keys())):
    outFile.write(kwd + '\n')

def writeCountedList(counted_keywords, outFile):
  for kwd in sorted(list(counted_keywords.keys())):
    outFile.write(kwd + '\t' + str(counted_keywords[kwd]) + '\n')

def filterNGramLength(keywords, min_ngram_length):
  kw_list = list(keywords.keys())
  ret_dict = {}
  for kw in kw_list:
    tokens = tknzr.tokenize(kw)
    if len(tokens) >= min_ngram_length:
      ret_dict[kw] = keywords[kw]
  return ret_dict


def countOccurrences(keywords, corpus):
  kw_list = list(keywords.keys())
  count_dict = {}
  for kw in kw_list:
    count_dict[kw] = corpus.count(kw)
  return count_dict


def filterOccurrences(count_dict, min_occurrences):
  kw_list = list(count_dict.keys())
  ret_dict = {}
  for kw in kw_list:
    if count_dict[kw] >= min_occurrences:
      ret_dict[kw] = count_dict[kw]
  return ret_dict


def getMultiWordLengths(list):
  lengths = dict()
  for entry in list:
    tokens = tknzr.tokenize(entry)
    num_tokens = len(tokens)
    if num_tokens in lengths:
      lengths[num_tokens] += 1
    else:
      lengths[num_tokens] = 1
  return lengths

#works with dict from getMultiWordLengths()
def getMultiWordDistribution(mw_lengths):
  max_length = max(mw_lengths.keys())
  for i in range(1,max_length+1):
    if i in mw_lengths:
      print(str(i)+": "+str(mw_lengths[i]))
    else:
      print(str(i)+": 0")


def getMultiWordDict(mw_list):
  """
  Convert list of keywords into token tree
  Every key token leads to a dict of next word key tokens
  """
  mw_dict = dict()
  for entry in mw_list:
    tokens = tknzr.tokenize(entry)
    if len(tokens) <= 1:
      continue
    cur = mw_dict
    for token in tokens:
      if token in cur:
        cur = cur[token]
      else:
        cur[token] = dict()
        cur = cur[token]
    cur['!end'] = None
  return mw_dict


def checkCurrentToken(tokens, i, cur_dict, ann_token_buffer, cur_delta):
  if i < len(tokens) and tokens[i] in cur_dict: #token found
    if cur_delta == 1: #first token
      ann_token_buffer.append('B')
    else: #later token
      ann_token_buffer.append('I')
    if '.' in cur_dict[tokens[i]]:
      print(tokens, cur_dict)
    return checkCurrentToken(tokens, i+1, cur_dict[tokens[i]], ann_token_buffer, cur_delta + 1)
  else: #token not found
    if '!end' in cur_dict: #valid
      if cur_delta == 1:
        raise ValueError('Invalid !end token in kw_dict')
      return ann_token_buffer, cur_delta-1
    else: #invalid
      return ['O'], 1

def annotateSentence(tokens, kw_dict):
  i = 0
  ann_tokens = []
  while i < len(tokens):
    ann_token_buffer, delta = checkCurrentToken(tokens, i, kw_dict, [], 1)
    #print(tokens[i:i+delta], ann_token_buffer)
    ann_tokens.extend(ann_token_buffer)
    i += delta
  return ann_tokens


def annotateFile(filePath, kw_dict):
  infile = io.open(filePath, 'r', encoding='utf8')
  outfile = io.open(filePath[:len(filePath)-5]+'ann', 'w', encoding='utf8')
  for line in infile:
    line_tokens = tknzr.tokenize(line)
    ann_tokens = annotateSentence(line_tokens, kw_dict)
    outfile.write(" ".join(line_tokens) + "\n")
    outfile.write(" ".join(ann_tokens) + "\n")
    #print(str(list(zip(line_tokens, ann_tokens))))
  infile.close()
  outfile.close()

keywords = {}

fileList = os.listdir(DATA_FOLDER)
for fname in fileList:
  if fname[len(fname)-3:] == 'ann':
    readKeywords(DATA_FOLDER + fname, keywords)



with io.open(OUT_FOLDER_KWDS + 'keywords', 'w', encoding='utf8') as outfile:
  writeKeywordList(keywords, outfile)


fileList = os.listdir(OUT_FOLDER)
#keyword counts
corpus = ""
for fname in fileList:
  if fname[len(fname)-5:] == 'sents':
    infile = io.open(OUT_FOLDER + fname, 'r', encoding='utf8')
    for line in infile:
      corpus += line

mwu_kws = filterNGramLength(keywords, 2)
count_dict = countOccurrences(mwu_kws, corpus)
filtered_kws = filterOccurrences(count_dict, 3)

with io.open(OUT_FOLDER_KWDS + 'keywords_filtered', 'w', encoding='utf8') as outfile:
  writeCountedList(filtered_kws, outfile)

kw_dict = getMultiWordDict(list(filtered_kws.keys()))


#annotation
for fname in fileList:
  if fname[len(fname)-5:] == 'sents':
    annotateFile(OUT_FOLDER + fname, kw_dict)




