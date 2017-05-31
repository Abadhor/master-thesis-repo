import io
import csv
import os
import nltk
import nltk.tokenize.punkt as punkt
import re
import xml.etree.ElementTree as ET

DATA_FOLDER = "D:\\Uni\\MasterThesis\\Data\\SemEval2017_AutomaticKeyphraseExtraction\\scienceie2017_train\\train2\\"
OUT_FOLDER = "D:\\Uni\\MasterThesis\\Data\\SemEval2017_AutomaticKeyphraseExtraction\\scienceie2017_train\\train2_modified\\"
OUT_FOLDER_KWDS = "D:\\Uni\\MasterThesis\\Data\\SemEval2017_AutomaticKeyphraseExtraction\\scienceie2017_train\\train2_keywords\\"

#TODO: split all keywords into tokens and create dictionary chains out of all tokens 
def readKeywords(filePath, keywords):
  infile = io.open(filePath, 'r', encoding='utf8')
  reader = csv.reader(infile, delimiter='\t')
  for row in reader:
    if (row[0] == '*') or (row[0][0] == 'R'):
      continue
    if row[2] not in keywords:
      keywords[row[2]] = []
    keywords[row[2]].append({'ID':row[0], 'Info':row[1]})
  infile.close()
  

def writeKeywordList(keywords, outFile):
  for kwd in sorted(list(keywords.keys())):
    outFile.write(kwd + '\n')

def getMultiWordLengths(list):
  lengths = dict()
  for entry in list:
    tokens = entry.split(" ")
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
    tokens = entry.split(" ")
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


def annotateSentence(tokens, kw_dict):
  cur_dict = kw_dict
  ann_tokens = []
  ann_token_buffer = []
  for i in range(0, len(tokens)):
    if tokens[i] in cur_dict:
      if len(ann_token_buffer) == 0: #beginning of potential mwu
        ann_token_buffer.append('B')
      else:
        ann_token_buffer.append('I') #inside potential mwu
      cur_dict = cur_dict[tokens[i]] #get dict for next token
    else:
      if len(ann_token_buffer) == 0: #current word outside
        ann_tokens.append('O')
      else:
        if '!end' in cur_dict: #current n-gram is mwu
          ann_tokens.extend(ann_token_buffer)
          ann_token_buffer = []
          cur_dict = kw_dict #go back to beginning of dict
          if tokens[i] in cur_dict: #check cur token again
            ann_token_buffer.append('B')
          else:
            ann_tokens.append('O')
        else: #current n-gram is not mwu
          for entry in ann_token_buffer:
            ann_tokens.append('O')
          ann_token_buffer = []
          cur_dict = kw_dict #go back to beginning of dict
          if tokens[i] in cur_dict: #check cur token again
            ann_token_buffer.append('B')
          else:
            ann_tokens.append('O')
    #print(tokens[i]+ str(ann_token_buffer))
  #finalize end of sentence
  if '!end' in cur_dict: #current n-gram is mwu
    ann_tokens.extend(ann_token_buffer)
    ann_token_buffer = []
    cur_dict = kw_dict #go back to beginning of dict
    #no need to check again
    #if tokens[i] in cur_dict: 
    #  ann_token_buffer.append('B')
    #else:
    #  ann_tokens.append('O')
  else: #current n-gram is not mwu
    for entry in ann_token_buffer:
      ann_tokens.append('O')
    ann_token_buffer = []
    cur_dict = kw_dict #go back to beginning of dict
  return ann_tokens


def annotateFile(filePath, kw_dict):
  infile = io.open(filePath, 'r', encoding='utf8')
  outfile = io.open(filePath[:len(filePath)-5]+'ann', 'w', encoding='utf8')
  for line in infile:
    ann_tokens = annotateSentence(line.split(" "), kw_dict)
    outfile.write(line)
    outfile.write(" ".join(ann_tokens) + "\n")
    #print(str(list(zip(line.split(" "), ann_tokens))))
  infile.close()
  outfile.close()

keywords = {}

fileList = os.listdir(DATA_FOLDER)
for fname in fileList:
  if fname[len(fname)-3:] == 'ann':
    readKeywords(DATA_FOLDER + fname, keywords)



with io.open(OUT_FOLDER_KWDS + 'keywords', 'w', encoding='utf8') as outfile:
  writeKeywordList(keywords, outfile)


kw_dict = getMultiWordDict(list(keywords.keys()))

fileList = os.listdir(OUT_FOLDER)
for fname in fileList:
  if fname[len(fname)-5:] == 'sents':
    annotateFile(OUT_FOLDER + fname, kw_dict)




