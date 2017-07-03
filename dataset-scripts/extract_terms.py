import io
import os
import nltk
import math
from operator import itemgetter
from nltk.tokenize import TweetTokenizer
from collections import Counter

DATA_FOLDER = "D:\\Uni\\MasterThesis\\Data\\SemEval2017_AutomaticKeyphraseExtraction\\scienceie2017_train\\train2_modified\\"
OUT_FOLDER_KWDS = "D:\\Uni\\MasterThesis\\Data\\SemEval2017_AutomaticKeyphraseExtraction\\scienceie2017_train\\train2_keywords\\"
MIN_OCCURRENCES = 3
MAX_LENGTH = 6
MIN_AVERAGE_CHARS_PER_WORD = 2

number_of_tokens = 0
number_of_files = 0

tknzr = TweetTokenizer()
#grammar from the c-value paper
grammar_original = r"""
    NP: {(<ADJ|NOUN>+|(<ADJ|NOUN>*<ADP>?)<ADJ|NOUN>*)<NOUN>}
"""
#this will always extract the longest NP that contains a preposition first
#then it extracts all NP without preposition
grammar_improved = r"""
    NP: {((<ADJ|NOUN>+<ADP>)+<ADJ|NOUN>*)<NOUN>}
    NP: {<ADJ|NOUN>+<NOUN>}
"""

def writeCountedList(counted_keywords, outFile):
  for kwd in sorted(list(counted_keywords.keys())):
    outFile.write(kwd + '\t' + str(counted_keywords[kwd]) + '\n')

def writeDictDesc(numbers_dict, outFile):
  for tuple in sorted(list(numbers_dict.items()), key=itemgetter(1), reverse=True):
    outFile.write(tuple[0] + '\t' + str(tuple[1]) + '\n')

def writeTupleList(list, outFile):
  for tuple in list:
    strs = []
    for item in tuple:
      strs.append(str(item))
    outFile.write("\t".join(strs)+'\n')

def tagFile(filePath, candidateDict):
  global number_of_files
  global number_of_tokens
  infile = io.open(filePath, 'r', encoding='utf8')
  number_of_files += 1
  for line in infile:
    line_tokens = tknzr.tokenize(line)
    number_of_tokens += len(line_tokens)
    print("Tokens: ",number_of_tokens)
    print("File: ",number_of_files, end='\033[1A\r')
    if len(line_tokens) == 0:
      continue
    sent = nltk.pos_tag(line_tokens, tagset='universal')
    cp = nltk.RegexpParser(grammar_improved)
    tree = cp.parse(sent)
    for node in tree:
      if type(node) is nltk.tree.Tree:
        term_candidate = []
        for tuple in list(node):
          term_candidate.append(tuple[0])
        candidateString = " ".join(term_candidate)
        if candidateString in candidateDict:
          candidateDict[candidateString] = (candidateDict[candidateString][0],candidateDict[candidateString][1]+1)
        else:
          candidateDict[candidateString] = (term_candidate, 1)
  infile.close()

def isInList(s_list, biggerList):
  for i in range(0, len(biggerList)):
    if s_list == biggerList[i:i+len(s_list)]:
      return True
  return False

"""
def getSublists(base_list):
  retList = []
  for size in range(2,len(base_list)):
    retList.extend([sublist for sublist in (base_list[x:x+size] for x in range(0,len(base_list)-size+1))])
  return retList
"""

#dict of term candidate token lengths
candidateDict = dict()

print("Tagging...")
fileList = os.listdir(DATA_FOLDER)
for fname in fileList:
  if fname[len(fname)-5:] == 'sents':
    tagFile(DATA_FOLDER + fname, candidateDict)

print("File: ",number_of_files)
print("Tokens: ",number_of_tokens)

#List each candidate by number of tokens, longest first
#Important so other loops can stop when length is too small
#(0=string, 1=tokenList, 2=length, 3=frequencies)
candidateDict_items = [{'name':x[0],'list':x[1][0], 'length':len(x[1][0]), 'freq':x[1][1]} for x in candidateDict.items()]
candidateDict_items = sorted(candidateDict_items, key=itemgetter('name'))
candidateDict_items = sorted(candidateDict_items, key=itemgetter('length'), reverse=True)

#with io.open(OUT_FOLDER_KWDS + 'candidate_lengths', 'w', encoding='utf8') as outfile:
#  writeTupleList(candidateDict_items, outfile)

print("Number of Candidates: ", len(candidateDict_items))
print("Filtering Formulas...")


#filter out formulas wrongly tagged as candidates
candidateDict_items = [candidate for candidate in candidateDict_items if sum([len(x) for x in candidate['list']])/candidate['length'] >= MIN_AVERAGE_CHARS_PER_WORD]


print("Filtering by length...")
#only use candidates that do not exceed MAX_LENGTH
candidateDict_items_filtered = [x for x in candidateDict_items if not x['length'] > MAX_LENGTH]

print("Counting Total Frequencies...")
candidate_counts = {}
#Count total frequencies of candidates
#process only longer candidates, then break
for idx, candidate in enumerate(candidateDict_items_filtered):
  print(idx, '/', len(candidateDict_items_filtered), end='\r')
  freq = candidate['freq']
  for previous in candidateDict_items:
    if previous['length'] <= candidate['length']:
      break
    if isInList(candidate['list'], previous['list']):
      freq += previous['freq']
  candidate_counts[candidate['name']] = freq

#Update all frequencies
for candidate in candidateDict_items_filtered:
  candidate['freq'] = candidate_counts[candidate['name']]


#only use candidates that occur at least MIN_OCCURRENCES times
candidates = [x for x in candidateDict_items_filtered if not x['freq'] < MIN_OCCURRENCES]

print("Number of Candidates: ", len(candidates))


#Calculate candidate nested frequencies
print("Calculating candidate nested frequencies...")
for idx, candidate in enumerate(candidates):
  print(idx, '/', len(candidates), end='\r')
  candidate['nested_count'] = 0
  candidate['nested_freq'] = 0
  for previous in candidates:
    if previous['length'] <= candidate['length']:
      break
    if isInList(candidate['list'], previous['list']):
      candidate['nested_count'] += 1
      candidate['nested_freq'] += previous['freq'] - previous['nested_freq']

c_vals = {}

print("Calculating C-Values...")
for candidate in candidates:
  if candidate['nested_count'] == 0:
    c_vals[candidate['name']] = math.log(candidate['length'],2)*candidate['freq']
  else:
    c_vals[candidate['name']] = math.log(candidate['length'],2)*(candidate['freq'] - 1/candidate['nested_count'] * candidate['nested_freq'])

with io.open(OUT_FOLDER_KWDS + 'candidates', 'w', encoding='utf8') as outfile:
  writeTupleList(candidates, outfile)

with io.open(OUT_FOLDER_KWDS + 'candidate_c_vals', 'w', encoding='utf8') as outfile:
  writeDictDesc(c_vals, outfile)




