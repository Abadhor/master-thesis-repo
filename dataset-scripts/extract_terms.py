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

def writeTupleList(list, outFile):
  for tuple in list:
    strs = []
    for item in tuple:
      strs.append(str(item))
    outFile.write("\t".join(strs)+'\n')

def tagFile(filePath, candidateList, candidateLengths):
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
        candidateLengths[candidateString] = len(term_candidate)
        candidateList.append(candidateString)
  infile.close()

def calcCValue(candidate, nested_set, candidate_counts):
  tokens = tknzr.tokenize(candidate)
  if len(nested_set) == 0:
    return math.log(len(tokens),2)*candidate_counts[candidate]
  else:
    sum = 0
    for nested in nested_set:
      sum += candidate_counts[nested]
    nested_res = 1/len(nested_set) * sum
    return math.log(len(tokens),2)*(candidate_counts[candidate] - nested_res)

#list of term candidates
candidateList = []
#dict of term candidate token lengths
candidateLengths = dict()

print("Tagging...")
fileList = os.listdir(DATA_FOLDER)
for fname in fileList:
  if fname[len(fname)-5:] == 'sents':
    tagFile(DATA_FOLDER + fname, candidateList, candidateLengths)

print("File: ",number_of_files)
print("Tokens: ",number_of_tokens)

#Count each candidate
candidate_counts = Counter(candidateList)
#List each candidate by number of tokens, longest first
candidateLengths_items = candidateLengths.items()
candidateLengths_items = sorted(candidateLengths_items, key=itemgetter(0))
candidateLengths_items = sorted(candidateLengths_items, key=itemgetter(1), reverse=True)

#with io.open(OUT_FOLDER_KWDS + 'candidate_lengths', 'w', encoding='utf8') as outfile:
#  writeTupleList(candidateLengths_items, outfile)

print("Number of Candidates: ", len(candidateLengths_items))
print("Filtering...")

#only use candidates that don't exceed MAX_LENGTH
candidateLengths_items = [x for x in candidateLengths_items if not x[1] > MAX_LENGTH]

#only use candidates that occur at least MIN_OCCURRENCES times
candidates = [x + (candidate_counts[x[0]],) for x in candidateLengths_items if not candidate_counts[x[0]] < MIN_OCCURRENCES]

print("Number of Candidates: ", len(candidates))

#TODO Continue here with c value tuple --> find f(b) 
#--> look through whole corpus

candidate_sets = {}
c_vals = {}

print("Creating Sets and calculating...")
count = 0
for candidate in candidate_counts.keys():
  count += 1
  candidate_sets[candidate] = set()
  for other in candidate_counts.keys():
    if candidate == other:
      continue
    if candidate in other:
      candidate_sets[candidate].add(other)
      #candidate_counts[candidate] += candidate_counts[other]
  val = calcCValue(candidate, candidate_sets[candidate], candidate_counts)
  c_vals[candidate] = val
  print(count, "/", len(candidate_counts.keys()), end="\r")
  

with io.open(OUT_FOLDER_KWDS + 'candidate_counts', 'w', encoding='utf8') as outfile:
  writeCountedList(candidate_counts, outfile)

with io.open(OUT_FOLDER_KWDS + 'candidate_c_vals', 'w', encoding='utf8') as outfile:
  writeCountedList(c_vals, outfile)




