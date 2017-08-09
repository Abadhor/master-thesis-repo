import math
import nltk
import io
import numpy as np
from operator import attrgetter
from term_perplexity import Perplexity
from StatFeatures import StatFeatures

class Candidate:
  
  def __init__(self):
    self.name = None
    self.list = None
    self.length = None
    self.freq = None
    self.NPFrequency = None
    self.TF = None
    self.IDF = None
    self.nested_count = None
    self.nested_freq = None
    self.c_value = None
    self.log_pp = None
    self.t_score = None
    self.pmi = None
    self.dice = None

class Statistics:
  
  def __init__(self):
    self.precision = None
    self.recall = None
    self.F1 = None

class BaselineFeatures:
  
  def __init__(self, dataset, addKeywords=False):
    """
      addKeywords adds the keywords to the candidates
    """
    self.ds_keywords, self.ds_text_files = dataset.getDataset()
    self.dictionary = dataset.getDictionary()
    self.statFeatures = StatFeatures(self.dictionary)
    self.addKeywords = addKeywords
  
  def calculateStatistics(self, candidates):
    true_positive = 0
    for c in candidates:
      if c.name in self.ds_keywords:
        true_positive += 1
    positive = len(self.ds_keywords)
    detected_positive = len(candidates) #true + false positive
    s = Statistics()
    s.precision = true_positive / detected_positive
    s.recall = true_positive / positive
    s.F1 = 2 * s.precision * s.recall / (s.precision + s.recall)
    return s
  
  def normalizeFeatures(self, candidates, attrs):
    for candidate in candidates:
      for attr in attrs:
        setattr(candidate, attr, attrgetter(attr)(candidate)/len(self.ds_text_files))
  
  def extractNounPhrases(self, verbose=False):
    """Create a list of candidates from the dataset by extracting the longest NPs."""
    grammar_improved = r"""
        NP: {((<ADJ|NOUN>+<ADP>)+<ADJ|NOUN>*)<NOUN>}
        NP: {<ADJ|NOUN>+<NOUN>}
    """
    candidateDict = {}
    self.tokenized_files = {}
    for idx, fname in enumerate(self.ds_text_files.keys()):
      if verbose:
        print("File:", idx+1, '/', len(self.ds_text_files.keys()), end='\r')
      file = self.ds_text_files[fname]
      file_sents = []
      for line in file:
        line_tokens = nltk.word_tokenize(line)
        file_sents.append(" ".join(line_tokens))
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
      self.tokenized_files[fname] = file_sents
    #add keywords to candidates on demand
    if self.addKeywords:
      for kwd in self.ds_keywords:
        if kwd in candidateDict:
          continue
        term_candidate = nltk.word_tokenize(kwd)
        candidateDict[kwd] = (term_candidate, 1)
    #convert to candidate objects
    candidates = []
    for item in candidateDict.items():
      c = Candidate()
      c.name = item[0]
      c.list = item[1][0]
      c.length = len(item[1][0])
      c.freq = item[1][1]
      candidates.append(c)
    if verbose:
      print()
    return candidates
  
  def sortCandidates(self, candidates):
    """Sort candidates by length (number of tokens) and by name if same length"""
    ret = sorted(candidates, key=attrgetter('name'))
    ret = sorted(ret, key=attrgetter('length'), reverse=True)
    return ret
  
  def filterFormulas(self, candidates, MIN_AVERAGE_CHARS_PER_WORD=2):
    #filter out formulas wrongly tagged as candidates
    return [candidate for candidate in candidates if sum([len(x) for x in candidate.list])/candidate.length >= MIN_AVERAGE_CHARS_PER_WORD]
  
  def filterLength(self, candidates, MAX_LENGTH=6):
    #only use candidates that do not exceed MAX_LENGTH
    return [x for x in candidates if not x.length > MAX_LENGTH]
  
  def filterTF(self, candidates, MIN_OCCURRENCES=3):
    #only use candidates that occur at least MIN_OCCURRENCES times
    return [x for x in candidates if not x.freq < MIN_OCCURRENCES]
  
  def calculateTotalNPFrequencies(self, candidates, verbose=False):
    """Count total frequencies of candidate occurrences in the collected Noun Phrases"""
    #process only longer candidates, then break
    for idx, candidate in enumerate(candidates):
      if verbose:
        print(idx, '/', len(candidates), end='\r')
      candidate.NPFrequency = candidate.freq
      #for previous in candidateDict_items:
      for previous in candidates:
        if previous.length <= candidate.length:
          break
        if candidate.name in previous.name:
          candidate.NPFrequency += previous.freq
    if verbose:
      print()
  
  def calculateRawTF_IDF(self, candidates, verbose=False):
    """Count term frequencies and inverse document frequencies of candidates"""
    #process only longer candidates, then break
    for idx, candidate in enumerate(candidates):
      if verbose:
        print(idx, '/', len(candidates), end='\r')
      candidate.TF = 0
      DF = 0
      for fkey in self.tokenized_files.keys():
        indoc = False
        for sent in self.tokenized_files[fkey]:
          if candidate.name in sent:
            candidate.TF += 1
            indoc = True
        if indoc:
          DF += 1
      if DF == 0:
        DF = 1
        candidate.TF = 1
      candidate.IDF = len(self.tokenized_files)/DF
    if verbose:
      print()
  
  def isInList(self, s_list, biggerList):
    for i in range(0, len(biggerList)):
      if s_list == biggerList[i:i+len(s_list)]:
        return True
    return False
  
  def calculateNestedFrequencies(self, candidates, verbose=False):
    """Requires total frequencies"""
    for idx, candidate in enumerate(candidates):
      if verbose:
        print(idx, '/', len(candidates), end='\r')
      candidate.nested_count = 0
      candidate.nested_freq = 0
      for previous in candidates:
        if previous.length <= candidate.length:
          break
        if self.isInList(candidate.list, previous.list):
          candidate.nested_count += 1
          candidate.nested_freq += previous.NPFrequency - previous.nested_freq
    if verbose:
      print()
  
  def calculateCVals(self, candidates, verbose=False):
    """Calculate the C-value for each candidate. Requires Nested Frequencies"""
    for candidate in candidates:
      if candidate.nested_count == 0:
        candidate.c_value = math.log(candidate.length,2)*candidate.NPFrequency
      else:
        candidate.c_value = math.log(candidate.length,2)*(candidate.NPFrequency - 1/candidate.nested_count * candidate.nested_freq)
  
  def calculatePerplexity(self, candidates, verbose=False):
    """Calculate the perplexity for each candidate."""
    p = Perplexity(self.ds_text_files, verbose)
    for candidate in candidates:
      candidate.log_pp = p.calculateLogPerplexity(candidate.list)
  
  def calculateStatFeatures(self, candidates, verbose=False):
    """Calculate t-score, pmi and dice for each candidate."""
    for candidate in candidates:
      candidate.t_score = self.statFeatures.t_score(candidate.list, candidate.TF)
      candidate.pmi = self.statFeatures.pmi(candidate.list, candidate.TF)
      candidate.dice = self.statFeatures.dice(candidate.list, candidate.TF)
  
  def exportARFF(self, candidates, attrs, name, folder, fileName):
    """Export candidates as ARFF"""
    f = "{:.4}"
    with io.open(folder + fileName, 'w', encoding='utf8') as outfile:
      outfile.write("@RELATION\t\"" + name + "\"\n\n")
      for attr in attrs:
        outfile.write("@ATTRIBUTE\t" + attr + "\tNUMERIC\n")
      outfile.write("@ATTRIBUTE\tkeyphrase\t{yes,no}\n\n")
      outfile.write("@DATA\n")
      for candidate in candidates:
        for attr in attrs:
          outfile.write(f.format(attrgetter(attr)(candidate)*1.0) + ",")
        if candidate.name in self.ds_keywords:
          outfile.write("yes\n")
        else:
          outfile.write("no\n")
  
  def asNumpy(self, candidates, attrs):
    """Convert candidate features and keywords into numpy matrices, whereas each  candidate is represented by a row and each feature by a column."""
    X = []
    y = []
    for candidate in candidates:
      row = []
      for attr in attrs:
        row.append(attrgetter(attr)(candidate))
      X.append(row)
      if candidate.name in self.ds_keywords:
        y.append(1)
      else:
        y.append(0)
    X = np.array(X, dtype='float64')
    y = np.array(y, dtype='float64')
    return X, y
    
    