import math
import nltk
from operator import attrgetter

class Candidate:
  
  def __init__(self):
    self.name = None
    self.list = None
    self.length = None
    self.freq = None
    self.totalFrequency = None
    self.nested_count = None
    self.nested_freq = None

class Statistics:
  
  def __init__(self):
    self.precision = None
    self.recall = None
    self.F1 = None

class Baseline:
  
  def __init__(self, dataset):
    self.ds_keywords, self.ds_text_files = dataset.getDataset()
  
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
  
  
  def extractNounPhrases(self, debug=False):
    """Create a list of candidates from the dataset by extracting the longest NPs."""
    grammar_improved = r"""
        NP: {((<ADJ|NOUN>+<ADP>)+<ADJ|NOUN>*)<NOUN>}
        NP: {<ADJ|NOUN>+<NOUN>}
    """
    candidateDict = {}
    for idx, fname in enumerate(self.ds_text_files.keys()):
      if debug:
        print("File:", idx+1, '/', len(self.ds_text_files.keys()), end='\r')
      file = self.ds_text_files[fname]
      for line in file:
        line_tokens = nltk.word_tokenize(line)
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
    candidates = []
    for item in candidateDict.items():
      c = Candidate()
      c.name = item[0]
      c.list = item[1][0]
      c.length = len(item[1][0])
      c.freq = item[1][1]
      candidates.append(c)
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
  
  def filterTotalFrequency(self, candidates, MIN_OCCURRENCES=3):
    #only use candidates that occur at least MIN_OCCURRENCES times
    return [x for x in candidates if not x.freq < MIN_OCCURRENCES]
  
  def calculateTotalFrequencies(self, candidates, debug=False):
    """Count total frequencies of candidates, including when a candidate appears as part of another candidate"""
    candidate_counts = {}
    #process only longer candidates, then break
    for idx, candidate in enumerate(candidates):
      if debug:
        print(idx, '/', len(candidates), end='\r')
      freq = candidate.freq
      #for previous in candidateDict_items:
      for previous in candidates:
        if previous.length <= candidate.length:
          break
        if candidate.name in previous.name:
          freq += previous.freq
      candidate_counts[candidate.name] = freq
    
    #Set total frequencies
    for candidate in candidates:
      candidate.totalFrequency = candidate_counts[candidate.name]
  
  def isInList(self, s_list, biggerList):
    for i in range(0, len(biggerList)):
      if s_list == biggerList[i:i+len(s_list)]:
        return True
    return False
  
  def calculateNestedFrequencies(self, candidates, debug=False):
    """Requires total frequencies"""
    for idx, candidate in enumerate(candidates):
      if debug:
        print(idx, '/', len(candidates), end='\r')
      candidate.nested_count = 0
      candidate.nested_freq = 0
      for previous in candidates:
        if previous.length <= candidate.length:
          break
        if self.isInList(candidate.list, previous.list):
          candidate.nested_count += 1
          candidate.nested_freq += previous.totalFrequency - previous.nested_freq
  
  def calculateCVals(self, candidates, debug=False):
    """Return a list of C-values for each candidate. Requires Nested Frequencies"""
    c_vals = {}
    for candidate in candidates:
      if candidate.nested_count == 0:
        c_vals[candidate.name] = math.log(candidate.length,2)*candidate.totalFrequency
      else:
        c_vals[candidate.name] = math.log(candidate.length,2)*(candidate.totalFrequency - 1/candidate.nested_count * candidate.nested_freq)
    return c_vals
    
    
    
    