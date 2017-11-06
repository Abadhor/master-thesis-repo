import math
import nltk
import io
import numpy as np
from operator import attrgetter
from term_perplexity import Perplexity
from StatFeatures import StatFeatures
from nltk.stem import WordNetLemmatizer

class Candidate:
  
  def __init__(self):
    self.name = None
    self.list = None
    self.length = None
    self.syntax = None
    self.syntaxClass = None
    self.syntaxFrequency = None
    self.freq = None
    self.NPFrequency = None
    self.TF = None
    self.IDF = None
    self.DF = None
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
  
  def __init__(self, dataset, addKeywords=False, syntaxDict=None):
    """
      addKeywords adds the keywords to the candidates
    """
    self.syntaxDict = syntaxDict
    self.ds_keywords, self.ds_text_files = dataset.getDataset()
    self.dictionary = dataset.getDictionary()
    self.statFeatures = StatFeatures(self.dictionary)
    self.addKeywords = addKeywords
  
  def calculateStatistics(self, candidates, names_set):
    true_positive = 0
    for c in candidates:
      if c.list in names_set:
        true_positive += 1
    positive = len(names_set)
    detected_positive = len(candidates) #true + false positive
    s = Statistics()
    s.precision = true_positive / detected_positive
    s.recall = true_positive / positive
    s.F1 = 2 * s.precision * s.recall / (s.precision + s.recall)
    return s
  
  def calculateNestedRecall(self, candidates, names_set):
    true_positive = 0
    for c in candidates:
      tokens = c.list
      for idx in range(len(tokens)):
        ls = names_set.getAll(tokens[idx:])
        true_positive += len(ls)
    positive = len(names_set)
    return true_positive / positive
  
  def normalizeFeatures(self, candidates, attrs):
    """Replace selected attributes with themselves divided by the number of files in dataset"""
    for candidate in candidates:
      for attr in attrs:
        setattr(candidate, attr, attrgetter(attr)(candidate)/len(self.ds_text_files))
  
  def logFeatures(self, candidates, attrs, base=2):
    """Replace selected attributes with the log_base of themselves"""
    for candidate in candidates:
      for attr in attrs:
        setattr(candidate, attr, math.log(attrgetter(attr)(candidate),base))
  
  def getAllNGramms(self, line_tokens, min_length, max_length):
    nGramms = []
    for l in range(min_length,max_length+1):
      if l > len(line_tokens):
        break
      for i in range(0, len(line_tokens) - l + 1):
        nGramms.append(line_tokens[i:i+l])
    return nGramms
  
  def extractNGramms(self, min_length=2, max_length=2, verbose=False):
    """Create a list of n-gramms that may be used as term candidates from the dataset."""
    if min_length > max_length:
      max_length = min_length
    wn = WordNetLemmatizer()
    #tokenized and then joined by spaces
    self.tokenized_files = {}
    #dict of n-gramms
    NGrammDict = {}
    for idx, fname in enumerate(self.ds_text_files.keys()):
      if verbose:
        print("File:", idx+1, '/', len(self.ds_text_files.keys()), end='\r')
      file = self.ds_text_files[fname]
      file_sents = []
      #has the 
      doc_ngramms = set()
      for line in file:
        line_tokens = nltk.word_tokenize(line)
        line_tokens = [ wn.lemmatize(x.strip()) for x in line_tokens]
        file_sents.append(" ".join(line_tokens))
        if len(line_tokens) == 0:
          continue
        line_NGramms = self.getAllNGramms(line_tokens, min_length, max_length)
        for ng in line_NGramms:
          ng_string = " ".join(ng)
          if ng_string not in NGrammDict:
            c = Candidate()
            c.name = ng_string
            c.list = ng
            c.length = len(ng)
            c.freq = 1
            c.TF = 1
            c.DF = 1
            NGrammDict[ng_string] = c
          else:
            NGrammDict[ng_string].freq += 1
            NGrammDict[ng_string].TF += 1
            if ng_string not in doc_ngramms:
              NGrammDict[ng_string].DF += 1
              doc_ngramms.add(ng_string)
      self.tokenized_files[fname] = file_sents
    if verbose:
      print()
    NGrammLS = list(NGrammDict.values())
    for c in NGrammLS:
      c.IDF = len(self.ds_text_files)/c.DF
    return NGrammLS
    
  
  def extractNounPhrases(self, verbose=False):
    """Create a list of noun phrases that may be used as term candidates from the dataset by extracting the longest NPs."""
    wn = WordNetLemmatizer()
    grammar_improved = r"""
        NP: {((<ADJ|NOUN>+<ADP>)+<ADJ|NOUN>*)<NOUN>}
        NP: {<ADJ|NOUN>+<NOUN>}
    """
    NPDict = {}
    self.tokenized_files = {}
    for idx, fname in enumerate(self.ds_text_files.keys()):
      if verbose:
        print("File:", idx+1, '/', len(self.ds_text_files.keys()), end='\r')
      file = self.ds_text_files[fname]
      file_sents = []
      for line in file:
        line_tokens = line
        #dont lemmatize line_tokens used for PoS tagging
        file_sents.append(" ".join([ wn.lemmatize(x.strip()) for x in line_tokens]))
        #file_sents.append(" ".join(line_tokens))
        if len(line_tokens) == 0:
          continue
        sent = nltk.pos_tag(line_tokens, tagset='universal')
        cp = nltk.RegexpParser(grammar_improved)
        tree = cp.parse(sent)
        for node in tree:
          if type(node) is nltk.tree.Tree:
            NP = []
            NP_POS = []
            for tuple in list(node):
              NP.append(tuple[0])
              NP_POS.append(tuple[1])
            #lemmatize each token
            NP = [ wn.lemmatize(x.strip()) for x in NP]
            NPString = " ".join(NP)
            if NPString in NPDict:
              NPDict[NPString] = (NPDict[NPString][0],NPDict[NPString][1],NPDict[NPString][2]+1)
            else:
              NPDict[NPString] = (NP," ".join(NP_POS), 1)
      self.tokenized_files[fname] = file_sents
    #add keywords to Noun Phrases on demand
    if self.addKeywords:
      for kwd in self.ds_keywords:
        if kwd in NPDict:
          continue
        NP = nltk.word_tokenize(kwd)
        NPDict[kwd] = (NP, 1)
    #convert to candidate objects
    noun_phrases = []
    for item in NPDict.items():
      c = Candidate()
      c.name = item[0]
      c.list = item[1][0]
      c.length = len(item[1][0])
      c.syntax = item[1][1]
      c.freq = item[1][2]
      noun_phrases.append(c)
    if verbose:
      print()
    return noun_phrases
  
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
        print(idx+1, '/', len(candidates), end='\r')
      candidate.NPFrequency = candidate.freq
      #for previous in candidateDict_items:
      for previous in candidates:
        if previous.length <= candidate.length:
          break
        if candidate.name in previous.name:
          candidate.NPFrequency += previous.freq
    if verbose:
      print()

  def calculateSyntaxFrequencies(self, candidates, verbose=False):
    """Count total frequencies of syntax based on term frequencies"""
    syntaxDict = {}
    for idx, candidate in enumerate(candidates):
      if verbose:
        print(idx+1, '/', len(candidates), end='\r')
      if candidate.syntax in syntaxDict:
        syntaxDict[candidate.syntax] += candidate.TF
      else:
        syntaxDict[candidate.syntax] = candidate.TF
    for idx, candidate in enumerate(candidates):
      if verbose:
        print(idx+1, '/', len(candidates), end='\r')
      candidate.syntaxFrequency = syntaxDict[candidate.syntax]
    if verbose:
      print()
  
  def convertSyntaxToNumerical(self, candidates, verbose=False):
    """Convert syntax string classes to numerical classes"""
    syntaxDict = {}
    classNum = 0
    if self.syntaxDict == None:
      for idx, candidate in enumerate(candidates):
        if verbose:
          print(idx+1, '/', len(candidates), end='\r')
        if candidate.syntax not in syntaxDict:
          syntaxDict[candidate.syntax] = classNum
          classNum += 1
      self.syntaxDict = syntaxDict
    else:
      syntaxDict = self.syntaxDict
    for idx, candidate in enumerate(candidates):
      if verbose:
        print(idx+1, '/', len(candidates), end='\r')
      if candidate.syntax in syntaxDict:
        candidate.syntaxClass = syntaxDict[candidate.syntax]
      else:
        candidate.syntaxClass = -1
    if verbose:
      print()
  
  def calculateRawTF_IDF(self, candidates, verbose=False):
    """Count term frequencies and inverse document frequencies of candidates"""
    #process only longer candidates, then break
    for idx, candidate in enumerate(candidates):
      if verbose:
        print(idx+1, '/', len(candidates), end='\r')
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
        print(idx+1, '/', len(candidates), end='\r')
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
  
  def asNumpy(self, candidates, gt_names, attrs):
    """Convert candidate features and keywords into numpy matrices, whereas each  candidate is represented by a row and each feature by a column."""
    X = []
    y = []
    for candidate in candidates:
      row = []
      for attr in attrs:
        if attr == "syntaxClass": #one-hot categorical
          classes = len(self.syntaxDict)
          classNum = candidate.syntaxClass
          row.extend([1 if x == classNum else 0 for x in range(0,classes)])
        else:
          row.append(attrgetter(attr)(candidate))
      X.append(row)
      if candidate.list in gt_names:
        y.append(1)
      else:
        y.append(0)
    X = np.array(X, dtype='float64')
    y = np.array(y, dtype='float64')
    return X, y
    
    