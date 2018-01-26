import io
import os
import nltk
import csv
import nltk.tokenize.punkt as punkt
import re
import xml.etree.ElementTree as ET
from KeywordDataset import KeywordDataset
from MWUHashTree import MWUHashTree
from nltk.stem import WordNetLemmatizer

class SemEval2017Dataset(KeywordDataset):
  
  def __init__(self, spacyNLP = None, lemmatize = False):
    super().__init__()
    self.nlp = spacyNLP
    self.lemmatize = lemmatize
    if self.nlp:
      self.useSpacy = True
    if self.lemmatize:
      self.wn = WordNetLemmatizer()
    
    # NLTK sentence splitter
    self.ABBREV_TYPES = set(['e.g', 'eq', 'eqs', 'etc', 'refs', 'ref', 'fig', 'figs', 'i.e', 'al', 'inc', 'sec', 'cf', 'i.v', 'adapt'])
    if not self.useSpacy:
      self.sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    
    # regex definitions
    self.sentence_tokenizer._params.abbrev_types.update(self.ABBREV_TYPES)
    self.RE_REF = re.compile(r'\[[\d\W]*?\]')
    self.RE_FIG = re.compile(r'[F|f]igs?\. *\d+')
    self.RE_NEWLINE = re.compile(r'\n')
    self.RE_ET_AL = re.compile(r' et al\.')
    self.RE_ET_AL_SUB = ' et al'
    self.RE_CA = re.compile(r' ca\.')
    self.RE_CA_SUB = ' ca'
    
    # namespaces for xml extraction
    self.ns_doc = {'xocs':'http://www.elsevier.com/xml/xocs/dtd',
    'xs':'http://www.w3.org/2001/XMLSchema',
    'xsi':'http://www.w3.org/2001/XMLSchema-instance',
    'xlmns':'http://www.elsevier.com/xml/ja/dtd',
    'ja':'http://www.elsevier.com/xml/ja/dtd',
    'mml':'http://www.w3.org/1998/Math/MathML',
    'tb':'http://www.elsevier.com/xml/common/table/dtd',
    'sb':'http://www.elsevier.com/xml/common/struct-bib/dtd',
    'ce':'http://www.elsevier.com/xml/common/dtd',
    'xlink':'http://www.w3.org/1999/xlink',
    'cals':'http://www.elsevier.com/xml/common/cals/dtd'}
    
    self.ns_fulltext = {'xocs':'http://www.elsevier.com/xml/xocs/dtd',
    'xsi':'http://www.w3.org/2001/XMLSchema-instance',
    'xlmns':'http://www.elsevier.com/xml/svapi/article/dtd',
    'prism':'http://prismstandard.org/namespaces/basic/2.0/',
    'dcterms':'http://purl.org/dc/terms/',
    'dc':'http://purl.org/dc/elements/1.1/',
    'xlink':'http://www.w3.org/1999/xlink',
    'tb':'http://www.elsevier.com/xml/common/table/dtd',
    'sb':'http://www.elsevier.com/xml/common/struct-bib/dtd',
    'sa':'http://www.elsevier.com/xml/common/struct-aff/dtd',
    'mml':'http://www.w3.org/1998/Math/MathML',
    'ja':'http://www.elsevier.com/xml/ja/dtd',
    'ce':'http://www.elsevier.com/xml/common/dtd',
    'cals':'http://www.elsevier.com/xml/common/cals/dtd',
    'bk':'http://www.elsevier.com/xml/bk/dtd'}
  
  
  def extractKeywords(self, folder, ext):
    """Extract Keywords from files in folder with specified extension"""
    self.keywords = MWUHashTree()
    fileList = os.listdir(folder)
    for fname in fileList:
      if fname[len(fname)-len(ext):] == ext:
        self.extractKeywordsFromFile(folder, fname, ext)
  
  def dumpKeywords(self, folder, fileName):
    """Write keyword set to file"""
    # TODO: change to JSON
    with io.open(folder + fileName, 'w', encoding='utf8') as outfile:
      for kwd in sorted(list(self.keywords.keys())):
        outfile.write(" ".join(kwd) + '\n')
  
  def loadKeywords(self, folder, fileName):
    """Read keyword set from file"""
    # TODO: change to JSON
    self.keywords = set()
    with io.open(folder + fileName, 'r', encoding='utf8') as infile:
      for line in infile:
        self.keywords.add(line.strip())
  
  def extractKeywordsFromFile(self, folder, fname, ext):
    filePath = folder + fname
    infile = io.open(filePath, 'r', encoding='utf8')
    reader = csv.reader(infile, delimiter='\t')
    for row in reader:
      if (row[0] == '*') or (row[0][0] == 'R') or (row[1][:4] == 'Task'):
        continue
      if self.useSpacy:
        kw = self.word_tokenize_Spacy(row[2])
      else:
        kw = nltk.word_tokenize(row[2])
      if self.lemmatize:
        kw = [self.wn.lemmatize(x) for x in kw]
      # debug: safe last known file
      self.keywords[kw] = fname[:len(fname)-(len(ext)+1)]
    infile.close()
  
  def extractSentences(self, folder, ext, xml=False, verbose=False):
    """Extract Sentences from xml files in folder with specified extension"""
    self.text_files = dict()
    fileList = os.listdir(folder)
    f_fileList = [fname for fname in fileList if fname[len(fname)-len(ext):] == ext]
    for idx, fname in enumerate(f_fileList):
      if verbose:
        print("File:",idx+1, '/', len(f_fileList), end='\r')
      key = fname[:len(fname)-(len(ext)+1)]
      file_sentences = []
      if xml:
        paras = self.extractXMLfromFile(folder + fname)
      else:
        paras = self.extractFromText(folder + fname)
      for para in paras:
        if self.useSpacy:
          sents = self.split_sentences_Spacy(para)
        else:
          sents = self.split_sentences_NLTK(para)
        file_sentences.extend(sents)
      self.text_files[key] = file_sentences
    if verbose:
      print()
  
  def dumpSentences(self, folder, ext):
    """Dump split sentences to files"""
    for fname_key in self.text_files.keys():
      with io.open(folder + fname_key + '.' + ext, 'w', encoding='utf8') as outfile:
        for sent in self.text_files[fname_key]:
          outfile.write(sent+'\n')
  
  def loadSentences(self, folder, ext):
    """Read split sentences from files"""
    self.text_files = dict()
    fileList = os.listdir(folder)
    for fname in fileList:
      if fname[len(fname)-len(ext):] == ext:
        key = fname[:len(fname)-(len(ext)+1)]
        file_sentences = []
        with io.open(folder + fname, 'r', encoding='utf8') as infile:
          for line in infile:
            file_sentences.append(line.strip())
        self.text_files[key] = file_sentences
  
  def tokenize(self):
    """Tokenize the dataset and create a dictionary with word frequencies for the whole dataset"""
    self.corpus = None
    dictionary = {}
    for fname in self.text_files.keys():
      file = self.text_files[fname]
      for i in range(len(file)):
        sentence = file[i]
        if self.useSpacy:
          line_tokens = self.word_tokenize_Spacy(sentence)
        else:
          line_tokens = nltk.word_tokenize(sentence)
        file[i] = line_tokens
        if len(line_tokens) == 0:
          continue
        for token in line_tokens:
          if token not in dictionary:
            dictionary[token] = 1
          else:
            dictionary[token] += 1
    self.dictionary = dictionary
  
  def extractXMLfromFile(self, filePath):
    """Read a SemEval2017 xml file and return paragraph texts.
    (Takes all sub node texts and adds them together. Does not
    handle footnotes or similar)"""
    tree = ET.parse(filePath)
    root = tree.getroot()
    paras = []
    if "full-text-retrieval-response" in root.tag:
      for para in root.findall(".//xocs:rawtext", self.ns_fulltext):
        paras.append(para.text)
    for para in root.findall(".//ce:sections//ce:para", self.ns_doc):
      paras.append(self.getNodeText(para))
    #domain = root.findall(".//xocs:normalized-srctitle", ns)[0].text
    return paras#, domain
  
  def getNodeText(self, node):
    retString = ''
    if node.text != None: 
      retString += node.text + ' '
    if node.tail != None: 
      retString += node.tail + ' '
    for child in node:
      retString += self.getNodeText(child)
    return retString
  
  def split_sentences_NLTK(self, text):
    text = self.RE_NEWLINE.sub('', text)
    text = self.RE_REF.sub('', text)
    text = self.RE_FIG.sub('', text)
    #text = self.RE_OTHERS.sub('', text)
    sentences = self.sentence_tokenizer.tokenize(text)
    sentences = [sent.strip() for sent in sentences]
    return sentences
  
  def split_sentences_Spacy(self, text):
    text = text.replace(chr(8211), '-')
    text = text.replace('"', '')
    text = text.replace('“', '')
    text = text.replace('”', '')
    text = self.RE_ET_AL.sub(self.RE_ET_AL_SUB, text)
    text = self.RE_CA.sub(self.RE_CA_SUB, text)
    doc = self.nlp(text)
    sentences = [sent.string.strip() for sent in doc.sents]
    return sentences
  
  def word_tokenize_Spacy(self, text):
    """Enhance spacy tokenization"""
    text = self.preprocess(text, toLower=True, doubleDash=True)
    doc = self.nlp(text)
    tokens = [token.text.strip() for token in doc]
    tokens = self.postprocess(tokens, removeHyphens=True)
    # lemmatize
    if self.lemmatize:
      tokens = [self.wn.lemmatize(x) for x in tokens]
    return tokens
  
  def extractFromText(self, filePath):
    with io.open(filePath, 'r', encoding='utf8') as infile:
      para = []
      for line in infile:
        para.append(line)
      return para
  
  def filterUnigramKeywords(self):
    items = self.keywords.items()
    self.keywords = MWUHashTree()
    for item in items:
      key = item[0]
      value = item[1]
      if len(key) > 1:
        self.keywords[key] = value


