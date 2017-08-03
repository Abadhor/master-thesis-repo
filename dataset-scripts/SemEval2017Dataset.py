import io
import os
import nltk
import csv
import nltk.tokenize.punkt as punkt
import re
import xml.etree.ElementTree as ET
from KeywordDataset import KeywordDataset

class SemEval2017Dataset(KeywordDataset):
  
  def __init__(self):
    self.ABBREV_TYPES = set(['e.g', 'eq', 'eqs', 'etc', 'refs', 'ref', 'fig', 'figs', 'i.e', 'al', 'inc', 'sec', 'cf', 'i.v'])
    self.sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    #add extra
    self.sentence_tokenizer._params.abbrev_types.update(self.ABBREV_TYPES)
    self.RE_REF = re.compile(r'\[[\d\W]*?\]')
    self.RE_FIG = re.compile(r'[F|f]igs?\. *\d+')
    self.RE_NEWLINE = re.compile(r'\n')
    self.RE_OTHERS = re.compile(r'[^a-zA-Z0-9_\-\. ]+')
    
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
    self.keywords = set()
    fileList = os.listdir(folder)
    for fname in fileList:
      if fname[len(fname)-len(ext):] == ext:
        self.extractKeywordsFromFile(folder + fname)
  
  def dumpKeywords(self, folder, fileName):
    """Write keyword set to file"""
    with io.open(folder + fileName, 'w', encoding='utf8') as outfile:
      for kwd in sorted(list(self.keywords)):
        outfile.write(kwd + '\n')
  
  def loadKeywords(self, folder, fileName):
    """Read keyword set from file"""
    self.keywords = set()
    with io.open(folder + fileName, 'r', encoding='utf8') as infile:
      for line in infile:
        self.keywords.add(line.strip())
  
  def extractKeywordsFromFile(self, filePath):
    infile = io.open(filePath, 'r', encoding='utf8')
    reader = csv.reader(infile, delimiter='\t')
    for row in reader:
      if (row[0] == '*') or (row[0][0] == 'R'):
        continue
      kw = " ".join(nltk.word_tokenize(row[2]))
      self.keywords.add(kw)
    infile.close()
  
  def extractSentencesFromXML(self, folder, ext='xml', debug=False):
    """Extract Sentences from xml files in folder with specified extension"""
    self.text_files = dict()
    fileList = os.listdir(folder)
    f_fileList = [fname for fname in fileList if fname[len(fname)-len(ext):] == ext]
    for idx, fname in enumerate(f_fileList):
      if debug:
        print("File:",idx+1, '/', len(f_fileList), end='\r')
      key = fname[:len(fname)-(len(ext)+1)]
      file_sentences = []
      paras = self.extractXMLfromFile(folder + fname)
      for para in paras:
        sents = self.split_sentences(para)
        file_sentences.extend(sents)
        for i in range(0, len(file_sentences)):
          file_sentences[i] = file_sentences[i].strip()
      self.text_files[key] = file_sentences
  
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
  
  def getDictionary(self):
    """Get a dictionary with word frequencies for the whole dataset"""
    dictionary = {}
    for fname in self.text_files.keys():
      file = self.text_files[fname]
      for line in file:
        line_tokens = nltk.word_tokenize(line)
        if len(line_tokens) == 0:
          continue
        for token in line_tokens:
          if token not in dictionary:
            dictionary[token] = 1
          else:
            dictionary[token] += 1
    return dictionary
  
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
  
  def split_sentences(self, text):
    text = self.RE_NEWLINE.sub('', text)
    text = self.RE_REF.sub('', text)
    text = self.RE_FIG.sub('', text)
    #text = self.RE_OTHERS.sub('', text)
    return self.sentence_tokenizer.tokenize(text)
  
  def filterUnigramKeywords(self):
    self.keywords = {x for x in self.keywords if len(nltk.word_tokenize(x)) >= 2}


