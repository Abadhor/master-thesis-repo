import io
import csv
import os
import nltk
import nltk.tokenize.punkt as punkt
import re
import xml.etree.ElementTree as ET

DATA_FOLDER = "D:\\Uni\\MasterThesis\\Data\\SemEval2017_AutomaticKeyphraseExtraction\\scienceie2017_train\\train2\\"
OUT_FOLDER = "D:\\Uni\\MasterThesis\\Data\\SemEval2017_AutomaticKeyphraseExtraction\\scienceie2017_train\\train2_modified\\"

ABBREV_TYPES = set(['e.g', 'eq', 'eqs', 'etc', 'refs', 'ref', 'fig', 'figs', 'i.e', 'al', 'inc', 'sec', 'cf', 'i.v'])
sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
#add extra
sentence_tokenizer._params.abbrev_types.update(ABBREV_TYPES)
p = re.compile(r'\[[\d\W]*?\]')

ns_doc = {'xocs':'http://www.elsevier.com/xml/xocs/dtd',
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

ns_fulltext = {'xocs':'http://www.elsevier.com/xml/xocs/dtd',
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


def getNodeText(node):
  retString = ''
  if node.text != None: 
    retString += node.text
  if node.tail != None: 
    retString += node.tail
  for child in node:
    retString += getNodeText(child)
  return retString


def readSentences(filePath, sentences):
  infile = io.open(DATA_FOLDER + fname, 'r', encoding='utf8')
  #reader = csv.reader(infile, delimiter='\t')
  text = infile.read()
  #res = sentence_tokenizer.sentences_from_text(text)
  sentences.extend(split_sentences(text))
  

def readXML(filePath):
  """Read a SemEval2017 xml file and return paragraph texts.
  (Takes all sub node texts and adds them together. Does not
  handle footnotes or similar)"""
  tree = ET.parse(filePath)
  root = tree.getroot()
  paras = []
  if "full-text-retrieval-response" in root.tag:
    for para in root.findall(".//xocs:rawtext", ns_fulltext):
      paras.append(para.text)
  for para in root.findall(".//ce:sections//ce:para", ns_doc):
    paras.append(getNodeText(para))
  #domain = root.findall(".//xocs:normalized-srctitle", ns)[0].text
  return paras#, domain

def split_sentences(text):
  text = p.sub('#REF', text)
  return sentence_tokenizer.tokenize(text)


def writeSentences(sentences, outFile):
  for sent in sentences:
    outFile.write(sent + '\n')

def writePara(para, outFile):
  first = True
  for sent in para:
    if first:
      outFile.write('\'' + sent + '\'')
      first = False
    else:
      outFile.write(',\'' + sent + '\'')
  outFile.write('\n')

def countProblematic(sentences):
  count = 0
  for i in range(0, len(sentences)):
    s = sentences[i].strip()
    if s[0] == s[0].lower():
      #print(sentences[i-1]+ '####' +sentences[i])
      count += 1
  return count

sentences = []

sentCount = 0
problemCount = 0

fileList = os.listdir(DATA_FOLDER)
for fname in fileList:
  if fname[len(fname)-3:] == 'xml':
    paras = readXML(DATA_FOLDER + fname)
    with io.open(OUT_FOLDER + fname[:len(fname)-3] + 'sents', 'w', encoding='utf8') as outfile:
      for para in paras:
        sents = split_sentences(para)
        writeSentences(sents, outfile)
        #writePara(sents, outfile)
        sentCount += len(sents)
        problemCount += countProblematic(sents)
  #elif fname[len(fname)-3:] == 'txt':
  #  readSentences(DATA_FOLDER + fname, sentences)


print("Total Sentences: {0}".format(sentCount))
print("Problematic Sentences: {:.2%}".format(problemCount/sentCount))


      




