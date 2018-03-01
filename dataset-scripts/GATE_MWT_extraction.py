import io
import xml.etree.ElementTree as ET
import argparse
import os
import pickle
import spacy
from nltk.tokenize import RegexpTokenizer


parser = argparse.ArgumentParser()
parser.add_argument(
      "--gate_dir",
      type=str,
      default="",
      required=True,
      help="The location of the GATE inline MWT annotated text files.")
parser.add_argument(
      "--out_file",
      type=str,
      default="",
      required=True,
      help="The name of the output file of the pickled set.")
args = parser.parse_args()

def extractMWTS(file):
  with io.open(file, 'r', encoding='utf-8') as fp:
    file = fp.read()
  xmlString = '<doc xmlns:gate="gate">' + file + '</doc>'
  root = ET.fromstring(xmlString)
  mwts = []
  for node in root.findall("./MWT"):
    mwts.append(getNodeText(node, True))
  return mwts

def getNodeText(node, first):
  retString = ''
  if node.text != None: 
    retString += node.text
  for child in node:
    retString += getNodeText(child, False)
  if node.tail != None and not first: 
    retString += node.tail
  return retString

def cleansetokenize(inputtext):
    text=inputtext.lower()
    cleanr =re.compile('<.*?>')
    text=re.sub(cleanr,'', text)
    text=re.sub('\d+', ' ', text)
    tokenizer = RegexpTokenizer(r'\w+')
    processed_tokens = tokenizer.tokenize(text)
    return processed_tokens

nlp = spacy.load('en_core_web_md')

# MWT Extraction
mwt_set_raw = set()
fileList = os.listdir(args.gate_dir)
for fname in fileList:
  ls = extractMWTS(args.gate_dir+'/'+fname)
  for item in ls:
    mwt_set_raw.add(item)
# Pre-processing
mwt_dict = {}
for mwt in mwt_set_raw:
  doc = nlp(mwt)
  # lemma + remove whitespaces and empty tokens
  tokens = [x.lemma_.strip() for x in doc if len(x.lemma_.strip()) > 0]
  mwt_dict[" ".join(tokens)] = tokens
with io.open(args.out_file, "wb") as fp:
  pickle.dump(mwt_dict, fp)