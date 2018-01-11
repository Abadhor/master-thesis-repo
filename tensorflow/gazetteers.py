import json
import io
import spacy
import math
from gensim.models.word2vec import Word2Vec
from operator import itemgetter

FILE_PATH = "D:/data/other/gazetteer/all-gazetteers.json"
OUT_PATH = "D:/data/other/gazetteer/selected-gazetteers.json"
WORD2VEC = "D:/data/other/wikipedia/300-1/skipgram.model"

def norm_2(A):
  sum = 0
  for i in range(len(A)):
    sum += A[i]*A[i]
  return math.sqrt(sum)

def dot(A,B):
  dot = 0
  for i in range(len(A)):
    dot += A[i] * B[i]
  return dot

def cosine(A,B):
  return dot(A,B) / (norm_2(A) * norm_2(B))

with io.open(FILE_PATH, "r") as fp:
  root = json.load(fp)

strings = []
for k in root.keys():
  if k == "freebase_project":
    pass
  elif k == "freebase_material":
    pass
  elif k == "freebase_invention":
    pass
  elif k == "peter_turney_list":
    pass
  elif k == "freebase_chemical_element":
    pass
  elif k == "freebase_patent":
    pass
  else:
    continue
  if type(root[k][0]) == type(""):
    strings.extend(root[k])

# remove duplicates
strings = list(set(strings))

model = Word2Vec.load(WORD2VEC)
dictionary = model.wv.vocab
dictionary = {k:v.count for k, v in dictionary.items()}

nlp = spacy.load('en_core_web_md')

gazetteers = []
MWT_gazetteers = []
for idx, s in enumerate(strings):
  print(idx, "/", len(strings), end="\r")
  doc = nlp(s)
  tokens = [token.text.strip() for token in doc]
  valid = True
  for token in tokens:
    if token not in dictionary:
      valid = False
  if valid:
    gazetteers.append(tokens)
    if len(tokens) > 1:
      sum = 0
      for i in range(len(tokens)-1):
        sum += cosine(model.wv[tokens[i]],model.wv[tokens[i+1]])
      score = sum/(len(tokens)-1)
      MWT_gazetteers.append((tokens, score, len(tokens)))

MWT_gazetteers_sorted = sorted(MWT_gazetteers, key=itemgetter(1), reverse=True)

with io.open(OUT_PATH, "w") as fp:
  root = json.dump(MWT_gazetteers_sorted[:1000], fp)
