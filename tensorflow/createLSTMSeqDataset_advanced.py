
import random
import numpy as np
import io
import pickle

random.seed(5)

PATH = "./data/"
TEST = "test_seq_data_adv.pickle"
DEV = "dev_seq_data_adv.pickle"
TRAIN = "train_seq_data_adv.pickle"

WORDS = ['a','b','c','d','e','f','g']
CLASS_OUTSIDE = 'O'
CLASS_VALID = 'X'
CLASS_INVALID = 'I'
LABELS = [CLASS_OUTSIDE, CLASS_VALID, CLASS_INVALID]

SPECIAL = 'X'
MOD = 'M'
SENT_NUM = 10000
TEST_RATIO = 0.1
DEV_RATIO = 0.1
TRAIN_RATIO = 0.8
SENT_MAX_LEN = 20
SENT_MIN_LEN = 10
MAX_SPECIAL = 4
MAX_MOD = 4


DICT = WORDS.copy()
DICT.append(SPECIAL)
DICT.append(MOD)


def createSentence(length):
  sent = []
  for i in range(0,length):
    sent.append(random.choice(WORDS))
  sp_num = random.randrange(1,MAX_SPECIAL+1)
  mod_num = random.randrange(1,MAX_MOD+1)
  for i in range(mod_num):
    sent[random.randrange(0,length)] = MOD
  for i in range(sp_num):
    sent[random.randrange(0,length)] = SPECIAL
  return sent

def createAnnotation(sent):
  """If there is a MOD in the sentence before SPECIAL, SPECIAL is valid.
  Otherwise SPECIAL is invalid"""
  flag = False
  ann = []
  for w in sent:
    if w in WORDS:
      ann.append(CLASS_OUTSIDE)
    if w == MOD:
      flag = True
      ann.append(CLASS_OUTSIDE)
    if w == SPECIAL:
      if flag:
        ann.append(CLASS_VALID)
        flag = False
      else:
        ann.append(CLASS_INVALID)
  return ann

def encode(ls):
  data = np.zeros((len(ls),SENT_MAX_LEN, len(DICT)))
  labels = np.zeros((len(ls),SENT_MAX_LEN, len(LABELS)))
  for s_idx in range(0,len(ls)):
    s = ls[s_idx][0]
    a = ls[s_idx][1]
    s_len = len(s)
    for w_idx in range(0,s_len):
      dict_idx = DICT.index(s[w_idx])
      labels_idx = LABELS.index(a[w_idx])
      data[s_idx,w_idx,dict_idx] = 1
      labels[s_idx,w_idx,labels_idx] = 1
  return data, labels

sent_dict = {}
while len(sent_dict) < SENT_NUM:
  s = createSentence(random.randrange(SENT_MIN_LEN, SENT_MAX_LEN+1))
  a = createAnnotation(s)
  s_str = " ".join(s)
  sent_dict[s_str] = (s,a)

sents = list(sent_dict.values())
cur = 0

test = []
for i in range(cur, cur+int(SENT_NUM*TEST_RATIO)):
  test.append(sents[i])
cur = cur+int(SENT_NUM*TEST_RATIO)

dev = []
for i in range(cur, cur+int(SENT_NUM*DEV_RATIO)):
  dev.append(sents[i])
cur = cur+int(SENT_NUM*DEV_RATIO)

train = []
for i in range(cur, cur+int(SENT_NUM*TRAIN_RATIO)):
  train.append(sents[i])

data, labels = encode(test)
with io.open(PATH+TEST, "wb") as fp:
  pickle.dump({"data":data, "labels":labels, "dictionary":DICT, "label_names":LABELS}, fp)

data, labels = encode(dev)
with io.open(PATH+DEV, "wb") as fp:
  pickle.dump({"data":data, "labels":labels, "dictionary":DICT, "label_names":LABELS}, fp)

data, labels = encode(train)
with io.open(PATH+TRAIN, "wb") as fp:
  pickle.dump({"data":data, "labels":labels, "dictionary":DICT, "label_names":LABELS}, fp)

