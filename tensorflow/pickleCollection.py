import io
import pickle
import numpy as np
from gensim.models.word2vec import Word2Vec
from SemEval2017Collection import SemEval2017Collection 
import options

PATH = "./data/"
TEST = "test_sem2017.pickle"
DEV = "dev_sem2017.pickle"
TRAIN = "train_sem2017.pickle"
META = "meta_sem2017.pickle"
LM_DICT = "vocab-2016-09-10.txt"
LM_DATA = "embeddings_char_cnn.npy"

WORD2VEC = options.WORD2VEC
TRAIN_FOLDER = options.TRAIN_FOLDER
DEV_FOLDER = options.DEV_FOLDER
TEST_FOLDER = options.TEST_FOLDER
GAZETTEERS_PATH = options.GAZETTEERS_PATH
GAZETTEERS_OUT_PATH = options.GAZETTEERS_OUT_PATH

WORD_MAX_LEN = 16

model = Word2Vec.load(WORD2VEC)
dictionary = model.wv.vocab
dictionary = {k:v.count for k, v in dictionary.items()}

collection = SemEval2017Collection(TRAIN_FOLDER, DEV_FOLDER, TEST_FOLDER, verbose=True)
collection.setDictionary(dictionary)
collection.loadGazetteers(GAZETTEERS_PATH)
collection.cropGazetteers(GAZETTEERS_OUT_PATH)

def save(ds, labels, path):
  chars, word_lengths = collection.encodeCharacters(ds.corpus, WORD_MAX_LEN)
  data, labels, lengths = collection.encode(ds.corpus_unk, labels)
  gazetteers = collection.encodeGazetteers(ds.corpus_unk)
  with io.open(path, "wb") as fp:
    pickle.dump({"data":data,
                 "labels":labels,
                 "lengths":lengths,
                 "chars":chars,
                 "word_lengths":word_lengths,
                 "gazetteers":gazetteers
                }, fp)



save(collection.train, collection.train_labels, PATH+TRAIN)
save(collection.dev, collection.dev_labels, PATH+DEV)
save(collection.test, collection.test_labels, PATH+TEST)


# create matrix with word embeddings, indexed based on the collection dictionary
invDict = collection.getInverseDictionary()
no_vector_count = len(invDict) - model.wv.syn0.shape[0]
word_vectors = np.zeros(model.wv.syn0.shape)
for i in range(model.wv.syn0.shape[0]):
  print("Vector:", i+1, "/", model.wv.syn0.shape[0], end='\r')
  word = invDict[i]
  if word in model.wv:
    v = model.wv[word].reshape((1,model.wv.syn0.shape[1]))
    word_vectors[i,:] = v
  else:
    v = np.zeros((1,model.wv.syn0.shape[1]))
    word_vectors[i,:] = v
print()

with io.open(PATH+LM_DICT, "r", encoding='utf-8') as fp:
  lm_dict = fp.readlines()
  lm_dict = {x.strip():idx for idx,x in enumerate(lm_dict)}

matches = []
non_matches = 0
for w in invDict:
  if w in lm_dict:
    matches.append((w,lm_dict[w]))
  else:
    matches.append((w,lm_dict['<UNK>']))
    non_matches += 1

print("Non-Matches:",non_matches)
print("Non-Matches: {:.3%}".format(non_matches/len(invDict)))

emb_char_cnn = np.load(PATH+LM_DATA)

emb_out = np.zeros((len(matches),emb_char_cnn.shape[1]))

for i in range(len(matches)):
  id = matches[i][1]
  emb_out[i,:] = emb_char_cnn[id]



with io.open(PATH+META, "wb") as fp:
    pickle.dump({"sent_length": collection.getSentenceLength(),
                 "invDict":invDict,
                 "labelNames":collection.getInverseLabelDict(),
                 "labelDict":collection.getLabelDict(),
                 "alphabet":collection.getAlphabet(),
                 "word_length": WORD_MAX_LEN,
                 "word_vectors":word_vectors,
                 "char_CNN_vectors":emb_out,
                 "gazetteer_count":len(collection.gazetteers),
                 "no_vector_count": no_vector_count}, fp)

