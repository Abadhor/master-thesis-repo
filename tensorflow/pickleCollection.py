import io
import pickle
import numpy as np
from gensim.models.word2vec import Word2Vec
from SemEval2017Collection import SemEval2017Collection 

PATH = "./data/"
TEST = "test_sem2017.pickle"
DEV = "dev_sem2017.pickle"
TRAIN = "train_sem2017.pickle"
META = "meta_sem2017.pickle"

WORD2VEC = "D:/data/other/wikipedia/300-1/skipgram.model"
TRAIN_FOLDER = "D:/Uni/MasterThesis/Data/SemEval2017_AutomaticKeyphraseExtraction/scienceie2017_train/train2/"
DEV_FOLDER = "D:/Uni/MasterThesis/Data/SemEval2017_AutomaticKeyphraseExtraction/scienceie2017_dev/dev/"
TEST_FOLDER = "D:/Uni/MasterThesis/Data/SemEval2017_AutomaticKeyphraseExtraction/semeval_articles_test/"

model = Word2Vec.load(WORD2VEC)
dictionary = model.wv.vocab
dictionary = {k:v.count for k, v in dictionary.items()}

collection = SemEval2017Collection(TRAIN_FOLDER, DEV_FOLDER, TEST_FOLDER, verbose=True)
collection.setDictionary(dictionary)

def save(ds, labels, path):
  data, labels, lengths = collection.encode(ds.corpus, labels)
  with io.open(path, "wb") as fp:
    pickle.dump({"data":data, "labels":labels, "lengths":lengths}, fp)



save(collection.train, collection.train_labels, PATH+TRAIN)
save(collection.dev, collection.dev_labels, PATH+DEV)
save(collection.test, collection.test_labels, PATH+TEST)


# create matrix with word embeddings, indexed based on the collection dictionary
invDict = collection.getInverseDictionary()
word_vectors = np.zeros((len(invDict),model.wv.syn0.shape[1]))
for i in range(len(invDict)):
  print("Vector:", i+1, "/", len(invDict), end='\r')
  word = invDict[i]
  if word in model.wv:
    v = model.wv[word].reshape((1,model.wv.syn0.shape[1]))
    word_vectors[i,:] = v
  else:
    v = np.zeros((1,model.wv.syn0.shape[1]))
    word_vectors[i,:] = v
print()

with io.open(PATH+META, "wb") as fp:
    pickle.dump({"sent_length": collection.getSentenceLength(),
                 "invDict":invDict,
                 "labelNames":collection.getInverseLabelDict(),
                 "labelDict":collection.getLabelDict(),
                 "word_vectors":word_vectors}, fp)

