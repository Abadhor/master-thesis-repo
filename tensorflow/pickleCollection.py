import io
import pickle
from SemEval2017Collection import SemEval2017Collection

PATH = "./data/"
TEST = "test_sem2017.pickle"
DEV = "dev_sem2017.pickle"
TRAIN = "train_sem2017.pickle"
META = "meta_sem2017.pickle"

TRAIN_FOLDER = "D:\\Uni\\MasterThesis\\Data\\SemEval2017_AutomaticKeyphraseExtraction\\scienceie2017_train\\train2\\"
DEV_FOLDER = "D:/Uni/MasterThesis/Data/SemEval2017_AutomaticKeyphraseExtraction/scienceie2017_dev/dev/"
TEST_FOLDER = "D:/Uni/MasterThesis/Data/SemEval2017_AutomaticKeyphraseExtraction/semeval_articles_test/"

collection = SemEval2017Collection(TRAIN_FOLDER, DEV_FOLDER, TEST_FOLDER, verbose=True)
collection.setDictionary()

def save(ds, labels, path):
  data, labels, lengths = collection.encode(ds.corpus, labels)
  with io.open(path, "wb") as fp:
    pickle.dump({"data":data, "labels":labels, "lengths":lengths}, fp)



save(collection.train, collection.train_labels, PATH+TRAIN)
save(collection.dev, collection.dev_labels, PATH+DEV)
save(collection.test, collection.test_labels, PATH+TEST)

with io.open(PATH+META, "wb") as fp:
    pickle.dump({"sent_length": collection.getSentenceLength(),
                 "invDict":collection.getInverseDictionary(),
                 "labelNames":collection.getInverseLabelDict(),
                 "labelDict":collection.getLabelDict()}, fp)

