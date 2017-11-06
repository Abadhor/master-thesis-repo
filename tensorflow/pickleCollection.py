
from SemEval2017Collection import SemEval2017Collection

PATH = "./data/"

TRAIN_FOLDER = "D:\\Uni\\MasterThesis\\Data\\SemEval2017_AutomaticKeyphraseExtraction\\scienceie2017_train\\train2\\"
DEV_FOLDER = "D:/Uni/MasterThesis/Data/SemEval2017_AutomaticKeyphraseExtraction/scienceie2017_dev/dev/"
TEST_FOLDER = "D:/Uni/MasterThesis/Data/SemEval2017_AutomaticKeyphraseExtraction/semeval_articles_test/"

collection = SemEval2017Collection(TRAIN_FOLDER, DEV_FOLDER, TEST_FOLDER, verbose=True)
#collection.pickle(PATH)