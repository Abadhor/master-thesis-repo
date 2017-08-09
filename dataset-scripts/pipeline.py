from SemEval2017Dataset import SemEval2017Dataset
from BaselineFeatures import BaselineFeatures
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report
import numpy as np

DATA_FOLDER = "D:\\Uni\\MasterThesis\\Data\\SemEval2017_AutomaticKeyphraseExtraction\\scienceie2017_train\\train2\\"
OUT_FOLDER = "D:\\Uni\\MasterThesis\\Data\\SemEval2017_AutomaticKeyphraseExtraction\\scienceie2017_train\\train2_modified\\"
OUT_FOLDER_KWDS = "D:\\Uni\\MasterThesis\\Data\\SemEval2017_AutomaticKeyphraseExtraction\\scienceie2017_train\\train2_keywords\\"
OUT_FOLDER_FEATURES ="D:/Uni/MasterThesis/Data/SemEval2017_AutomaticKeyphraseExtraction/dump/"
TEST_FOLDER = "D:/Uni/MasterThesis/Data/SemEval2017_AutomaticKeyphraseExtraction/scienceie2017_dev/dev/"
DEBUG = False
f = "{:.4}"
ds = SemEval2017Dataset()

def getBaselineFeatures(folder, addKeywords, verbose=False):
  
  ds.extractKeywords(folder, "ann")
  ds.filterUnigramKeywords()
  #ds.dumpKeywords(OUT_FOLDER_KWDS, "keywords")
  
  #ds.extractSentences(folder, "xml", xml=True, verbose=verbose)
  ds.extractSentences(folder, "txt", xml=False, verbose=verbose)
  #ds.dumpSentences(OUT_FOLDER, "sents")
  
  baseline = BaselineFeatures(ds, addKeywords=addKeywords)
  print("Tagging...")
  candidates = baseline.extractNounPhrases(verbose=verbose)
  
  s = baseline.calculateStatistics(candidates)
  print("Recall", f.format(s.recall))
  print("Precision", f.format(s.precision))
  print("F1", f.format(s.F1))
  
  print("Sorting...")
  candidates = baseline.sortCandidates(candidates)
  print("Number of Candidates: ", len(candidates))
  
  print("Filtering Formulas...")
  candidates = baseline.filterFormulas(candidates)
  candidates = baseline.filterLength(candidates)
  
  print("Counting Total NP Frequencies...")
  baseline.calculateTotalNPFrequencies(candidates, verbose=verbose)
  #candidates = baseline.filterTotalFrequency(candidates)
  print("Counting raw TF and IDF...")
  baseline.calculateRawTF_IDF(candidates, verbose=verbose)
  print("Number of Candidates: ", len(candidates))
  
  s = baseline.calculateStatistics(candidates)
  print("Recall", f.format(s.recall))
  print("Precision", f.format(s.precision))
  print("F1", f.format(s.F1))
  
  print("Calculating candidate nested frequencies...")
  baseline.calculateNestedFrequencies(candidates, verbose=verbose)
  print("Calculating C-Values...")
  baseline.calculateCVals(candidates, verbose=verbose)
  print("Calculating Perplexity...")
  baseline.calculatePerplexity(candidates, verbose=verbose)
  print("Calculating Statistical Features...")
  baseline.calculateStatFeatures(candidates, verbose=verbose)
  
  norm = ['TF', 'IDF']
  baseline.normalizeFeatures(candidates, norm)
  attrs = ['length', 'TF', 'IDF', 'c_value', 'log_pp', 't_score', 'pmi', 'dice']
  #baseline.exportARFF(candidates, attrs, "baseline", OUT_FOLDER_FEATURES, "baseline.arff")
  
  return baseline.asNumpy(candidates, attrs)

features_train, labels_train = getBaselineFeatures(DATA_FOLDER, addKeywords=True, verbose=True)
np.savetxt(OUT_FOLDER_FEATURES + "train.npy", features_train, fmt='%10.5f',delimiter=',')
features_test, labels_test = getBaselineFeatures(TEST_FOLDER, addKeywords=False, verbose=True)
np.savetxt(OUT_FOLDER_FEATURES + "test.npy", features_test, fmt='%10.5f', delimiter=',')
clf = svm.SVC()
clf.fit(features_train, labels_train)

pred = clf.predict(features_test)
acc = accuracy_score(labels_test , pred)
conf = confusion_matrix(labels_test , pred, labels=[1,0])
print(conf)
print(classification_report(labels_test , pred, labels=[1,0], target_names=['yes','no']))
#prec = precision_score(labels_test, pred, average=None)
#recall = recall_score(labels_test, pred, average=None)

#DEBUG
if not DEBUG:
  exit()
test = SemEval2017Dataset()
test.loadKeywords(OUT_FOLDER_KWDS, "keywords")
test.loadSentences(OUT_FOLDER, "sents")
t_keywords, t_text = test.getDataset()
ds_keywords, ds_text = ds.getDataset()
print("Keywords are equal: ",ds_keywords == t_keywords)

print("Text Files are equal: ",ds_text == t_text)
if ds_text != t_text:
  for key in ds_text.keys():
    for idx, line in enumerate(ds_text[key]):
      if line != t_text[key][idx]:
        print('[',idx,']', key,'\n',line, '\n', t_text[key][idx])