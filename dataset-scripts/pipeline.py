from SemEval2017Dataset import SemEval2017Dataset
from Baseline import Baseline

DATA_FOLDER = "D:\\Uni\\MasterThesis\\Data\\SemEval2017_AutomaticKeyphraseExtraction\\scienceie2017_train\\train2\\"
OUT_FOLDER = "D:\\Uni\\MasterThesis\\Data\\SemEval2017_AutomaticKeyphraseExtraction\\scienceie2017_train\\train2_modified\\"
OUT_FOLDER_KWDS = "D:\\Uni\\MasterThesis\\Data\\SemEval2017_AutomaticKeyphraseExtraction\\scienceie2017_train\\train2_keywords\\"
DEBUG = True
ds = SemEval2017Dataset()


ds.extractKeywords(DATA_FOLDER, "ann")
ds.filterUnigramKeywords()
ds.dumpKeywords(OUT_FOLDER_KWDS, "keywords")

ds.extractSentencesFromXML(DATA_FOLDER, "xml", debug=True)
ds.dumpSentences(OUT_FOLDER, "sents")

baseline = Baseline(ds)
print("Tagging...")
candidates = baseline.extractNounPhrases(debug=True)

s = baseline.calculateStatistics(candidates)
print("Recall", s.recall)
print("Precision", s.precision)
print("F1", s.F1)

print("Sorting...")
candidates = baseline.sortCandidates(candidates)
print("Number of Candidates: ", len(candidates))

print("Filtering Formulas...")
candidates = baseline.filterFormulas(candidates)
candidates = baseline.filterLength(candidates)

print("Counting Total Frequencies...")
baseline.calculateTotalFrequencies(candidates, debug=True)
candidates = baseline.filterTotalFrequency(candidates)
print("Number of Candidates: ", len(candidates))

s = baseline.calculateStatistics(candidates)
print("Recall", s.recall)
print("Precision", s.precision)
print("F1", s.F1)

print("Calculating candidate nested frequencies...")
baseline.calculateNestedFrequencies(candidates, debug=True)
print("Calculating C-Values...")
c_vals = baseline.calculateCVals(candidates, debug=True)








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