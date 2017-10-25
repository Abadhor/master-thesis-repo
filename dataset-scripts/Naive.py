from SemEval2017Dataset import SemEval2017Dataset
from BaselineFeatures import BaselineFeatures
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report, make_scorer
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import math
import random

DATA_FOLDER = "D:\\Uni\\MasterThesis\\Data\\SemEval2017_AutomaticKeyphraseExtraction\\scienceie2017_train\\train2\\"
OUT_FOLDER = "D:\\Uni\\MasterThesis\\Data\\SemEval2017_AutomaticKeyphraseExtraction\\scienceie2017_train\\train2_modified\\"
OUT_FOLDER_KWDS = "D:\\Uni\\MasterThesis\\Data\\SemEval2017_AutomaticKeyphraseExtraction\\scienceie2017_train\\train2_keywords\\"
OUT_FOLDER_FEATURES ="D:/Uni/MasterThesis/Data/SemEval2017_AutomaticKeyphraseExtraction/dump/"
TEST_FOLDER = "D:/Uni/MasterThesis/Data/SemEval2017_AutomaticKeyphraseExtraction/scienceie2017_dev/dev/"
#TEST_FOLDER = "D:/Uni/MasterThesis/Data/SemEval2017_AutomaticKeyphraseExtraction/semeval_articles_test/"
DEBUG = False
f = "{:.4}"
ds = SemEval2017Dataset()
random.seed(5)

def keyword_f1_function(y_true, y_pred, np_recall=1):
  """
  Function that calculates the f1 score for detected keywords.
  Also considers recall of Noun Phrase filtering step in score calculation.
  """
  prec = precision_score(y_true, y_pred, average=None)
  recall = recall_score(y_true, y_pred, average=None)
  #class 1: keyword = yes
  kw_prec = prec[1]
  kw_rec = recall[1]
  kw_f1 = 2*(kw_prec*kw_rec*np_recall)/(kw_prec + (kw_rec * np_recall))
  return kw_f1

def oversampleClass(X, y, c_name, factor):
  a_x = []
  a_y = []
  for i in range(1,math.floor(factor)):
    for idx,r in enumerate(X):
      if y[idx] == c_name:
        a_x.append(r)
        a_y.append(c_name)
  prob = factor - math.floor(factor)
  for idx,r in enumerate(X):
    if y[idx] == c_name:
      if random.random() < prob:
        a_x.append(r)
        a_y.append(c_name)
  return np.append(X, np.array(a_x), axis=0), np.append(y, np.array(a_y), axis=0)

def undersampleClass(X, y, c_name, factor):
  a_x = []
  a_y = []
  for idx,r in enumerate(X):
    if y[idx] == c_name:
      if random.random() < factor:
        a_x.append(r)
        a_y.append(c_name)
      else:
        a_x.append(r)
        a_y.append(c_name)
  return np.array(a_x), np.array(a_y)

def countClass(X, y, c_name):
  count = 0
  for idx,r in enumerate(X):
    if y[idx] == c_name:
      count += 1
  return count


def getBaselineFeatures(folder, addKeywords, verbose=False, syntaxDict=None):
  
  ds.extractKeywords(folder, "ann")
  ds.filterUnigramKeywords()
  ds.dumpKeywords(OUT_FOLDER_KWDS, "keywords")
  
  #ds.extractSentences(folder, "xml", xml=True, verbose=verbose)
  ds.extractSentences(folder, "txt", xml=False, verbose=verbose)
  ds.dumpSentences(OUT_FOLDER, "sents")
  
  baseline = BaselineFeatures(ds, addKeywords=addKeywords, syntaxDict=syntaxDict)
  print("Tagging...")
  candidates = baseline.extractNounPhrases(verbose=verbose)
  
  s = baseline.calculateStatistics(candidates, baseline.ds_keywords)
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
  print("Counting Syntax Frequencies...")
  baseline.calculateSyntaxFrequencies(candidates, verbose=verbose)
  print("Convert Syntax to numerical SyntaxClass...")
  baseline.convertSyntaxToNumerical(candidates, verbose=verbose)
  print("Number of Candidates: ", len(candidates))
  
  s = baseline.calculateStatistics(candidates, baseline.ds_keywords)
  print("Recall", f.format(s.recall))
  print("Precision", f.format(s.precision))
  print("F1", f.format(s.F1))
  nested_recall = baseline.calculateNestedRecall(candidates, baseline.ds_keywords)
  print("Nested Recall", f.format(nested_recall))
  
  print("Calculating candidate nested frequencies...")
  baseline.calculateNestedFrequencies(candidates, verbose=verbose)
  print("Calculating C-Values...")
  baseline.calculateCVals(candidates, verbose=verbose)
  print("Calculating Perplexity...")
  baseline.calculatePerplexity(candidates, verbose=verbose)
  print("Calculating Statistical Features...")
  baseline.calculateStatFeatures(candidates, verbose=verbose)
  
  return baseline, candidates


def plotAttr(X,y, plotAttr, attrs):
  h = .02  # step size in the mesh
  plt.subplots_adjust(wspace=0.4, hspace=0.4)
  for i in range(0,len(attrs)):
    plt.subplot(len(attrs)/2,2,i+1)
    # Plot the data points
    plt.scatter(X[:,plotAttr], X[:,i],c=y, cmap=plt.cm.coolwarm)
    plt.xlabel(attrs[plotAttr])
    plt.ylabel(attrs[i])
  plt.show()

#TRAIN
baseline, candidates = getBaselineFeatures(DATA_FOLDER, addKeywords=False, verbose=True)
#divide some features by number of files in data set
#to make training and test set compatible
norm = ['TF']
baseline.normalizeFeatures(candidates, norm)
log_features = ['IDF']
baseline.logFeatures(candidates,log_features, base=2)
#attrs = ['length', 'TF', 'IDF', 'c_value', 'log_pp', 't_score', 'pmi', 'dice']
#attrs = ['c_value']
#attrs = ['length', 'syntaxClass', 'syntaxFrequency', 'pmi']
#attrs = ['length', 'TF', 'IDF', 'log_pp', 't_score', 'pmi', 'dice']
attrs = ['length', 'c_value', 'IDF', 'log_pp', 't_score', 'pmi', 'dice', 'syntaxClass', 'syntaxFrequency']

features_train, labels_train = baseline.asNumpy(candidates, baseline.ds_keywords, attrs)
print("Training set size, class 0:", countClass(features_train, labels_train, 0))
print("Training set size, class 1:", countClass(features_train, labels_train, 1))
#oversample keywords
features_train, labels_train = oversampleClass(features_train, labels_train, 1, 2.33)
#scale training set to 0 mean and unit variance
#scaler = StandardScaler()
#scaler.fit(features_train)
#scaler.transform(features_train)

print("Oversampled size, class 0:", countClass(features_train, labels_train, 0))
print("Oversampled size, class 1:", countClass(features_train, labels_train, 1))
#plotAttr(features_train, labels_train, 1, attrs)
#exit()

#TEST
baseline2, candidates2 = getBaselineFeatures(TEST_FOLDER, addKeywords=False, verbose=True, syntaxDict=baseline.syntaxDict)
#divide some features by number of files in data set
#to make training and test set compatible
baseline2.normalizeFeatures(candidates2, norm)
baseline2.logFeatures(candidates2,log_features, base=2)
features_test, labels_test = baseline2.asNumpy(candidates2, baseline2.ds_keywords, attrs)
#scaler.transform(features_test)
s = baseline2.calculateStatistics(candidates2, baseline2.ds_keywords)
print("Recall", f.format(s.recall))
np_recall = s.recall
print("Precision", f.format(s.precision))
print("F1", f.format(s.F1))

#Scorer
#create scorer that maximises F1 of keyword=yes class
kwargs={'np_recall':np_recall}
score = make_scorer(keyword_f1_function, greater_is_better=True, **kwargs)

#Classify
#Definition of grid search parameters
#class 0: keyword=no, class 1: keyword=yes
param_grid = [
  {
    'C': [1, 10, 100], 
    'kernel': ['linear'], 
    'class_weight':[{0:1.0, 1:1.0},{0:1.0, 1:1.25},{0:1.0, 1:1.5}]
  },
  {
    'C': [1, 10, 100], 
    'gamma': [0.1,0.01,0.001], 
    'kernel': ['rbf'],
    'class_weight':[{0:1.0, 1:1.0},{0:1.0, 1:1.25},{0:1.0, 1:1.5}]
  }
]
param_grid_MLP = [
  {
    'hidden_layer_sizes': [(4,), (8,), (16,),(3,), (40,), (100,)], 
    'activation': ['tanh','relu'], 
    'alpha':[0.01,0.001,0.0001],
    'solver':['adam'],
    'random_state':[5]
  },
  {
    'hidden_layer_sizes': [(4,), (8,), (16,),(3,), (40,), (100,)], 
    'activation': ['tanh','relu'], 
    'alpha':[0.01,0.001,0.0001],
    'solver':['sgd'],
    'learning_rate':['constant','adaptive'],
    'early_stopping':[True],
    'random_state':[5]
  }
]
param_randomForest = [
  {
    'n_estimators': [100], 
    'max_features': ['sqrt',None], 
    'min_samples_leaf': [1,3,5],
    'random_state':[5]
  }
]
k = 10
print("Support Vector Classifier Grid Search with ",k,"-fold cross validation")
print("Using Attributes:")
print(attrs)
#clf = GridSearchCV(svm.SVC(), param_grid, cv=k,scoring=score, verbose=1)
#clf = GridSearchCV(svm.SVC(), param_grid, cv=k,scoring="accuracy", verbose=2)
#clf = GridSearchCV(MLPClassifier(), param_grid_MLP, cv=k,scoring="accuracy", verbose=1)
clf = GridSearchCV(RandomForestClassifier(), param_randomForest, cv=k,scoring="accuracy", verbose=1)

clf.fit(features_train, labels_train)
print("Best parameters set found on development set:")
print(clf.best_params_)

print("Detailed classification report:")
pred = clf.predict(features_test)
print("Accuracy Score:")
print(accuracy_score(labels_test , pred))
print("Confusion Matrix:")
print(confusion_matrix(labels_test , pred, labels=[1,0]))

print(classification_report(labels_test , pred, labels=[1,0], target_names=['yes','no']))
print("F1 score with regards to all keywords:")
print(keyword_f1_function(labels_test , pred, np_recall=np_recall))


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