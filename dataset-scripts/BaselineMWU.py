from SemEval2017Dataset import SemEval2017Dataset
from BaselineFeatures import BaselineFeatures
from sklearn import svm
from sklearn.neural_network import MLPClassifier
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
DEBUG = False
f = "{:.4}"
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
  for i in range(0,math.floor(factor)):
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


def getNP(ds, addKeywords, verbose=False):
    
  baseline = BaselineFeatures(ds, addKeywords=addKeywords)
  print("Tagging...")
  candidates = baseline.extractNounPhrases(verbose=verbose)

  print("Number of Candidates: ", len(candidates))
  
  print("Filtering Formulas...")
  candidates = baseline.filterFormulas(candidates)
  candidates = baseline.filterLength(candidates)
  
  return {x.name for x in candidates}

def getBaselineFeaturesNGramms(ds, min_length=2, max_length=2, verbose=False):
  
  baseline = BaselineFeatures(ds, addKeywords=False)
  print("Tagging...")
  candidates = baseline.extractNGramms(min_length=min_length, max_length=max_length, verbose=verbose)

  print("Number of Candidates: ", len(candidates))
  
  print("Filtering Formulas...")
  candidates = baseline.filterFormulas(candidates)
  candidates = baseline.filterLength(candidates)
  
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
ds = SemEval2017Dataset()
ds.extractKeywords(DATA_FOLDER, "ann")
ds.filterUnigramKeywords()
ds.dumpKeywords(OUT_FOLDER_KWDS, "keywords")

ds.extractSentences(DATA_FOLDER, "txt", xml=False, verbose=True)
ds.dumpSentences(OUT_FOLDER, "sents")

b_train, c_train = getBaselineFeaturesNGramms(ds, min_length=2, max_length=6, verbose=True)
np_train = getNP(ds, addKeywords=True, verbose=True)
s = b_train.calculateStatistics(c_train, np_train)
print("Recall", f.format(s.recall))
np_recall = s.recall
print("Precision", f.format(s.precision))
print("F1", f.format(s.F1))
#oversample rate -> set so equal number of true and false labels
os_rate = 1/s.precision - 2

#divide some features by number of files in data set
#to make training and test set compatible
norm = ['TF']
b_train.normalizeFeatures(c_train, norm)
log_features = ['IDF']
b_train.logFeatures(c_train,log_features, base=2)
attrs = ['length', 'TF', 'IDF', 'log_pp', 't_score', 'pmi', 'dice']

features_train, labels_train = b_train.asNumpy(c_train, np_train, attrs)
print("Not NP: ", len([x for x in labels_train if x == 0]))
print("NP: ", len([x for x in labels_train if x == 1]))
#oversample keywords
features_train, labels_train = oversampleClass(features_train, labels_train, 1, os_rate)
print("Oversample NP: ", len([x for x in labels_train if x == 1]))
#scale training set to 0 mean and unit variance
scaler = StandardScaler()
scaler.fit(features_train)
scaler.transform(features_train)


#TEST
ds = SemEval2017Dataset()
ds.extractKeywords(TEST_FOLDER, "ann")
ds.filterUnigramKeywords()
ds.dumpKeywords(OUT_FOLDER_KWDS, "keywords")

ds.extractSentences(TEST_FOLDER, "txt", xml=False, verbose=True)
ds.dumpSentences(OUT_FOLDER, "sents")

b_test, c_test = getBaselineFeaturesNGramms(ds, min_length=2, max_length=6, verbose=True)
np_test = getNP(ds, addKeywords=True, verbose=True)
s = b_test.calculateStatistics(c_test, np_test)
print("Recall", f.format(s.recall))
np_recall = s.recall
print("Precision", f.format(s.precision))
print("F1", f.format(s.F1))

#divide some features by number of files in data set
#to make training and test set compatible
b_test.normalizeFeatures(c_test, norm)
b_test.logFeatures(c_test,log_features, base=2)
attrs = ['length', 'TF', 'IDF', 'log_pp', 't_score', 'pmi', 'dice']

features_test, labels_test = b_test.asNumpy(c_test, np_test, attrs)
#oversample keywords
#scale training set to 0 mean and unit variance
scaler.transform(features_test)

#Scorer
#create scorer that maximises F1 of keyword=yes class
kwargs={'np_recall':1}
score = make_scorer(keyword_f1_function, greater_is_better=True, **kwargs)

#Classify
#Definition of grid search parameters
#class 0: keyword=no, class 1: keyword=yes
param_grid = [
  {
    'C': [1, 10, 100], 
    'kernel': ['linear'], 
    'class_weight':[{0:1.0, 1:1.0},{0:1.0, 1:2.0},{0:1.0, 1:3.0}]
  },
  {
    'C': [1, 10, 100], 
    'gamma': [0.1,0.01,0.001], 
    'kernel': ['rbf'],
    'class_weight':[{0:1.0, 1:1.0},{0:1.0, 1:2.0},{0:1.0, 1:3.0}]
  }
]
param_grid_MLP = [
  {
    'hidden_layer_sizes': [(4,), (8,), (16,),(32,), (50,), (100,)], 
    'activation': ['tanh','relu'], 
    'alpha':[0.01,0.001,0.0001],
    'solver':['adam']
  },
  {
    'hidden_layer_sizes': [(4,), (8,), (16,),(32,), (50,), (100,)], 
    'activation': ['tanh','relu'], 
    'alpha':[0.01,0.001,0.0001],
    'solver':['sgd'],
    'learning_rate':['constant','adaptive'],
    'early_stopping':[True]
  }
]
k = 5
print("Multi Layer Perceptron Classifier Grid Search with ",k,"-fold cross validation")
#clf = GridSearchCV(svm.SVC(), param_grid, cv=k,scoring=score, verbose=2)
clf = GridSearchCV(MLPClassifier(), param_grid_MLP, cv=k,scoring=score, verbose=2)
#clf = svm.SVC(class_weight=class_weights)
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
print(keyword_f1_function(labels_test , pred, np_recall=1))
#plotAttr(features_train, labels_train, 0, attrs)

