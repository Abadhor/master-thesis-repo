
import math

class StatFeatures:
  
  def __init__(self, dictionary):
    self.c = dictionary
    self.N = sum(dictionary.values())
  
  def expected(self, tokens):
    res = 1
    for w in tokens:
      if w not in self.c:
        self.c[w] = 1
      res *= self.c[w] / self.N
    res *= self.N
    return res
  
  def t_score(self, tokens, freq):
    return (freq - self.expected(tokens)) / math.pow(freq , 1/2)
  
  def pmi(self, tokens, freq):
    return math.log(freq / self.expected(tokens) , 2)
  
  def dice(self, tokens, freq):
    sum = 0
    for w in tokens:
      sum += self.c[w]
    return (len(tokens) * freq) / sum
  
  
  