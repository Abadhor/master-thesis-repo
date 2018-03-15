import tensorflow as tf
import spacy
import numpy as np

sess = tf.Session()
nlp = spacy.load('en_core_web_md')

def _fun(x):
  doc = nlp(x.decode('utf-8'))
  print([token.text for token in doc])
  return np.array([1,2,3])

filenames = ['./00.txt']
dataset = tf.data.TextLineDataset(filenames)
#dataset = dataset.map(lambda x: tf.py_func(_fun, [x],[tf.int32]))
get_next = dataset.make_one_shot_iterator().get_next()
shape = tf.shape(get_next)

for i in range(5):
  print(sess.run(get_next))

def tokenize(text):
  doc = nlp(str(text))
  return [token.text for token in doc]

def vectorize(tokens):
  vector = [1 for word in tokens]
  vector = vector[:10]
  return vector


def _featurize_py_func(text):
    print(text)
    tokens = tokenize(text)
    print(tokens)
    #vector = vectorize(tokens)
    #return np.array(vector, dtype=np.int32)
    return tokens


dataset = tf.data.TextLineDataset(filenames)
    
dataset = dataset.map(lambda text: tf.py_func(_featurize_py_func, [text], [tf.string]))
get_next = dataset.make_one_shot_iterator().get_next()


#def _incArr(x):
#  return [[x,x+1]]

def _incArr(x):
  text = "This is a long sentence."
  doc = nlp(text)
  ls = [token.text.encode('utf-8') for token in doc]
  ls.append(str(x).encode('utf-8'))
  return np.array(ls, dtype='object')

inc_dataset = tf.data.Dataset.range(5)
inc_dataset = inc_dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(tf.py_func(_incArr, [x], tf.string)))
get_next = inc_dataset.make_one_shot_iterator().get_next()
sess.run(get_next)

const = tf.constant(['First sent','', 'Second'], dtype=tf.string)
dataset = tf.data.Dataset.from_tensor_slices(['First sent','', 'Second'])
get_next = dataset.make_one_shot_iterator().get_next()
sess.run(get_next)



