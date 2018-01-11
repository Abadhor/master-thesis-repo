


import tensorflow as tf
import numpy as np

sent_length = 2
hidden_size = 10
vocab_size = 8
batch_size = 64
LSTM_layer_count = 1

session = tf.Session()

#sentences = tf.placeholder(tf.float32, [None, sent_length, vocab_size])

#labels = tf.placeholder(tf.float32, [None, sent_length])

sentences = tf.placeholder(tf.float32, [None, sent_length, 3])

labels = tf.placeholder(tf.float32, [None, sent_length, 2])

W = tf.constant([[[1,0,0],[0,0,1]]], dtype=tf.float32)
cur_batch_size = tf.shape(sentences)[0]
W = tf.tile(W, multiples=[cur_batch_size,1,1])
b = tf.constant([0.5,1], dtype=tf.float32)

res = tf.transpose(tf.matmul(W, tf.transpose(sentences, perm=[0,2,1])),perm=[0,2,1]) + b

correct_prediction = tf.equal(tf.argmax(res,2), tf.argmax(labels,2))
sent_sum = tf.reduce_sum(tf.cast(correct_prediction, tf.float32),1)
correct_sent = tf.equal(sent_sum, tf.cast(tf.shape(correct_prediction[1]), tf.float32))
#loss = accuracy =  tf.reduce_mean(tf.cast(correct_sent, tf.float32))

loss = tf.nn.softmax_cross_entropy_with_logits(logits=res, labels=labels)

#loss = tf.reduce_sum(loss, 1)

#loss = tf.reduce_mean(loss)

#loss = tf.reduce_mean(loss)

batch_sent = np.array([
                        [[100,0,0],[0,0,100]],
                        [[0,0,1],[0,0,1]],
                        [[0,0,0],[0,0,0]]
                        ])

batch_labels = np.array([
                        [[0,1],[0,1]],
                        [[0,1],[0,1]],
                        [[0,0],[0,0]]
                        ])

indices = tf.constant([[0, 0], [1, 1]], tf.int64)

mask = tf.reduce_max(sentences, 2)

#params = tf.constant([['a', 'b'], ['c', 'd']], tf.string)
#params = tf.constant(['a', 'b', 'c', 'd'], tf.string)

#g = tf.gather(params, indices)

#print(session.run(g))

np_emb = np.reshape(np.arange(3*2), (3,2))

emb= tf.constant(np_emb, dtype=tf.float32)

indices = [75,76,77,78]
tokens = tf.constant(indices, dtype=tf.int32)
limit = tf.constant(76, dtype=tf.int32)
t2 = tokens - limit
t3 = tokens * limit
t_ft = tf.nn.embedding_lookup(emb, t2)


#print(session.run(loss, data))

data = {sentences: batch_sent, labels: batch_labels}

list = session.run([emb, tokens, t2, t3, t_ft])
for idx,v in enumerate(list):
  print(idx,v)




"""
DECORATOR
>>> def some_bs(func):
...   def bs(text):
...     return "bs"
...   return bs
...
>>> @some_bs
... def hello(text):
...   return "Hello " + text
...
>>> hello("World")
'bs'
"""


"""
IF THERE WAS BROADCASTING FOR MATMUL


shape IF x3 == y2
(x1,x2,x3) x (y1,y2,y3) => (y1, x2, y3)

1 dimension from x taken, 2 from y


(batch_size, sent_size, depth)
y: we want batch_size and sent_size(1<->2)
(batch_size, depth, sent_size)
use transpose [0,2,1] for this


(1, depth, num_classes)
x: we want num_classes (1<->2)
(1, num_classes, depth)
use transpose [0,2,1] for this

(1, num_classes, depth) x (batch_size, depth, sent_size) => (batch_size, num_classes, sent_size)
transpose [0,2,1]
(batch_size, sent_size, num_classes)
"""

