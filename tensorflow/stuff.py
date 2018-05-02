


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



def input_fn():
  dataset = tf.data.Dataset.range(10)
  dataset = dataset.map(lambda x: ({'x':x},[1]))
  return dataset.make_one_shot_iterator().get_next()

def my_model_fn(
   features, # This is batch_features from input_fn
   labels,   # This is batch_labels from input_fn
   mode,     # An instance of tf.estimator.ModeKeys
   params):  # Additional configuration
  print("Here is stuff\n\n\nHere too!")
  print(params)
  print(features)
  logits = tf.constant([-1,1], dtype=tf.float32)
  print("1")
  print(features['x'].shape)
  print(logits.shape)
  predicted_classes = tf.argmax(logits, 0)
  print("2")
  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
          'class_ids': predicted_classes,
          'probabilities': tf.nn.softmax(logits),
          'logits': logits,
    }
    return tf.estimator.EstimatorSpec(mode, predictions=predictions)
  # Compute loss.
  var1 = tf.get_variable("log", shape=(1,), dtype=tf.float32)
  loss = tf.cast(features['x'], dtype=tf.float32) + var1
  # Compute evaluation metrics.
  print("3")
  accuracy = tf.metrics.accuracy(labels=labels,
                               predictions=predicted_classes,
                               name='acc_op')
  print("3.1")
  metrics = {'accuracy': accuracy}
  print("3.2")
  tf.summary.scalar('accuracy', accuracy[1])
  print("4")
  if mode == tf.estimator.ModeKeys.EVAL:
      return tf.estimator.EstimatorSpec(
          mode, loss=loss, eval_metric_ops=metrics)
  optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
  train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
  return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

class MyHook(tf.train.SessionRunHook):
  
  def __init__(self):
    super().__init__()
    self.i = 0
  
  def after_run(self, run_context, run_values):
    print(self.i)
    if self.i > 3:
      run_context.request_stop()
    self.i += 1

# Build 2 hidden layer DNN with 10, 10 units respectively.
classifier = tf.estimator.Estimator(
    model_fn=my_model_fn,
    model_dir='./tmp/',
    params={
        'feature_columns': 'my_feature_columns',
        # Two hidden layers of 10 nodes each.
        'hidden_units': [10, 10],
        # The model must choose between 3 classes.
        'n_classes': 3,
    })
classifier.train(input_fn, hooks=[MyHook()], steps=5)
classifier.evaluate(input_fn)
classifier.predict(input_fn)

# filter1 = tf.get_variable("char_cnn_filter1",shape=
      # [
        # params['char_cnn_filter_width'],
        # in_depth,
        # out_depth
      # ], dtype=tf.float32)
filter1 = tf.constant(
    [
      [
        [1,0,0],
        [0,1,0]
      ],
      [
        [0,0,2],
        [0,0,0]
      ],
      [
        [1,0,0],
        [0,0,0]
      ]
    ],
    dtype=tf.float32)
input = tf.constant(
    [
      [
        [0,1],
        [1,0],
        [0,1],
        [1,0],
        [0,1]
      ]
    ],
    dtype=tf.float32)
char_cnn1 = tf.nn.conv1d(
        input,
        filter1,
        1,
        padding='SAME',
        data_format="NWC",
        name="cnn1"
      )
s.run(char_cnn1)
