import tensorflow as tf


class TFLogger:
  
  def __init__(self, summary_names, session):
    self.log_ops = {}
    self.session = session
    # register summaries
    for name in summary_names:
      if name not in self.log_ops:
        ph = tf.placeholder(dtype=tf.float32, shape=(), name='ph_'+name)
        summary = tf.summary.scalar(
            name=name,
            tensor=ph
        )
        self.log_ops[name] = ph
    self.merged = tf.summary.merge_all()
  
  def create_writer(self, log_dir):
    return tf.summary.FileWriter(log_dir, self.session.graph)
  
  def log(self, writer, log_dict, step):
    feed_dict = {}
    for name, value in log_dict.items():
      if name not in self.log_ops:
        continue
      # fill the placeholders
      feed_dict[self.log_ops[name]] = value
    fetches = {'summary':self.merged}
    fetched = self.session.run(fetches, feed_dict=feed_dict)
    writer.add_summary(fetched['summary'], global_step=step)
    writer.flush()