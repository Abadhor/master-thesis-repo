import io
import pickle
import numpy as np

mwt_file = "D:/data/datasets/patent_mwt/mwts/mwts.dict"
with io.open(mwt_file, 'rb') as fp:
  mwt_dict = pickle.load(fp)

print("Unique MWTS:", len(mwt_dict))
print("count 1 MWTS:", len([k for k,v in mwt_dict.items() if v == 1]))
print("count >= 5 MWTS:", len([k for k,v in mwt_dict.items() if v >= 5]))
v = np.array(list(mwt_dict.values()), dtype='float32')
print("Mean count:", np.mean(v))
print("Stddev count:", np.std(v))
x = [len(' '.join(m.split('-')).split()) for m in list(mwt_dict.keys())]
v = np.array(x, dtype='float32')
print("Mean len:", np.mean(v))
print("Stddev len:", np.std(v))