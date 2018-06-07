import io
import pickle

mwt_file = "D:/data/datasets/patent_mwt/mwts/mwts.set"
with io.open(mwt_file, 'rb') as fp:
  mwt_dict = pickle.load(fp)

print("Unique MWTS:", len(mwt_dict))
print("len 1 MWTS:", len([k for k,v in mwt_dict.items() if v == 1]))
print("len >= 5 MWTS:", len([k for k,v in mwt_dict.items() if v >= 5]))
print("Mean len:", sum(mwt_dict.values())/len(mwt_dict))