import os
import sys
import cv2
import numpy as np
import cPickle as pickle
import load

_dig_classlist_ = ['0','1','2','3','4','5','6','7','8','9'] 
rawdata = load.loadData(_dig_classlist_, os.path.join('data', 'test'))

data = load.test_dataset(rawdata)
with open('nets/diff.obj', 'rb') as fp:
    diff_model = pickle.load(fp)

bin_models = []
for dig in _dig_classlist_:
    with open('nets/'+str(dig)+'.obj', 'rb') as fp:
        bin_models.append(pickle.load(fp))
        
def apply_bins (data):
    outs = np.array((10000,10))
    for i in xrange(10):
        outs[i] = 
    
    
    def apply_single (model, data):
        out = np.array(10000)
        for i in xrange(100):
            out[i*100 : (i+1)*100] = model.apply(data, i)
        return out

    
