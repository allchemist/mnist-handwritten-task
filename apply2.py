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
    model = pickle.load(fp)


def apply_whole (model, data):
    out = []
    for i in xrange(100):
        res = model.apply(data, i).tolist()
        out += res
    return out

def apply_team (models, data):
    res = []
    for m in models:
        res.append(np.array(apply_whole(m, data)))
    return np.sum(res, axis=0)/len(models)

answers = rawdata[:,-1]

def test_whole (diff_model, data, answers):
    diff_out = apply_model(diff_model, data)
    my_answers = np.zeros(10000)
    for i in xrange(10000):
        my_answers[i] = np.argmax(diff_out[i])
    print len(filter(lambda x: x != 0, my_answers-answers))
    return my_answers
