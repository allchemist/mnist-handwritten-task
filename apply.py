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

def img (arr, flag=True):
    res = (arr*255).astype(np.uint8).reshape((28,28))
    if not flag:
        res = 255-res
    return res

def show (img, name="image"):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return()

def group_imgs (arrs):
    ncols = 100
    nrows = len(arrs)/ncols
    rows = []
    for i in xrange(ncols):
        row = []
        res = map(np.argmax, model.apply(data, i))
        orig = answers[i*100: (i+1)*100]
        for j in xrange(nrows):
            row.append(img(arrs[i*100+j], res[j] == orig[j]))
        rows.append(np.vstack(row))
    return np.hstack(rows)

def draw_diff (path='diff_out.png'):
    answers = rawdata[:,-1]
    global fail_count
    fail_count = 0
    def fn (x,y):
        global fail_count
        if x != y:
            fail_count += 1
    for i in xrange(100):
        res = model.apply(data, i)
        res = map(np.argmax, res)
        orig = answers[i*100: (i+1)*100]
        tmp = map(fn, res, orig)
    cv2.imwrite(path, group_imgs(data.apply_set))
    return (10000-fail_count)/100.

def apply_whole (model, data):
    out = []
    for i in xrange(100):
        res = model.apply(data, i).tolist()
        out += res
    return out

def padding (arr):
    return np.vstack(map(lambda x: arr, range(100)))

import theano

def test_whole (diff_model, bin_models, data, thr=0.5):
    susp = []
    diff_out = apply_whole(diff_model, data)
    my_answers = np.zeros(10000)
    for i in xrange(10000):
        if np.max(diff_out[i]) > thr:
            my_answers[i] = np.argmax(diff_out[i])
        else:
            susp.append(np.argmax(diff_out[i]))
            bin_outs = []
            for dig in xrange(10):
                b_out = bin_models[dig].apply_model(padding(data.apply_set[i]).astype(theano.config.floatX))[0][0]
                bin_outs.append(b_out)
            my_answers[i] = np.argmax(bin_outs)
    print susp
    print len(susp)
    return my_answers

my_answers = test_whole(diff_model, bin_models, data, 0.7)
len(filter(lambda x: x != 0, my_answers-answers))
