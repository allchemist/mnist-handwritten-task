import os
import sys
import cPickle
import numpy as np
import load
import lenet5

_max_epochs_ = 40
digs = ['0','1','2','3','4','5','6','7','8', '9']

for dig in digs:
    print dig+' char'
    good_train = np.random.permutation(load.loadData([dig], os.path.join('data', 'train'), '0'))
    bad_train  = np.random.permutation(load.loadData(filter(lambda x: x != dig, digs), os.path.join('data', 'train'), '1'))
    good_train = good_train[:len(good_train)/100*100]
    bad_train = bad_train[:len(good_train)]
    good_valid = np.random.permutation(load.loadData([dig], os.path.join('data', 'valid'), '0'))
    bad_valid  = np.random.permutation(load.loadData(filter(lambda x: x != dig, digs), os.path.join('data', 'valid'), '1'))
    good_valid = good_valid[:len(good_valid)/100*100]
    bad_valid = bad_valid[:len(good_valid)]
    data = load.train_dataset(np.vstack((good_train, bad_train)), np.vstack((good_valid, bad_valid)))
    model = lenet5.Model(batch_size=100, outsize=2)
    model.loadData(data)
    model.prepareLearning()
    with open(os.path.join('nets', str(dig)+'.obj'), 'wb') as fp:
        cPickle.dump(model.train(_max_epochs_), fp, protocol=cPickle.HIGHEST_PROTOCOL)
