import os
import sys
import cPickle
import numpy as np
import load
import lenet5

_max_epochs_ = 50
_dig_netpath_ = os.path.join('nets','diff.obj')
_dig_classlist_ = ['0','1','2','3','4','5','6','7','8','9'] 



dataset = load.train_dataset(load.loadData(_dig_classlist_, os.path.join('data', 'train')),
                             load.loadData(_dig_classlist_, os.path.join('data', 'valid')))

### digits train

print 'Training digits difference classifier'
model = lenet5.Model(batch_size=100, outsize=10)
model.loadData(dataset)
model.prepareLearning()

with open(_dig_netpath_, 'wb') as fp:
    cPickle.dump(model.train(_max_epochs_), fp, protocol=cPickle.HIGHEST_PROTOCOL)
