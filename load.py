import numpy as np
import os
import theano
import cv2

_imsize_ = (28, 28)


def shared_dataset (data_xy, borrow=True):             
            data_x, data_y = data_xy
            shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                    borrow=borrow)
            shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                    borrow=borrow)        
            return shared_x, theano.tensor.cast(shared_y, 'int32')      

class train_dataset ():
    def __init__(self, trainset, validset):
        self.train_set  = np.random.permutation(trainset)
        self.validation_set  = np.random.permutation(validset)
        self.train_set = shared_dataset((self.train_set[:,:-1].astype(theano.config.floatX)/255.0, self.train_set[:,-1]))
        self.validation_set = shared_dataset((self.validation_set[:,:-1].astype(theano.config.floatX)/255.0, self.validation_set[:,-1]))

def zeropadding (inputs, batchsize):
    # checks needed
    realsize = inputs.shape[0]
    if (batchsize > realsize):
        inputs = np.vstack((inputs, np.zeros((batchsize-realsize, inputs.shape[1])))).astype(theano.config.floatX)        
    return inputs
        
class test_dataset ():
    def __init__ (self, testset):
        self.apply_set = testset[:,:-1].astype(theano.config.floatX)/255.0
    
def loadDataClass (classname, path, altname=None):
    path = os.path.join(path,classname)
    filenames = filter(os.path.isfile, [os.path.join(path,f) for f in os.listdir(path)])
    chars  = map(lambda (x): cv2.imread(x,0).flatten(), filenames)
    if (chars):
        inputs = np.vstack(chars)
        output = np.array(([altname or classname]*inputs.shape[0])).astype(np.int32).reshape((inputs.shape[0],1))
        images = np.hstack((inputs,output))
        return images

def loadData (classlist, path, altname=None):
    chars = filter(lambda (ch): ch != None,
                    map(lambda (cl): loadDataClass(cl, path, altname), classlist))
    return np.vstack(chars)
