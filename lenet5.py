# -*- coding: utf-8 -*-
import numpy as np
import time
import sys
import theano
import theano.tensor as T
import theano.printing as P
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

rng = np.random.RandomState(23455)

class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=np.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX),
                                name='W', borrow=True)
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=np.zeros((n_out,),
                                                 dtype=theano.config.floatX),
                               name='b', borrow=True)

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        
        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        
        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()	
            
    #def __getstate__(self):
    #    (self.W.get_value(),self.b.get_value())
    #def __setstate__(self,data):
    #    W,b = data
    #    (self.W.set_value(W),self.b.set_value(b))
        
class HiddenLayer(object):
    def __init__(self, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        self.input = input
        if W is None:
            W_values = np.asarray(rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values, name='W', borrow=True)
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
            
        self.W = W
        self.b = b
        
        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        # parameters of the model
        self.params = [self.W, self.b]
    #    def __getstate__(self):
    #        (self.W.get_value(),self.b.get_value())
    #    def __setstate__(self,data):
    #        W,b = data
    #        (self.W.set_value(W),self.b.set_value(b))
	
class LeNetConvPoolLayer(object):    

    def __init__(self, input, filter_shape, image_shape, poolsize=(2, 2)):        
        assert image_shape[1] == filter_shape[1]
        self.input = input

        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
                   np.prod(poolsize))        
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(np.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)
        
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)
        
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape)
        
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize, ignore_border=True)

        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]
        #def __getstate__(self):
        #    (self.W.get_value(),self.b.get_value())
        #def __setstate__(self,data):
        #    W,b = data
        #    (self.W.set_value(W),self.b.set_value(b))    
    
class Model(object):                    
    def __init__(self,ishape=(28,28),nkerns=[20, 50],batch_size=100, outsize=10):    
        self.index = T.lscalar()  # index to a [mini]batch
        self.x = T.matrix('x')   # the data is presented as rasterized images
        self.y = T.ivector('y') 
        self.batch_size=batch_size
        self.layer0_input = self.x.reshape((self.batch_size, 1, 28, 28))
        self.layer0 = LeNetConvPoolLayer(input=self.layer0_input,
            image_shape=(batch_size, 1, 28, 28),
            filter_shape=(nkerns[0], 1, 5, 5), poolsize=(2, 2))

        self.layer1 = LeNetConvPoolLayer(input=self.layer0.output,
            image_shape=(batch_size, nkerns[0], 12, 12),
            filter_shape=(nkerns[1], nkerns[0], 5, 5), poolsize=(2, 2))

        self.layer2_input = self.layer1.output.flatten(2)

    
        self.layer2 = HiddenLayer(input=self.layer2_input, n_in=nkerns[1] * 4 * 4,
                         n_out=500, activation=T.tanh)

        self.layer3 = LogisticRegression(input=self.layer2.output, n_in=500, n_out=outsize)

        self.cost = self.layer3.negative_log_likelihood(self.y)
        self.params = self.layer3.params + self.layer2.params + self.layer1.params + self.layer0.params    
        self.grads = T.grad(self.cost, self.params)        
        self.apply_model = theano.function([self.x],self.layer3.p_y_given_x)
    def loadData(self,dataset):    
        self.dataset = dataset
    def prepareLearning(self,learning_rate=0.05):    
        self.n_train_batches = self.dataset.train_set[0].get_value(borrow=True).shape[0]
        self.n_valid_batches = self.dataset.validation_set[0].get_value(borrow=True).shape[0]        
        self.n_train_batches /= self.batch_size
        self.n_valid_batches /= self.batch_size        
        
        train_set_x, train_set_y = self.dataset.train_set
        valid_set_x, valid_set_y = self.dataset.validation_set               
        
        self.validate_model = theano.function([self.index], self.layer3.errors(self.y),
            givens={
                self.x: valid_set_x[self.index * self.batch_size: (self.index + 1) * self.batch_size],
                self.y: valid_set_y[self.index * self.batch_size: (self.index + 1) * self.batch_size]})

        updates = []
        for param_i, grad_i in zip(self.params, self.grads):
            updates.append((param_i, param_i - learning_rate * grad_i))

        self.train_model = theano.function([self.index], self.cost, updates=updates,
          givens={
                self.x: train_set_x[self.index * self.batch_size: (self.index + 1) * self.batch_size],
                self.y: train_set_y[self.index * self.batch_size: (self.index + 1) * self.batch_size]})       
        #self.apply_model = theano.function([self.x],self.layer3.p_y_given_x) # Замени y_pred на p_y_given_x если нужны оценки вероятности классо
    def prepareDumping():
        self.dataset = None
    def train(self,n_epochs,callback=None):
            patience = 10000  # look as this many examples regardless
            patience_increase = 2  # wait this much longer when a new best is
                           # found
            improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
            validation_frequency = min(self.n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

            best_params = None
            best_validation_loss = np.inf
            best_iter = 0
            test_score = 0.
            start_time = time.clock()

            epoch = 0
            done_looping = False

            while (epoch < n_epochs) and (not done_looping):
                epoch = epoch + 1
                for minibatch_index in xrange(self.n_train_batches):

                    iter = (epoch - 1) * self.n_train_batches + minibatch_index

#                    if iter % 100 == 0:
#                        print 'training @ iter = ', iter

                    cost_ij = self.train_model(minibatch_index)

                    if (iter + 1) % validation_frequency == 0:

                        # compute zero-one loss on validation set
                        validation_losses = [self.validate_model(i) for i
                                             in xrange(self.n_valid_batches)]
                        this_validation_loss = np.mean(validation_losses)
#                        print('epoch %i, minibatch %i/%i, validation error %f %%' % \
#                              (epoch, minibatch_index + 1, self.n_train_batches, \
#                               this_validation_loss * 100.))

                        # if we got the best validation score until now
                        if this_validation_loss < best_validation_loss:

                            #improve patience if loss improvement is good enough
                            if this_validation_loss < best_validation_loss *  \
                               improvement_threshold:
                                patience = max(patience, iter * patience_increase)

                            # save best validation score and iteration number
                            best_validation_loss = this_validation_loss
                            best_iter = iter
                            
                            #print(('     epoch %i, error of best '
                                  # 'model %f %%') %
                            print (epoch, minibatch_index + 1, self.n_train_batches, best_validation_loss * 100.)

                    if patience <= iter:
                        done_looping = True
                        break
                if callable(callback):
                        done_looping = callback(epoch,best_validation_loss) # Здесь можно подавать в callback все, что угодно, равно как и останавливать обучение
            end_time = time.clock()
            print('Optimization complete.')
            print('Best validation score of %f %% obtained at iteration %i' %
                  (best_validation_loss * 100., best_iter + 1))
            return self      
            #print >> sys.stderr, ('The code for file ' +
            #                      os.path.split(__file__)[1] +
            #                      ' ran for %.2fm' % ((end_time - start_time) / 60.))            
    def apply(self,data, pos):
        results = self.apply_model(data.apply_set[pos*100:(pos+1)*100].astype(theano.config.floatX))
        #results = results[0:data.apply_size]
        return results
        
    def __getstate__(self):
        return (self.layer0.W.get_value(),self.layer0.b.get_value(),
                self.layer1.W.get_value(),self.layer1.b.get_value(),
                self.layer2.W.get_value(),self.layer2.b.get_value(),
                self.layer3.W.get_value(),self.layer3.b.get_value(),self.batch_size)
    def __setstate__(self,data):
        (l0W,l0b,l1W,l1b,l2W,l2b,l3W,l3b,batch_size) = data
        self.__init__(batch_size=batch_size)
        self.layer0.W.set_value(l0W)
        self.layer0.b.set_value(l0b)
        self.layer1.W.set_value(l1W)
        self.layer1.b.set_value(l1b)
        self.layer2.W.set_value(l2W)
        self.layer2.b.set_value(l2b)
        self.layer3.W.set_value(l3W)
        self.layer3.b.set_value(l3b)        
        self.apply_model = theano.function([self.x],self.layer3.p_y_given_x)   
        
#def chunks(l, n):
#    """ Yield successive n-sized chunks from l.
#    """
#    for i in xrange(0, len(l), n):
#        yield l[i:i+n]
